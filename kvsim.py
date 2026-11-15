"""KV cache simulator: no cache vs contiguous vs paged (vLLM-style) vs paged + sliding window.

Run:  python kvsim.py            (benchmark, prompt lengths 100..10,000)
      python kvsim.py --help
"""
import argparse
import math
from dataclasses import dataclass

import numpy as np

SIM_DIM = 64          # shortcut: stored arrays are one small head; memory figures use ModelCfg below
MAX_CONTEXT = 10_240  # max sequence length the server supports
C_COMPUTE = 1.0       # shortcut: latency cost units, compute K/V for one token (all layers)
C_READ = 0.02         #           read one cached token's K/V from memory. Replace with measured numbers for a real GPU.


@dataclass
class ModelCfg:  # Llama-2-7B-like, fp16. Used only for byte accounting.
    layers: int = 32
    heads: int = 32
    head_dim: int = 128
    dtype_bytes: int = 2

    @property
    def bytes_per_token(self):
        return 2 * self.layers * self.heads * self.head_dim * self.dtype_bytes  # K and V


# ---------------------------------------------------------------- caches

class ContiguousCache:
    """Pre-vLLM style: reserve one contiguous max_len slab per sequence up front."""

    def __init__(self, max_len=MAX_CONTEXT, dim=SIM_DIM):
        self.k = np.zeros((max_len, dim), np.float32)
        self.v = np.zeros((max_len, dim), np.float32)
        self.n = 0

    def append(self, k, v):
        if self.n == len(self.k):
            raise MemoryError("sequence exceeded its reserved slab")
        self.k[self.n], self.v[self.n] = k, v
        self.n += 1

    def view(self):
        """(K, V, logical position of first cached token)."""
        return self.k[:self.n], self.v[:self.n], 0

    @property
    def slots_reserved(self):
        return len(self.k)

    @property
    def slots_used(self):
        return self.n


class BlockPool:
    """Shared physical memory split into fixed-size blocks, with a free list."""

    def __init__(self, num_blocks, block_size, dim=SIM_DIM):
        self.block_size = block_size
        self.k = np.zeros((num_blocks, block_size, dim), np.float32)
        self.v = np.zeros((num_blocks, block_size, dim), np.float32)
        self.free = list(range(num_blocks - 1, -1, -1))

    def alloc(self):
        if not self.free:
            raise MemoryError("block pool exhausted")
        return self.free.pop()

    def release(self, block):
        self.free.append(block)

    @property
    def in_use(self):
        return len(self.k) - len(self.free)


class PagedCache:
    """One sequence: a block table mapping logical blocks -> physical blocks in a shared pool.

    Memory grows one block at a time, so waste is at most block_size-1 slots in the last block.
    With `window`, blocks that fall entirely outside the last `window` tokens are returned to the
    pool (rolling buffer), so memory stays bounded however long generation runs.
    """

    def __init__(self, pool, window=None):
        self.pool, self.window = pool, window
        self.table = []       # physical block ids, oldest first
        self.n = 0            # logical length (tokens ever appended)
        self.first_block = 0  # logical block index of table[0]

    def append(self, k, v):
        bs = self.pool.block_size
        if self.n % bs == 0:
            self.table.append(self.pool.alloc())
        b, off = self.table[-1], self.n % bs
        self.pool.k[b, off], self.pool.v[b, off] = k, v
        self.n += 1
        if self.window:
            while (self.first_block + 1) * bs <= self.n - self.window:
                self.pool.release(self.table.pop(0))
                self.first_block += 1

    def view(self):
        bs, dim = self.pool.block_size, self.pool.k.shape[-1]
        start = self.first_block * bs
        held = self.n - start
        k = self.pool.k[self.table].reshape(-1, dim)[:held]  # gather blocks via the table
        v = self.pool.v[self.table].reshape(-1, dim)[:held]
        if self.window and held > self.window:  # partially-evicted first block: mask the stale part
            cut = held - self.window
            k, v, start = k[cut:], v[cut:], start + cut
        return k, v, start

    def free(self):
        for b in self.table:
            self.pool.release(b)
        self.table = []

    @property
    def slots_reserved(self):
        return len(self.table) * self.pool.block_size

    @property
    def slots_used(self):
        return self.n - self.first_block * self.pool.block_size


# ---------------------------------------------------------------- generation loop

def attend(q, k, v):
    s = k @ q / math.sqrt(len(q))
    w = np.exp(s - s.max())
    return (w / w.sum()) @ v


def fake_model(total, n_queries, seed=0):
    """Stand-in for the transformer: deterministic K, V per position and one query per decode step."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((total, SIM_DIM), dtype=np.float32),
            rng.standard_normal((total, SIM_DIM), dtype=np.float32),
            rng.standard_normal((n_queries, SIM_DIM), dtype=np.float32))


def simulate(strategy, prompt_len, gen_len=64, block_size=16, window=2048, cfg=ModelCfg(), seed=0):
    """strategy: 'nocache' | 'contiguous' | 'paged' | 'paged+window'. Returns metrics + attention outputs."""
    total = prompt_len + gen_len
    K, V, Q = fake_model(total, gen_len, seed)

    if strategy == "contiguous":
        cache = ContiguousCache()
    elif strategy in ("paged", "paged+window"):
        pool = BlockPool(math.ceil(MAX_CONTEXT / block_size), block_size)
        cache = PagedCache(pool, window if strategy == "paged+window" else None)
    elif strategy != "nocache":
        raise ValueError(strategy)

    computed = reads = requested = 0
    peak_reserved = peak_used = 0
    outputs = []

    if strategy != "nocache":  # prefill: compute the prompt's K/V once
        for p in range(prompt_len):
            cache.append(K[p], V[p])
        computed += prompt_len

    for step in range(gen_len):
        pos = prompt_len + step
        requested += pos + 1  # full attention wants every past token
        if strategy == "nocache":  # recompute the whole prefix every step
            k, v = K[:pos + 1], V[:pos + 1]
            computed += pos + 1
            peak_reserved = peak_used = max(peak_used, pos + 1)  # transient, freed after the step
        else:
            cache.append(K[pos], V[pos])
            computed += 1
            k, v, _ = cache.view()
            reads += len(k)
            peak_reserved = max(peak_reserved, cache.slots_reserved)
            peak_used = max(peak_used, cache.slots_used)
        outputs.append(attend(Q[step], k, v))

    mb = cfg.bytes_per_token / 2**20
    return {
        "strategy": strategy,
        "prompt": prompt_len,
        "peak_mem_mb": peak_reserved * mb,
        "hit_rate": reads / requested,                       # past K/V served from cache
        "waste_pct": 100 * (peak_reserved - peak_used) / peak_reserved,  # reserved but unused
        "latency_per_tok": (computed * C_COMPUTE + reads * C_READ) / gen_len,
        "outputs": np.array(outputs),
    }


def batch_capacity(budget_tokens, lengths, block_size):
    """How many of `lengths` (admitted in order) fit in memory at once.

    Contiguous must reserve MAX_CONTEXT per request because final length is unknown at admission;
    paged only holds the blocks actually filled.
    """
    contiguous = min(len(lengths), budget_tokens // MAX_CONTEXT)
    used = paged = 0
    for n in lengths:
        used += math.ceil(n / block_size) * block_size
        if used > budget_tokens:
            break
        paged += 1
    return contiguous, paged


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", type=int, nargs="+", default=[100, 500, 1000, 2000, 5000, 10000])
    ap.add_argument("--gen", type=int, default=64, help="tokens generated per run")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--window", type=int, default=2048)
    args = ap.parse_args()

    cfg = ModelCfg()
    print(f"Model: {cfg.layers} layers x {cfg.heads} heads x {cfg.head_dim} dim, fp16 "
          f"= {cfg.bytes_per_token / 2**20:.2f} MB of KV per token")
    print(f"block={args.block_size}  window={args.window}  generate={args.gen}  max_context={MAX_CONTEXT}\n")
    print(f"{'prompt':>7} {'strategy':<13} {'peak MB':>9} {'hit rate':>9} {'waste %':>8} {'latency/tok':>12}")
    for p in args.prompts:
        for s in ("nocache", "contiguous", "paged", "paged+window"):
            r = simulate(s, p, args.gen, args.block_size, args.window, cfg)
            print(f"{p:>7} {s:<13} {r['peak_mem_mb']:>9.0f} {r['hit_rate']:>9.1%} "
                  f"{r['waste_pct']:>8.1f} {r['latency_per_tok']:>12.1f}")
        print()

    budget = 80 * 2**30 // cfg.bytes_per_token  # tokens of KV that fit in 80 GB
    lengths = np.random.default_rng(1).integers(100, 10_000, 500).tolist()
    c, pg = batch_capacity(budget, lengths, args.block_size)
    print(f"Concurrent requests in 80 GB of KV memory (lengths 100-10,000): "
          f"contiguous={c}  paged={pg}  ({pg / max(c, 1):.1f}x)")


if __name__ == "__main__":
    main()
