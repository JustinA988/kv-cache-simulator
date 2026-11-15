"""Run: python test_kvsim.py  (or pytest)"""
import numpy as np

from kvsim import BlockPool, PagedCache, attend, batch_capacity, fake_model, shared_prefix, simulate


def test_paged_matches_contiguous_and_nocache():
    outs = [simulate(s, 333, gen_len=20, block_size=16)["outputs"] for s in ("nocache", "contiguous", "paged")]
    assert np.allclose(outs[0], outs[1]) and np.allclose(outs[1], outs[2])


def test_hit_rates():
    assert simulate("nocache", 200, 10)["hit_rate"] == 0
    assert simulate("paged", 200, 10)["hit_rate"] == 1
    assert simulate("paged+window", 5000, 10, window=1024)["hit_rate"] < 0.25


def test_window_keeps_exactly_last_tokens_and_bounds_memory():
    K, V, Q = fake_model(3000, 1)
    pool = BlockPool(100, 16)
    c = PagedCache(pool, window=500)
    for i in range(3000):
        c.append(K[i], V[i])
        assert pool.in_use <= 500 // 16 + 2
    k, v, start = c.view()
    assert start == 2500 and np.array_equal(k, K[2500:]) and np.array_equal(v, V[2500:])
    assert np.allclose(attend(Q[0], k, v), attend(Q[0], K[2500:], V[2500:]))


def test_pool_free_and_exhaustion():
    pool = BlockPool(4, 8)
    c = PagedCache(pool)
    z = np.zeros(pool.k.shape[-1])
    for _ in range(32):
        c.append(z, z)
    try:
        c.append(z, z)
        raise AssertionError("expected MemoryError")
    except MemoryError:
        pass
    c.free()
    assert pool.in_use == 0


def test_paged_waste_under_one_block():
    r = simulate("paged", 1000, 7, block_size=16)
    assert r["waste_pct"] < 16 / 1007 * 100
    assert simulate("contiguous", 1000, 7)["waste_pct"] > 80


def test_prefix_sharing_refcounts():
    pool = BlockPool(10, 4)
    parent = PagedCache(pool)
    z = np.zeros(pool.k.shape[-1])
    for _ in range(10):  # 2 full blocks + half a block
        parent.append(z, z)
    child, reused = PagedCache.from_prefix(parent, 10)
    assert reused == 8 and child.table == parent.table[:2]
    child.append(z, z)  # child's tail goes in a new private block
    assert child.table[2] != parent.table[2] and pool.in_use == 4
    parent.free()
    assert pool.in_use == 3  # shared blocks survive while child holds them
    child.free()
    assert pool.in_use == 0


def test_shared_prefix_saves_memory_same_outputs():
    s = shared_prefix(n_requests=8, prefix_len=500, suffix_len=50)
    assert np.allclose(s["private"]["outputs"], s["shared"]["outputs"])
    assert s["shared"]["blocks"] < s["private"]["blocks"] / 3
    assert s["shared"]["computed"] < s["private"]["computed"] / 3


def test_batch_capacity():
    assert batch_capacity(20_480, [100] * 50, 16) == (2, 50)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
