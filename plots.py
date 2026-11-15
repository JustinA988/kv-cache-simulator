"""Regenerate docs/results.png:  python plots.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kvsim import shared_prefix, simulate

PROMPTS = [100, 250, 500, 1000, 2000, 3000, 5000, 7500, 10000]
STRATS = {"nocache": "#999999", "contiguous": "#d1495b", "paged": "#2e86ab", "paged+window": "#3bb273"}


def main():
    runs = {s: [simulate(s, p) for p in PROMPTS] for s in STRATS}
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    panels = [
        (ax[0, 0], "peak_mem_mb", "Peak KV memory (MB)", "Peak memory per sequence"),
        (ax[0, 1], "latency_per_tok", "Simulated cost units / token", "Decode latency per token"),
        (ax[1, 0], "hit_rate", "Fraction of context served from cache", "Cache hit rate"),
    ]
    for a, key, ylabel, title in panels:
        for s, color in STRATS.items():
            ls = "--" if s == "paged+window" else "-"
            a.plot(PROMPTS, [r[key] for r in runs[s]], ls, marker="o", ms=4, color=color, label=s)
        a.set(xscale="log", xlabel="Prompt length (tokens)", ylabel=ylabel, title=title)
        a.grid(alpha=0.3)
    ax[0, 1].set_yscale("log")
    ax[0, 0].legend()

    sp = shared_prefix()
    a = ax[1, 1]
    labels = ["KV memory (GB)", "Tokens computed (k)"]
    private = [sp["private"]["mem_mb"] / 1024, sp["private"]["computed"] / 1000]
    shared = [sp["shared"]["mem_mb"] / 1024, sp["shared"]["computed"] / 1000]
    x = range(len(labels))
    a.bar([i - 0.2 for i in x], private, 0.4, color="#d1495b", label="private copies")
    a.bar([i + 0.2 for i in x], shared, 0.4, color="#2e86ab", label="shared prefix blocks")
    a.set_xticks(list(x), labels)
    a.set_title(f"Prefix sharing: {sp['requests']} requests, {sp['prefix']}-token shared prompt")
    a.legend()
    a.grid(alpha=0.3, axis="y")

    fig.suptitle("KV cache simulator (Llama-2-7B-sized KV, block=16, window=2048, 64 generated tokens)")
    fig.tight_layout()
    Path("docs").mkdir(exist_ok=True)
    fig.savefig("docs/results.png", dpi=120)
    print("wrote docs/results.png")


if __name__ == "__main__":
    main()
