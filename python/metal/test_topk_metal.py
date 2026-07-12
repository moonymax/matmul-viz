"""Correctness + speed of the Metal approx top-K vs torch.topk (MPS)."""
from __future__ import annotations

import time
import torch
from topk_metal import approx_topk

DEV = torch.device("mps")
P, D, K = 96, 2048, 9216
WARMUP, REPS = 30, 200


def bench(fn, v):
    for _ in range(WARMUP):
        fn(v)
    torch.mps.synchronize(); t0 = time.perf_counter()
    for _ in range(REPS):
        r = fn(v)
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / REPS, r


def main():
    torch.manual_seed(0)
    v = torch.randn(P, D, device=DEV).reshape(-1)
    n = v.numel()

    true_idx = torch.topk(v.abs(), K, sorted=False).indices
    true_set = set(true_idx.tolist())

    idx = approx_topk(v, K)
    got = set(idx.tolist())
    overlap = len(true_set & got) / K
    # energy captured by our set vs the true top-K set
    e_got = v[idx.long()].pow(2).sum().item()
    e_true = v[true_idx].pow(2).sum().item()

    print(f"n={n}  K={K}")
    print(f"  approx_topk: returned {idx.numel()} idx, overlap@K={overlap:.3f}, "
          f"energy vs true topK={e_got/e_true:.4f}")

    t_torch, _ = bench(lambda v: torch.topk(v.abs(), K, sorted=False).indices, v)
    t_metal, _ = bench(lambda v: approx_topk(v, K), v)
    print(f"  torch.topk : {1e6*t_torch:8.1f} us")
    print(f"  metal approx:{1e6*t_metal:8.1f} us   ({t_torch/t_metal:.1f}x faster)")


if __name__ == "__main__":
    main()
