"""Diagnose the sparse-forward slowdown (Sd=4096, k=1M). Isolate causes via env:
  SPARSE_FWD_MODE = global | cached | seq
  SPARSE_FWD_TG   = threadgroup size (parallelism / occupancy)
  - global vs cached      : DRAM-gather cost (already known ~2x)
  - cached vs seq          : random shared-mem access / bank conflicts
  - TG sweep on cached     : occupancy / parallelism
Reference: dense Linear forward (same MAC count as sparse, k = D_dense^2).
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "metal"))

D = 1024
SD = 4096
B = 256
K = D * D
WARMUP, REPS = 15, 60
DEV = torch.device("mps")


def sync():
    torch.mps.synchronize()


def timed(fn):
    for _ in range(WARMUP):
        fn()
    sync(); t0 = time.perf_counter()
    for _ in range(REPS):
        fn()
    sync()
    return 1e3 * (time.perf_counter() - t0) / REPS


def main():
    from sparse_metal import SparseLinearMetal
    torch.manual_seed(0)
    layer = SparseLinearMetal(SD, SD, K, generator=torch.Generator().manual_seed(0)).to(DEV)
    x = torch.randn(B, SD, device=DEV)

    # dense reference (same MACs: B*D^2 = B*k)
    W = torch.randn(D, D, device=DEV)
    xd = torch.randn(B, D, device=DEV)
    t_dense = timed(lambda: xd @ W.t())

    def run(mode, tg):
        os.environ["SPARSE_FWD_MODE"] = mode
        os.environ["SPARSE_FWD_TG"] = str(tg)
        with torch.no_grad():
            return timed(lambda: layer(x))

    print(f"device={DEV}  B={B}  Sd={SD}  k={K:,}  (dense-ref D={D}, same MACs)\n")
    print(f"  dense Linear forward: {t_dense:.3f} ms   (MACs = B*k = {B*K:,})\n")

    print("  mode / TG sweep (ms):")
    print(f"    {'config':>22} {'time(ms)':>10} {'vs dense':>9}")
    configs = [("global", 256), ("cached", 64), ("cached", 128), ("cached", 256),
               ("cached", 512), ("cached", 1024), ("seq", 256), ("seq", 1024)]
    for mode, tg in configs:
        t = run(mode, tg)
        label = f"{mode} TG={tg}" if mode != "global" else "global (uncached)"
        print(f"    {label:>22} {t:>10.3f} {t/t_dense:>8.1f}x")
    print("\n  cached vs seq (same TG) isolates random-shared-access/bank-conflict cost;")
    print("  TG sweep isolates occupancy; global vs cached isolates DRAM gather.")


if __name__ == "__main__":
    main()
