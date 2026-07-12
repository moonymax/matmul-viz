"""Isolate the contribution of each optimization to metal fwd+bwd.

Configs (env toggles read per backward call):
  A baseline : separate commits + serial grad_val   (FUSED=0, TG=0)
  B +WIN1    : fused single-commit backward, serial  (FUSED=1, TG=0)
  C +WIN1+2  : fused + threadgroup grad_val          (FUSED=1, TG=1)
"""
from __future__ import annotations
import os, sys, time
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sparse_metal import SparseLinearMetal  # noqa: E402

DEV = "mps"
RANK = 16


def timed(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    return 1000.0 * (time.perf_counter() - t0) / iters


def run():
    print(f"metal fwd+bwd (ms) by config; rank={RANK}")
    print("  A baseline=separate commits+serial gradval, B=+win1(fused commit),"
          " C=+win1+win2(tg gradval)")
    print(f"  {'n':>6}{'B':>6}{'k':>9}{'A base':>10}{'B +win1':>10}{'C +win1+2':>11}"
          f"{'win1 %':>9}{'win2 %':>9}")
    configs = [("A", "0", "0"), ("B", "1", "0"), ("C", "1", "1")]
    for n in (512, 1024, 2048):
        for B in (128, 512, 2048):
            k = 2 * n * RANK
            x = torch.randn(B, n, device=DEV)
            metal = SparseLinearMetal(n, n, k, generator=torch.Generator().manual_seed(2)).to(DEV)

            def fb():
                metal.zero_grad(set_to_none=True)
                out = metal(x.detach().requires_grad_(True))
                out.pow(2).sum().backward()

            res = {}
            for name, fused, tg in configs:
                os.environ["SPARSE_METAL_FUSED_BWD"] = fused
                os.environ["SPARSE_METAL_TG_GRADVAL"] = tg
                res[name] = timed(fb)
            win1 = 100.0 * (res["A"] - res["B"]) / res["A"]
            win2 = 100.0 * (res["B"] - res["C"]) / res["B"]
            print(f"  {n:>6}{B:>6}{k:>9}{res['A']:>10.3f}{res['B']:>10.3f}{res['C']:>11.3f}"
                  f"{win1:>8.1f}%{win2:>8.1f}%")
        print()


if __name__ == "__main__":
    run()
