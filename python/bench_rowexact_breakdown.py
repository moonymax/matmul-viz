"""Where does rowexact's wall-clock go? Profile each sub-step (D=2048, MPS).

Tells us what a fused kernel should actually target (matmuls vs topk vs gather).
"""
from __future__ import annotations

import time
import torch

D = 2048
K = 9216
P = 96
B = 32
WARMUP, REPS = 30, 200
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def sync():
    if DEV.type == "mps":
        torch.mps.synchronize()


def timed(fn, *a):
    for _ in range(WARMUP):
        fn(*a)
    sync(); t0 = time.perf_counter()
    for _ in range(REPS):
        r = fn(*a)
    sync()
    return (time.perf_counter() - t0) / REPS, r


def main():
    torch.manual_seed(0)
    x = torch.randn(B, D, device=DEV)
    da = torch.randn(B, D, device=DEV)

    print(f"device={DEV}  D={D}  B={B}  P(rows)={P}  K={K}  reps={REPS}\n")

    # step 1: Gram M = x x^T
    t_gram, M = timed(lambda: x @ x.t())
    # step 2: row scores RS = (da * (M @ da)).sum(0)
    t_score, RS = timed(lambda: (da * (M @ da)).sum(0))
    # step 3: top-P rows
    t_topr, topR = timed(lambda: torch.topk(RS, P).indices)
    # step 4: gather + exact rows Grows = da[:, topR].t() @ x
    t_grows, Grows = timed(lambda: da[:, topR].t() @ x)
    # step 5: top-K cells among the p x D strip
    t_topk, loc = timed(lambda: torch.topk(Grows.abs().reshape(-1), K).indices)
    # step 6: map to global indices
    t_idx, _ = timed(lambda: topR[loc // D] * D + (loc % D))

    # full rowexact (single fused-in-python call) and bare GEMM for reference
    def rowexact():
        M = x @ x.t()
        RS = (da * (M @ da)).sum(0)
        tR = torch.topk(RS, P).indices
        Gr = da[:, tR].t() @ x
        lc = torch.topk(Gr.abs().reshape(-1), K).indices
        return tR[lc // D] * D + (lc % D)
    t_all, _ = timed(rowexact)
    t_gemm, _ = timed(lambda: da.t() @ x)

    steps = [
        ("1 Gram  x@x^T", t_gram),
        ("2 score (M@da)", t_score),
        ("3 topk rows", t_topr),
        ("4 exact rows (matmul)", t_grows),
        ("5 topk cells (p*D)", t_topk),
        ("6 index map", t_idx),
    ]
    print(f"  {'step':>24} {'time(us)':>10} {'% of steps-sum':>15}")
    ssum = sum(t for _, t in steps)
    for name, t in steps:
        print(f"  {name:>24} {1e6*t:>10.1f} {100*t/ssum:>14.1f}%")
    print(f"  {'-'*24} {'-'*10}")
    print(f"  {'sum of steps':>24} {1e6*ssum:>10.1f}")
    print(f"  {'rowexact (one call)':>24} {1e6*t_all:>10.1f}")
    print(f"  {'bare GEMM da^T@x':>24} {1e6*t_gemm:>10.1f}")


if __name__ == "__main__":
    main()
