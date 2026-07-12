"""Wall-clock: rowexact grow-gradient approximation vs the full gradient (D=2048).

Compares, for a D x D layer with batch B, three ways to produce the grow signal:
  full-grad    : G = da^T @ x                       (the full gradient matmul)
  full+select  : G = da^T @ x, then top-K cells      (full grow pipeline)
  rowexact-p   : top-p rows by exact grad-row norm, those rows computed exactly
                 over all cols, then top-K cells      (the approximation)

Reports analytic ops AND measured wall-clock (median over many reps, after warmup;
device synchronized around each timed batch). rowexact has far fewer ops; whether
it is faster in wall-clock depends on kernel-launch / topk overhead vs the single
fused full matmul -- which is exactly what we want to see.
"""

from __future__ import annotations

import time

import torch

D = 2048
K = 9216                     # grow budget (cells)
BATCHES = [8, 32, 128]
ROWS = [96, 192]
WARMUP, REPS = 20, 100
SEED = 0


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def full_grad(x, da):
    return da.t() @ x


def full_select(x, da):
    G = da.t() @ x
    return torch.topk(G.abs().reshape(-1), K, largest=True).indices


def make_rowexact(p):
    def fn(x, da):
        B, Dloc = x.shape
        RS = (da * ((x @ x.t()) @ da)).sum(0)             # ||g_i||^2 via B x B Gram
        topR = torch.topk(RS, p).indices
        Grows = da[:, topR].t() @ x                        # (p, D) exact rows
        loc = torch.topk(Grows.abs().reshape(-1), K, largest=True).indices
        return topR[loc // Dloc] * Dloc + (loc % Dloc)
    return fn


def ops_full(B):
    return D * D * B


def ops_rowexact(B, p):
    return 2 * B * B * D + p * B * D                       # gram scores + p exact rows


def bench(fn, x, da, dev):
    for _ in range(WARMUP):
        fn(x, da)
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(REPS):
        fn(x, da)
    sync(dev)
    return (time.perf_counter() - t0) / REPS


def run(dev):
    torch.manual_seed(SEED)
    print(f"\n==== device = {dev} ====  (D={D}, grow budget K={K}, reps={REPS})")
    for B in BATCHES:
        x = torch.randn(B, D, device=dev)
        da = torch.randn(B, D, device=dev)
        t_full = bench(full_grad, x, da, dev)
        t_sel = bench(full_select, x, da, dev)
        fo = ops_full(B)
        print(f"\n  batch={B}:")
        print(f"    {'method':>14} {'ops':>14} {'%full':>7} {'time(us)':>10} {'vs full-grad':>12}")
        print(f"    {'full-grad':>14} {fo:>14,} {'100%':>7} {1e6*t_full:>10.1f} {'1.00x':>12}")
        print(f"    {'full+select':>14} {fo:>14,} {'100%':>7} {1e6*t_sel:>10.1f} "
              f"{t_full/t_sel:>11.2f}x")
        for p in ROWS:
            fn = make_rowexact(p)
            t = bench(fn, x, da, dev)
            o = ops_rowexact(B, p)
            print(f"    {'rowexact'+str(p):>14} {o:>14,} {100*o/fo:>6.1f}% {1e6*t:>10.1f} "
                  f"{t_full/t:>11.2f}x")


def main():
    run(torch.device("cpu"))
    if torch.backends.mps.is_available():
        run(torch.device("mps"))


if __name__ == "__main__":
    main()
