"""Correctness + wall-clock of the moveable block-sparse layer vs dense (MPS).

Same setup as bench_step.py: dense-small (D, equal params), and block-sparse
(Sd, block b, n_active blocks with n_active*b^2 = D^2 params). Times fwd/bwd/step
+ grow-prune. The question: does the bmm-based block forward run near dense speed
(matmul units), unlike the ~95x scalar CSR kernel?
"""
from __future__ import annotations

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from block_sparse import BlockSparseLinear

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
D = 1024
SD = 4096
B = 256
K = D * D                       # equal params
WARMUP, REPS = 15, 60
RIGL_EVERY = 100
DROP_FRAC = 0.01


def sync():
    if DEV.type == "mps":
        torch.mps.synchronize()


def timed(fn, reps=REPS):
    for _ in range(WARMUP):
        fn()
    sync(); t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()
    return 1e3 * (time.perf_counter() - t0) / reps


def correctness():
    g = torch.Generator().manual_seed(0)
    lay = BlockSparseLinear(64, 48, block=16, n_active=(64 // 16) * (48 // 16), generator=g)
    x = torch.randn(8, 64)
    ref = x @ lay.to_dense().t()
    out = lay(x)
    print(f"  correctness (all blocks active): max abs diff {(out - ref).abs().max():.2e}")
    # partial
    lay2 = BlockSparseLinear(64, 48, block=16, n_active=3, generator=g)
    x2 = torch.randn(8, 64)
    print(f"  correctness (sparse blocks):     max abs diff "
          f"{(lay2(x2) - x2 @ lay2.to_dense().t()).abs().max():.2e}")


def dense_phases(width):
    lin = nn.Linear(width, width, bias=False).to(DEV)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3)
    x = torch.randn(B, width, device=DEV)
    go = torch.randn(B, width, device=DEV)

    def fwd():
        with torch.no_grad():
            return lin(x)

    def fb():
        lin.zero_grad(set_to_none=True)
        lin(x).backward(go)

    fb()
    return timed(fwd), max(timed(fb) - timed(fwd), 0.0), timed(lambda: opt.step())


def block_phases(block):
    n_active = K // (block * block)
    g = torch.Generator().manual_seed(0)
    lay = BlockSparseLinear(SD, SD, block, n_active, generator=g).to(DEV)
    opt = torch.optim.AdamW(lay.parameters(), lr=1e-3)
    x = torch.randn(B, SD, device=DEV)
    go = torch.randn(B, SD, device=DEV)

    def fwd():
        with torch.no_grad():
            return lay(x)

    def fb():
        lay.zero_grad(set_to_none=True)
        lay(x).backward(go)

    fb()
    t_f, t_fb, t_s = timed(fwd), timed(fb), timed(lambda: opt.step())
    n_swap = max(1, int(DROP_FRAC * n_active))
    lay.zero_grad(set_to_none=True); lay(x).backward(go)
    t_gp = timed(lambda: lay.grow_prune(opt, x, go, n_swap), reps=20)
    return t_f, max(t_fb - t_f, 0.0), t_s, t_gp, n_active


def main():
    print("== correctness ==")
    correctness()
    print(f"\n== wall-clock (MPS)  B={B}  dense D={D}  block Sd={SD}  k={K:,} ==\n")

    ds = dense_phases(D)
    dw = dense_phases(SD)
    print(f"  {'model':>16} {'fwd':>8} {'bwd':>8} {'step':>8} {'grow-pr':>9} {'total':>9} {'x dense-s':>10}")
    print(f"  {'dense-small(D)':>16} {ds[0]:>8.3f} {ds[1]:>8.3f} {ds[2]:>8.3f} {'-':>9} {sum(ds):>9.3f} {1.0:>9.1f}x")
    print(f"  {'dense-wide(Sd)':>16} {dw[0]:>8.3f} {dw[1]:>8.3f} {dw[2]:>8.3f} {'-':>9} {sum(dw):>9.3f} {sum(dw)/sum(ds):>9.1f}x")
    for block in (32, 64, 128, 256):
        bf, bb, bs, bgp, na = block_phases(block)
        amort = bf + bb + bs + bgp / RIGL_EVERY
        print(f"  {'block b='+str(block):>16} {bf:>8.3f} {bb:>8.3f} {bs:>8.3f} "
              f"{bgp/RIGL_EVERY:>9.3f} {amort:>9.3f} {amort/sum(ds):>9.1f}x   "
              f"(n_active={na}, grid {SD//block}x{SD//block})")


if __name__ == "__main__":
    main()
