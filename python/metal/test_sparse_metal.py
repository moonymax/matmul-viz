"""Correctness tests + benchmark for the fused Metal sparse layer.

Run with:
  PATH="$PWD/.venv/bin:$PATH" ./.venv/bin/python python/metal/test_sparse_metal.py
"""

from __future__ import annotations

import math
import os
import sys
import time

import torch

# make python/ importable so we can grab the reference SparseLinear
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sparse_layer import SparseLinear                      # noqa: E402
from sparse_metal import SparseLinearMetal                 # noqa: E402

DEV = "mps"


def make_matched_pair(in_f, out_f, k, seed=0):
    """Build a reference SparseLinear (CPU) and a Metal layer (MPS) that share
    the exact same positions and values."""
    g = torch.Generator().manual_seed(seed)
    ref = SparseLinear(in_f, out_f, k, generator=g)
    metal = SparseLinearMetal.from_positions(
        in_f, out_f, ref.rows, ref.cols, ref.values.detach()
    )
    return ref, metal.to(DEV)


def test_correctness():
    print("=" * 70)
    print("CORRECTNESS  (Metal float32 on MPS  vs  reference SparseLinear CPU)")
    print("=" * 70)
    cases = [(6, 5, 7), (64, 48, 100), (128, 256, 800), (512, 512, 4096)]
    all_ok = True
    for in_f, out_f, k in cases:
        ref, metal = make_matched_pair(in_f, out_f, k, seed=in_f)
        B = 32
        xg = torch.Generator().manual_seed(123)
        x_cpu = torch.randn(B, in_f, generator=xg)
        x_ref = x_cpu.clone().requires_grad_(True)
        x_met = x_cpu.to(DEV).clone().requires_grad_(True)

        out_ref = ref(x_ref)
        out_met = metal(x_met)
        fdiff = (out_met.cpu() - out_ref).abs().max().item()

        # dense reference: x @ W.T
        W = ref.to_dense()
        dense = x_cpu @ W.t()
        ddiff = (out_met.cpu() - dense).abs().max().item()

        # backward on same loss out.pow(2).sum()
        out_ref.pow(2).sum().backward()
        out_met.pow(2).sum().backward()

        # grad_values: reference stores values in (rows,cols) order; metal in CSR
        # order. Map ref grads into CSR order via the same perm metal used.
        from sparse_metal import build_index_structures
        idx = build_index_structures(ref.rows, ref.cols, in_f, out_f)
        gv_ref_csr = ref.values.grad[idx["csr_perm"]]
        gvdiff = (metal.values.grad.cpu() - gv_ref_csr).abs().max().item()
        gxdiff = (x_met.grad.cpu() - x_ref.grad).abs().max().item()

        ok = max(fdiff, ddiff, gvdiff, gxdiff) < 1e-3
        all_ok &= ok
        print(f"  in={in_f:>4} out={out_f:>4} k={k:>5}  "
              f"fwd={fdiff:.2e} dense={ddiff:.2e} dval={gvdiff:.2e} dx={gxdiff:.2e}  "
              f"{'OK' if ok else 'FAIL'}")
    print(f"\n  ALL CORRECTNESS: {'PASS' if all_ok else 'FAIL'}\n")
    return all_ok


def bench():
    print("=" * 70)
    print("BENCHMARK on MPS  (ms, mean over timed iters; lower is better)")
    print("=" * 70)

    def sync():
        torch.mps.synchronize()

    def timed(fn, warmup=10, iters=50):
        for _ in range(warmup):
            fn()
        sync()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        sync()
        return 1000.0 * (time.perf_counter() - t0) / iters

    RANK = 16
    B = 128
    rows = ["n", "k", "density",
            "dense.f", "gath.f", "metal.f",
            "dense.fb", "gath.fb", "metal.fb"]
    print("  " + "".join(f"{h:>10}" for h in rows))

    for n in (512, 1024, 2048):
        k = 2 * n * RANK
        density = k / (n * n)
        g = torch.Generator().manual_seed(0)

        x = torch.randn(B, n, device=DEV)

        dense = torch.nn.Linear(n, n, bias=False).to(DEV)
        gath = SparseLinear(n, n, k, generator=torch.Generator().manual_seed(1)).to(DEV)
        metal = SparseLinearMetal(n, n, k, generator=torch.Generator().manual_seed(2)).to(DEV)

        def fwd(m):
            def f():
                with torch.no_grad():
                    m(x)
            return f

        def fwdbwd(m):
            def f():
                m.zero_grad(set_to_none=True)
                out = m(x.detach().requires_grad_(True))
                out.pow(2).sum().backward()
            return f

        d_f = timed(fwd(dense)); g_f = timed(fwd(gath)); m_f = timed(fwd(metal))
        d_fb = timed(fwdbwd(dense)); g_fb = timed(fwdbwd(gath)); m_fb = timed(fwdbwd(metal))

        vals = [f"{n}", f"{k}", f"{density:.3%}",
                f"{d_f:.3f}", f"{g_f:.3f}", f"{m_f:.3f}",
                f"{d_fb:.3f}", f"{g_fb:.3f}", f"{m_fb:.3f}"]
        print("  " + "".join(f"{v:>10}" for v in vals))
    print()


if __name__ == "__main__":
    torch.manual_seed(0)
    if not torch.backends.mps.is_available():
        print("MPS not available"); sys.exit(1)
    ok = test_correctness()
    bench()
    sys.exit(0 if ok else 1)
