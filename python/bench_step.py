"""End-to-end training-STEP wall-clock: dense vs bigger-sparse (norm-gate grow).

Per the phase analysis:
  forward : dense_small does B*D^2; sparse (wide, k active) does O(B*k). At equal
            params (k = D_small^2) these are ~equal mults -- sparse's win is
            representational (approximates a D_wide^2 operator with k params), not
            fewer flops. dense_wide shows the operator sparse stands in for.
  backward: dense computes the full outer-product grad; sparse computes grads only
            at active cells (O(B*k)).
  step    : dense = AdamW over D^2; sparse = AdamW over k (cheap) PLUS grow-prune:
            (a) deadness search over k active, (b) norm-gate candidate gen + exact
            grad on candidates + top-K search, (c) rewrite positions (CSR rebuild).
            These searches are the sparse-only tax; grow-prune runs every T steps.

We time each phase (warmup + many reps + device sync) for:
  dense_small (D)   equal params to sparse
  dense_wide  (Sd)  the operator sparse approximates
  sparse      (Sd, k=D^2 active)   fused-Metal fwd/bwd + norm-gate grow-prune
"""
from __future__ import annotations

import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "metal"))
from sparse_metal import SparseLinearMetal, build_index_structures  # noqa: E402

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
D = 1024                 # dense-small width (params = D^2)
SD = 4096                # sparse/dense-wide width
B = 256
K = D * D                # active weights (equal params to dense-small)
DENSITY = K / (SD * SD)
DROP_FRAC = 0.01         # grow-prune swap fraction -> n_swap
GATE_C_MULT = 10         # norm-gate candidate pool = GATE_C_MULT * n_swap
RIGL_EVERY = 100         # grow-prune period (amortization)
WARMUP, REPS = 10, 50


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
    return 1e3 * (time.perf_counter() - t0) / reps       # ms


def dense_phases(width):
    lin = nn.Linear(width, width, bias=False).to(DEV)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3)
    x = torch.randn(B, width, device=DEV, requires_grad=True)
    go = torch.randn(B, width, device=DEV)

    def fwd():
        with torch.no_grad():
            return lin(x)

    def fwd_bwd():
        lin.zero_grad(set_to_none=True)
        if x.grad is not None: x.grad = None
        out = lin(x); out.backward(go)

    fwd_bwd()  # populate grads for step
    t_fwd = timed(fwd)
    t_fb = timed(fwd_bwd)
    t_step = timed(lambda: opt.step())
    return t_fwd, max(t_fb - t_fwd, 0.0), t_step


def sparse_phases():
    gen = torch.Generator().manual_seed(0)
    layer = SparseLinearMetal(SD, SD, K, generator=gen).to(DEV)
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    x = torch.randn(B, SD, device=DEV, requires_grad=True)
    go = torch.randn(B, SD, device=DEV)

    def fwd():
        with torch.no_grad():
            return layer(x)

    def fwd_bwd():
        layer.zero_grad(set_to_none=True)
        if x.grad is not None: x.grad = None
        out = layer(x); out.backward(go)

    fwd_bwd()
    t_fwd = timed(fwd)
    t_fb = timed(fwd_bwd)
    t_step = timed(lambda: opt.step())

    # ---- grow-prune (norm gate), timed as one block ----
    n_swap = max(1, int(DROP_FRAC * K))
    Cpool = min(GATE_C_MULT * n_swap, SD * SD)

    # capture a real (x, da) for the gate
    layer.zero_grad(set_to_none=True)
    out = layer(x); out.backward(go)
    xb = x.detach()
    # da = grad wrt layer output; recompute cleanly
    xr = x.detach().requires_grad_(False)
    out2 = layer(xr)
    da = go  # for a linear-ish layer the incoming grad IS da

    def grow_prune():
        rows = layer.csr_rowidx.long(); cols = layer.csr_col.long()
        vals = layer.values.detach()
        # (a) deadness search over k active: |value| (proxy) -> smallest n_swap
        _, prune_slots = torch.topk(vals.abs(), n_swap, largest=False)
        # (b) norm-gate candidate pool: top-Cpool cells by ||da[:,i]||*||x[:,j]||
        r = da.norm(dim=0); c = xb.norm(dim=0)                 # (Sd,), (Sd,)
        bound = (r.unsqueeze(1) * c.unsqueeze(0)).reshape(-1)   # Sd^2 (materialized)
        cand = torch.topk(bound, Cpool).indices                # search
        cr, cc = cand // SD, cand % SD
        # exact gradient on candidates
        gcand = (da[:, cr] * xb[:, cc]).sum(0)                  # Cpool*B
        # top n_swap candidates by |grad| (search); (skip active-overlap check cost here)
        grow = torch.topk(gcand.abs(), n_swap).indices
        gr, gc = cr[grow], cc[grow]
        # (c) rewrite pruned positions -> new cells, rebuild CSR
        rows[prune_slots] = gr; cols[prune_slots] = gc
        vals2 = vals.clone(); vals2[prune_slots] = 0.0
        idx = build_index_structures(rows, cols, SD, SD)       # CSR rebuild (argsort)
        return idx

    t_gp = timed(grow_prune, reps=10)
    return t_fwd, max(t_fb - t_fwd, 0.0), t_step, t_gp, n_swap, Cpool


def main():
    print(f"device={DEV}  B={B}  dense-small D={D} ({D*D:,} params)  "
          f"sparse/wide Sd={SD}  k={K:,} (density {DENSITY:.1%})")
    print(f"grow-prune: n_swap={int(DROP_FRAC*K):,}  gate pool={GATE_C_MULT}x  every {RIGL_EVERY}\n")

    ds_f, ds_b, ds_s = dense_phases(D)
    dw_f, dw_b, dw_s = dense_phases(SD)
    sp_f, sp_b, sp_s, sp_gp, n_swap, Cpool = sparse_phases()

    step_ds = ds_f + ds_b + ds_s
    step_dw = dw_f + dw_b + dw_s
    step_sp_base = sp_f + sp_b + sp_s
    step_sp_amort = step_sp_base + sp_gp / RIGL_EVERY

    print(f"  {'model':>16} {'fwd':>8} {'bwd':>8} {'step':>8} {'grow-prune':>11} {'step total':>11}")
    print(f"  {'dense-small(D)':>16} {ds_f:>8.3f} {ds_b:>8.3f} {ds_s:>8.3f} {'-':>11} {step_ds:>11.3f}")
    print(f"  {'dense-wide(Sd)':>16} {dw_f:>8.3f} {dw_b:>8.3f} {dw_s:>8.3f} {'-':>11} {step_dw:>11.3f}")
    print(f"  {'sparse(base)':>16} {sp_f:>8.3f} {sp_b:>8.3f} {sp_s:>8.3f} {'-':>11} {step_sp_base:>11.3f}")
    print(f"  {'sparse(amort)':>16} {'':>8} {'':>8} {'':>8} {sp_gp/RIGL_EVERY:>11.3f} {step_sp_amort:>11.3f}")
    print(f"\n  grow-prune full cost: {sp_gp:.2f} ms (every {RIGL_EVERY} steps)")
    print(f"  vs dense-small step: sparse-amort is {step_sp_amort/step_ds:.2f}x")
    print(f"  vs dense-wide step:  sparse-amort is {step_sp_amort/step_dw:.2f}x  "
          f"(the operator it approximates)")


if __name__ == "__main__":
    main()
