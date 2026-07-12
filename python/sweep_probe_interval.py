"""Does the dense-gradient probe interval (delta_T) actually matter?

The exact DynamicSparseLinear now runs an O(k) forward and only materializes the
dense weight gradient (G = da^T @ x) on probe steps, to pick where to grow. The
open question: does probing OFTEN help, or does probing every 100 steps land in
the same place as probing every step? If frequency doesn't matter, the "expensive
dense probe" is amortized to nothing and we can probe rarely for free.

Same too-big synthetic problem as the other sweeps (D=128 isotropic teacher,
2-layer tanh student, exact grow-and-prune at a fixed density). We sweep the probe
interval and report final PURE data MSE + total probes + wall-clock. Bracketed by
static (never grow) as the floor and full dense as the ceiling.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamic_sparse import (
    DynamicSparseLinear, ApproxDynamicSparseLinear,
    set_capture, rigl_update, cosine_drop_fraction,
)

D = 128
SAMPLES, BATCH, STEPS, LR = 4096, 128, 1500, 1e-3
DENSITY = 0.10
APPROX_TOPK = 48
SEED = 0
INTERVALS = [1, 5, 10, 25, 50, 100, 200, 500]   # None = static (never grow)


class Big(nn.Module):
    def __init__(self, mk):
        super().__init__(); self.W1 = mk(); self.W2 = mk()
    def forward(self, x):
        return torch.tanh(self.W2(torch.tanh(self.W1(x))))


class FullDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(D, D, bias=False); self.W2 = nn.Linear(D, D, bias=False)
    def forward(self, x):
        return torch.tanh(self.W2(torch.tanh(self.W1(x))))


def make(kind, k):
    gen = torch.Generator().manual_seed(SEED + 2)
    cls = DynamicSparseLinear if kind == "exact" else ApproxDynamicSparseLinear
    return Big(lambda: cls(D, D, k, init_scale=math.sqrt(D / k), generator=gen))


def train(model, X, Y, interval):
    """interval None -> static (no grow-and-prune). Else probe every `interval`.
    rigl_update handles both layer types (exact uses the dense-grad probe, approx
    the O(k) top-p x top-q estimate)."""
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(SEED + 7)
    probes = 0
    t0 = time.time()
    for step in range(STEPS):
        idx = torch.randint(0, SAMPLES, (BATCH,), generator=g)
        frac = cosine_drop_fraction(step, STEPS)
        probe = interval is not None and step % interval == 0 and frac > 0
        if probe:
            set_capture(model, True)
        opt.zero_grad(set_to_none=True)
        F.mse_loss(model(X[idx]), Y[idx]).backward()
        if probe:
            rigl_update(model, opt, frac, topk=APPROX_TOPK)
            set_capture(model, False)
            probes += 1
        opt.step()
    with torch.no_grad():
        mse = F.mse_loss(model(X), Y).item()
    return mse, probes, time.time() - t0


def main():
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED + 999); s = 1.0 / math.sqrt(D)
    A1 = torch.randn(D, D, generator=g) * s; A2 = torch.randn(D, D, generator=g) * s
    gx = torch.Generator().manual_seed(SEED + 1); X = torch.randn(SAMPLES, D, generator=gx)
    Y = torch.tanh(torch.tanh(X @ A1.t()) @ A2.t()).detach()
    k = round(DENSITY * D * D)

    dense_mse, _, _ = train(FullDense(), X, Y, None)

    print(f"too-big teacher D={D}, grow-and-prune @ density {DENSITY:.0%} "
          f"(k={k}), {STEPS} steps\n")
    print(f"  dense ceiling : {dense_mse:.5f}\n")

    for kind in ("exact", "approx"):
        stat_mse, _, _ = train(make(kind, k), X, Y, None)  # never grow = static mask
        print(f"  [{kind}]  static (no grow-and-prune): {stat_mse:.5f}")
        print(f"  {'probe every':>12} {'probes':>7} {'final MSE':>10} {'vs static':>10} {'time(s)':>8}")
        for iv in INTERVALS:
            mse, probes, dt = train(make(kind, k), X, Y, iv)
            rel = (mse / stat_mse - 1) * 100
            print(f"  {iv:>12} {probes:>7} {mse:>10.5f} {rel:>+9.1f}% {dt:>8.2f}")
        print()


if __name__ == "__main__":
    main()
