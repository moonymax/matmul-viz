"""Fair test: full-rank small dense vs big sparse, at equal params.

Previously "dense but smaller" had to bottleneck a fixed 256->256 task, which
IS low-rank. To give the small dense model a genuinely FULL-RANK form, we make
the intrinsic problem low-dimensional:

  teacher:  y = tanh(A2 tanh(A1 x)),  A: d x d,  x in R^d      (dense, full-rank)

The BIG models get a FROZEN dense up-projection U (D x d) and operate in R^D:
  big(x) = tanh( Uup^T . ( W2 . tanh( W1 . (U x) ) ) )         (W: D x D)

Students (all predict R^d, tanh-tanh, param-matched where noted):
  small_dense : native d x d dense              2*d^2 params   [full rank, no low-rank]
  big_dense   : D x D dense (frozen lift)        2*D^2 params   [over-dimensioned ceiling]
  big_static  : D x D sparse, fixed random mask  2*k   params
  big_dynamic : D x D sparse + exact grow/prune  2*k   params
  big_approx  : D x D sparse + O(k) grow/prune   2*k   params

Param match: k = density * D^2 = d^2  ->  density = (d/D)^2.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from sparse_layer import SparseLinear
from dynamic_sparse import (
    DynamicSparseLinear, ApproxDynamicSparseLinear,
    set_capture, rigl_update, cosine_drop_fraction, approx_grow_prune,
)

D_SMALL = 64
D_BIG = 256
DENSITY = (D_SMALL / D_BIG) ** 2          # = 0.0625 -> k = d^2, param-matched
K = round(DENSITY * D_BIG * D_BIG)        # 4096 per big layer  (= d^2)
SAMPLES, BATCH, STEPS, LR = 4096, 128, 1500, 1e-3
RIGL_EVERY, APPROX_TOPK, SEED = 25, 48, 0


class SmallDense(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.l1 = nn.Linear(d, d, bias=False)
        self.l2 = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return torch.tanh(self.l2(torch.tanh(self.l1(x))))


class BigModel(nn.Module):
    """Operates in R^D via a frozen up/down projection U (D x d)."""

    def __init__(self, make_layer, U):
        super().__init__()
        self.register_buffer("U", U)          # (D, d), frozen
        self.W1 = make_layer()
        self.W2 = make_layer()

    def forward(self, x):                      # x: (B, d)
        z = x @ self.U.t()                     # (B, D) lift
        h1 = torch.tanh(self.W1(z))            # (B, D)
        a2 = self.W2(h1)                       # (B, D)
        u = a2 @ self.U                        # (B, d) unlift
        return torch.tanh(u)


def make_big_layer(kind, gen):
    scale = math.sqrt(D_BIG / K)
    if kind == "big_dense":
        return nn.Linear(D_BIG, D_BIG, bias=False)
    if kind == "big_static":
        return SparseLinear(D_BIG, D_BIG, K, init_scale=scale, generator=gen)
    if kind == "big_dynamic":
        return DynamicSparseLinear(D_BIG, D_BIG, K, init_scale=scale, generator=gen)
    if kind == "big_approx":
        return ApproxDynamicSparseLinear(D_BIG, D_BIG, K, init_scale=scale, generator=gen)
    raise ValueError(kind)


def n_params(m):
    # dynamic-sparse layers store exactly their k active weights as `values`;
    # frozen U is a buffer, so a plain trainable-parameter count is correct.
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build(kind, U, gen):
    if kind == "small_dense":
        return SmallDense(D_SMALL)
    return BigModel(lambda: make_big_layer(kind, gen), U)


def train(kind, U, X, Y):
    gen = torch.Generator().manual_seed(SEED + 2)
    model = build(kind, U, gen)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(SEED + 7)
    for step in range(STEPS):
        idx = torch.randint(0, SAMPLES, (BATCH,), generator=g)
        frac = cosine_drop_fraction(step, STEPS)
        probe = step % RIGL_EVERY == 0 and frac > 0
        if probe:
            set_capture(model, True)
        opt.zero_grad(set_to_none=True)
        F.mse_loss(model(X[idx]), Y[idx]).backward()
        if probe:
            if kind == "big_dynamic":
                rigl_update(model, opt, frac)
            elif kind == "big_approx":
                approx_grow_prune(model, opt, n_swap=max(1, round(frac * K)), topk=APPROX_TOPK)
            set_capture(model, False)
        opt.step()
    with torch.no_grad():
        return model, F.mse_loss(model(X), Y).item()


def main():
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED + 999)
    s = 1.0 / math.sqrt(D_SMALL)
    A1 = torch.randn(D_SMALL, D_SMALL, generator=g) * s
    A2 = torch.randn(D_SMALL, D_SMALL, generator=g) * s
    gx = torch.Generator().manual_seed(SEED + 1)
    X = torch.randn(SAMPLES, D_SMALL, generator=gx)
    Y = torch.tanh(torch.tanh(X @ A1.t()) @ A2.t()).detach()

    # frozen orthonormal up-projection U (D x d)
    gu = torch.Generator().manual_seed(SEED + 5)
    U, _ = torch.linalg.qr(torch.randn(D_BIG, D_SMALL, generator=gu))

    print(f"intrinsic dim d={D_SMALL} (full-rank dense teacher), lifted D={D_BIG}, "
          f"density={DENSITY:.2%} (k={K}/layer = d^2), {STEPS} steps\n")
    order = ["small_dense", "big_dense", "big_static", "big_dynamic", "big_approx"]
    for kind in order:
        model, mse = train(kind, U, X, Y)
        print(f"  {kind:>12}: params={n_params(model):>8,}  final MSE={mse:.5f}")


if __name__ == "__main__":
    main()
