"""Hold the full-rank small-dense baseline fixed; sweep the EXACT grow-and-prune
sparse density upward and watch how fast it overtakes dense as params grow.

Same lifted setup as test_lifted_sparse.py: intrinsic d=64 full-rank dense
teacher, frozen up-projection to D=256, big model W1/W2 in R^D.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamic_sparse import (
    DynamicSparseLinear, set_capture, rigl_update, cosine_drop_fraction,
)

D_SMALL, D_BIG = 64, 256
SAMPLES, BATCH, STEPS, LR = 4096, 128, 1500, 1e-3
RIGL_EVERY, SEED = 25, 0
DENSITIES = [0.0625, 0.08, 0.10, 0.125, 0.1875, 0.25, 0.375, 0.5, 1.0]


class SmallDense(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.l1 = nn.Linear(d, d, bias=False)
        self.l2 = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return torch.tanh(self.l2(torch.tanh(self.l1(x))))


class BigDynamic(nn.Module):
    def __init__(self, U, k, gen):
        super().__init__()
        self.register_buffer("U", U)
        scale = math.sqrt(D_BIG / k)
        self.W1 = DynamicSparseLinear(D_BIG, D_BIG, k, init_scale=scale, generator=gen)
        self.W2 = DynamicSparseLinear(D_BIG, D_BIG, k, init_scale=scale, generator=gen)

    def forward(self, x):
        z = x @ self.U.t()
        h1 = torch.tanh(self.W1(z))
        a2 = self.W2(h1)
        return torch.tanh(a2 @ self.U)


def data_and_U():
    g = torch.Generator().manual_seed(SEED + 999)
    s = 1.0 / math.sqrt(D_SMALL)
    A1 = torch.randn(D_SMALL, D_SMALL, generator=g) * s
    A2 = torch.randn(D_SMALL, D_SMALL, generator=g) * s
    gx = torch.Generator().manual_seed(SEED + 1)
    X = torch.randn(SAMPLES, D_SMALL, generator=gx)
    Y = torch.tanh(torch.tanh(X @ A1.t()) @ A2.t()).detach()
    gu = torch.Generator().manual_seed(SEED + 5)
    U, _ = torch.linalg.qr(torch.randn(D_BIG, D_SMALL, generator=gu))
    return X, Y, U


def run(model, X, Y, dynamic=False):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(SEED + 7)
    for step in range(STEPS):
        idx = torch.randint(0, SAMPLES, (BATCH,), generator=g)
        frac = cosine_drop_fraction(step, STEPS)
        probe = dynamic and step % RIGL_EVERY == 0 and frac > 0
        if probe:
            set_capture(model, True)
        opt.zero_grad(set_to_none=True)
        F.mse_loss(model(X[idx]), Y[idx]).backward()
        if probe:
            rigl_update(model, opt, frac)
            set_capture(model, False)
        opt.step()
    with torch.no_grad():
        return F.mse_loss(model(X), Y).item()


def sparse_params(model):
    return sum(m.values.numel() for m in model.modules()
               if isinstance(m, DynamicSparseLinear))


def main():
    torch.manual_seed(SEED)
    X, Y, U = data_and_U()

    small = SmallDense(D_SMALL)
    base_mse = run(small, X, Y)
    base_params = sum(p.numel() for p in small.parameters())
    print(f"baseline  small_dense (full-rank {D_SMALL}x{D_SMALL}): "
          f"params={base_params:,}  MSE={base_mse:.5f}\n")

    print(f"  exact grow-and-prune sparse (D={D_BIG}), sweeping density:")
    print(f"    {'density':>8} {'k/layer':>9} {'params':>9} {'MSE':>9} {'vs dense':>10} {'x params':>9}")
    crossover = None
    for d in DENSITIES:
        k = round(d * D_BIG * D_BIG)
        gen = torch.Generator().manual_seed(SEED + 2)
        model = BigDynamic(U, k, gen)
        mse = run(model, X, Y, dynamic=True)
        p = sparse_params(model)
        beats = "BEATS" if mse < base_mse else ""
        if mse < base_mse and crossover is None:
            crossover = (d, p, mse)
        print(f"    {d:>8.4f} {k:>9,} {p:>9,} {mse:>9.5f} {beats:>10} {p / base_params:>8.2f}x")

    if crossover:
        d, p, mse = crossover
        print(f"\n  first to beat dense: density {d:.2%} "
              f"({p:,} params = {p / base_params:.2f}x the dense baseline), MSE {mse:.5f} < {base_mse:.5f}")
    else:
        print(f"\n  sparse never beat the dense baseline ({base_mse:.5f}) in this sweep")


if __name__ == "__main__":
    main()
