"""Does L1 hurt the NON-grow/prune models on pure data loss?

L1 belongs only on the dynamic-sparse training (it pushes weights toward 0 so
grow-and-prune has safe-to-prune candidates). The dense and static-sparse models
don't prune, so L1 is pure regularization for them — it should only hurt their
data fit. We train each with L1=0 and L1=1e-4 and compare PURE data MSE.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from sparse_layer import SparseLinear

D, M = 128, 64
SAMPLES, BATCH, STEPS, LR = 4096, 128, 1500, 1e-3
RECON_STEPS, L1_COEF, SEED = 1000, 1e-4, 0
STATIC_DENSITY = 0.25


class Recon(nn.Module):
    def __init__(s, D, m):
        super().__init__(); s.P = nn.Linear(D, m, bias=False); s.Q = nn.Linear(m, D, bias=False)
    def forward(s, x): return s.Q(s.P(x))


class TooSmallDense(nn.Module):
    def __init__(s, P, Q, m):
        super().__init__()
        s.register_buffer("P", P); s.register_buffer("Q", Q)
        s.v1 = nn.Linear(m, m, bias=False); s.v2 = nn.Linear(m, m, bias=False)
    def forward(s, x):
        z = x @ s.P.t()
        core = torch.tanh(s.v2(torch.tanh(s.v1(z))))
        return torch.tanh(core @ s.Q.t())


class FullDense(nn.Module):
    def __init__(s):
        super().__init__(); s.W1 = nn.Linear(D, D, bias=False); s.W2 = nn.Linear(D, D, bias=False)
    def forward(s, x): return torch.tanh(s.W2(torch.tanh(s.W1(x))))


class StaticSparse(nn.Module):
    def __init__(s, k, gen):
        super().__init__()
        sc = math.sqrt(D / k)
        s.W1 = SparseLinear(D, D, k, init_scale=sc, generator=gen)
        s.W2 = SparseLinear(D, D, k, init_scale=sc, generator=gen)
    def forward(s, x): return torch.tanh(s.W2(torch.tanh(s.W1(x))))


def generic_l1(model):
    return sum(p.abs().sum() for p in model.parameters() if p.requires_grad)


def train(model, X, Y, l1coef):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(SEED + 7)
    for _ in range(STEPS):
        idx = torch.randint(0, SAMPLES, (BATCH,), generator=g)
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(X[idx]), Y[idx])
        if l1coef:
            loss = loss + l1coef * generic_l1(model)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return F.mse_loss(model(X), Y).item()   # PURE data MSE


def main():
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED + 999); s = 1.0 / math.sqrt(D)
    A1 = torch.randn(D, D, generator=g) * s; A2 = torch.randn(D, D, generator=g) * s
    gx = torch.Generator().manual_seed(SEED + 1); X = torch.randn(SAMPLES, D, generator=gx)
    Y = torch.tanh(torch.tanh(X @ A1.t()) @ A2.t()).detach()

    rec = Recon(D, M); opt = torch.optim.Adam(rec.parameters(), 1e-3)
    gg = torch.Generator().manual_seed(SEED + 3)
    for _ in range(RECON_STEPS):
        idx = torch.randint(0, SAMPLES, (BATCH,), generator=gg)
        opt.zero_grad(set_to_none=True); F.mse_loss(rec(X[idx]), X[idx]).backward(); opt.step()
    P = rec.P.weight.detach().clone(); Q = rec.Q.weight.detach().clone()

    k = round(STATIC_DENSITY * D * D)
    builders = {
        "full_dense": lambda: FullDense(),
        "too_small_dense": lambda: TooSmallDense(P, Q, M),
        "static_sparse": lambda: StaticSparse(k, torch.Generator().manual_seed(SEED + 2)),
    }
    print(f"pure data MSE, L1 coef {L1_COEF} applied to ALL weights (these models don't prune):\n")
    print(f"    {'model':>16} {'no L1':>10} {'with L1':>10} {'change':>10}")
    for name, mk in builders.items():
        m0 = train(mk(), X, Y, 0.0)
        m1 = train(mk(), X, Y, L1_COEF)
        print(f"    {name:>16} {m0:>10.5f} {m1:>10.5f} {(m1/m0-1)*100:>+9.1f}%")


if __name__ == "__main__":
    main()
