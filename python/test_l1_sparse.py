"""Does L1 actually HELP the grow-and-prune sparse models (pure data MSE)?

Same too-big problem (D=128 isotropic teacher). For the exact and approx
dynamic-sparse models we hold everything fixed (deadness pruning, schedule) and
vary ONLY the L1 coefficient, comparing pure data MSE. L1 should help here (it
creates genuinely-dead cells for safe pruning) — the opposite of its effect on
the non-pruning models.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamic_sparse import (
    DynamicSparseLinear, ApproxDynamicSparseLinear,
    enforce_masks, rigl_update, cosine_drop_fraction,
    set_capture, approx_grow_prune, l1_penalty,
)

D = 128
SAMPLES, BATCH, STEPS, LR = 4096, 128, 1200, 1e-3
RIGL_EVERY, APPROX_TOPK, SEED = 25, 48, 0
DENSITIES = [0.10, 0.25]
L1_COEF = 1e-4


class Big(nn.Module):
    def __init__(self, mk):
        super().__init__(); self.W1 = mk(); self.W2 = mk()
    def forward(self, x):
        return torch.tanh(self.W2(torch.tanh(self.W1(x))))


def make(kind, k, gen):
    sc = math.sqrt(D / k)
    cls = DynamicSparseLinear if kind == "exact" else ApproxDynamicSparseLinear
    return Big(lambda: cls(D, D, k, init_scale=sc, generator=gen))


def train(kind, k, l1coef, X, Y):
    gen = torch.Generator().manual_seed(SEED + 2)
    model = make(kind, k, gen)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(SEED + 7)
    for step in range(STEPS):
        idx = torch.randint(0, SAMPLES, (BATCH,), generator=g)
        frac = cosine_drop_fraction(step, STEPS)
        probe = step % RIGL_EVERY == 0 and frac > 0
        if kind == "approx" and probe:
            set_capture(model, True)
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(X[idx]), Y[idx])
        if l1coef:
            loss = loss + l1coef * l1_penalty(model)
        loss.backward()
        if kind == "exact" and probe:
            rigl_update(model, opt, frac)
        if kind == "approx" and probe:
            approx_grow_prune(model, opt, n_swap=max(1, round(frac * k)), topk=APPROX_TOPK)
            set_capture(model, False)
        opt.step()
        if kind == "exact":
            enforce_masks(model)
    with torch.no_grad():
        return F.mse_loss(model(X), Y).item()


def main():
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED + 999); s = 1.0 / math.sqrt(D)
    A1 = torch.randn(D, D, generator=g) * s; A2 = torch.randn(D, D, generator=g) * s
    gx = torch.Generator().manual_seed(SEED + 1); X = torch.randn(SAMPLES, D, generator=gx)
    Y = torch.tanh(torch.tanh(X @ A1.t()) @ A2.t()).detach()

    print(f"pure data MSE for grow-and-prune models, L1 {L1_COEF} vs none "
          f"(deadness pruning fixed):\n")
    print(f"    {'model':>8} {'density':>8} {'no L1':>10} {'with L1':>10} {'change':>9}")
    for kind in ("exact", "approx"):
        for d in DENSITIES:
            k = round(d * D * D)
            m0 = train(kind, k, 0.0, X, Y)
            m1 = train(kind, k, L1_COEF, X, Y)
            tag = "helps" if m1 < m0 else "hurts"
            print(f"    {kind:>8} {d:>8.2f} {m0:>10.5f} {m1:>10.5f} {(m1/m0-1)*100:>+7.1f}%  {tag}")


if __name__ == "__main__":
    main()
