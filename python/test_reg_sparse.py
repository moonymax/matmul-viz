"""Does L2 penalty or weight decay help the grow-and-prune sparse models?
(L1 did not.) Same too-big problem; vary only the regularizer; compare PURE
data MSE (penalty always excluded from the reported number).

  none : Adam, no regularization
  L2   : Adam + coef * sum(active w^2)   (penalty in the loss, adapted by Adam)
  WD   : AdamW(weight_decay)             (decoupled decay, not in the loss)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamic_sparse import (
    DynamicSparseLinear, ApproxDynamicSparseLinear,
    enforce_masks, rigl_update, cosine_drop_fraction,
    set_capture, approx_grow_prune, l2_penalty,
)

D = 128
SAMPLES, BATCH, STEPS, LR = 4096, 128, 1000, 1e-3
RIGL_EVERY, APPROX_TOPK, SEED = 25, 48, 0
DENSITY = 0.25
L2_COEFS = [1e-5, 1e-4, 1e-3]
WD_COEFS = [1e-2, 1e-1, 3e-1]


class Big(nn.Module):
    def __init__(s, mk):
        super().__init__(); s.W1 = mk(); s.W2 = mk()
    def forward(s, x):
        return torch.tanh(s.W2(torch.tanh(s.W1(x))))


def make(kind, k):
    sc = math.sqrt(D / k)
    cls = DynamicSparseLinear if kind == "exact" else ApproxDynamicSparseLinear
    return Big(lambda: cls(D, D, k, init_scale=sc, generator=torch.Generator().manual_seed(SEED + 2)))


def train(kind, k, X, Y, reg="none", coef=0.0):
    model = make(kind, k)
    if reg == "wd":
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=coef)
    else:
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
        if reg == "l2":
            loss = loss + coef * l2_penalty(model)
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
        return F.mse_loss(model(X), Y).item()          # PURE data MSE


def main():
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED + 999); s = 1.0 / math.sqrt(D)
    A1 = torch.randn(D, D, generator=g) * s; A2 = torch.randn(D, D, generator=g) * s
    gx = torch.Generator().manual_seed(SEED + 1); X = torch.randn(SAMPLES, D, generator=gx)
    Y = torch.tanh(torch.tanh(X @ A1.t()) @ A2.t()).detach()
    k = round(DENSITY * D * D)

    print(f"pure data MSE, grow-and-prune @ density {DENSITY:.0%} (k={k}), D={D}:\n")
    for kind in ("exact", "approx"):
        base = train(kind, k, X, Y, "none")
        print(f"  [{kind}]  none: {base:.5f}")
        for c in L2_COEFS:
            v = train(kind, k, X, Y, "l2", c)
            print(f"          L2 coef={c:<6}: {v:.5f}  ({(v/base-1)*100:+.1f}%  {'helps' if v<base else 'hurts'})")
        for c in WD_COEFS:
            v = train(kind, k, X, Y, "wd", c)
            print(f"          WD    ={c:<6}: {v:.5f}  ({(v/base-1)*100:+.1f}%  {'helps' if v<base else 'hurts'})")
        print()


if __name__ == "__main__":
    main()
