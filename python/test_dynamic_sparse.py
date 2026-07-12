"""Two-layer synthetic experiment: does grow-and-prune (dynamic sparse) beat a
fixed random mask (static sparse) at the same nonzero budget?

Target: a frozen random 2-layer dense teacher  y = tanh(W2 tanh(W1 x)),  x in R^n.
Students (2-layer, tanh between):
  - dense    : nn.Linear (full n*n per layer)          [capacity ceiling]
  - static   : SparseLinear, fixed random k positions
  - dynamic  : DynamicSparseLinear, same k, RigL grow-and-prune
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from sparse_layer import SparseLinear
from dynamic_sparse import (
    DynamicSparseLinear, ApproxDynamicSparseLinear,
    enforce_masks, rigl_update, cosine_drop_fraction,
    set_capture, approx_grow_prune,
)

APPROX_TOPK = 48        # top-p rows x top-q cols scored by the approximate probe

N = 256                 # matrix dim
DENSITY = 0.05          # sparse budget (both static & dynamic)
K = round(DENSITY * N * N)
SAMPLES = 4096
BATCH = 128
STEPS = 1500
LR = 1e-3
RIGL_EVERY = 50
SEED = 0
DEVICE = "cpu"          # small; CPU is deterministic and fine


class TwoLayer(nn.Module):
    def __init__(self, l1, l2):
        super().__init__()
        self.l1, self.l2 = l1, l2

    def forward(self, x):
        return torch.tanh(self.l2(torch.tanh(self.l1(x))))


def make_layer(kind, gen):
    scale = math.sqrt(N / K)
    if kind == "dense":
        return nn.Linear(N, N, bias=False)
    if kind == "static":
        return SparseLinear(N, N, K, init_scale=scale, generator=gen)
    if kind == "dynamic":
        return DynamicSparseLinear(N, N, K, init_scale=scale, generator=gen)
    if kind == "approx":
        return ApproxDynamicSparseLinear(N, N, K, init_scale=scale, generator=gen)
    raise ValueError(kind)


def teacher_fn():
    g = torch.Generator().manual_seed(SEED + 999)
    s = 1.0 / math.sqrt(N)
    W1 = (torch.randn(N, N, generator=g) * s)
    W2 = (torch.randn(N, N, generator=g) * s)

    def f(x):
        return torch.tanh(torch.tanh(x @ W1.t()) @ W2.t())
    return f


def n_params(m):
    """Effective trainable params: a DynamicSparseLinear stores a masked *dense*
    weight for the grow/prune probe, but only its `k` active entries count."""
    dyn_ids = {id(mod.weight) for mod in m.modules() if isinstance(mod, DynamicSparseLinear)}
    total = sum(int(mod.mask.sum().item()) for mod in m.modules()
                if isinstance(mod, DynamicSparseLinear))
    total += sum(p.numel() for p in m.parameters() if id(p) not in dyn_ids)
    return total


def train(kind, X, Y):
    gen = torch.Generator().manual_seed(SEED + 2)
    model = TwoLayer(make_layer(kind, gen), make_layer(kind, gen)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Nn = X.shape[0]
    g = torch.Generator().manual_seed(SEED + 7)
    curve = []
    for step in range(STEPS):
        idx = torch.randint(0, Nn, (BATCH,), generator=g)
        frac = cosine_drop_fraction(step, STEPS)
        probe = step % RIGL_EVERY == 0 and frac > 0
        if kind == "approx" and probe:
            set_capture(model, True)
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(X[idx]), Y[idx])
        loss.backward()
        if kind == "dynamic" and probe:
            rigl_update(model, opt, frac)
        if kind == "approx" and probe:
            approx_grow_prune(model, opt, n_swap=max(1, round(frac * K)), topk=APPROX_TOPK)
            set_capture(model, False)
        opt.step()
        if kind == "dynamic":
            enforce_masks(model)
        if step % 100 == 0 or step == STEPS - 1:
            with torch.no_grad():
                full = F.mse_loss(model(X), Y).item()
            curve.append((step, full))
    with torch.no_grad():
        final = F.mse_loss(model(X), Y).item()
    return model, final, curve


def main():
    torch.manual_seed(SEED)
    teacher = teacher_fn()
    gx = torch.Generator().manual_seed(SEED + 1)
    X = torch.randn(SAMPLES, N, generator=gx)
    Y = teacher(X).detach()

    print(f"teacher: 2-layer tanh, dim {N}   |   sparse budget k={K:,}/layer "
          f"({DENSITY:.1%} density)   |   {STEPS} steps\n")

    results = {}
    for kind in ("dense", "static", "dynamic", "approx"):
        model, final, curve = train(kind, X, Y)
        results[kind] = (n_params(model), final, curve)
        print(f"  {kind:>8}: params={n_params(model):>9,}  final MSE={final:.5f}")

    print("\n  loss curve (full-dataset MSE):")
    steps = [s for s, _ in results["dense"][2]]
    print("    " + "step".rjust(6) + "".join(f"{k:>11}" for k in results))
    curmap = {k: dict(results[k][2]) for k in results}
    for s in steps:
        row = "".join(f"{curmap[k].get(s, float('nan')):>11.5f}" for k in results)
        print(f"    {s:>6}{row}")

    d_static = results["static"][1]
    d_dyn = results["dynamic"][1]
    d_approx = results["approx"][1]
    d_dense = results["dense"][1]
    print(f"\n  static (fixed mask)         : {d_static:.4f}")
    print(f"  dynamic (exact, masked-dense): {d_dyn:.4f}   ({d_static/d_dyn:.2f}x better than static)")
    print(f"  approx (top-p x top-q, O(k)) : {d_approx:.4f}   ({d_static/d_approx:.2f}x better than static)")
    print(f"  approx vs exact grow: {abs(d_approx-d_dyn)/d_dyn*100:.0f}% apart "
          f"(does the cheap probe match the full-gradient one?)")
    print(f"  dense ceiling ({d_dense:.4f}): static {d_static/d_dense:.1f}x, "
          f"dynamic {d_dyn/d_dense:.1f}x, approx {d_approx/d_dense:.1f}x above")


if __name__ == "__main__":
    main()
