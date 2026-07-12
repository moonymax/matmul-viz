"""How close is the APPROXIMATE grow gradient to the DENSE gradient, vs batch size?

Isolated question (grow-and-prune frequency held FIXED at every 5 steps, not
studied here): at a grow event we need the weight gradient  G = da^T @ x  (out x
in) to decide which inactive cells to grow. The O(k) approximation never forms G;
it computes G exactly only on a top-p error-rows x top-q activation-cols block and
treats the rest as zero. We measure how well that approximate gradient matches the
full dense gradient, as a function of BATCH SIZE.

Accuracy = MSE between the approximate gradient and the dense gradient, reported
as a *relative* MSE  ||G_approx - G||^2 / ||G||^2  (= fraction of gradient energy
missed) so it is comparable across batch sizes (raw MSE scales with the ~1/B
gradient magnitude and would be confounded).

Hypothesis: G = sum_b da_b (x) x_b is a sum of B rank-1 terms. At B=1 it is exactly
rank-1, so top-|da_i| rows x top-|x_j| cols hit the largest entries by
construction; as B grows the rank rises and norm-ranking stops predicting the big
entries -> the approximation should get worse with batch size.

Reference line: ORACLE same-budget = keep the true top (p*q) cells of |G|. That is
the best any p*q-cell approximation could do (pure coverage floor); the gap
approx - oracle is the pure *selection* error of the norm heuristic.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamic_sparse import (
    DynamicSparseLinear, set_capture, rigl_update, cosine_drop_fraction,
    _DynamicSparseBase,
)

D = 128
SAMPLES, STEPS, LR = 4096, 1500, 1e-3
DENSITY = 0.10
RIGL_EVERY = 5                 # FIXED grow-and-prune frequency (not studied here)
TRAJ_BATCH = 128               # batch driving the (exact-grow) trajectory
P = Q = 48                     # the approximation's fixed block size
SEED = 0
MEAS_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


class Big(nn.Module):
    def __init__(self, mk):
        super().__init__(); self.W1 = mk(); self.W2 = mk()
    def forward(self, x):
        return torch.tanh(self.W2(torch.tanh(self.W1(x))))


def block_rel_mse(x, da, p, q):
    """Relative MSE of the top-p rows x top-q cols approximation vs dense G."""
    out_f, in_f = da.shape[1], x.shape[1]
    p, q = min(p, out_f), min(q, in_f)
    G = da.t() @ x                                        # (out, in) dense gradient
    topR = torch.topk(da.pow(2).sum(0), p).indices
    topC = torch.topk(x.pow(2).sum(0), q).indices
    G_ap = torch.zeros_like(G)
    G_ap[topR.unsqueeze(1), topC.unsqueeze(0)] = G[topR.unsqueeze(1), topC.unsqueeze(0)]
    denom = G.pow(2).sum().item() + 1e-30
    approx_rel = (G - G_ap).pow(2).sum().item() / denom
    # oracle: keep the true top (p*q) largest-|G| cells (best possible at this budget)
    budget = p * q
    flat = G.abs().reshape(-1)
    keep = torch.topk(flat, min(budget, flat.numel()), largest=True).indices
    kept_energy = G.reshape(-1)[keep].pow(2).sum().item()
    oracle_rel = 1.0 - kept_energy / denom
    return approx_rel, oracle_rel


def main():
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED + 999); s = 1.0 / math.sqrt(D)
    A1 = torch.randn(D, D, generator=g) * s; A2 = torch.randn(D, D, generator=g) * s
    gx = torch.Generator().manual_seed(SEED + 1); X = torch.randn(SAMPLES, D, generator=gx)
    Y = torch.tanh(torch.tanh(X @ A1.t()) @ A2.t()).detach()
    k = round(DENSITY * D * D)

    gen = torch.Generator().manual_seed(SEED + 2)
    model = Big(lambda: DynamicSparseLinear(D, D, k, init_scale=math.sqrt(D / k), generator=gen))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = torch.Generator().manual_seed(SEED + 7)
    mrng = torch.Generator().manual_seed(SEED + 33)

    # accumulators per batch size: [sum approx_rel, sum oracle_rel, count]
    acc = {B: [0.0, 0.0, 0] for B in MEAS_BATCHES}
    layers = lambda: [m for m in model.modules() if isinstance(m, _DynamicSparseBase)]

    for step in range(STEPS):
        frac = cosine_drop_fraction(step, STEPS)
        probe = step % RIGL_EVERY == 0 and frac > 0

        if probe:
            # --- measurement: fresh independent batch per size, current model state ---
            for B in MEAS_BATCHES:
                idx = torch.randint(0, SAMPLES, (B,), generator=mrng)
                set_capture(model, True)
                opt.zero_grad(set_to_none=True)
                F.mse_loss(model(X[idx]), Y[idx]).backward()
                for layer in layers():
                    ar, orc = block_rel_mse(layer._x, layer._da, P, Q)
                    acc[B][0] += ar; acc[B][1] += orc; acc[B][2] += 1
                set_capture(model, False)

        # --- trajectory step (exact grow every RIGL_EVERY) ---
        idx = torch.randint(0, SAMPLES, (TRAJ_BATCH,), generator=rng)
        if probe:
            set_capture(model, True)
        opt.zero_grad(set_to_none=True)
        F.mse_loss(model(X[idx]), Y[idx]).backward()
        if probe:
            rigl_update(model, opt, frac)
            set_capture(model, False)
        opt.step()

    print(f"approx vs dense gradient  (D={D}, density {DENSITY:.0%}, k={k}, "
          f"grow/prune every {RIGL_EVERY}, block p=q={P})")
    print(f"relative MSE = ||G_approx - G||^2 / ||G||^2  (0 = perfect, 1 = no signal)\n")
    print(f"  {'batch':>6} {'approx relMSE':>14} {'oracle relMSE':>14} {'selection gap':>14}")
    for B in MEAS_BATCHES:
        ar, orc, c = acc[B]
        print(f"  {B:>6} {ar/c:>14.4f} {orc/c:>14.4f} {(ar-orc)/c:>14.4f}")


if __name__ == "__main__":
    main()
