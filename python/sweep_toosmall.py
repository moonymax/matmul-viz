"""Make the dense model genuinely TOO SMALL, then sweep sparse density to find
when sparse (full-dimensional) overtakes it.

Setup:
  * Problem is deliberately "too big": x ~ N(0, I_D) isotropic in D dims (full
    rank / incompressible), teacher y = tanh(A2 tanh(A1 x)), A: D x D.
  * A linear down/up projection (P: D->m, Q: m->D) is trained with pure
    reconstruction loss ||Q P x - x||^2 to keep as much of x as possible at m<D
    (best m-dim view; it even overfits the finite dataset a bit). Then FROZEN.
  * "too-small dense": frozen P -> dense m x m core (2 layers) -> frozen Q.
    Doubly bottlenecked through m dims -> hard error floor. Trainable = 2*m^2.
  * "sparse": full D-dim, exact grow-and-prune, swept over density. Full access
    to all D dims, just sparse. Params = 2 * density * D^2.

Question: how much density (params) does sparse need to overtake the too-small
dense model that was given the best possible compressed view?
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamic_sparse import (
    DynamicSparseLinear, set_capture, rigl_update, cosine_drop_fraction, l1_penalty,
)

D = 128            # "too big" problem dimension (isotropic, full rank)
M = 64             # too-small dense bottleneck (best m-dim view via reconstruction)
SAMPLES, BATCH, STEPS, LR = 4096, 128, 1500, 1e-3
RECON_STEPS = 1000
RIGL_EVERY, SEED = 25, 0
L1_COEF = 0.0      # L1 found to HURT pure data loss (see findings); use AdamW weight
                   # decay instead if regularizing the sparse models. Kept as a knob.
DENSITIES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.0]


class Recon(nn.Module):
    def __init__(self, D, m):
        super().__init__()
        self.P = nn.Linear(D, m, bias=False)   # down
        self.Q = nn.Linear(m, D, bias=False)   # up

    def forward(self, x):
        return self.Q(self.P(x))


class TooSmallDense(nn.Module):
    def __init__(self, P, Q, m):
        super().__init__()
        self.register_buffer("P", P)           # (m, D) frozen
        self.register_buffer("Q", Q)           # (D, m) frozen
        self.v1 = nn.Linear(m, m, bias=False)
        self.v2 = nn.Linear(m, m, bias=False)

    def forward(self, x):
        z = x @ self.P.t()                     # (B, m)  compress (lossy)
        core = torch.tanh(self.v2(torch.tanh(self.v1(z))))
        return torch.tanh(core @ self.Q.t())   # (B, D)  expand back into m-subspace


class SparseBig(nn.Module):
    def __init__(self, k, gen):
        super().__init__()
        scale = math.sqrt(D / k)
        self.W1 = DynamicSparseLinear(D, D, k, init_scale=scale, generator=gen)
        self.W2 = DynamicSparseLinear(D, D, k, init_scale=scale, generator=gen)

    def forward(self, x):
        return torch.tanh(self.W2(torch.tanh(self.W1(x))))


class FullDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(D, D, bias=False)
        self.W2 = nn.Linear(D, D, bias=False)

    def forward(self, x):
        return torch.tanh(self.W2(torch.tanh(self.W1(x))))


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
        loss = F.mse_loss(model(X[idx]), Y[idx])
        if dynamic:
            loss = loss + L1_COEF * l1_penalty(model)
        loss.backward()
        if probe:
            rigl_update(model, opt, frac)
            set_capture(model, False)
        opt.step()
    with torch.no_grad():
        return F.mse_loss(model(X), Y).item()


def main():
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED + 999)
    s = 1.0 / math.sqrt(D)
    A1 = torch.randn(D, D, generator=g) * s
    A2 = torch.randn(D, D, generator=g) * s
    gx = torch.Generator().manual_seed(SEED + 1)
    X = torch.randn(SAMPLES, D, generator=gx)          # isotropic -> incompressible
    Y = torch.tanh(torch.tanh(X @ A1.t()) @ A2.t()).detach()

    # --- train the best m-dim reconstruction projection, then freeze ---
    rec = Recon(D, M)
    opt = torch.optim.Adam(rec.parameters(), lr=1e-3)
    gg = torch.Generator().manual_seed(SEED + 3)
    for _ in range(RECON_STEPS):
        idx = torch.randint(0, SAMPLES, (BATCH,), generator=gg)
        opt.zero_grad(set_to_none=True)
        F.mse_loss(rec(X[idx]), X[idx]).backward()
        opt.step()
    with torch.no_grad():
        rec_mse = F.mse_loss(rec(X), X).item()
        var = X.var().item()
    print(f"problem D={D} (isotropic, full-rank), too-small bottleneck m={M}")
    print(f"reconstruction: kept {100*(1-rec_mse/var):.1f}% of input variance "
          f"(recon MSE {rec_mse:.4f} vs var {var:.4f})\n")
    P = rec.P.weight.detach().clone()   # (m, D)
    Q = rec.Q.weight.detach().clone()   # (D, m)

    # --- too-small dense baseline ---
    small = TooSmallDense(P, Q, M)
    base = run(small, X, Y)
    base_params = sum(p.numel() for p in small.parameters() if p.requires_grad)
    full = run(FullDense(), X, Y)
    print(f"  too-small dense (m={M}, frozen best projection): params={base_params:,}  MSE={base:.5f}")
    print(f"  full dense (D={D}, ceiling):                     params={2*D*D:,}  MSE={full:.5f}\n")

    print("  exact grow-and-prune sparse (full D), sweeping density:")
    print(f"    {'density':>8} {'params':>9} {'xdense':>7} {'MSE':>9} {'beats?':>7}")
    crossover = None
    for d in DENSITIES:
        k = round(d * D * D)
        gen = torch.Generator().manual_seed(SEED + 2)
        mse = run(SparseBig(k, gen), X, Y, dynamic=True)
        p = 2 * k
        beats = "BEATS" if mse < base else ""
        if mse < base and crossover is None:
            crossover = (d, p, mse)
        print(f"    {d:>8.2f} {p:>9,} {p/base_params:>6.2f}x {mse:>9.5f} {beats:>7}")

    if crossover:
        d, p, mse = crossover
        print(f"\n  sparse overtakes the too-small dense at density {d:.0%} "
              f"({p:,} params = {p/base_params:.2f}x it), MSE {mse:.5f} < {base:.5f}")
    else:
        print(f"\n  sparse never beat the too-small dense ({base:.5f}) in this sweep")


if __name__ == "__main__":
    main()
