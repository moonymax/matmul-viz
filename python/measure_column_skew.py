"""Gating check for MIPS-style screening of the grow gradient G = da^T @ x.

Cauchy-Schwarz column screening (|G[i,j]| <= ||da[:,i]|| * ||x[:,j]||) only pays
if the column norms ||x[:,j]|| are heavy-tailed AND the true top-K cells actually
concentrate in a few rows/columns. If norms are ~uniform / cells are spread over
all rows and columns, screening keeps almost everything and buys nothing.

We measure, for realistic (x, da) captured from a training 2-layer net:
  * skew of column norms ||x[:,j]|| and row norms ||da[:,i]||  (top-1%/10% energy,
    max/median)
  * concentration of the true top-K |G| cells: how many DISTINCT rows and columns
    they occupy (few => row/col screening can shrink the candidate set a lot)
Contrasted across input types: isotropic (worst case) vs a heavy-tailed input.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 2048
SAMPLES, STEPS, B, LR = 4096, 200, 32, 1e-3
K = 9216
SEED = 0
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class TwoLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(D, D, bias=False); self.W2 = nn.Linear(D, D, bias=False)
    def forward(self, x):
        self.x1 = x
        self.pre1 = self.W1(x)
        self.x2 = torch.tanh(self.pre1)
        self.pre2 = self.W2(self.x2)
        return torch.tanh(self.pre2)


def skew(norms):
    """Heavy-tail summary of a 1-D vector of norms."""
    v = norms.sort(descending=True).values
    tot = v.sum().item() + 1e-30
    n = v.numel()
    top1 = v[: max(1, n // 100)].sum().item() / tot
    top10 = v[: max(1, n // 10)].sum().item() / tot
    med = v.median().item() + 1e-30
    return top1, top10, v[0].item() / med


def concentration(G, K):
    """How many distinct rows / cols the true top-K |G| cells occupy."""
    in_f = G.shape[1]
    idx = torch.topk(G.abs().reshape(-1), K, largest=False).indices if False else \
        torch.topk(G.abs().reshape(-1), K).indices
    rows = torch.unique(idx // in_f).numel()
    cols = torch.unique(idx % in_f).numel()
    return rows, cols


def make_data(kind):
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED + 999); s = 1.0 / math.sqrt(D)
    A1 = torch.randn(D, D, generator=g) * s; A2 = torch.randn(D, D, generator=g) * s
    gx = torch.Generator().manual_seed(SEED + 1)
    if kind == "isotropic":
        X = torch.randn(SAMPLES, D, generator=gx)
    else:  # heavy-tailed: scale features by a power-law so a few dominate
        X = torch.randn(SAMPLES, D, generator=gx)
        scale = torch.logspace(0, -3, D, base=10.0)          # feature j ~ 10^(-3 j/D)
        X = X * scale
    Y = torch.tanh(torch.tanh(X @ A1.t()) @ A2.t()).detach()
    return X.to(DEV), Y.to(DEV)


def run(kind):
    X, Y = make_data(kind)
    model = TwoLayer().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = torch.Generator().manual_seed(SEED + 7)
    acc = {}
    probes = 0
    for step in range(STEPS):
        idx = torch.randint(0, SAMPLES, (B,), generator=rng).to(DEV)
        opt.zero_grad(set_to_none=True)
        out = model(X[idx])
        model.pre1.retain_grad(); model.pre2.retain_grad()
        F.mse_loss(out, Y[idx]).backward()
        if step % 40 == 0:
            for lname, x, da in (("L1", model.x1.detach(), model.pre1.grad),
                                 ("L2", model.x2.detach(), model.pre2.grad)):
                cn = x.norm(dim=0)                 # ||x[:,j]||  (D,)
                rn = da.norm(dim=0)                # ||da[:,i]|| (D,)
                G = da.t() @ x
                rr, cc = concentration(G, K)
                a = acc.setdefault(lname, [0.0] * 7 + [0])
                c1, c10, cr = skew(cn); r1, r10, rr_ = skew(rn)
                a[0] += c1; a[1] += c10; a[2] += cr
                a[3] += r1; a[4] += r10; a[5] += rr_
                a[6] += rr; a[7] += 1
                acc[lname + "_conc"] = acc.get(lname + "_conc", [0, 0, 0])
                acc[lname + "_conc"][0] += rr; acc[lname + "_conc"][1] += cc
                acc[lname + "_conc"][2] += 1
        opt.step()

    print(f"\n==== input = {kind}  (D={D}, B={B}, K={K}) ====")
    print(f"  {'layer':>6} {'col top1%':>9} {'col top10%':>10} {'col max/med':>11} "
          f"{'row top10%':>10} {'row max/med':>11} {'topK rows':>9} {'topK cols':>9}")
    for lname in ("L1", "L2"):
        a = acc[lname]; c = a[7]
        cc = acc[lname + "_conc"]
        print(f"  {lname:>6} {a[0]/c:>9.3f} {a[1]/c:>10.3f} {a[2]/c:>11.1f} "
              f"{a[4]/c:>10.3f} {a[5]/c:>11.1f} {cc[0]//cc[2]:>9} {cc[1]//cc[2]:>9}")
    print(f"  (uniform baseline: top10% share ~0.10; topK rows/cols near {D} means spread)")


def main():
    print("Cauchy-Schwarz screening pays iff column norms are heavy-tailed AND")
    print("the top-K cells occupy few distinct rows/cols. Measuring both:")
    run("isotropic")
    run("heavytailed")


if __name__ == "__main__":
    main()
