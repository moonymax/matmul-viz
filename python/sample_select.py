"""Sample-subset gradient approximation: can a few high-error samples reconstruct
the full-batch grow gradient G = da^T @ x = sum_b da[b] (x) x[b] ?

If yes, compute the EXACT gradient from just those m samples (m*D^2 instead of
B*D^2 mults) -- and since G_m is rank <= m, the cheap norm-block approx is then
near-perfect on it too. Questions:
  * ceiling : how few samples m recover the full G's top-K cells?
  * selector: does ranking samples by error norm ||da[b]|| (cheap, no x needed)
              pick the right ones, vs contribution ||da[b]||*||x[b]|| vs random?

Metric (grow-relevant, scale-free): top-K overlap of G_m vs full-batch G.
Also rescaled relMSE ||(B/m)G_m - G||^2/||G||^2 for reconstruction fidelity.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 2048
SAMPLES, STEPS, LR = 8192, 250, 1e-3
B_MEAS = 256
K = 9216
MS = [4, 8, 16, 32, 64, 128]
SEED = 0
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class TwoLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(D, D, bias=False); self.W2 = nn.Linear(D, D, bias=False)
    def forward(self, x):
        self.x1 = x; self.pre1 = self.W1(x)
        self.x2 = torch.tanh(self.pre1); self.pre2 = self.W2(self.x2)
        return torch.tanh(self.pre2)


def topk_set(G, K):
    return torch.topk(G.abs().reshape(-1), K).indices


def main():
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED + 999); s = 1.0 / math.sqrt(D)
    A1 = torch.randn(D, D, generator=g) * s; A2 = torch.randn(D, D, generator=g) * s
    gx = torch.Generator().manual_seed(SEED + 1); X = torch.randn(SAMPLES, D, generator=gx)
    Y = torch.tanh(torch.tanh(X @ A1.t()) @ A2.t()).detach()
    X, Y = X.to(DEV), Y.to(DEV)

    model = TwoLayer().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = torch.Generator().manual_seed(SEED + 7)
    mrng = torch.Generator().manual_seed(SEED + 33)

    heur = ["error", "contrib", "random"]
    # acc[m][h] = [sum overlap, sum relMSE, count]
    acc = {m: {h: [0.0, 0.0, 0] for h in heur} for m in MS}
    contrib_conc = [0.0, 0.0, 0]   # top-m/B contribution energy share for a couple m

    for step in range(STEPS):
        if step % 25 == 0:
            idx = torch.randint(0, SAMPLES, (B_MEAS,), generator=mrng).to(DEV)
            opt.zero_grad(set_to_none=True)
            out = model(X[idx])
            model.pre2.retain_grad()
            F.mse_loss(out, Y[idx]).backward()
            x, da = model.x2.detach(), model.pre2.grad          # (B, D), (B, D)
            G = da.t() @ x
            denom = G.pow(2).sum().item() + 1e-30
            full_set = set(topk_set(G, K).tolist())

            enorm = da.norm(dim=1)                               # ||da[b]|| error per sample
            cnorm = enorm * x.norm(dim=1)                        # contribution norm
            order = {
                "error": torch.argsort(enorm, descending=True),
                "contrib": torch.argsort(cnorm, descending=True),
                "random": torch.randperm(B_MEAS, generator=torch.Generator().manual_seed(step)),
            }
            # contribution concentration: share of total (approx) energy in top-32 samples
            cs = (cnorm ** 2).sort(descending=True).values
            contrib_conc[0] += cs[:32].sum().item() / (cs.sum().item() + 1e-30)
            contrib_conc[2] += 1

            for m in MS:
                for h in heur:
                    sel = order[h][:m]
                    Gm = da[sel].t() @ x[sel]
                    ov = len(full_set & set(topk_set(Gm, K).tolist())) / K
                    rel = ((B_MEAS / m) * Gm - G).pow(2).sum().item() / denom
                    a = acc[m][h]; a[0] += ov; a[1] += rel; a[2] += 1
            opt.zero_grad(set_to_none=True)

        bidx = torch.randint(0, SAMPLES, (128,), generator=rng).to(DEV)
        opt.zero_grad(set_to_none=True)
        F.mse_loss(model(X[bidx]), Y[bidx]).backward()
        opt.step()

    cc = contrib_conc[0] / contrib_conc[2]
    print(f"sample-subset grow gradient (D={D}, meas batch B={B_MEAS}, K={K})")
    print(f"top-32/{B_MEAS} samples hold {100*cc:.1f}% of contribution energy\n")
    print(f"  {'m':>4} {'m/B':>5} | {'overlap: error':>14} {'contrib':>8} {'random':>7} "
          f"| {'relMSE error':>12} {'contrib':>8}")
    for m in MS:
        e, c, r = acc[m]["error"], acc[m]["contrib"], acc[m]["random"]
        n = e[2]
        print(f"  {m:>4} {m/B_MEAS:>5.2f} | {e[0]/n:>14.3f} {c[0]/n:>8.3f} {r[0]/n:>7.3f} "
              f"| {e[1]/n:>12.3f} {c[1]/n:>8.3f}")
    print("\n  overlap = top-K(G_m) ∩ top-K(full G) / K   (1.0 = m samples recover full grow signal)")


if __name__ == "__main__":
    main()
