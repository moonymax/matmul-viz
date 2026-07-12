"""Efficient grow-gradient approximation at LARGE D, tracked by OPS and wall-clock.

The grow step needs the large-|G| cells of the weight gradient G = da^T @ x
(out x in). Approx is only worth it if it costs less than forming G (D^2*B mults).
We run at a large D (where sparse layers actually live) and report, per method:
  * relMSE : energy missed vs the dense gradient (0 = perfect)
  * ops    : analytic mult-count, and as % of the full gradient
  * time   : wall-clock (incl. the full-gradient baseline time)

Methods (all avoid forming the full G):
  norm       : top-p rows by ||da_i||^2 x top-q cols by ||x_j||^2, keep block.       ~p*q*B
  gram       : same block but rows/cols by EXACT grad-row/col norm (alignment).       2B^2*D
  rowexact-p : top-p rows by grad-row norm, compute those rows EXACTLY over ALL
               cols, keep the true top-(P*Q) cells (scattered, no col restriction).   B^2*D + p*B*D
  crossexact : exact top-p rows AND exact top-q cols (a cross), keep top-(P*Q).       2B^2*D + (p+q)*B*D
  oracle     : true top-(P*Q) cells of |G| (reference; needs full G).
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

D = 2048
SAMPLES, STEPS, LR = 4096, 300, 1e-3
MEAS_EVERY = 10
P = Q = 96                 # block budget = P*Q cells
SEED = 0
MEAS_BATCHES = [1, 8, 32, 128]
ROWEXACT_PS = [48, 96, 192]
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def sync():
    if DEV.type == "mps":
        torch.mps.synchronize()


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


# ---- selection strategies: return (kept_flat_indices, analytic_ops) ---------- #
def sel_norm(x, da, in_f, out_f):
    B = x.shape[0]
    topR = torch.topk(da.pow(2).sum(0), min(P, out_f)).indices
    topC = torch.topk(x.pow(2).sum(0), min(Q, in_f)).indices
    kept = (topR.unsqueeze(1) * in_f + topC.unsqueeze(0)).reshape(-1)
    return kept, (out_f + in_f) * B + topR.numel() * topC.numel() * B


def sel_gram(x, da, in_f, out_f):
    B = x.shape[0]
    RS = (da * ((x @ x.t()) @ da)).sum(0)
    CS = (x * ((da @ da.t()) @ x)).sum(0)
    topR = torch.topk(RS, min(P, out_f)).indices
    topC = torch.topk(CS, min(Q, in_f)).indices
    kept = (topR.unsqueeze(1) * in_f + topC.unsqueeze(0)).reshape(-1)
    return kept, 2 * B * B * (in_f + out_f) + topR.numel() * topC.numel() * B


def sel_rowexact(p):
    def fn(x, da, in_f, out_f):
        B = x.shape[0]
        RS = (da * ((x @ x.t()) @ da)).sum(0)              # ||g_i||^2, B^2*(in+out)
        topR = torch.topk(RS, min(p, out_f)).indices
        Grows = da[:, topR].t() @ x                         # (p,in) EXACT, p*B*in
        loc = torch.topk(Grows.abs().reshape(-1), min(P * Q, Grows.numel())).indices
        kept = topR[loc // in_f] * in_f + (loc % in_f)
        return kept, B * B * (in_f + out_f) + topR.numel() * B * in_f
    return fn


def sel_crossexact(p, q):
    def fn(x, da, in_f, out_f):
        B = x.shape[0]
        RS = (da * ((x @ x.t()) @ da)).sum(0)
        CS = (x * ((da @ da.t()) @ x)).sum(0)
        topR = torch.topk(RS, min(p, out_f)).indices
        topC = torch.topk(CS, min(q, in_f)).indices
        Grows = da[:, topR].t() @ x                         # (p,in)
        Gcols = da.t() @ x[:, topC]                          # (out,q)
        ar_in = torch.arange(in_f, device=x.device)
        ar_out = torch.arange(out_f, device=x.device)
        rflat = (topR.unsqueeze(1) * in_f + ar_in).reshape(-1)
        cflat = (ar_out.unsqueeze(1) * in_f + topC).reshape(-1)
        cand = torch.cat([rflat, cflat])
        val = torch.cat([Grows.reshape(-1), Gcols.reshape(-1)])
        loc = torch.topk(val.abs(), min(P * Q, val.numel())).indices
        kept = torch.unique(cand[loc])
        return kept, 2 * B * B * (in_f + out_f) + topR.numel() * B * in_f + topC.numel() * B * out_f
    return fn


def relmse(G, denom, kept):
    return 1.0 - G.reshape(-1)[kept].pow(2).sum().item() / denom


def oracle_relmse(G, denom, budget):
    keep = torch.topk(G.abs().reshape(-1), min(budget, G.numel()), largest=True).indices
    return 1.0 - G.reshape(-1)[keep].pow(2).sum().item() / denom


def main():
    torch.manual_seed(SEED)
    print(f"device={DEV}  D={D}  block P=Q={P}  budget={P*Q} cells\n", flush=True)
    g = torch.Generator().manual_seed(SEED + 999); s = 1.0 / math.sqrt(D)
    A1 = torch.randn(D, D, generator=g) * s; A2 = torch.randn(D, D, generator=g) * s
    gx = torch.Generator().manual_seed(SEED + 1); X = torch.randn(SAMPLES, D, generator=gx)
    Y = torch.tanh(torch.tanh(X @ A1.t()) @ A2.t()).detach()
    X, Y = X.to(DEV), Y.to(DEV)

    model = TwoLayer().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = torch.Generator().manual_seed(SEED + 7)
    mrng = torch.Generator().manual_seed(SEED + 33)

    methods = [("norm", sel_norm), ("gram", sel_gram)]
    methods += [(f"rowexact{p}", sel_rowexact(p)) for p in ROWEXACT_PS]
    methods += [("crossexact", sel_crossexact(P, Q))]
    names = [m[0] for m in methods] + ["oracle", "full"]
    # [sum relMSE, sum ops, sum time, count, sum overlap-with-oracle]
    acc = {B: {n: [0.0, 0, 0.0, 0, 0.0] for n in names} for B in MEAS_BATCHES}

    for step in range(STEPS):
        if step % MEAS_EVERY == 0:
            for B in MEAS_BATCHES:
                idx = torch.randint(0, SAMPLES, (B,), generator=mrng).to(DEV)
                opt.zero_grad(set_to_none=True)
                out = model(X[idx])
                model.pre1.retain_grad(); model.pre2.retain_grad()
                F.mse_loss(out, Y[idx]).backward()
                in_f = out_f = D
                for x, da in ((model.x1.detach(), model.pre1.grad),
                              (model.x2.detach(), model.pre2.grad)):
                    sync(); t0 = time.perf_counter()
                    G = da.t() @ x                                   # full gradient baseline
                    sync(); tfull = time.perf_counter() - t0
                    denom = G.pow(2).sum().item() + 1e-30
                    oracle_keep = torch.topk(G.abs().reshape(-1), P * Q, largest=True).indices
                    a = acc[B]["full"]; a[1] += in_f * out_f * B; a[2] += tfull; a[3] += 1
                    for n, fn in methods:
                        sync(); t0 = time.perf_counter()
                        kept, ops = fn(x, da, in_f, out_f)
                        sync(); dt = time.perf_counter() - t0
                        aa = acc[B][n]
                        aa[0] += relmse(G, denom, kept); aa[1] += ops; aa[2] += dt; aa[3] += 1
                        aa[4] += torch.isin(kept, oracle_keep).float().mean().item()
                    ao = acc[B]["oracle"]
                    ao[0] += 1.0 - G.reshape(-1)[oracle_keep].pow(2).sum().item() / denom; ao[3] += 1
                opt.zero_grad(set_to_none=True)

        idx = torch.randint(0, SAMPLES, (128,), generator=rng).to(DEV)
        opt.zero_grad(set_to_none=True)
        F.mse_loss(model(X[idx]), Y[idx]).backward()
        opt.step()

    for B in MEAS_BATCHES:
        full_ops = D * D * B
        ft = acc[B]["full"]; full_t = 1e6 * ft[2] / ft[3]
        oracle_captured = 1.0 - acc[B]["oracle"][0] / acc[B]["oracle"][3]   # energy frac oracle keeps
        print(f"  batch={B}:  full-grad G = {full_ops:,} ops, wall-clock {full_t:.1f} us  "
              f"(oracle keeps {100*oracle_captured:.1f}% of energy in {P*Q} cells)")
        print(f"    {'method':>12} {'selQ':>7} {'overlap':>8} {'ops':>14} {'%full':>7} {'time(us)':>9}")
        print(f"    {'oracle':>12} {1.000:>7.3f} {1.000:>8.3f} {'(full G)':>14} {'-':>7} {'-':>9}")
        for n in [m[0] for m in methods]:
            r, ops, tsum, c, ov = acc[B][n]
            captured = 1.0 - r / c
            selq = captured / (oracle_captured + 1e-30)       # 1.0 = matches oracle at same budget
            oe = ops // c; tm = 1e6 * tsum / c
            print(f"    {n:>12} {selq:>7.3f} {ov/c:>8.3f} {oe:>14,} {100*oe/full_ops:>6.1f}% {tm:>9.1f}")
        print(f"    {'full (G)':>12} {1.000:>7.3f} {1.000:>8.3f} {full_ops:>14,} {'100.0%':>7} {full_t:>9.1f}")
        print()
    print("  selQ = energy captured / oracle energy at same budget (1.0 = optimal selection)")
    print("  overlap = fraction of kept cells that are in the oracle's top-(P*Q) set")


if __name__ == "__main__":
    main()
