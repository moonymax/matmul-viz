"""Random-projection SKETCH gate for the grow gradient (vs the norm gate).

Finding 16: the magnitude-only Cauchy-Schwarz gate (row_i*col_j, h=1/cell) is too
loose -- it ignores alignment. We have B mults/cell of budget, so spend b'<<B on a
sketch that captures alignment: with P: R^B -> R^b' (JL),
  G[i,j] = <da[:,i], x[:,j]>  ~=  <P da[:,i], P x[:,j]>   (b' mults/cell)
i.e. Ghat = (P da)^T (P x), a rank-b' estimate of G. Rank cells by |Ghat|, compute
the top-C exactly. Compare recall(true top-K) vs the norm gate, and the ops:
  dense    = ncell*B
  norm     = ncell*1 + (out+in)*B + C*B
  sketch   = ncell*b' + (out+in)*B*b' + C*B
speedup = dense / method. Real CIFAR, D=2048, cached model.
"""
from __future__ import annotations

import os
import torch
import torch.nn.functional as F

import cifar_sparse_vs_dense as C
from cifar_sparse_vs_dense import ResFFN

DEV = C.DEVICE
WIDTH = 2048
TRAIN_EPOCHS = 6
MEAS_BATCHES = 2
K_FRAC = 0.01
BPRIMES = [8, 16, 32]
C_GRID = [0.01, 0.02, 0.05, 0.10, 0.20]     # fraction of cells computed exactly
TARGET = 0.95
SEED = 0


def recall_curve(score_flat, true_scores, K, ncell):
    """recall of the K true cells within the top-(frac*ncell) cells by score."""
    maxC = int(C_GRID[-1] * ncell)
    tv = torch.topk(score_flat, maxC).values          # sorted desc
    out = {}
    for f in C_GRID:
        thr = tv[min(int(f * ncell), maxC) - 1]
        out[f] = (true_scores >= thr).float().mean().item()
    return out


def speedup_at(recalls, ncell, out_f, in_f, B, bprime):
    """min C (frac) reaching TARGET recall, and the speedup there."""
    frac = next((f for f in C_GRID if recalls[f] >= TARGET), None)
    if frac is None:
        return None, None
    Cc = frac * ncell
    dense = ncell * B
    if bprime == 0:  # norm gate
        ops = ncell * 1 + (out_f + in_f) * B + Cc * B
    else:
        ops = ncell * bprime + (out_f + in_f) * B * bprime + Cc * B
    return frac, dense / ops


def main():
    tr, te = C.load_data()
    trl = torch.utils.data.DataLoader(tr, batch_size=C.BATCH, shuffle=True,
                                      num_workers=4, drop_last=True, persistent_workers=True)
    tel = torch.utils.data.DataLoader(te, batch_size=512, shuffle=False,
                                      num_workers=2, persistent_workers=True)
    model = ResFFN(WIDTH, sparse=False).to(DEV)
    cache = f"data/cifar_dense_w{WIDTH}_e{TRAIN_EPOCHS}.pt"
    if os.path.exists(cache):
        model.load_state_dict(torch.load(cache, map_location=DEV))
        print(f"loaded cached dense ResFFN (width {WIDTH})\n", flush=True)
    else:
        C.EPOCHS = TRAIN_EPOCHS
        acc = C.train(model, False, trl, tel, "dense"); torch.save(model.state_dict(), cache)
        print(f"trained {acc:.1f}% (cached)\n", flush=True)

    layers = {"stem": model.stem}
    for i, b in enumerate(model.blocks):
        layers[f"b{i}.w1(LN)"] = b.w1
        layers[f"b{i}.w2(GELU)"] = b.w2
    inp, gout = {}, {}
    for name, m in layers.items():
        m.register_forward_hook(lambda mod, i, o, n=name: inp.__setitem__(n, i[0].detach()))
        m.register_full_backward_hook(lambda mod, gi, go, n=name: gout.__setitem__(n, go[0].detach()))

    methods = ["norm"] + [f"sk{bp}" for bp in BPRIMES]
    # per layer per method: [sum recall@5%, sum recall@10%, sum speedup@95, sum frac@95(or nan), cnt]
    acc = {n: {m: [0.0, 0.0, 0.0, 0.0, 0] for m in methods} for n in layers}
    gen = torch.Generator(device="cpu").manual_seed(SEED)

    it = iter(trl)
    for _ in range(MEAS_BATCHES):
        xb, yb = next(it); xb, yb = xb.to(DEV), yb.to(DEV)
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb), yb).backward()
        for name in layers:
            x, da = inp[name], gout[name]
            in_f, out_f, B = x.shape[1], da.shape[1], x.shape[0]
            ncell = in_f * out_f
            G = da.t() @ x
            Gflat = G.reshape(-1); Gabs = Gflat.abs()
            K = max(1, int(K_FRAC * ncell))
            true_idx = torch.topk(Gabs, K).indices

            scorings = {"norm": (da.norm(dim=0).unsqueeze(1) * x.norm(dim=0).unsqueeze(0)).reshape(-1)}
            for bp in BPRIMES:
                P = (torch.randn(bp, B, generator=gen) / (bp ** 0.5)).to(DEV)
                Ghat = (P @ da).t() @ (P @ x)                      # rank-b' estimate of G
                scorings[f"sk{bp}"] = Ghat.abs().reshape(-1)

            for mth, sflat in scorings.items():
                rc = recall_curve(sflat, sflat[true_idx], K, ncell)
                bp = 0 if mth == "norm" else int(mth[2:])
                frac, sp = speedup_at(rc, ncell, out_f, in_f, B, bp)
                a = acc[name][mth]
                a[0] += rc[0.05]; a[1] += rc[0.10]
                a[2] += sp if sp else 0.0; a[3] += frac if frac else 0.0; a[4] += 1

    print(f"real CIFAR D={WIDTH}, sketch gate vs norm gate (K={K_FRAC:.0%}, B={C.BATCH}, target recall {TARGET})")
    print(f"  each method: recall@5% / recall@10% cells | C@95%recall, speedup@95%\n")
    print(f"  {'layer':>14} | " + "  ".join(f"{m:>18}" for m in methods))
    for name in layers:
        cells = []
        for mth in methods:
            a = acc[name][mth]; c = a[4]
            frac = a[3] / c; sp = a[2] / c
            tag = f"{a[0]/c:.2f}/{a[1]/c:.2f} {sp:.1f}x@{frac:.0%}" if frac > 0 else f"{a[0]/c:.2f}/{a[1]/c:.2f}  none"
            cells.append(tag)
        print(f"  {name:>14} | " + "  ".join(f"{v:>18}" for v in cells))
    print("\n  cell = recall@5%/recall@10% + speedup@95%recall (none = needs >20% of cells)")


if __name__ == "__main__":
    main()
