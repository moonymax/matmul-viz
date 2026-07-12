"""Cheap rank-1 grow-gradient approximations on REAL CIFAR activations.

G[i,j] = sum_b da[b,i]*x[b,j]. A rank-1 approximation scores cell (i,j) as
row_i * col_j for some per-row / per-col reduction of the batch, and reads off
top cells as top-rows x top-cols (essentially free, O(B*D)). We compare batch
reductions:
  sum_signed : row_i=|sum_b da[b,i]|,  col_j=|sum_b x[b,j]|   (the "add the batch"
               idea -- uses direction, subject to cancellation; exact if inputs
               are aligned, since then G = da_sum (x) x)
  sum_abs    : row_i=sum_b|da[b,i]|,    col_j=sum_b|x[b,j]|    (L1, no cancellation)
  l2norm     : row_i=||da[:,i]||,       col_j=||x[:,j]||       (Cauchy-Schwarz bound)
Reference ceiling: rowexact (top rows by l2 grad-row norm, those rows exact over
all cols). Scored by top-K overlap and selQ (energy) vs the oracle top-K of |G|.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import cifar_sparse_vs_dense as C
from cifar_sparse_vs_dense import ResFFN, D_DENSE

DEV = C.DEVICE
WIDTH = 2048              # residual width (larger -> candidates are genuinely sparse vs D^2)
TRAIN_EPOCHS = 6
MEAS_BATCHES = 3
K_FRAC = 0.01            # grow budget ~ realistic (1% of D^2, ~ what we'd actually grow)
ROWEXACT_P = 256          # top rows kept exact (p*in must exceed K)
PERSAMPLE_M = 32          # samples whose per-sample rank-1 top cells we union
PERSAMPLE_AB = 64         # top-a rows x top-b cols per sample


def topk_idx(t, K):
    flat = t.reshape(-1)
    return torch.topk(flat, min(K, flat.numel())).indices


def main():
    C.EPOCHS = TRAIN_EPOCHS
    tr, te = C.load_data()
    trl = torch.utils.data.DataLoader(tr, batch_size=C.BATCH, shuffle=True,
                                      num_workers=4, drop_last=True, persistent_workers=True)
    tel = torch.utils.data.DataLoader(te, batch_size=512, shuffle=False,
                                      num_workers=2, persistent_workers=True)
    import os
    model = ResFFN(WIDTH, sparse=False).to(DEV)
    cache = f"data/cifar_dense_w{WIDTH}_e{TRAIN_EPOCHS}.pt"
    if os.path.exists(cache):
        model.load_state_dict(torch.load(cache, map_location=DEV))
        print(f"loaded cached dense ResFFN (width {WIDTH})\n", flush=True)
    else:
        print(f"training dense ResFFN (width {WIDTH}) {TRAIN_EPOCHS} epochs...", flush=True)
        acc = C.train(model, False, trl, tel, "dense")
        torch.save(model.state_dict(), cache)
        print(f"  {acc:.1f}% test (cached)\n", flush=True)

    layers = {"stem": model.stem}
    for i, b in enumerate(model.blocks):
        layers[f"b{i}.w1(LN)"] = b.w1
        layers[f"b{i}.w2(GELU)"] = b.w2
    inp, gout = {}, {}
    for name, m in layers.items():
        m.register_forward_hook(lambda mod, i, o, n=name: inp.__setitem__(n, i[0].detach()))
        m.register_full_backward_hook(lambda mod, gi, go, n=name: gout.__setitem__(n, go[0].detach()))

    methods = ["sum_signed", "sum_abs", "l2norm", "persample", "rowexact"]
    acc_ov = {n: {mth: [0.0, 0.0, 0] for mth in methods} for n in layers}  # overlap, selQ, cnt
    union_frac = {n: [0.0, 0] for n in layers}   # |union| / (in*out)

    it = iter(trl)
    for _ in range(MEAS_BATCHES):
        xb, yb = next(it); xb, yb = xb.to(DEV), yb.to(DEV)
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb), yb).backward()
        for name in layers:
            x, da = inp[name], gout[name]
            in_f, out_f = x.shape[1], da.shape[1]
            G = da.t() @ x
            Gabs = G.abs()
            K = max(1, int(K_FRAC * in_f * out_f))
            oracle = topk_idx(Gabs, K)
            oset = set(oracle.tolist())
            oenergy = G.reshape(-1)[oracle].pow(2).sum().item() + 1e-30

            def score_block(r, c):
                s = r.unsqueeze(1) * c.unsqueeze(0)          # (out,in) rank-1 cell scores
                idx = topk_idx(s, K)
                ov = len(oset & set(idx.tolist())) / K
                selq = G.reshape(-1)[idx].pow(2).sum().item() / oenergy
                return ov, selq

            scorings = {
                "sum_signed": (da.sum(0).abs(), x.sum(0).abs()),
                "sum_abs": (da.abs().sum(0), x.abs().sum(0)),
                "l2norm": (da.norm(dim=0), x.norm(dim=0)),
            }
            for mth, (r, c) in scorings.items():
                ov, sq = score_block(r, c)
                a = acc_ov[name][mth]; a[0] += ov; a[1] += sq; a[2] += 1

            # per-sample rank-1 union: each of M samples contributes its exact
            # top-(a x b) cells (rank-1 -> norm-ranking exact); union them, then
            # evaluate the exact FULL-BATCH G on the union and take top-K.
            contrib = da.norm(dim=1) * x.norm(dim=1)
            samp = torch.topk(contrib, min(PERSAMPLE_M, x.shape[0])).indices
            a_, b_ = min(PERSAMPLE_AB, out_f), min(PERSAMPLE_AB, in_f)
            cand = set()
            for si in samp.tolist():
                tr = torch.topk(da[si].abs(), a_).indices
                tc = torch.topk(x[si].abs(), b_).indices
                flat = (tr.unsqueeze(1) * in_f + tc.unsqueeze(0)).reshape(-1)
                cand.update(flat.tolist())
            cand_t = torch.tensor(sorted(cand), device=DEV)
            union_frac[name][0] += cand_t.numel() / (in_f * out_f); union_frac[name][1] += 1
            R, Cc = cand_t // in_f, cand_t % in_f
            Gcells = (da[:, R] * x[:, Cc]).sum(0)               # exact full-batch G on union
            loc = torch.topk(Gcells.abs(), min(K, Gcells.numel())).indices
            kept = cand_t[loc]
            ov = len(oset & set(kept.tolist())) / K
            sq = G.reshape(-1)[kept].pow(2).sum().item() / oenergy
            a = acc_ov[name]["persample"]; a[0] += ov; a[1] += sq; a[2] += 1

            # rowexact reference
            rownorm = da.norm(dim=0)
            topR = torch.topk(rownorm, min(ROWEXACT_P, out_f)).indices
            Grows = da[:, topR].t() @ x
            loc = topk_idx(Grows.abs(), K)
            kept = topR[loc // in_f] * in_f + (loc % in_f)
            ov = len(oset & set(kept.tolist())) / K
            sq = G.reshape(-1)[kept].pow(2).sum().item() / oenergy
            a = acc_ov[name]["rowexact"]; a[0] += ov; a[1] += sq; a[2] += 1

    print(f"real CIFAR, rank-1 approximations, top-K overlap / selQ vs oracle "
          f"(K={int(100*K_FRAC)}% cells, rowexact p={ROWEXACT_P}):\n")
    print(f"  {'layer':>14} | " + "  ".join(f"{m:>10}" for m in methods) + f" | {'union|/D2':>9}")
    for name in layers:
        cells = []
        for mth in methods:
            a = acc_ov[name][mth]; c = a[2]
            cells.append(f"{a[0]/c:.2f}/{a[1]/c:.2f}")
        uf = union_frac[name]; ufrac = uf[0] / uf[1] if uf[1] else 0.0
        print(f"  {name:>14} | " + "  ".join(f"{v:>10}" for v in cells) + f" | {ufrac:>8.1%}")
    print("\n  each cell = overlap / selQ  (selQ = energy captured vs oracle at same budget)")
    print("  persample cost ~ |union|*B mults; if |union|/D2 is small it is a genuine sparse win,")
    print(f"  if near 100% it just covers the matrix. (K budget = {int(100*K_FRAC)}% of cells)")


if __name__ == "__main__":
    main()
