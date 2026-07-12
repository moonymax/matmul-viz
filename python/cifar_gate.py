"""Per-cell compute-gating for the grow gradient: does an exact Cauchy-Schwarz
gate let an efficient kernel beat the dense gradient?

Each cell G[i,j] = <da[:,i], x[:,j]> costs B mults. Gate: skip cell (i,j) if
  bound(i,j) = ||da[:,i]|| * ||x[:,j]||  <  tau_K   (the true K-th largest |G|)
This is EXACT (bound >= |G|), so no true top-K cell is ever skipped. The gate is
~1 mult/cell (norms precomputed, (out+in)*B amortized). Ops:
  dense  = out*in*B
  gated  = out*in*1 (gate) + (out+in)*B (norms) + C_pass*B (survivors)
where C_pass = #cells with bound >= tau_K. Speedup ~ out*in / C_pass when the
survivor term dominates. C_pass measures how LOOSE the CS bound is (false
positives from high-norm but poorly-aligned cells). We report C_pass and speedup
at recall targets on REAL CIFAR (D=2048, cached model).
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
CMULTS = [1, 2, 5, 10, 20, 50, 100]      # C_pass tested as multiples of K


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
        acc = C.train(model, False, trl, tel, "dense")
        torch.save(model.state_dict(), cache)
        print(f"trained {acc:.1f}% (cached)\n", flush=True)

    layers = {"stem": model.stem}
    for i, b in enumerate(model.blocks):
        layers[f"b{i}.w1(LN)"] = b.w1
        layers[f"b{i}.w2(GELU)"] = b.w2
    inp, gout = {}, {}
    for name, m in layers.items():
        m.register_forward_hook(lambda mod, i, o, n=name: inp.__setitem__(n, i[0].detach()))
        m.register_full_backward_hook(lambda mod, gi, go, n=name: gout.__setitem__(n, go[0].detach()))

    # accumulators: recall at each C-multiple, and exact C_pass at tau_K
    rec = {n: {cm: [0.0, 0] for cm in CMULTS} for n in layers}
    cpass = {n: [0.0, 0] for n in layers}

    it = iter(trl)
    for _ in range(MEAS_BATCHES):
        xb, yb = next(it); xb, yb = xb.to(DEV), yb.to(DEV)
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb), yb).backward()
        for name in layers:
            x, da = inp[name], gout[name]
            in_f, out_f = x.shape[1], da.shape[1]
            B = x.shape[0]
            G = da.t() @ x
            Gabs = G.abs().reshape(-1)
            ncell = in_f * out_f
            K = max(1, int(K_FRAC * ncell))
            tauK = torch.topk(Gabs, K).values[-1].item()          # true K-th largest |G|
            true_set = set(torch.topk(Gabs, K).indices.tolist())
            bound = (da.norm(dim=0).unsqueeze(1) * x.norm(dim=0).unsqueeze(0)).reshape(-1)
            # exact survivors under the CS gate at tau_K
            cp = int((bound >= tauK).sum().item())
            cpass[name][0] += cp / ncell; cpass[name][1] += 1
            # recall of true top-K within the top-(cm*K) cells by bound
            for cm in CMULTS:
                Ctest = min(cm * K, ncell)
                topb = set(torch.topk(bound, Ctest).indices.tolist())
                r = len(true_set & topb) / K
                a = rec[name][cm]; a[0] += r; a[1] += 1

    print(f"real CIFAR D={WIDTH}, per-cell Cauchy-Schwarz gate (K={K_FRAC:.0%} of cells, B={C.BATCH})")
    print(f"C_pass = cells surviving the exact gate at tau_K (as % of all cells);")
    print(f"speedup ~ 1 / (C_pass_frac)  when survivor compute dominates.\n")
    hdr = "  ".join(f"{cm}K" for cm in CMULTS)
    print(f"  {'layer':>14} {'C_pass%':>8} {'speedup':>8} | recall@ topC = {hdr}")
    for name in layers:
        cf = cpass[name][0] / cpass[name][1]
        sp = 1.0 / cf if cf > 0 else float("inf")
        recs = "  ".join(f"{rec[name][cm][0]/rec[name][cm][1]:.2f}" for cm in CMULTS)
        print(f"  {name:>14} {cf:>7.2%} {sp:>7.1f}x | {recs}")
    print("\n  gate wins if C_pass% << 100%; recall@C shows how many cells you must")
    print("  actually compute (top-C by bound) to capture the true top-K.")


if __name__ == "__main__":
    main()
