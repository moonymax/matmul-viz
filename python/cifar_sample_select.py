"""Do a few high-error REAL CIFAR samples reconstruct the grow gradient?

Isotropic synthetic (sample_select.py) said no -- but there every sample is
equally hard. Real classification has heavy-tailed per-sample loss, so a few
hard samples may dominate the gradient G = da^T @ x. Train the dense ResFFN,
then on a large real batch measure, per layer:
  * sample-contribution concentration (energy share of top-m/B samples)
  * top-K overlap of G_m (m samples) vs full-batch G, selecting samples by
    error norm ||da[b]|| vs contribution ||da[b]||*||x[b]|| vs random.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import cifar_sparse_vs_dense as C
from cifar_sparse_vs_dense import ResFFN, D_DENSE

DEV = C.DEVICE
TRAIN_EPOCHS = 8
B_MEAS = 512
K_FRAC = 0.05
MS = [4, 8, 16, 32, 64]


def topk_set(G, K):
    return set(torch.topk(G.abs().reshape(-1), K).indices.tolist())


def main():
    C.EPOCHS = TRAIN_EPOCHS
    tr, te = C.load_data()
    trl = torch.utils.data.DataLoader(tr, batch_size=C.BATCH, shuffle=True,
                                      num_workers=4, drop_last=True, persistent_workers=True)
    tel = torch.utils.data.DataLoader(te, batch_size=512, shuffle=False,
                                      num_workers=2, persistent_workers=True)
    model = ResFFN(D_DENSE, sparse=False).to(DEV)
    print(f"training dense ResFFN {TRAIN_EPOCHS} epochs...", flush=True)
    acc = C.train(model, False, trl, tel, "dense")
    print(f"  {acc:.1f}% test\n", flush=True)

    layers = {"stem": model.stem}
    for i, b in enumerate(model.blocks):
        layers[f"b{i}.w2(GELU)"] = b.w2
    inp, gout = {}, {}
    for name, m in layers.items():
        m.register_forward_hook(lambda mod, i, o, n=name: inp.__setitem__(n, i[0].detach()))
        m.register_full_backward_hook(lambda mod, gi, go, n=name: gout.__setitem__(n, go[0].detach()))

    # one large real batch
    xb, yb = next(iter(torch.utils.data.DataLoader(tr, batch_size=B_MEAS, shuffle=True)))
    xb, yb = xb.to(DEV), yb.to(DEV)
    model.zero_grad(set_to_none=True)
    logits = model(xb)
    per_sample_loss = F.cross_entropy(logits, yb, reduction="none")
    per_sample_loss.mean().backward()

    print(f"real CIFAR, sample-subset grow-gradient reconstruction (B={B_MEAS})")
    print(f"  per-sample CE loss: mean {per_sample_loss.mean():.2f}  max {per_sample_loss.max():.2f}  "
          f"top-32/{B_MEAS} loss share {per_sample_loss.sort(descending=True).values[:32].sum()/per_sample_loss.sum():.2f}\n")

    for name in layers:
        x, da = inp[name], gout[name]
        in_f, out_f = x.shape[1], da.shape[1]
        G = da.t() @ x
        K = max(1, int(K_FRAC * in_f * out_f))
        full = topk_set(G, K)
        enorm = da.norm(dim=1); cnorm = enorm * x.norm(dim=1)
        cs = (cnorm ** 2).sort(descending=True).values
        conc32 = cs[:32].sum().item() / (cs.sum().item() + 1e-30)
        order = {"error": torch.argsort(enorm, descending=True),
                 "contrib": torch.argsort(cnorm, descending=True),
                 "random": torch.randperm(B_MEAS, device=DEV)}
        print(f"  [{name}]  top-32/{B_MEAS} contribution energy {100*conc32:.1f}%")
        print(f"    {'m':>4} | {'error':>7} {'contrib':>8} {'random':>7}  (top-K overlap vs full G)")
        for m in MS:
            row = []
            for h in ("error", "contrib", "random"):
                sel = order[h][:m]
                Gm = da[sel].t() @ x[sel]
                row.append(len(full & topk_set(Gm, K)) / K)
            print(f"    {m:>4} | {row[0]:>7.3f} {row[1]:>8.3f} {row[2]:>7.3f}")
        print()


if __name__ == "__main__":
    main()
