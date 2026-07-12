"""Do REAL CIFAR-10 activations have the heavy-tailed column norms that make
MIPS/Cauchy-Schwarz grow-gradient screening pay? (finding 13 said: no on the
isotropic synthetic testbed; only heavy-tailed input-facing activations help.)

Trains the dense ResFFN briefly, then on real CIFAR batches captures, for every
Linear layer, its input x (columns = in-features) and output-gradient da, and
measures:
  * column-norm ||x[:,j]|| skew (top-1%/10% energy share, max/median)
  * how many DISTINCT rows/cols the true top-K |G| cells occupy (concentration)
Layer types: stem (raw pixels), block W1 (input = LayerNorm output),
block W2 (input = GELU output).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import cifar_sparse_vs_dense as C
from cifar_sparse_vs_dense import ResFFN, D_DENSE

DEV = C.DEVICE
TRAIN_EPOCHS = 8
MEAS_BATCHES = 5
K_FRAC = 0.05          # budget = 5% of cells for the concentration metric


def skew(norms):
    v = norms.sort(descending=True).values
    tot = v.sum().item() + 1e-30
    n = v.numel()
    top1 = v[: max(1, n // 100)].sum().item() / tot
    top10 = v[: max(1, n // 10)].sum().item() / tot
    med = v.median().item() + 1e-30
    return top1, top10, v[0].item() / med


def concentration(G, K):
    in_f = G.shape[1]
    idx = torch.topk(G.abs().reshape(-1), K).indices
    return torch.unique(idx // in_f).numel(), torch.unique(idx % in_f).numel()


def main():
    C.EPOCHS = TRAIN_EPOCHS
    train_set, test_set = C.load_data()
    trl = torch.utils.data.DataLoader(train_set, batch_size=C.BATCH, shuffle=True,
                                      num_workers=4, drop_last=True, persistent_workers=True)
    tel = torch.utils.data.DataLoader(test_set, batch_size=512, shuffle=False,
                                      num_workers=2, persistent_workers=True)

    model = ResFFN(D_DENSE, sparse=False).to(DEV)
    print(f"training dense ResFFN {TRAIN_EPOCHS} epochs for realistic activations...", flush=True)
    acc = C.train(model, False, trl, tel, "dense")
    print(f"  trained to {acc:.1f}% test\n", flush=True)

    layers = {"stem": model.stem}
    for i, b in enumerate(model.blocks):
        layers[f"b{i}.w1(LN)"] = b.w1
        layers[f"b{i}.w2(GELU)"] = b.w2
    layers["head"] = model.head

    inp, gout = {}, {}
    for name, m in layers.items():
        m.register_forward_hook(lambda mod, i, o, n=name: inp.__setitem__(n, i[0].detach()))
        m.register_full_backward_hook(lambda mod, gi, go, n=name: gout.__setitem__(n, go[0].detach()))

    acc_stats = {n: [0.0] * 5 + [0, 0, 0] for n in layers}   # c1,c10,cmax, rows,cols, in,out,cnt
    model.train()
    it = iter(trl)
    for _ in range(MEAS_BATCHES):
        xb, yb = next(it)
        xb, yb = xb.to(DEV), yb.to(DEV)
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb), yb).backward()
        for name in layers:
            x, da = inp[name], gout[name]
            in_f, out_f = x.shape[1], da.shape[1]
            cn = x.norm(dim=0)
            G = da.t() @ x
            K = max(1, int(K_FRAC * in_f * out_f))
            rr, cc = concentration(G, K)
            c1, c10, cmax = skew(cn)
            a = acc_stats[name]
            a[0] += c1; a[1] += c10; a[2] += cmax
            a[3] += rr; a[4] += cc
            a[5], a[6] = in_f, out_f; a[7] += 1

    print(f"real CIFAR activations, column-norm skew + top-{int(100*K_FRAC)}%-cell concentration:")
    print(f"  {'layer':>14} {'in':>6} {'out':>5} {'col top1%':>9} {'col top10%':>10} "
          f"{'col max/med':>11} {'topK rows':>10} {'topK cols':>10}")
    for name in layers:
        a = acc_stats[name]; c = a[7]
        in_f, out_f = a[5], a[6]
        print(f"  {name:>14} {in_f:>6} {out_f:>5} {a[0]/c:>9.3f} {a[1]/c:>10.3f} "
              f"{a[2]/c:>11.1f} {a[3]//c:>7}/{out_f:<4} {a[4]//c:>6}/{in_f:<4}")
    print("\n  heavy tail => col top10% >> 0.10 and/or col max/med >> 2, and topK cols << in")


if __name__ == "__main__":
    main()
