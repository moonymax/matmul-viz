"""Representational capacity: block-sparse vs cell-sparse at EQUAL params (real CIFAR).

Same residual-FFN, same operator size S x S in the body, same active-param budget
(density rho). Body layers are one of:
  dense    : full S x S               (ceiling; 1/rho x more params)
  cell     : DynamicSparseLinear       (free scattered cells, grow-and-prune)
  block-b  : BlockSparseLinear b x b   (moveable dense blocks, grow-and-prune)
cell vs block at the SAME params answers: does forcing connections into blocks
cost capacity vs free cells? (finding 15 -> importance may cluster -> little cost.)
"""
from __future__ import annotations

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import cifar_sparse_vs_dense as C
from dynamic_sparse import DynamicSparseLinear, set_capture, rigl_update, cosine_drop_fraction
from block_sparse import BlockSparseLinear

DEV = C.DEVICE
S = 512
L = 4
IN, CLASSES = 3072, 10
DENSITY = 0.25
EPOCHS = 20
BATCH = 256
LR, WD = 1e-3, 0.05
RIGL_EVERY = 50
SEED = 0
SUBSET_TRAIN = 5000     # 1/10 of CIFAR-10 for fast iteration (~2 min total)
SUBSET_TEST = 2000


def make_body(kind, gen, width):
    k = round(DENSITY * width * width)
    if kind == "dense":
        return nn.Linear(width, width, bias=False)
    if kind == "cell":
        return DynamicSparseLinear(width, width, k, generator=gen)
    if kind.startswith("block"):
        b = int(kind[5:])
        return BlockSparseLinear(width, width, b, n_active=k // (b * b), generator=gen)
    raise ValueError(kind)


def dense_param_count(width):
    return 2 * L * width * width + (IN + 1 + 2 * L + 2 + CLASSES) * width + CLASSES


class Net(nn.Module):
    def __init__(self, kind, gen, width=S):
        super().__init__()
        self.kind = kind
        self.width = width
        self.stem = nn.Linear(IN, width)
        self.lns = nn.ModuleList([nn.LayerNorm(width) for _ in range(L)])
        self.w1 = nn.ModuleList([make_body(kind, gen, width) for _ in range(L)])
        self.w2 = nn.ModuleList([make_body(kind, gen, width) for _ in range(L)])
        self.head_ln = nn.LayerNorm(width)
        self.head = nn.Linear(width, CLASSES)
        self._cap = {}      # for block layers: (layer -> (x, grad_out))
        if kind.startswith("block"):
            for m in list(self.w1) + list(self.w2):
                m.register_forward_hook(self._fhook)
                m.register_full_backward_hook(self._bhook)

    def _fhook(self, mod, inp, out):
        self._cap.setdefault(id(mod), [None, None])[0] = inp[0].detach()

    def _bhook(self, mod, gin, gout):
        self._cap.setdefault(id(mod), [None, None])[1] = gout[0].detach()

    def forward(self, x):
        x = self.stem(x)
        for i in range(L):
            h = self.lns[i](x)
            x = x + self.w2[i](F.gelu(self.w1[i](h)))
        return self.head(self.head_ln(x))

    def grow(self, opt, frac):
        if self.kind == "cell":
            rigl_update(self, opt, frac)
        elif self.kind.startswith("block"):
            for m in list(self.w1) + list(self.w2):
                cx, cg = self._cap.get(id(m), [None, None])
                if cx is not None and cg is not None:
                    m.grow_prune(opt, cx, cg, n_swap=max(1, round(frac * m.n_active)))


def n_params(model):
    tot = 0
    for m in model.modules():
        if isinstance(m, DynamicSparseLinear):
            tot += m.values.numel()
        elif isinstance(m, BlockSparseLinear):
            tot += m.blocks.numel()
        elif len(list(m.children())) == 0:
            tot += sum(p.numel() for p in m.parameters(recurse=False) if p.requires_grad)
    return tot


def train(label, kind, trl, tel, width=S):
    gen = torch.Generator().manual_seed(SEED + 2)
    model = Net(kind, gen, width).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = EPOCHS * len(trl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    is_sparse = kind != "dense"
    step, best = 0, 0.0
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in trl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            frac = cosine_drop_fraction(step, steps)
            grow = is_sparse and step % RIGL_EVERY == 0 and frac > 0
            if grow and kind == "cell":
                set_capture(model, True)
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(xb), yb).backward()
            if grow:
                model.grow(opt, frac)
                if kind == "cell":
                    set_capture(model, False)
            opt.step(); sched.step(); step += 1
        # eval
        model.eval(); cor = tot = 0
        with torch.no_grad():
            for xb, yb in tel:
                xb, yb = xb.to(DEV), yb.to(DEV)
                cor += (model(xb).argmax(1) == yb).sum().item(); tot += yb.numel()
        acc = 100.0 * cor / tot; best = max(best, acc)
        print(f"    [{label}] epoch {epoch+1:2d}/{EPOCHS}  test {acc:5.2f}%  "
              f"({time.time()-t0:5.1f}s)", flush=True)
    return best, n_params(model), time.time() - t0


def main():
    torch.manual_seed(SEED)
    tr, te = C.load_data()
    g = torch.Generator().manual_seed(SEED)
    tr = torch.utils.data.Subset(tr, torch.randperm(len(tr), generator=g)[:SUBSET_TRAIN].tolist())
    te = torch.utils.data.Subset(te, torch.randperm(len(te), generator=g)[:SUBSET_TEST].tolist())
    trl = torch.utils.data.DataLoader(tr, batch_size=BATCH, shuffle=True, num_workers=4,
                                      drop_last=True, persistent_workers=True)
    tel = torch.utils.data.DataLoader(te, batch_size=512, shuffle=False, num_workers=2,
                                      persistent_workers=True)
    # param-match a full dense to the sparse/block models (equal-budget dense).
    target = n_params(Net("cell", torch.Generator().manual_seed(SEED + 2), S))
    pm_w = min(range(32, S + 1), key=lambda w: abs(dense_param_count(w) - target))
    print(f"CIFAR block-vs-cell capacity  S={S} L={L} density={DENSITY:.0%} epochs={EPOCHS} "
          f"(subset {SUBSET_TRAIN} train / {SUBSET_TEST} test)")
    print(f"  sparse/block budget = {target:,} params; equal-param full-dense width = {pm_w}\n", flush=True)
    print(f"  {'body':>12} {'width':>6} {'params':>10} {'best test':>10} {'train(s)':>9}")
    runs = [
        ("dense-512", "dense", S),         # over-parameterized reference (3.68M)
        ("dense-pm", "dense", pm_w),        # EQUAL params to sparse/block (narrow full dense)
        ("cell", "cell", S),
        ("block32", "block32", S),
    ]
    for label, kind, width in runs:
        acc, p, secs = train(label, kind, trl, tel, width)
        print(f"  {label:>12} {width:>6} {p:>10,} {acc:>9.2f}% {secs:>8.1f}", flush=True)
    print("\n  dense-pm = full dense at EQUAL params (narrow); cell/block32 = wide-but-sparse")


if __name__ == "__main__":
    main()
