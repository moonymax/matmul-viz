"""CIFAR-10: parameter-equal dense FFN vs. dynamic-sparse FFN.

CIFAR-10 is a problem that is *too big* for a small plain FFN — exactly the
regime where our synthetic findings say a wider-but-sparse layer should beat a
narrow dense one at equal parameter count.

Both models are the same residual FFN:
    stem  Linear(3072 -> S)   (dense, has bias)
    L x   residual block:  x = x + W2( gelu( W1( LN(x) ) ) )    W1,W2: S x S
    head  Linear(S -> 10)     (dense, has bias) after a final LayerNorm

  * DENSE baseline: S = D_DENSE, W1/W2 are full dense (density 1.0).
  * SPARSE model:   W1/W2 are DynamicSparseLinear (grow-and-prune, deadness
    pruning). We sweep density; for each density we solve for the matrix size S
    that makes the SPARSE model's TOTAL trainable params equal the dense
    baseline. Higher density -> smaller S (a linear param constraint rho*S^2 =
    const on the body), lower density -> wider-but-sparser residual stream.

Both use AdamW (weight decay) and skip connections. Sparse pruning removes only
weights that are simultaneously small in magnitude, momentum, and gradient.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from dynamic_sparse import (
    DynamicSparseLinear, set_capture, rigl_update, cosine_drop_fraction,
)

IN, CLASSES = 3072, 10
D_DENSE = 256            # dense baseline residual width
L_BLOCKS = 4             # residual blocks
EPOCHS = 25
BATCH = 256
LR = 1e-3
WD = 0.05                # AdamW weight decay (both models)
RIGL_EVERY = 100         # grow-and-prune probe interval (sparse only)
SEED = 0
DENSITIES = [1.0, 0.5, 0.25, 0.10, 0.05, 0.02]

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    def __init__(self, S, sparse, density, gen):
        super().__init__()
        self.ln = nn.LayerNorm(S)
        if sparse:
            k = round(density * S * S)
            scale = math.sqrt(S / k)
            self.w1 = DynamicSparseLinear(S, S, k, init_scale=scale, generator=gen)
            self.w2 = DynamicSparseLinear(S, S, k, init_scale=scale, generator=gen)
        else:
            self.w1 = nn.Linear(S, S, bias=False)
            self.w2 = nn.Linear(S, S, bias=False)

    def forward(self, x):
        return x + self.w2(F.gelu(self.w1(self.ln(x))))


class ResFFN(nn.Module):
    def __init__(self, S, sparse=False, density=1.0, gen=None):
        super().__init__()
        self.stem = nn.Linear(IN, S)
        self.blocks = nn.ModuleList(
            [Block(S, sparse, density, gen) for _ in range(L_BLOCKS)]
        )
        self.head_ln = nn.LayerNorm(S)
        self.head = nn.Linear(S, CLASSES)

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        return self.head(self.head_ln(x))


def analytic_params(S, sparse, density):
    """Total trainable params (counting a sparse layer as its k active weights)."""
    stem = IN * S + S
    head_ln = 2 * S
    head = S * CLASSES + CLASSES
    per_block_ln = 2 * S
    per_matrix = round(density * S * S) if sparse else S * S
    body = L_BLOCKS * (per_block_ln + 2 * per_matrix)
    return stem + body + head_ln + head


def solve_S(target, density):
    """Smallest S whose sparse model's total params >= target (monotone in S)."""
    lo, hi = 2, 8192
    while lo < hi:
        mid = (lo + hi) // 2
        if analytic_params(mid, True, density) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def count_trainable(model):
    # a DynamicSparseLinear stores exactly its k active weights as `values`, so
    # a plain parameter count already reflects the true sparse param budget.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
ROOT = "./data/cifar10"


def _preload(split):
    """Decode all PNGs of a split into a uint8 tensor (N,3,32,32) once; cache."""
    cache = f"{ROOT}/{split}_u8.pt"
    import os
    if os.path.exists(cache):
        d = torch.load(cache)
        return d["x"], d["y"]
    ds = torchvision.datasets.ImageFolder(f"{ROOT}/{split}", transform=T.PILToTensor())
    xs = torch.empty(len(ds), 3, 32, 32, dtype=torch.uint8)
    ys = torch.empty(len(ds), dtype=torch.long)
    for i in range(len(ds)):
        img, y = ds[i]
        xs[i], ys[i] = img, y
    torch.save({"x": xs, "y": ys}, cache)
    return xs, ys


class CIFAR(torch.utils.data.Dataset):
    """In-memory CIFAR-10 with cheap tensor-side augmentation (pad+crop+flip)."""
    def __init__(self, x_u8, y, train):
        self.x = x_u8
        self.y = y
        self.train = train

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        img = self.x[i].float() / 255.0
        if self.train:
            img = F.pad(img.unsqueeze(0), (4, 4, 4, 4), mode="reflect").squeeze(0)
            top = torch.randint(0, 9, (1,)).item()
            left = torch.randint(0, 9, (1,)).item()
            img = img[:, top:top + 32, left:left + 32]
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[2])
        img = (img - MEAN) / STD
        return img.reshape(-1), self.y[i]


def load_data():
    xtr, ytr = _preload("train")
    xte, yte = _preload("test")
    return CIFAR(xtr, ytr, train=True), CIFAR(xte, yte, train=False)


# --------------------------------------------------------------------------- #
# train / eval
# --------------------------------------------------------------------------- #
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return 100.0 * correct / total


def train(model, sparse, train_loader, test_loader, label):
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    total_steps = EPOCHS * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=total_steps, pct_start=0.1
    )
    step = 0
    best = 0.0
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            frac = cosine_drop_fraction(step, total_steps)
            probe = sparse and step % RIGL_EVERY == 0 and frac > 0
            if probe:
                set_capture(model, True)   # capture x + da for the dense-grad probe
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            if probe:
                rigl_update(model, opt, frac)
                set_capture(model, False)
            opt.step()
            sched.step()
            step += 1
        acc = evaluate(model, test_loader)
        best = max(best, acc)
        print(f"    [{label}] epoch {epoch+1:2d}/{EPOCHS}  test {acc:5.2f}%  "
              f"(best {best:5.2f}%)  {time.time()-t0:5.1f}s", flush=True)
    return best


def main():
    torch.manual_seed(SEED)
    print(f"device={DEVICE}  dense width D={D_DENSE}  blocks={L_BLOCKS}  "
          f"epochs={EPOCHS}  batch={BATCH}  lr={LR}  wd={WD}\n", flush=True)

    train_set, test_set = load_data()
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH, shuffle=True, num_workers=4,
        drop_last=True, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=512, shuffle=False, num_workers=2,
        persistent_workers=True)

    target = analytic_params(D_DENSE, False, 1.0)
    print(f"parameter budget (dense baseline): {target:,}\n", flush=True)

    results = []

    # --- dense baseline ---
    print(">> DENSE baseline", flush=True)
    dense = ResFFN(D_DENSE, sparse=False)
    p = count_trainable(dense)
    print(f"   S={D_DENSE}  trainable params={p:,}", flush=True)
    acc = train(dense, False, train_loader, test_loader, f"dense S={D_DENSE}")
    results.append(("dense", D_DENSE, 1.0, p, acc))
    print(f"   -> best test {acc:.2f}%\n", flush=True)

    # --- sparse sweep (param-equal) ---
    for d in DENSITIES:
        S = solve_S(target, d)
        gen = torch.Generator().manual_seed(SEED + 2)
        print(f">> SPARSE density={d:.0%}  solved S={S}", flush=True)
        model = ResFFN(S, sparse=True, density=d, gen=gen)
        p = count_trainable(model)
        print(f"   S={S}  density={d:.0%}  trainable params={p:,}  "
              f"(dense-equiv {2*L_BLOCKS*S*S:,})", flush=True)
        acc = train(model, True, train_loader, test_loader, f"sparse {d:.0%} S={S}")
        results.append((f"sparse", S, d, p, acc))
        print(f"   -> best test {acc:.2f}%\n", flush=True)

    # --- summary ---
    print("\n" + "=" * 64)
    print(f"CIFAR-10  |  ~equal params ({target:,})  |  {L_BLOCKS} blocks")
    print("=" * 64)
    print(f"  {'model':>8} {'S':>6} {'density':>8} {'params':>11} {'best test':>10}")
    for name, S, d, p, acc in results:
        print(f"  {name:>8} {S:>6} {d:>7.0%} {p:>11,} {acc:>9.2f}%")


if __name__ == "__main__":
    main()
