"""CIFAR-10: fair sparse-vs-dense at equal params, with the INPUT STEM sparsified.

Follow-up to `cifar_sparse_vs_dense.py`. There, only the body was sparse and the
dense stem `Linear(3072->S)` dominated the budget — so holding params equal while
widening S poured the budget into the (growing) dense stem and starved the sparse
body. Result: dense won monotonically.

Here the *whole network* is sparse (stem AND body are grow-and-prune sparse
layers), so widening S no longer cannibalizes the body — it's a genuinely wide,
full-input-coverage sparse operator, the true analogue of the synthetic test.

  stem  DynamicSparseLinear(3072 -> S, density rho)
  L x   residual block:  x = x + W2( gelu( W1( LN(x) ) ) )   W1,W2 sparse S x S
  head  Linear(S -> 10)  (dense, tiny) after a final LayerNorm

Dense baseline = `cifar_sparse_vs_dense.ResFFN(256)`. For each density we solve for
S so the sparse model's TOTAL trainable params equal the dense baseline. Exact
grow-and-prune (O(k) forward; dense-gradient probe da^T@x only every few steps),
deadness pruning, AdamW + weight decay, skip connections.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import cifar_sparse_vs_dense as base
from cifar_sparse_vs_dense import IN, CLASSES, L_BLOCKS, D_DENSE, ResFFN
from dynamic_sparse import ApproxDynamicSparseLinear

DENSITIES = [1.0, 0.5, 0.25, 0.10, 0.05, 0.02]
SEED = 0

# The probe-interval sweep (sweep_probe_interval.py) showed grow-and-prune needs
# ~150-240 probes to reach its sweet spot; RIGL_EVERY=20 gives ~195 over 4875
# steps. Approx (O(k)) layers keep the probe cheap even at very wide S, where the
# exact dense-gradient probe (~S^2 per probe) would dominate.
base.RIGL_EVERY = 20

# Sparse layer type: O(k) forward + O(k) approximate grow (top-p x top-q).
Sparse = ApproxDynamicSparseLinear


class SparseBlock(nn.Module):
    def __init__(self, S, density, gen):
        super().__init__()
        self.ln = nn.LayerNorm(S)
        k = round(density * S * S)
        self.w1 = DynamicSparseLinear(S, S, k, generator=gen)   # default init_scale = sqrt(S/k)
        self.w2 = DynamicSparseLinear(S, S, k, generator=gen)

    def forward(self, x):
        return x + self.w2(F.gelu(self.w1(self.ln(x))))


class SparseNet(nn.Module):
    """Fully-sparse residual FFN: sparse stem + sparse blocks, dense head."""
    def __init__(self, S, density, gen):
        super().__init__()
        k_stem = round(density * IN * S)
        self.stem = DynamicSparseLinear(IN, S, k_stem, generator=gen)  # default init scale
        self.blocks = nn.ModuleList([SparseBlock(S, density, gen) for _ in range(L_BLOCKS)])
        self.head_ln = nn.LayerNorm(S)
        self.head = nn.Linear(S, CLASSES)

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        return self.head(self.head_ln(x))


def sparse_params(S, density):
    """Total trainable params of the fully-sparse net (each sparse layer = its k)."""
    k_stem = round(density * IN * S)
    k_body = round(density * S * S)
    body = L_BLOCKS * (2 * S + 2 * k_body)          # per-block LayerNorm(2S) + 2 sparse matrices
    head = S * CLASSES + CLASSES
    head_ln = 2 * S
    return k_stem + body + head_ln + head


def solve_S(target, density):
    lo, hi = 2, 16384
    while lo < hi:
        mid = (lo + hi) // 2
        if sparse_params(mid, density) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def main():
    torch.manual_seed(SEED)
    print(f"device={base.DEVICE}  dense width D={D_DENSE}  blocks={L_BLOCKS}  "
          f"epochs={base.EPOCHS}  batch={base.BATCH}  lr={base.LR}  wd={base.WD}  "
          f"(FULLY-SPARSE: stem+body)\n", flush=True)

    train_set, test_set = base.load_data()
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=base.BATCH, shuffle=True, num_workers=4,
        drop_last=True, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=512, shuffle=False, num_workers=2, persistent_workers=True)

    target = base.analytic_params(D_DENSE, False, 1.0)
    print(f"parameter budget (dense baseline): {target:,}\n", flush=True)

    results = []

    print(">> DENSE baseline", flush=True)
    dense = ResFFN(D_DENSE, sparse=False)
    p = base.count_trainable(dense)
    print(f"   S={D_DENSE}  trainable params={p:,}", flush=True)
    acc = base.train(dense, False, train_loader, test_loader, f"dense S={D_DENSE}")
    results.append(("dense", D_DENSE, 1.0, p, acc))
    print(f"   -> best test {acc:.2f}%\n", flush=True)

    for d in DENSITIES:
        S = solve_S(target, d)
        gen = torch.Generator().manual_seed(SEED + 2)
        print(f">> FULLY-SPARSE density={d:.0%}  solved S={S}", flush=True)
        model = SparseNet(S, d, gen)
        p = base.count_trainable(model)
        print(f"   S={S}  density={d:.0%}  trainable params={p:,}  "
              f"(k_stem={round(d*IN*S):,}  k_body/mat={round(d*S*S):,})", flush=True)
        acc = base.train(model, True, train_loader, test_loader, f"fsparse {d:.0%} S={S}")
        results.append(("fsparse", S, d, p, acc))
        print(f"   -> best test {acc:.2f}%\n", flush=True)

    print("\n" + "=" * 66)
    print(f"CIFAR-10  |  fully-sparse (stem+body)  |  ~equal params ({target:,})")
    print("=" * 66)
    print(f"  {'model':>8} {'S':>6} {'density':>8} {'params':>11} {'best test':>10}")
    for name, S, d, p, acc in results:
        print(f"  {name:>8} {S:>6} {d:>7.0%} {p:>11,} {acc:>9.2f}%")


if __name__ == "__main__":
    main()
