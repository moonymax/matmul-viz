# Two-layer sparse vs. dense: accuracy and operation efficiency

The `/train` demo trains three student networks on data from a fixed **2-layer
teacher**:

```
y = tanh(W2ᵗ · tanh(W1ᵗ · x))          (two 4×4 matrices, tanh hidden + output)
```

All three students are 2-layer with the same two tanh activations:

| Tab | Architecture | Trainable params |
|---|---|---:|
| Dense 4×4 | `tanh(W₂ · tanh(W₁ · x))`, two 4×4 | 32 |
| Sparse | `tanh(Uᵀ·(W₂·tanh(W₁·U·x)))`, two 16×16, `k` cells/layer, grow-and-prune + L1 | `2k` |
| Dense 16×16 | same lifted form, both 16×16 fully trainable | 512 |

`U` is a fixed orthonormal `16×4` lift and `Uᵀ` the down-projection; they exist
only to give the lifted models a larger scaffold on comparable 4-D data and are
excluded from the op counts below.

## ⚠️ The L1 penalty must be netted out of the sparse "final loss"

**The sparse model is trained with an L1 penalty on its active weights; the two
dense models have no L1. So the sparse tab's reported "Total / Final loss"
includes a term the dense tabs never pay.** To compare *fit quality* fairly you
must subtract it:

```
data loss (sparse) = total loss − L1 penalty
```

Worked example, sparse with `k = 144` cells/layer at the end of training:

| Component | Value |
|---|---:|
| Total loss (the "Final loss" stat) | 0.0309 |
| − L1 penalty | 0.0224 |
| **= data loss (comparable to the dense tabs)** | **≈ 0.008** |

Here the L1 term is ~72% of the reported number, so comparing raw totals across
tabs overstates the sparse model's disadvantage by roughly that much. The UI
therefore reports **"Final data loss"** = `dataFullLoss` (L1 already removed) as
the cross-tab comparison metric, alongside **"Total loss"** = the training
objective (data + L1) and the **"L1 penalty"** term itself, so the split is
always visible.

## Accuracy — data loss only (fair)

| Student | Params | Data loss |
|---|---:|---:|
| Dense 4×4 (teacher-matched) | 32 | 0.0051 |
| Sparse (`k = 144`/layer) | 288 | ≈ 0.008 |
| Dense 16×16 | 512 | 0.00067 |

- **Dense 4×4** fits well with few parameters because its architecture *is* the
  teacher's — the 32 weights are perfectly placed.
- **Dense 16×16** fits best; large capacity nails a genuinely 2-layer target.
- **Sparse**, once the L1 term is removed, fits nearly as well as the matched
  net — capacity/placement, not raw count, is what separates the three.

### Sparse density sweep (reported total loss, i.e. *with* L1)

| `k` (cells/layer) | ≈ nonzeros/row | Total loss (incl. L1) |
|---:|---:|---:|
| 18 | 1.1 | 0.070 |
| 36 | 2.25 | 0.052 |
| 144 | 9.0 | 0.031 |

More connections per hidden unit → richer features → lower loss, with
diminishing returns. Remember these totals still include L1; the underlying data
loss is lower and closes most of the gap to the matched net.

## Nonzeros-per-row scaling

Average nonzeros per row `= k / n` for an `n×n` layer with `k` active cells.

| Budget policy | `k` | nonzeros/row | forward/backward ops |
|---|---|---|---|
| Fixed param count | `const` | `k/n` → O(1/n) | O(k) |
| Fixed sparsity fraction `s` | `s·n²` | `s·n` → O(n) | O(s·n²) |
| Fixed fanin `d` per row | `d·n` | `d` → constant | O(d·n) |

A hidden unit computes `tanh(rowᵢ · z)`, so it needs several nonzeros per row to
form a useful input direction. Only the **fixed-fanin** policy keeps per-unit
richness constant as the matrix grows, at `O(n)` params/ops vs dense `O(n²)`.

## Operation counts (MACs per example, mixing matrices only)

Per trainable layer, `n = 16`, `k` = active cells:

| Stage | Sparse (`k`) | Dense (`n²` = 256) |
|---|---:|---:|
| Forward `W·h` | `k` | 256 |
| Backward (weight grads) | `k` | 256 |
| Adam update | `k` | 256 |

Forward and backward both scale as `O(k)` (sparse) vs `O(n²)` (dense). The tanh
activations add one cheap elementwise op per unit (`O(n)`), negligible next to
the mixing. Two layers just double every count.

**Caveats.**

1. MAC counts, not wall-clock. Unstructured sparsity usually doesn't beat dense
   hardware until very high sparsity (~90%+) and typically needs *structured*
   patterns (block, 2:4) to map onto dense compute units.
2. At `n = 16` the absolute numbers are tiny; the `O(k)` vs `O(n²)` scaling is
   the point, not the specific integers.
