# Sparse vs. dense matrix: accuracy and operation efficiency

This note compares the three models in the `/train` demo. All three learn the
same task — approximate a **nonlinear** target `y = nonlinear(A·x)` (hidden 4×4
matrix `A`) with a **linear** model — on the same dataset, with the same Adam
optimizer.

| Tab | Model | Trainable params | Final full-dataset loss |
|---|---|---:|---:|
| Nonlinear data | plain dense `4×4` (`pred = W·x`) | 16 | **0.420** |
| Sparse matrix | lifted, `18` movable cells (grow-and-prune) | 18 | **0.509** |
| Dense 16×16 | lifted, all `256` cells trainable | 256 | **0.430** |

Both lifted models compute `pred = Uᵀ · W · (U · x)`, where `U` is a **fixed**
`16×4` up-projection (orthonormal columns) and `Uᵀ` is the down-projection. The
lift/unlift exist only to give a larger scaffold something to act on so the
three models see comparable data — they are **not** part of the model capacity
and are **excluded** from the op counts below.

## Why more parameters barely help

The effective transform is `Uᵀ W U`. Because `U` is `16×4` with rank 4, this
collapses to a **4×4 matrix no matter what `W` is**. So:

- The dense `16×16` (256 params) can represent *any* 4×4 effective map — but so
  can the plain `4×4` (16 params). They share the same optimum (~0.42).
- The extra 240 dense parameters live in the null space of the lift and buy
  nothing.
- The task's real ceiling is "best linear fit to a nonlinear map," i.e. a
  rank-4 bottleneck. Everyone hits roughly the same floor.

The sparse model (18 cells) lands close (0.51) despite ~14× fewer parameters,
because the dense capacity was redundant to begin with.

## Operation counts (MACs per example, mixing matrix `W` only)

Lift/unlift costs are excluded by request — they are shared, fixed overhead used
only to make the data comparable. What remains is the cost of the trainable
matrix `W` itself.

Let `n = 16` (scaffold side), `nnz = 18` (active cells in the sparse model).

| Stage | What is computed | Sparse (`nnz`) | Dense (`n²`) |
|---|---|---:|---:|
| Forward | `mixed = W · z` | **18** | **256** |
| Backward (weight grads) | `∂L/∂W[i,j] = back[i]·z[j]` | **18** | **256** |
| Optimizer update | Adam step per parameter | 18 | 256 |

- **Forward:** `O(nnz)` vs `O(n²)` → sparse is `256 / 18 ≈ 14×` cheaper.
- **Backward:** same `O(nnz)` vs `O(n²)` — the gradient only exists at the
  active cells for the sparse model, at all cells for the dense one.
- `back = U · (∂L/∂pred)` (turning the output error into a lifted-space error) is
  part of the unlift and is therefore excluded; it is identical for both models.

## Takeaway (linear case)

Sparse achieves comparable accuracy (0.51 vs 0.43) at **~14× fewer parameters
and ~14× fewer mixing operations**, on both the forward and backward pass. But
this is "nearly free" only because the dense matrix is over-provisioned for the
effective rank of the problem — exactly the regime where pruning real networks
works.

## Adding a nonlinearity

Both lifted models were then made genuinely nonlinear by applying an elementwise
activation `φ` in the 16-D space:

```
pred = Uᵀ · φ(W · U·x)
```

This is a 1-hidden-layer net (16 hidden units) with fixed lift/unlift and a
trainable middle matrix `W`. The plain `4×4` tab stays linear as the reference.

**Crucially, `φ` is the *same* nonlinearity that generated the target data:**

```
φ(v) = tanh(2.85·v) + 0.55·sin(5.5·v + 0.35) + 0.18·sin(11·v²)
```

| Model | Params | Linear loss | Nonlinear loss (φ) |
|---|---:|---:|---:|
| Nonlinear data (linear `4×4`) | 16 | 0.420 | 0.420 (unchanged) |
| Sparse (18 cells) | 18 | 0.509 | **0.355** |
| Dense 16×16 | 256 | 0.430 | **0.398** |

Two observations:

1. **Both nonlinear models break the 0.42 linear ceiling** — the activation
   gives them capacity the linear map fundamentally lacked.
2. **The 256-param dense model does *not* beat the 18-param sparse one** here
   (0.398 vs 0.355); they are comparable, and sparse is marginally ahead.

**Why the extra capacity doesn't dominate.** `φ` is strongly oscillatory — its
derivative `φ'(v)` contains a `3.96·v·cos(11·v²)` term that swings wildly once
`|v|` grows past ~1. The dense matrix drives larger pre-activations
`a = W·(U·x)` (16 contributions per unit, so bigger magnitude), landing many
hidden units in that chaotic regime where gradients are noisy and `φ` itself is
erratic. The sparse matrix keeps pre-activations small and tame, so despite far
fewer parameters it optimizes at least as cleanly. (This is sensitive to init
scale and learning rate — the point is that more parameters is not automatically
better when the activation is this irregular.)

### Aside: a *mismatched* activation flips the result

As a control, using a plain `tanh` activation instead of the true `φ` gives:
sparse 0.471, dense **0.193**. There the dense model wins decisively, because a
smooth monotone `tanh` rewards having 16 independent input directions (one rich
feature per hidden unit) and the sparse model (≈1 weight per row) can't form
them. So whether dense capacity helps depends heavily on the activation:
smooth/monotone → dense wins big; matched-but-oscillatory → the advantage
evaporates.

The op-count ratio is unchanged either way (the activation adds one cheap
elementwise op per hidden unit, `O(n)`, negligible next to the `O(nnz)` vs
`O(n²)` mixing): sparse is still ~14× cheaper per step.

**Caveats.**

1. These are MAC counts, not wall-clock. Unstructured sparsity typically does
   not beat dense hardware until very high sparsity (~90%+) and usually needs
   *structured* patterns (block, 2:4) to map onto dense compute units.
2. At `n = 16` the absolute numbers are tiny; it is the `O(nnz)` vs `O(n²)`
   scaling that matters, not these specific integers.
