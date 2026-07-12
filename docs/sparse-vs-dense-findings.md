# When do sparse layers beat dense? — synthetic findings

Teacher–student experiments in PyTorch (`python/*.py`). A frozen random teacher
generates data; students (dense / sparse / low-rank) approximate it. **All
losses reported are PURE data MSE** — any L1/L2 penalty is used only in the
training gradient and is subtracted out of every reported/compared number.

## Grand synthesis

**Sparse layers beat dense specifically when the network is *undersized* for the
problem.** A sparse `D×D` layer is a *sparse slice of the full-sized operator* —
it keeps access to every input/output dimension, just with few weights, so it
approximates the appropriately-sized network by sampling it broadly. A small
*dense* model, to hit the same budget, must shrink its dimension (a bottleneck /
low-rank projection) and throw whole dimensions away. On a high-dimensional /
incompressible problem, broad sparse sampling beats low-rank projection. But when
a full-rank dense model *can* be sized to the problem's intrinsic dimension, that
dense model ties or beats sparse — sparse only wins when you can't afford a
dimension-spanning dense model. The advantage is **representational coverage per
parameter**, not compute (a sparse `D`-layer still runs full-`D` activations).

## Findings since last commit

### 1. Grow-and-prune (dynamic sparse) — `dynamic_sparse.py`
- `DynamicSparseLinear` (exact, masked-dense: dense `weight.grad` gives the grow
  signal at inactive cells) and `ApproxDynamicSparseLinear` (O(k), never forms
  the `n²` gradient).
- **Grow-and-prune beats a fixed random mask** at equal budget (~1.2–1.3× lower
  MSE). Adaptivity — moving connections to where they matter — is the value, not
  sparsity per se (a static sparse model even loses to a small dense one).
- **Approximate probe = the POC method**: capture the layer input `x` and
  output-gradient `da` (only on probe steps, via a hook), score just the top-p
  error rows × top-q activation columns (a tiny `p×q` matmul), grow the best
  inactive cell(s). Lands within ~10% of the exact full-gradient grow, and adds
  only **~4% amortized wall-clock** (probe every 25 steps; the rest is O(k)).

### 2. Pruning criterion — deadness, not magnitude
- Prune only weights that are near-zero **AND** stable: rank by
  `|w| + |momentum| + |gradient|` (all normalized) so an actively-moving weight
  is never pruned. Matches the JS POC rule. On these problems it scored ≈ plain
  magnitude pruning (marginal difference); doesn't change any conclusion.

### 3. Equal params: full-rank dense ties/beats sparse — `test_lifted_sparse.py`, `sweep_density.py`
- The earlier "sparse beats dense-but-smaller" was a **low-rank confound** — the
  small dense had been forced into a bottleneck. With a genuinely *full-rank*
  small dense (intrinsic-dim problem + frozen lift for the big models), at equal
  params: dense **0.0170** ≈ grow-prune sparse **0.0176** (a tie, dense a hair
  ahead); static sparse **0.0203** is worst.
- Density sweep vs a fixed full-rank dense baseline: sparse needs **~1.6× the
  params (≈10% density)** just to overtake, and the gains are back-loaded
  (near-flat until high density). On a right-sizeable problem, sparse is *not*
  parameter-efficient.

### 4. Undersized dense: sparse wins decisively — `sweep_toosmall.py`
- Make the problem genuinely too big (isotropic, full-rank `D=128`) and force the
  dense model through the **best possible** learned reconstruction bottleneck
  (`m=64`). Reconstruction converges keeping only **56.7% of input variance** —
  43% is *unavoidably* lost — so the too-small dense floors at **0.177**.
- Sparse (full-`D`, grow-and-prune) **overtakes it at just 5% density / 0.2× the
  params**, and is **5.3× lower loss at equal params** (0.033 vs 0.177), rising
  smoothly to the full-dense ceiling at 100%. This is the regime where sparsity
  pays: the dense model can't be made wide enough to see the whole problem.

### 5. Regularization — weight decay yes, L1/L2 no — `test_l1_effect.py`, `test_l1_sparse.py`, `test_reg_sparse.py`
Measured on pure data MSE (penalty excluded — verified by decomposition):
- **L1** hurts *everything*: +31% for grow-prune sparse (coef 1e-4), +88% for the
  full dense. Its constant `λ·sign(w)` pull kills useful small weights.
- **L2** (loss penalty) is neutral at a tiny coef and catastrophic if too strong
  (Adam mishandles the `Σw²` scale).
- **Weight decay** (decoupled, AdamW) is the **only net-positive** regularizer for
  grow-and-prune sparse: **−5 to −7% at wd≈1.0** (clear peak; over-decays by 2.0).
  Proportional decoupled shrink keeps unimportant weights decaying to prunable
  near-zero while data-reinforced weights survive.
- **Recommendation:** regularize the sparse models with `AdamW(weight_decay≈1.0)`,
  not L1. (Retune wd for real-scale models; wd≈1.0 is high because these are tiny
  synthetic nets.) L1 is disabled by default in `sweep_toosmall.py`.

## Caveats
- Single-seed, small synthetic nets; regularization effects are modest (~few %).
- The headline conclusions (grow-prune > static; dense ties sparse when
  right-sizeable; sparse wins when dense is undersized) are robust to the
  pruning/regularization choices above.
