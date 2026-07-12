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

### 6. CIFAR-10 (real data): wider-but-sparse LOSES to dense — `cifar_sparse_vs_dense.py`
First test on real data. A residual FFN (stem `Linear(3072→S)`, 4 residual blocks
of two square `S×S` matrices with GELU + skip, head `S→10`), AdamW + weight decay,
25 epochs. Dense fixes `S=256`; the sparse model holds **total** trainable params
equal and sweeps density, solving for `S` at each (higher density → smaller `S`).
Deadness pruning. Log: `docs/cifar-sweep-v1.log`.

| model | S | density | params | best test |
|---|---|---|---|---|
| dense | 256 | 100% | 1,316,106 | **61.00%** |
| sparse | 256 | 100% | 1,316,106 | 61.35%  (anchor ≈ dense ✓) |
| sparse | 306 | 50% | 1,321,012 | 60.70% |
| sparse | 348 | 25% | 1,318,582 | 59.75% |
| sparse | 387 | 10% | 1,316,817 | 58.54% |
| sparse | 405 | 5% | 1,318,283 | 57.87% |
| sparse | 417 | 2% | 1,317,615 | 57.29% |

**Result: monotonic — dense wins, sparser is strictly worse.** The opposite of the
synthetic thesis.

**Why it doesn't transfer (the decisive confound):** the dense input stem
`Linear(3072→S)` dominates the budget and is *never* sparsified. As density drops
and `S` grows to hold total params equal, the budget flows into the (growing) dense
stem and **starves the sparse body** — the operator the thesis is actually about:

| density | S | stem params | sparse **body** params |
|---|---|---|---|
| 100% | 256 | 786k (60%) | 524k (40%) |
| 10% | 387 | 1.19M (90%) | 120k (9%) |
| 2% | 417 | 1.28M (97%) | **28k (2%)** |

So the sweep tested "a wider dense input projection with a progressively defunded
body," not "a wider sparse operator." The synthetic experiments had no such stem
(the `D×D` matrices *were* the whole model), which is exactly why the thesis held
there. **The fair analogue on CIFAR requires sparsifying the stem too** (a
full-width end-to-end sparse network), so widening doesn't cannibalize the body —
not yet run.

### 7. Sparse layer made genuinely O(k) — `dynamic_sparse.py` (rewrite)
The exact `DynamicSparseLinear` was "masked-dense": full `S×S` forward/backward,
paying dense cost to get the inactive-cell gradient for free. Rewritten so both
layers share sparse (values/rows/cols) storage and an O(B·k) gather-mul-scatter
forward/backward. The exact grow signal is now reconstructed **separately** on
probe steps as the outer product `G = daᵀ·x` from the captured error `da` and
input `x` — an `(out×in)` materialization, but only every ΔT steps, not on every
forward. Reproduces the pre-rewrite numbers (dynamic 0.164 vs static 0.204).

### 8. Grow-and-prune frequency is a dominant knob — `sweep_probe_interval.py`
How often we grow-and-prune (ΔT) drives most of the benefit. On the D=128 too-big
teacher (10% density), exact grow: probing every 500 steps beats static by only
−12%, every 100 by −31%, every 10 by **−49%** (sweet spot ~every 5–10; every-1
slightly *worse* — over-churn). It's really total connection turnover (each probe
swaps `frac·k`). Approx tracks the same curve ~8–11% behind exact at every
interval. Consequence: the CIFAR v1 sweep (ΔT=100) badly under-probed the sparse
models.

### 9. Approx grow gradient is only faithful at small batch — `probe_approx_quality.py`
Direct MSE of the approximate grow gradient vs the dense gradient `G = daᵀ·x`
(relative MSE = fraction of gradient energy missed), grow/prune fixed at every 5,
D=128, 10% density, top-p×top-q block p=q=48, swept over **batch size**:

| batch | approx relMSE | oracle relMSE (same budget) | selection gap |
|---|---|---|---|
| 1 | 0.261 | 0.212 | 0.048 |
| 8 | 0.602 | 0.335 | 0.267 |
| 32 | 0.678 | 0.355 | 0.323 |
| 128 | 0.711 | 0.361 | 0.350 |
| 1024 | 0.738 | 0.362 | 0.376 |

**The approximation degrades monotonically with batch size** — captures ~74% of
gradient energy at B=1 but misses ~71% at B=128. The oracle (best possible
p·q-cell keep) barely moves (0.21→0.36, a mild coverage effect), so the collapse
is almost entirely **selection error**: the gap approx−oracle blows up 0.05→0.38.

**Why:** `G = Σ_b da_b ⊗ x_b` is a sum of B rank-1 terms. At B=1 it's exactly
rank-1 so `|G[i,j]| = |da_i|·|x_j|` and top-norm rows×cols *are* the largest
entries (norm-ranking optimal). As B grows, large entries come from cross-batch
**correlation** between `da[:,i]` and `x[:,j]`, which per-row/col norms are blind
to. Fixing the approximation at real batch sizes means replacing norm-ranking with
a rank-aware (alignment-aware) selector — the open work item.

### 10. Efficient rank-aware grow gradient at scale — `approx_strategies.py`
Chasing finding 9's fix: at large D (2048) with a fixed candidate budget, measure
each method's **selection quality** = energy captured / *oracle* energy at the same
budget (isolates selection from the coverage term, which otherwise dominates: at
D=2048 a 9216-cell budget is 0.22% of the matrix, so oracle itself keeps only
~3% of total energy). Ops tracked as % of the full gradient (`D²·B`).

Two failed selectors and the fix (B=32, D=2048, budget 9216 cells):
- **norm** (top rows by `‖da_i‖²` × top cols by `‖x_j‖²`, keep block): selQ **0.21**
  at 0.3% ops. Cheap but the block's row×col product structure can't cover the
  scattered large entries of a high-rank gradient.
- **gram** (same block, rows/cols by *exact* grad-row/col norm): selQ 0.25 — the
  block structure, not the scoring, was the limit. Dominated (6.5% ops, no gain).
- **svd** (scattered union of top-singular-direction blocks): also ~no better;
  the largest entries of a sum of B rank-1 terms aren't captured by individual
  singular directions.
- **rowexact** (top-p rows by exact grad-row norm, compute *those rows exactly
  over ALL columns*, keep the true top cells — scattered, no column restriction):
  selQ **0.80 at p=96 (7.8% ops)**, **0.89 at p=192 (12.5% ops)**. This is the win.
  Cost `O(B²D + pBD)` — cheaper than the `O(D²B)` full gradient only for B≪D
  (i.e. only at scale, which is the regime that matters).

Scaling: rowexact degrades with batch (higher rank ⇒ need more exact rows):
p=192 gives selQ 0.95 / 0.89 / 0.80 at B=1 / 32 / 128, and its ops rise (the
`B²·D` gram row-scoring term). Open items: (a) cheaper row selector to cut the
`B²·D` term at large batch; (b) confirm rowexact grow *decisions* train as well as
exact downstream. NOTE: wall-clock on MPS doesn't yet reflect the ops win
(topk/gather overhead) — ops is the reported signal.

### 11. Wall-clock: the real cost of exact grow is the top-K, not the matmul — `bench_rowexact.py`
Benchmarked rowexact vs the full gradient at D=2048 (median of 100 reps, warmed,
device-synced). Producing the grow signal for a D x D layer, batch B=32, on MPS:

| method | ops | %full | wall-clock |
|---|---|---|---|
| full-grad (`daᵀx` matmul only) | 134M | 100% | **227 µs** |
| full + select (matmul + top-K over `D²`) | 134M | 100% | **17,280 µs** |
| rowexact96 | 10.5M | 7.8% | 795 µs |
| rowexact192 | 16.8M | 12.5% | 1,361 µs |

- **vs the bare `daᵀx` GEMM, rowexact is slower** (795 vs 227 µs) despite 13× fewer
  ops — the GEMM is one maximally-optimized fused kernel, rowexact is many small
  unfused ones (Gram, two topks, gather, 2nd matmul), so it is launch/overhead
  bound. The ops win doesn't yet translate to wall-clock.
- **vs the real grow pipeline (`full+select`), rowexact is ~22× faster.** To grow
  you must *select* cells, and top-K over the full `D²`=4.2M-cell matrix costs
  ~17 ms (dominant, ~constant in B). rowexact only sorts its `p×D`=197K-cell strip.
- **The genuine bottleneck of exact grow is the top-K over `D²`, not the matmul —
  and that is exactly what rowexact avoids.** CPU shows the same pattern.
- Implication: a fused rowexact kernel (collapse the Gram/score/gather/2nd-matmul
  launches, avoid materializing the `p×D` strip) should turn the ops advantage
  into a wall-clock one even against the bare GEMM. (next work item)

### 12. Fused Metal top-K selection kernel — `metal/topk_metal.{mm,py}`
Profiling rowexact (`bench_rowexact_breakdown.py`) showed **67% of its wall-clock
is `torch.topk` over the p×D strip** (521 µs); the matmuls are cheap (~5% each).
So the fused kernel targets *selection*, not the matmuls. Built an approximate
top-K on MPS (one command buffer, three chained kernels, no host round-trips):
histogram of `|v|` by **float-bit magnitude** (for x≥0 the raw bits are monotone
in value, so `bin = bits>>18` needs no max reduction) → one-thread scan for the
threshold bin giving ≥K → atomic-append compaction, returning exactly K indices
(so it pipelines like `topk`, no counter readback). **394 µs vs torch.topk 508 µs
(1.3×), overlap@K 0.94, energy 0.98** — plenty for a grow heuristic. Tries that
lost: separate submissions with host round-trips (0.6×); coarse `NB=1024` bins
(overlap 0.58); threadgroup-local histogram (barrier overhead, slower). The win is
modest because MPS is overhead-bound on these small ops; the bigger lever is
upstream (screening, finding 13).

### 13. MIPS screening is data-gated — useless on isotropic, big on heavy-tailed — `measure_column_skew.py`
Grow-cell selection is maximum-inner-product search: `G[i,j]=⟨da[:,i],x[:,j]⟩`, so
Cauchy–Schwarz (`|G[i,j]| ≤ ‖da[:,i]‖·‖x[:,j]‖`) lets you screen out low-norm
columns exactly — **but only if column norms are heavy-tailed and the top-K cells
concentrate in few rows/cols.** Measured (D=2048, B=32, K=9216):
- **Isotropic input (our testbed): screening buys nothing.** Column-norm top-10%
  share 0.12 (uniform=0.10), max/median 1.5; the true top-K cells spread over
  ~1300–1700 of 2048 rows *and* columns. High-rank, diffuse, no tail to prune.
- **Heavy-tailed input: dramatic at the input layer** — layer-1 top-K cells
  collapse to **191 columns** (9% of D), col max/median 40×. But the skew dies at
  the nonlinearity: layer-2 (input `tanh(pre1)`) is back to uniform.
Conclusion: norm-based screening pays only on genuinely heavy-tailed, input-facing
activations (real models have outlier channels — "massive activations"); it is a
no-op on the isotropic synthetic problem and on post-nonlinearity activations. For
the direction-not-magnitude case, LSH/IVF (approximate MIPS) would be the route.
Not implemented — the synthetic testbed is its worst case; revisit on real data.

### 13b. Real CIFAR activations have only a mild tail — `cifar_skew.py`
Followed up on real data: dense ResFFN trained to 55.7%, column-norm skew per
Linear layer over real CIFAR batches.
- **Raw pixels (stem) and LayerNorm-fed layers (W1): uniform** (col max/median
  1.2–1.7, top-K cells span ~98% of columns). LayerNorm *structurally* removes
  column skew (normalizes each sample) → screening is dead on the LN path.
- **GELU-fed layers (W2): a mild tail that grows with depth** — col max/median
  2.0 → 3.6 by block 2; top-5% grad cells in ~145–200 of 256 columns (57–78%).
  Worth maybe a ~1.5–2× column prune, not the ~10× of the power-law synthetic.
Conclusion: CIFAR-MLP is a mild case; screening would help only the deeper GELU
layers, modestly. The compelling regime remains large models with residual-stream
"massive activations" (bigger D, stronger outlier channels).

### 14. Sample-subset gradient: modest lever, high-rank sum resists it — `sample_select.py`, `cifar_sample_select.py`
`G = da^T@x = sum_b da[b](x)x[b]` is a sum of per-sample rank-1 terms; can a few
high-error samples reconstruct it (compute exact G from m<<B samples, m*D^2 ops)?
Measured top-K overlap of the m-sample gradient vs the full-batch gradient.
- **Isotropic synthetic: no.** Sample contributions near-uniform (top-32/256 =
  18% vs 12.5% uniform); even m=B/2 recovers 0.28 of top-K; error-select ≈ random.
  Structureless on the sample axis too (every sample equally hard).
- **Real CIFAR (trained ResFFN): concentrated but still insufficient.** Top-32/512
  samples hold ~21-30% of contribution energy (~5x uniform) -- genuinely
  heavy-tailed per-sample errors. But m=64/512 (12.5%) recovers only ~0.39 of the
  top-K grow cells; the largest cells come from the full-batch *sum*, not a few
  dominant terms, so dropping samples loses them.
- **Best cheap selector = contribution `‖da_b‖·‖x_b‖`**, ~2x better than random at
  small m and better than error norm `‖da_b‖` alone.
Conclusion: a real but modest lever (caps ~0.4 overlap by m=64), can't reach
rowexact's 0.8-0.89 on its own. Same underlying obstacle as findings 9/13: the
grow gradient is high-rank/diffuse, so no single cheap axis (rows, cols, or
samples) reconstructs its top cells cheaply.

### 15. On REAL data the grow gradient's rank is layer-dependent; cheapest method wins where it's low-rank — `cifar_approx.py`
Switched to real CIFAR (dense ResFFN, per user directive to stop using synthetic).
Compared cheap rank-1 cell-scorings (score(i,j)=row_i*col_j; top rows x top cols),
per-sample rank-1 union, and rowexact, vs the oracle top-K (K=1% of cells, a
realistic grow budget). Rank-1 batch reductions:
  sum_signed = |sum_b da[b,i]| x |sum_b x[b,j]|   (the "add the batch" idea)
  sum_abs / l2norm = L1 / L2 batch norms per row/col.
At D=2048 (overlap/selQ vs oracle):
- **`persample` is a coverage artifact** — at D=256 its union was ~90% of the
  matrix (looked great, 0.90); at D=2048 the union is a genuine 1.3-2.7% of D^2
  and it drops to ~0.3-0.44 overlap. Not a real win.
- **`sum_signed` (batch-sum rank-1, ~free O(B*D)) is BEST on deep GELU layers**,
  with selQ climbing with depth: 0.38 -> 0.49 -> 0.82 -> **0.96** (blocks 0-3).
  It is exact when the batch gradients align (G ≈ da_sum (x) x_sum); deep
  post-GELU activations after training are dominated by a few consistent
  directions -> the grow gradient there is near **rank-1**, so the cheapest method
  captures ~all of it. Cancellation only hurt sum_signed on high-rank layers.
- **LN-fed layers stay higher-rank**: rowexact (0.66-0.87 selQ) and the norm rank-1
  (0.67-0.89) win there; sum_signed weaker.
Big picture: findings 9/13/14 ("gradient high-rank, nothing cheap works") were the
**isotropic-synthetic worst case**. On real data the grow-gradient rank is
layer-dependent; where it is low-rank (deep GELU) the cheapest rank-1 approximation
is near-perfect. The approximation should be chosen per layer to match its rank.
NOTE: D=256 persample numbers earlier were a coverage artifact; disregard them.

### 16. Per-cell compute-gating: exact CS gate fails, approximate wins ~10-20x on deep layers — `cifar_gate.py`
Framed the approximation as a per-cell economic decision: each cell G[i,j] costs
B mults; a gate costing h<<B decides compute-or-skip. Exact Cauchy-Schwarz gate:
skip if `‖da[:,i]‖·‖x[:,j]‖ < tau_K` (never drops a true top-K cell). Real CIFAR
D=2048, K=1% budget:
- **Exact gate is a dud: C_pass = 73-100% of cells survive** -> speedup ~1x. The
  magnitude-only bound is too loose (ignores alignment/cancellation); one modest-
  norm true-top-K cell drops tau_K enough that almost everything passes. The gate
  cost h≈1<<B is fine -- the failure is *discrimination*, not gate cost.
- **Approximate grow wins on deep layers** (reading recall@top-C%-by-bound, i.e.
  compute only the top-C% of cells exactly): b3.w2 GELU hits 98% recall at 5% of
  cells (**~20x fewer computed**), 100% at 10% (10x); b2.w2 97% at 10%; deep LN
  similar. Shallow layers ~5x at 95%; **stem only ~2x** (needs 50% for 91%). Same
  depth trend as finding 15 (deep gradients concentrated -> norm bound is a good
  candidate generator).
- **Budget implication:** we have B=256 mults/cell of gate budget; the norm gate
  spends 1. Huge room for a tighter gate -- e.g. a random-projection sketch
  `G[i,j] ≈ <P da[:,i], P x[:,j]>`, P: B->b' (b'≈16-32), h=b'<<B, capturing
  *alignment* not just magnitude. Would tighten the bound and could make even
  exact-recall gating cheap. (next experiment)

### 17. Sketch gate does NOT beat the norm gate — `cifar_sketch_gate.py`
Tested the finding-16 idea: replace the magnitude-only norm gate with a
random-projection sketch `Ghat = (P da)^T (P x)`, P: B->b' (b'=8/16/32), a rank-b'
estimate of G capturing alignment at b' mults/cell. Real CIFAR D=2048. Result:
**the sketch is strictly worse than the plain norm gate** at every layer (lower
top-K recall; needs >20% of cells for 95% recall vs the norm gate's 5-18%). Why:
- low-b' JL is too noisy (dot-product estimate error ~1/sqrt(b'), ~25% at b'=16)
  -- the noise corrupts the ranking more than alignment info helps; de-noising
  needs b'->B, defeating the purpose.
- on the layers where a cheap gate works at all (deep, near rank-1 per finding
  15), the norm gate is already near-OPTIMAL: for G = sigma*u(x)v,
  `‖da[:,i]‖·‖x[:,j]‖ ∝ |u_i||v_j| = |G[i,j]|` exactly, so norm ranks cells
  perfectly (recall 1.00 at 5% on b3.w2). No room for the sketch to add value;
  it would only help on high-rank shallow layers, where b' is far too small.
Conclusion: the norm gate (finding 16) is the best cheap gate. Magnitude bound is
loose for *exact* top-K but excellent for ~95% recall on deep near-rank-1 layers
(5-18x speedup). The cheapest gate already spends its 1 mult/cell optimally where
gating is viable; the extra per-cell budget doesn't buy a better gate here.

### 18. Reality check: the sparse KERNEL is the bottleneck, not grow-prune — `bench_step.py`
End-to-end training-step wall-clock (MPS): dense-small (D=1024, 1.05M params),
dense-wide (Sd=4096, 16.8M params, the operator sparse approximates), and sparse
(Sd=4096, k=1.05M active = equal params to dense-small, density 6.2%) using the
fused-Metal CSR kernel + norm-gate grow-prune.

| phase | dense-small | dense-wide | sparse |
|---|---|---|---|
| fwd (ms) | 0.35 | 5.36 | **70.2** |
| bwd (ms) | 0.35 | 20.9 | **385.0** |
| step (ms) | 0.87 | 34.7 | 3.03 |
| grow-prune amort (ms) | - | - | 1.61 (160 ms /100) |
| **total** | **1.57** | 60.95 | **459.9** |

- **Sparse is ~292x slower than equal-param dense, and 7.5x slower than the 16x-
  bigger dense-wide it approximates.** Not competitive.
- **The grow-prune overhead is NOT the problem** — amortized ~1.6 ms/step (≈ one
  dense step). The searches (norm-gate topk, deadness, CSR rebuild) are cheap.
- **The sparse fwd/bwd kernels are the catastrophe.** Same flops as dense-small
  (k=D^2) but ~3.8 GMAC/s vs dense's ~766 GMAC/s — **~200x less efficient**. Root
  cause: unstructured sparse = random memory access (`x[csr_col[t]]` scatters),
  which is murder on GPU; dense GEMM is coalesced + MPS-optimized. This is the
  classic reason unstructured sparsity doesn't accelerate dense hardware.
Reframing: findings 9-17 (grow-gradient approximation) only matter if the sparse
matmul can be made fast. Right now it can't (unstructured, memory-bound). Levers
that would actually move wall-clock: **structured/block sparsity** (coalesced), a
far better sparse kernel, or accepting sparse wins only on accuracy-per-param
(finding 6) and in memory-bound / non-GPU regimes — not MPS wall-clock.

### 19. x-caching the sparse forward: 2.1x, confirms the DRAM-gather diagnosis — `sparse_metal.mm`
Per the finding-18 diagnosis (the only random access is the input gather
`x[b, csr_col[t]]`, and `x[b,:]` is small enough to cache on-chip), added an
x-cached forward kernel: one threadgroup per batch element stages `x[b,:]` into
threadgroup (SRAM) memory once, then grid-strides the output rows gathering from
SRAM. Same step benchmark (Sd=4096, k=1M):

| phase | dense-small | sparse before | sparse x-cached |
|---|---|---|---|
| fwd (ms) | 0.35 | 70.2 | **33.4** (2.1x) |
| bwd (ms) | 0.35 | 385 | 298 (untouched) |
| total (ms) | 1.43 | 458 | 335 |

- **Forward 2.1x faster** — the DRAM random-gather was a real chunk, as predicted.
  Still 95x dense (latency-bound serial row loop; likely shared-mem bank conflicts
  on `xs[col]`; only B=256 threadgroups -> modest occupancy). More forward left.
- **Backward now dominates (298 ms)** and was intentionally not touched: its
  `grad_val` reads `x` *by column across the batch*, which per-batch-row caching
  can't help -- that needs the transposed-layout representation change (next step).

## Caveats
- Single-seed, small synthetic nets; regularization effects are modest (~few %).
- The headline conclusions (grow-prune > static; dense ties sparse when
  right-sizeable; sparse wins when dense is undersized) are robust to the
  pruning/regularization choices above.
