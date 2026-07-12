# TinyGPT (`python/lm`)

A from-scratch GPT for TinyStories, ported from `tiny-lm/main_v3.py`. It keeps the
v3 features and adds three selectable FFN variants.

**Kept from v3:** RoPE, SwiGLU, PLE (per-layer embedding), AttnRes (softmax-weighted
combination of previous layer outputs via learned per-layer query vectors), tied
embeddings, KV-cache attention, and the checkpoint/resume + inference design.

**Changed from v3:**
- **Even head schedule.** Uniform `UNIFORM_HEADS = 6` heads per layer (no diamond).
  The design constraint `n_heads * head_dim != embed_dim` is preserved:
  `6 * 32 = 192 != 160`. (Uniform 5 heads would be exactly 160 and is disallowed.)
  Change it in one place: `model.UNIFORM_HEADS`.
- **AdamW instead of Muon.** Muon/Newton–Schulz removed entirely; every parameter is
  trained with `torch.optim.AdamW` at LR `3e-4` with the v3 warmup + cosine schedule.
  Param-group split: weight decay (`0.01`) on ≥2-D weight matrices only; no decay on
  1-D params, norms, embeddings, and `ar_proj`. (Muon can't orthogonalize the
  `sparse` variant's 1-D `SparseLinear` params, hence Adam for all.)

Config stays the same as v3 otherwise: `embed_dim=160`, `11` layers, `head_dim=32`,
vocab target `4096`, `block_size=512`, batch `64`.

## The three FFN variants (`GPTConfig.ffn_type`)

Everything except the per-block FFN (attention, RoPE, AttnRes, tied embeddings,
norms) is identical across variants.

- **`ple`** — v3 baseline: `down(silu(gate(x)) * layer_emb(token_ids))` with a
  per-layer `Embedding(vocab, dim)`. Dominated by the 11 large per-layer embeddings.
- **`sparse`** — SwiGLU (gate/up/down) where each projection is a `SparseLinear`
  (reused from `sparse_layer.py`, O(k) hand-written forward/backward) instead of
  dense. No per-layer embedding. Sized with a big hidden dim + low density
  (`sparse_hidden=6144`, `sparse_density=0.25`) so the count of *stored* weights is
  comparable to `ple`.
- **`moe`** — DeepSeek-V3-style Mixture-of-Experts: `moe_n_routed=16` fine-grained
  small SwiGLU routed experts + `moe_n_shared=1` always-on shared expert,
  `moe_top_k=4` routing with a **sigmoid** gate (top-k weights normalized to sum 1),
  and **aux-loss-free load balancing** via a per-expert routing *bias* buffer nudged
  each step toward equal load (`bias += u * sign(mean_load - load_e)`), plus a small
  optional switch-style aux load-balance loss (`moe_aux_coef`).
  See DeepSeek-V3, Qwen3-MoE, GLM-4.5, and Kimi K2 for the recipe.

## Parameter counts (real config, `vocab=4096`)

| variant  | total params | FFN notes |
|----------|-------------:|-----------|
| `ple`    |    9,800,096 | 11 × `Embedding(4096,160)` dominate |
| `sparse` |   10,134,496 | hidden 6144 @ 25% density (k≈737k stored weights/layer) |
| `moe`    |   10,669,536 | 16 routed + 1 shared experts, hidden 96, top-4 |

All three are ~9.8–10.7M params (within ~9%), i.e. roughly comparable.

## Run

Smoke test (no downloads, no `datasets`/`tokenizers` needed):

```bash
.venv/bin/python python/lm/smoke_test.py
```

Train (downloads TinyStories; needs `datasets` + `tokenizers`):

```bash
FFN_TYPE=ple    .venv/bin/python python/lm/train.py
FFN_TYPE=sparse .venv/bin/python python/lm/train.py
FFN_TYPE=moe    .venv/bin/python python/lm/train.py
```

Checkpoints are named per variant (`checkpoint_<ffn>.pt`, `..._resume.pt`,
`..._epoch{n}.pt`) and store the full config so inference rebuilds the exact model.

Infer (interactive):

```bash
.venv/bin/python python/lm/inference.py ple    # or sparse / moe
```

## Files
- `model.py` — `GPTConfig` + all model classes + the 3 FFN variants. Imports only `torch`.
- `sparse_linear.py` — reused `SparseLinear` (O(k) sparse linear).
- `train.py` — tokenizer/dataset/training loop/checkpointing; AdamW; `FFN_TYPE` env selects the variant. `datasets`/`tokenizers` imported lazily inside functions.
- `inference.py` — checkpoint load + KV-cache generation.
- `smoke_test.py` — builds + trains-one-step each variant on random ids.
