"""Width-only comparison (v2): DEPTH, STEPS, and ACTIVE params all held fixed;
only WIDTH varies. Removes depth as a confound (wide+shallow beat thin+deep in v1
mainly because depth hurt).

  fixed : depth D (all models), steps, active params ~= A
  families    : {dense, ple, moe(dense experts), moe_ple}
  expansion   : FFN hidden = EXP*embed  (narrow model, wide FFN)
  no-expansion: FFN hidden = embed (square) but the whole model is WIDER
                (larger embed) so active params still match A. Attention scales
                with embed too — a "fat square" transformer.
  MoE         : more PARALLEL routed experts (n_routed) for extra TOTAL capacity;
                active matched (fine-grained: expert hidden = H/(top_k+shared)).

Each (family, expansion-ratio) solves its embed_dim so ACTIVE params ~= A at the
fixed depth. MoE uses the batched grouped-GEMM dispatch (fair wall-clock).

Run:
  DRYRUN=1 ./.venv/bin/python python/lm/exp_moe_width.py     # sizing only
  STEPS=250 SEQ=128 ./.venv/bin/python python/lm/exp_moe_width.py
"""
from __future__ import annotations

import os, sys, json

os.environ.setdefault("STEPS", "250")
os.environ.setdefault("SEQ", "128")
os.environ.setdefault("N_STORIES", "20000")
os.environ.setdefault("CKPT_EVERY", "250")

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402
import exp_moe as E  # noqa: E402  (reuse data/optim/train_one/active_params/flops)
from model import GPTConfig  # noqa: E402

DEPTH = int(os.environ.get("DEPTH", "8"))
EXP = int(os.environ.get("EXP", "4"))            # expansion ratio (FFN hidden = EXP*embed)
N_ROUTED = int(os.environ.get("N_ROUTED", "16"))  # parallel experts (MoE total capacity)
TOPK, NSHARED = 2, 1
NACT = TOPK + NSHARED
VOCAB, HEAD_DIM = 4096, 32
OUT_JSON = os.path.join(HERE, "exp_moe_width_results.json")


def heads_for(embed):
    nh = max(2, round(embed / HEAD_DIM))
    if nh * HEAD_DIM == embed:
        nh += 1                                   # keep attn_dim != embed
    nkv = max(1, nh // 3)
    while nh % nkv != 0:
        nkv -= 1
    return nh, nkv


def make_cfg(family, embed, ratio, depth=DEPTH, n_routed=N_ROUTED):
    nh, nkv = heads_for(embed)
    common = dict(vocab_size=VOCAB, embed_dim=embed, head_dim=HEAD_DIM, uniform_heads=nh,
                  n_kv_heads=nkv, block_size=E.SEQ, n_mtp=0,
                  moe_n_routed=n_routed, moe_n_shared=NSHARED, moe_top_k=TOPK)
    H = ratio * embed
    if family == "dense":
        return GPTConfig(ffn_type="dense", dense_hidden=H, n_layers=depth, **common).validate()
    if family == "ple":
        return GPTConfig(ffn_type="ple", ple_hidden=H, n_layers=depth, **common).validate()
    if family == "moe":
        return GPTConfig(ffn_type="moe", moe_expert_hidden=max(1, round(H / NACT)),
                         n_layers=depth, **common).validate()
    if family == "moe_ple":
        return GPTConfig(ffn_type="moe_ple", moe_ple_hidden=max(1, round(H / NACT)),
                         n_layers=depth, **common).validate()
    raise ValueError(family)


def active_est(family, embed, ratio, depth=DEPTH, n_routed=N_ROUTED):
    """Analytic active-param estimate (no model construction) for the embed search.
    Matches count_parameters closely; final reporting uses the exact active_params."""
    nh, nkv = heads_for(embed)
    attn_dim, kv_dim = nh * HEAD_DIM, nkv * HEAD_DIM
    qkv = embed * (attn_dim + 2 * kv_dim) + (attn_dim + 2 * kv_dim)   # Linear w+b
    proj = attn_dim * embed + embed
    ln = 2 * (2 * embed)                                             # ln1 + ln2 (w+b)
    H = ratio * embed
    he = max(1, round(H / NACT))
    if family == "dense":                                           # bias-free SwiGLU
        ffn = 3 * embed * H
    elif family == "ple":                                           # gate/down have bias
        ffn = (embed * H + H) + (H * embed + embed) + H             # gate + down + emb row
    elif family == "moe":                                           # router + active experts
        ffn = embed * n_routed + NACT * (3 * embed * he)
    else:                                                           # moe_ple
        ffn = embed * n_routed + NACT * (2 * embed * he) + he
    tok_emb = VOCAB * embed
    return depth * (qkv + proj + ln + ffn) + tok_emb + (depth + 1) * embed + 2 * embed


def solve_embed(family, ratio, target, lo=64, hi=1024, step=16):
    best, best_err = lo, 1e18
    for embed in range(lo, hi + 1, step):
        a = active_est(family, embed, ratio)
        if abs(a - target) < best_err:
            best, best_err = embed, abs(a - target)
    return best


# 8 cells: (label, family, expansion-ratio)
CELLS = [
    ("dense_exp",      "dense",   EXP),
    ("dense_wide",     "dense",   1),
    ("ple_exp",        "ple",     EXP),
    ("ple_wide",       "ple",     1),
    ("moe_dense_exp",  "moe",     EXP),
    ("moe_dense_wide", "moe",     1),
    ("moe_ple_exp",    "moe_ple", EXP),
    ("moe_ple_wide",   "moe_ple", 1),
]


def build_all():
    A = active_est("dense", 160, EXP)                   # anchor: dense expansion @ embed160
    cfgs = {}
    for label, family, ratio in CELLS:
        embed = 160 if (family == "dense" and ratio == EXP) else solve_embed(family, ratio, A)
        cfgs[label] = (make_cfg(family, embed, ratio), family, ratio, embed)
    return A, cfgs


def main():
    torch.manual_seed(E.SEED)
    A, cfgs = build_all()
    print(f"WIDTH-ONLY v2  depth={DEPTH} (fixed)  steps={E.STEPS}  EXP={EXP}  n_routed={N_ROUTED} "
          f"top{TOPK}+{NSHARED}sh  active budget A={A:,}", flush=True)
    print(f"  {'model':16s} {'embed':>5} {'ffnH':>5} {'active':>10} {'resid':>7} {'total':>11} {'tot/act':>7}",
          flush=True)
    for label, (cfg, family, ratio, embed) in cfgs.items():
        a, t = E.active_params(cfg), __import__("model").TinyGPT(cfg).count_parameters()
        H = getattr(cfg, {"dense": "dense_hidden", "ple": "ple_hidden",
                          "moe": "moe_expert_hidden", "moe_ple": "moe_ple_hidden"}[family])
        print(f"  {label:16s} {embed:>5} {H:>5} {a:>10,} {100*(a-A)/A:>+6.1f}% {t:>11,} {t/a:>6.2f}x",
              flush=True)
    if os.environ.get("DRYRUN"):
        return

    tok = E.load_tokenizer(); V = tok.get_vocab_size()
    stories = E.get_stories()
    chunks = E.build_chunks(stories, tok, E.SEQ)
    g = torch.Generator().manual_seed(0)
    chunks = chunks[torch.randperm(chunks.shape[0], generator=g)]
    val, train = chunks[:320], chunks[320:]
    print(f"\nData: {len(stories):,} stories -> train {train.shape[0]:,} / val {val.shape[0]:,}\n", flush=True)

    results = {}
    for label, (cfg, family, ratio, embed) in cfgs.items():
        print(f"=== {label} (embed {embed}, {cfg.ffn_type}, {cfg.n_layers}L) ===", flush=True)
        results[label] = E.train_one(label, cfg, train, val, V)
        print("", flush=True)

    print("=" * 112)
    print(f"{'model':16s} {'embed':>5} {'active':>10} {'total':>11} {'Gf/tok':>7} {'val':>7} "
          f"{'wall_s':>7} {'tok/s':>8}")
    print("-" * 112)
    for label, _, _ in CELLS:
        r = results[label]
        print(f"{label:16s} {r['config']['embed_dim']:>5} {r['active_params']:>10,} {r['n_params']:>11,} "
              f"{r['flops_fwd_per_tok']/1e9:>7.3f} {r['final_val_loss']:>7.4f} {r['wall_s']:>7.1f} "
              f"{r['tok_s']:>8,.0f}")
    print("=" * 112)
    for label, _, _ in sorted(CELLS, key=lambda c: results[c[0]]["final_val_loss"]):
        r = results[label]
        print(f"  {label:16s} val {r['final_val_loss']:.4f} @ {r['wall_s']:6.1f}s", flush=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"active_budget": A, "depth": DEPTH, "exp": EXP, "n_routed": N_ROUTED,
                   "steps": E.STEPS, "results": results}, f, indent=2)
    print(f"\nSaved {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
