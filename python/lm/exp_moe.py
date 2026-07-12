"""MoE vs dense vs PLE, with expansion vs no-expansion (TinyStories).

2x2x2 = 8 param-matched TinyGPT variants, identical trunk (embed_dim, GQA, RoPE,
AttnRes, tied embeddings), differing ONLY in the FFN family and its hidden width:

  family     : {dense SwiGLU, PLE SwiGLU, MoE (DeepSeek-style), MoE-PLE}
  routing    : plain (single FFN) vs MoE (fine-grained routed + shared expert,
               normalized sigmoid gating, aux-loss-free bias load balancing)
  expansion  : H = EXP*embed_dim ("expansion") vs H = embed_dim ("no expansion")

Block-sparse is intentionally dropped. MoE-PLE = the experts share ONE per-layer
token embedding (up/value path); each expert has its own gate and down.

Parameter equivalence is by DEPTH: every variant's n_layers is solved so total
params ~= a common budget P (user's rule: "without expansion just has more
layers"). Because PLE/MoE are much heavier per layer, their solved depth is far
smaller than dense's — that heterogeneity is the point (spend the budget on
width/experts or on depth?). Actual params + depths + residuals are reported.

Run:
  PYTHONUNBUFFERED=1 DRYRUN=1 ./.venv/bin/python python/lm/exp_moe.py   # sizing only
  PYTHONUNBUFFERED=1 STEPS=600 SEQ=128 ./.venv/bin/python python/lm/exp_moe.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from model import TinyGPT, GPTConfig, MoEFFN, MoEPLEFFN, SwiGLUPLE  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 1234

STEPS = int(os.environ.get("STEPS", "600"))
BATCH = int(os.environ.get("BATCH", "16"))
SEQ = int(os.environ.get("SEQ", "128"))
LOG_EVERY = int(os.environ.get("LOG_EVERY", "50"))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "200"))
N_STORIES = int(os.environ.get("N_STORIES", "20000"))
REF_LAYERS = int(os.environ.get("REF_LAYERS", "8"))   # dense_exp depth -> sets the ACTIVE budget A
EXP = int(os.environ.get("EXP", "4"))            # expansion factor (H = EXP*embed_dim)
LMIN, LMAX = 2, int(os.environ.get("LMAX", "28"))
NROUTED, TOPK, NSHARED = 8, 2, 1
NACT = TOPK + NSHARED                             # experts active per token
LR, BETAS, WD, WARMUP, GRAD_CLIP, VAL_BATCHES = 3e-4, (0.9, 0.95), 0.01, 100, 1.0, 20

C = 160
TOKENIZER_PATH = os.path.join(HERE, "tokenizer_v3.json")
OUT_JSON = os.path.join(HERE, "exp_moe_results.json")
CKPT_DIR = os.path.join(HERE, "checkpoints_moe")

HEXP, HNOEXP = EXP * C, C


def base(**kw):
    return dict(vocab_size=4096, embed_dim=C, head_dim=32, uniform_heads=6,
                n_kv_heads=2, block_size=SEQ, n_mtp=0,
                moe_n_routed=NROUTED, moe_n_shared=NSHARED, moe_top_k=TOPK, **kw)


def make_cfg(family, H, L):
    """H is the ACTIVE hidden width (4C = expansion, C = no expansion). For MoE the
    experts are fine-grained: each expert hidden = H/NACT, so activating NACT
    experts costs the same active params as a dense FFN of width H (but total
    params are larger — the extra routed experts are free capacity at equal
    compute)."""
    if family == "dense":
        return GPTConfig(ffn_type="dense", dense_hidden=H, n_layers=L, **base()).validate()
    if family == "ple":
        return GPTConfig(ffn_type="ple", ple_hidden=H, n_layers=L, **base()).validate()
    if family == "moe":
        return GPTConfig(ffn_type="moe", moe_expert_hidden=max(1, round(H / NACT)),
                         n_layers=L, **base()).validate()
    if family == "moe_ple":
        return GPTConfig(ffn_type="moe_ple", moe_ple_hidden=max(1, round(H / NACT)),
                         n_layers=L, **base()).validate()
    raise ValueError(family)


def count(cfg):
    return TinyGPT(cfg).count_parameters()


def active_params(cfg):
    """Params on the per-token active path: total MINUS (a) the routed experts not
    activated and (b) the un-looked-up rows of any per-layer token embedding."""
    model = TinyGPT(cfg)
    total = model.count_parameters()
    inactive = 0
    for m in model.modules():
        if isinstance(m, MoEFFN):
            per_expert = sum(p.numel() for p in m.experts[0].parameters())
            inactive += (m.n_routed - m.top_k) * per_expert
        elif isinstance(m, MoEPLEFFN):
            per_expert = sum(p.numel() for p in list(m.gate[0].parameters()) + list(m.down[0].parameters()))
            inactive += (m.n_routed - m.top_k) * per_expert
            inactive += m.layer_emb.weight.numel() - m.layer_emb.weight.shape[1]
        elif isinstance(m, SwiGLUPLE):
            inactive += m.layer_emb.weight.numel() - m.layer_emb.weight.shape[1]
    return total - inactive


def solve_layers(family, H, target):
    """Integer n_layers in [LMIN, LMAX] minimizing |ACTIVE params - target|."""
    best, best_err = LMIN, 1e18
    for L in range(LMIN, LMAX + 1):
        p = active_params(make_cfg(family, H, L))
        if abs(p - target) < best_err:
            best, best_err = L, abs(p - target)
    return best


# the 8 cells: (label, family, H)
CELLS = [
    ("dense_exp",       "dense",   HEXP),
    ("dense_noexp",     "dense",   HNOEXP),
    ("ple_exp",         "ple",     HEXP),
    ("ple_noexp",       "ple",     HNOEXP),
    ("moe_dense_exp",   "moe",     HEXP),
    ("moe_dense_noexp", "moe",     HNOEXP),
    ("moe_ple_exp",     "moe_ple", HEXP),
    ("moe_ple_noexp",   "moe_ple", HNOEXP),
]


def build_all():
    A = active_params(make_cfg("dense", HEXP, REF_LAYERS))   # active budget from dense_exp@REF_LAYERS
    cfgs = {}
    for label, family, H in CELLS:
        L = solve_layers(family, H, A)
        cfgs[label] = (make_cfg(family, H, L), family, H, L)
    return A, cfgs


def print_sizing(A, cfgs):
    print(f"ACTIVE budget A = {A:,} (dense_exp @ {REF_LAYERS} layers)  EXP={EXP} "
          f"(H_exp={HEXP}, H_noexp={HNOEXP})  experts={NROUTED} top{TOPK}+{NSHARED}shared  "
          f"layers in [{LMIN},{LMAX}]", flush=True)
    print(f"  {'model':16s} {'family':8s} {'H':>4} {'lyr':>3} {'active':>10} {'resid':>7} {'total':>11} {'tot/act':>7}", flush=True)
    for label, (cfg, family, H, L) in cfgs.items():
        a, t = active_params(cfg), count(cfg)
        print(f"  {label:16s} {family:8s} {H:>4} {L:>3} {a:>10,} {100*(a-A)/A:>+6.1f}% "
              f"{t:>11,} {t/a:>6.2f}x", flush=True)


# ---------------------------------------------------------------- data ----
def load_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(TOKENIZER_PATH)


def get_stories():
    from datasets import load_dataset
    ds = load_dataset("karpathy/tinystories-gpt4-clean", split="train", streaming=True)
    stories = []
    for ex in ds:
        stories.append(ex["text"])
        if len(stories) >= N_STORIES:
            break
    return stories


def build_chunks(stories, tok, block_size):
    eos = tok.token_to_id("<eos>") or 2
    ids = []
    for t in stories:
        ids.extend(tok.encode(t).ids)
        ids.append(eos)
    stride = block_size + 1
    chunks = [ids[i:i + stride] for i in range(0, len(ids) - block_size, stride)
              if len(ids[i:i + stride]) == stride]
    return torch.tensor(chunks, dtype=torch.long)


def build_optimizer(model):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "emb" in name or "ar_proj" in name else decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": WD},
                              {"params": no_decay, "weight_decay": 0.0}], lr=LR, betas=BETAS)


def make_scheduler(opt, total):
    def f(step):
        if step < WARMUP:
            return step / max(1, WARMUP)
        prog = (step - WARMUP) / max(1, total - WARMUP)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def batch_iter(train, batch, seed):
    g = torch.Generator().manual_seed(seed)
    n = train.shape[0]
    while True:
        for i in range(0, n - batch + 1, batch):
            yield torch.randperm(n, generator=g)[i:i + batch]


@torch.no_grad()
def eval_val(model, val, vocab_size):
    model.eval()
    losses = []
    for i in range(min(VAL_BATCHES, val.shape[0] // BATCH)):
        chunk = val[i * BATCH:(i + 1) * BATCH].to(DEVICE)
        x, y = chunk[:, :-1], chunk[:, 1:]
        logits, _ = model(x)
        losses.append(F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1)).item())
    return float(sum(losses) / max(len(losses), 1))


def save_ckpt(model, cfg, label, step):
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"{label}_step{step}.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": cfg.to_dict(),
                "tokenizer_path": TOKENIZER_PATH, "global_step": step,
                "label": label, "experiment": "moe"}, path)


def train_one(label, cfg, train, val, vocab_size):
    torch.manual_seed(SEED)
    model = TinyGPT(cfg).to(DEVICE)
    n_params = model.count_parameters()
    opt = build_optimizer(model)
    sched = make_scheduler(opt, STEPS)
    it = batch_iter(train, BATCH, 42)
    curve = []
    model.train()
    t0, total_tokens, peak = time.time(), 0, 0.0
    for step in range(1, STEPS + 1):
        chunk = train[next(it)].to(DEVICE)
        x, y = chunk[:, :-1], chunk[:, 1:]
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        aux = model.collect_aux_loss()
        opt.zero_grad(set_to_none=True)
        (loss + aux).backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()
        total_tokens += x.numel()
        if DEVICE == "mps":
            peak = max(peak, torch.mps.current_allocated_memory() / 1e9)
        if step % LOG_EVERY == 0 or step == 1:
            lv = eval_val(model, val, vocab_size)
            model.train()
            tps = total_tokens / (time.time() - t0)
            curve.append({"step": step, "train_loss": float(loss.item()), "val_loss": lv, "tok_s": tps})
            print(f"  [{label:16s}] step {step:4d}/{STEPS}  train {loss.item():.4f}  "
                  f"val {lv:.4f}  aux {float(aux):.4f}  {tps:,.0f} tok/s", flush=True)
        if step % CKPT_EVERY == 0 or step == STEPS:
            save_ckpt(model, cfg, label, step)
    wall = time.time() - t0
    return {"label": label, "ffn_type": cfg.ffn_type, "n_layers": cfg.n_layers,
            "n_params": n_params, "active_params": active_params(cfg), "config": cfg.to_dict(),
            "curve": curve, "final_val_loss": curve[-1]["val_loss"],
            "final_train_loss": curve[-1]["train_loss"],
            "wall_s": wall, "tok_s": total_tokens / wall, "peak_mem_gb": peak}


def main():
    torch.manual_seed(SEED)
    A, cfgs = build_all()
    print_sizing(A, cfgs)
    if os.environ.get("DRYRUN"):
        return

    tok = load_tokenizer(); V = tok.get_vocab_size()
    stories = get_stories()
    chunks = build_chunks(stories, tok, SEQ)
    g = torch.Generator().manual_seed(0)
    chunks = chunks[torch.randperm(chunks.shape[0], generator=g)]
    val, train = chunks[:320], chunks[320:]
    print(f"\nData: {len(stories):,} stories -> train {train.shape[0]:,} / val {val.shape[0]:,}\n", flush=True)

    results = {}
    for label, (cfg, family, H, L) in cfgs.items():
        print(f"=== Training {label} ({cfg.ffn_type}, {L} layers, H={H}) ===", flush=True)
        results[label] = train_one(label, cfg, train, val, V)
        print("", flush=True)

    print("=" * 108)
    print(f"{'model':16s} {'ffn':8s} {'lyr':>3} {'active':>10} {'total':>11} {'val':>8} "
          f"{'train':>8} {'tok/s':>9} {'wall_s':>8} {'mem_gb':>7}")
    print("-" * 108)
    for label, _, _ in CELLS:
        r = results[label]
        print(f"{label:16s} {r['ffn_type']:8s} {r['n_layers']:>3} {r['active_params']:>10,} "
              f"{r['n_params']:>11,} {r['final_val_loss']:>8.4f} {r['final_train_loss']:>8.4f} "
              f"{r['tok_s']:>9,.0f} {r['wall_s']:>8.1f} {r['peak_mem_gb']:>7.2f}")
    print("=" * 108)
    best = min(results.values(), key=lambda r: r["final_val_loss"])
    print(f"\nbest val loss: {best['label']} = {best['final_val_loss']:.4f}  (equal ACTIVE params)")

    with open(OUT_JSON, "w") as f:
        json.dump({"active_budget": A, "exp": EXP, "steps": STEPS, "seq": SEQ,
                   "device": DEVICE, "results": results}, f, indent=2)
    print(f"Saved {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
