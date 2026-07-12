"""Deep-sparse vs shallow-dense LM experiment (TinyStories).

Four TinyGPT variants, identical trunk (embed_dim, GQA attention, RoPE, AttnRes,
tied embeddings); they differ ONLY in depth + FFN:

  1. small_dense : SHALLOW dense SwiGLU            -> defines the param budget P
  2. sparse      : DEEP block-sparse SwiGLU        -> ~P (block_density solved)
  3. big_dense   : DEEP dense SwiGLU at the sparse  -> >> P  (the "if the sparse
                   operator's wide hidden          were dense" CEILING)
  4. ple         : SHALLOW per-layer-embedding SwiGLU -> ~P (n_layers solved)

Only the sparse models are deeper (user's design): the question is whether, at
EQUAL params, spending them on a deep+block-sparse net beats a shallow+dense net
(and shallow+ple), and how far under the deep+dense ceiling it lands.

GQA is on (n_kv_heads < uniform_heads). MTP is OFF (n_mtp=0): modded-nanogpt
found MTP gave no gain at this scale, so we don't spend params/compute on it.

Wall-clock, tok/s and peak memory are recorded per model (always). Block-sparse
FFNs grow-and-prune whole dead blocks every RIGL_EVERY steps (cumulative moves
logged) and freeze in the last 20% of training.

Run (ninja on PATH only needed if a Metal variant is added; not here):
  PYTHONUNBUFFERED=1 ./.venv/bin/python python/lm/exp_deep_sparse.py
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

from model import TinyGPT, GPTConfig, BlockSparseSwiGLU  # noqa: E402
from block_sparse import annealed_n_swap  # noqa: E402

# ---------------------------------------------------------------- config ----
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 1234

STEPS = int(os.environ.get("STEPS", "1500"))
BATCH = int(os.environ.get("BATCH", "16"))
SEQ = int(os.environ.get("SEQ", "256"))
LOG_EVERY = int(os.environ.get("LOG_EVERY", "50"))
N_STORIES = int(os.environ.get("N_STORIES", "20000"))
RIGL_EVERY = int(os.environ.get("RIGL_EVERY", "50"))
GROW_STOP_FRAC = 0.8            # freeze block topology after this fraction of steps
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "200"))   # save intermediate checkpoints
CKPT_DIR = os.path.join(HERE, "checkpoints")

# architecture knobs
SHALLOW = int(os.environ.get("SHALLOW", "6"))     # dense / ple depth
DEEP = int(os.environ.get("DEEP", "10"))          # sparse / big-dense depth (deeper)
DENSE_HIDDEN = int(os.environ.get("DENSE_HIDDEN", "512"))   # small-dense FFN hidden
WIDE_HIDDEN = int(os.environ.get("WIDE_HIDDEN", "768"))     # deep models' FFN hidden (a bit wider)
BLOCK_B = int(os.environ.get("BLOCK_B", "16"))

LR = 3e-4
BETAS = (0.9, 0.95)
WD = 0.01
WARMUP = 100
GRAD_CLIP = 1.0
VAL_BATCHES = 20

TOKENIZER_PATH = os.path.join(HERE, "tokenizer_v3.json")
OUT_JSON = os.path.join(HERE, "exp_deep_sparse_results.json")

BASE = dict(vocab_size=4096, embed_dim=160, head_dim=32,
            uniform_heads=6, n_kv_heads=2, block_size=SEQ, n_mtp=0)


# ------------------------------------------------------------------ data ----
def load_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(TOKENIZER_PATH)


def get_stories():
    try:
        from datasets import load_dataset
        ds = load_dataset("karpathy/tinystories-gpt4-clean", split="train", streaming=True)
        stories = []
        for ex in ds:
            stories.append(ex["text"])
            if len(stories) >= N_STORIES:
                break
        if not stories:
            raise RuntimeError("no stories streamed")
        return stories, f"karpathy/tinystories-gpt4-clean[:{len(stories)}] (streaming)"
    except Exception as e:  # pragma: no cover
        print(f"[data] HF load failed ({type(e).__name__}: {e}); using fallback corpus")
        base = ("Once upon a time there was a little girl named Lily. She loved to "
                "play in the park with her dog. One day the sun was shining and the "
                "birds were singing. Lily and her dog ran and played all day long.")
        return [base] * 4000, "fallback-builtin-corpus"


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


# ----------------------------------------------------------- config build ---
def cfg_small_dense():
    return GPTConfig(ffn_type="dense", n_layers=SHALLOW, dense_hidden=DENSE_HIDDEN, **BASE).validate()


def cfg_big_dense():
    return GPTConfig(ffn_type="dense", n_layers=DEEP, dense_hidden=WIDE_HIDDEN, **BASE).validate()


def cfg_block(density):
    return GPTConfig(ffn_type="block", n_layers=DEEP, block_hidden=WIDE_HIDDEN,
                     block_b=BLOCK_B, block_density=density, **BASE).validate()


def cfg_ple(n_layers):
    return GPTConfig(ffn_type="ple", n_layers=n_layers, **BASE).validate()


def count(cfg):
    return TinyGPT(cfg).count_parameters()


def solve_block_density(target):
    """Binary-search block_density so total params ~= target (monotal increasing)."""
    lo, hi = 0.02, 1.0
    best, best_err = 0.25, 1e18
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        p = count(cfg_block(mid))
        err = abs(p - target)
        if err < best_err:
            best, best_err = mid, err
        if p < target:
            lo = mid
        else:
            hi = mid
    return best


def solve_ple_layers(target):
    best, best_err = SHALLOW, 1e18
    for nl in range(1, 13):
        p = count(cfg_ple(nl))
        if abs(p - target) < best_err:
            best, best_err = nl, abs(p - target)
    return best


# ------------------------------------------------------------- optimizer ----
def build_optimizer(model):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "emb" in name or "ar_proj" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "weight_decay": WD},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=LR, betas=BETAS)


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
        perm = torch.randperm(n, generator=g)
        for i in range(0, n - batch + 1, batch):
            yield perm[i:i + batch]


# --------------------------------------------------------------- training ---
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
                "label": label, "experiment": "deep_sparse"}, path)
    return path


def train_one(label, cfg, train, val, vocab_size):
    torch.manual_seed(SEED)
    model = TinyGPT(cfg).to(DEVICE)
    n_params = model.count_parameters()
    opt = build_optimizer(model)
    sched = make_scheduler(opt, STEPS)

    # block grow-prune: cosine-annealed swap count (from the block FFN's n_active)
    n_active, is_block = 0, cfg.ffn_type == "block"
    if is_block:
        bffn = next(m for m in model.modules() if isinstance(m, BlockSparseSwiGLU))
        n_active = bffn.gate.n_active

    it = batch_iter(train, BATCH, seed=42)
    curve, moved_total = [], 0
    model.train()
    t0 = time.time()
    total_tokens, peak_mem = 0, 0.0

    for step in range(1, STEPS + 1):
        idx = next(it)
        chunk = train[idx].to(DEVICE)
        x, y = chunk[:, :-1], chunk[:, 1:]
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        aux = model.collect_aux_loss()
        opt.zero_grad(set_to_none=True)
        (loss + aux).backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()

        # block-sparse grow-and-prune: cosine-annealed migration (explore early,
        # freeze late) — lowest-weight blocks move to highest-gradient positions
        if is_block and step % RIGL_EVERY == 0:
            ns = annealed_n_swap(step, STEPS, n_active, base_frac=0.3, stop_frac=GROW_STOP_FRAC)
            if ns > 0:
                moved_total += model.sparse_update(opt, n_swap=ns)

        total_tokens += x.numel()
        if DEVICE == "mps":
            peak_mem = max(peak_mem, torch.mps.current_allocated_memory() / 1e9)

        if step % LOG_EVERY == 0 or step == 1:
            lv = eval_val(model, val, vocab_size)
            model.train()
            elapsed = time.time() - t0
            tps = total_tokens / elapsed if elapsed > 0 else 0.0
            curve.append({"step": step, "train_loss": float(loss.item()),
                          "val_loss": lv, "tok_s": tps, "moved": moved_total})
            print(f"  [{label:11s}] step {step:4d}/{STEPS}  train {loss.item():.4f}  "
                  f"val {lv:.4f}  {tps:,.0f} tok/s  moved={moved_total}", flush=True)

        if step % CKPT_EVERY == 0 or step == STEPS:
            save_ckpt(model, cfg, label, step)

    elapsed = time.time() - t0
    return {"label": label, "ffn_type": cfg.ffn_type, "n_layers": cfg.n_layers,
            "n_params": n_params, "config": cfg.to_dict(), "curve": curve,
            "final_train_loss": curve[-1]["train_loss"], "final_val_loss": curve[-1]["val_loss"],
            "wall_s": elapsed, "tok_s": total_tokens / elapsed if elapsed > 0 else 0.0,
            "peak_mem_gb": peak_mem, "blocks_moved": moved_total}


def main():
    print(f"Device: {DEVICE}  seed={SEED}  steps={STEPS} batch={BATCH} seq={SEQ}", flush=True)
    tok = load_tokenizer()
    vocab_size = tok.get_vocab_size()

    # --- size the models: small_dense defines P; solve ple depth + block density ---
    P = count(cfg_small_dense())
    ple_nl = solve_ple_layers(P)
    dens = solve_block_density(P)
    cfgs = {
        "small_dense": cfg_small_dense(),
        "sparse":      cfg_block(dens),
        "big_dense":   cfg_big_dense(),
        "ple":         cfg_ple(ple_nl),
    }
    print(f"\nPARAM BUDGET P (small_dense) = {P:,}", flush=True)
    for name, c in cfgs.items():
        p = count(c)
        print(f"  {name:11s} ffn={c.ffn_type:6s} layers={c.n_layers:2d} "
              f"hidden={getattr(c, 'block_hidden' if c.ffn_type=='block' else 'dense_hidden', '-')}"
              f"  params={p:>10,}  ({100*(p-P)/P:+.1f}% vs P)", flush=True)
    print(f"  [solved] ple n_layers={ple_nl}, block_density={dens:.4f}  "
          f"attn_dim={cfgs['small_dense'].attn_dim} (GQA kv_heads={BASE['n_kv_heads']})\n", flush=True)

    # --- data ---
    stories, source = get_stories()
    print(f"Data: {source} ({len(stories):,} stories)", flush=True)
    chunks = build_chunks(stories, tok, SEQ)
    n_val = max(BATCH * VAL_BATCHES, int(0.02 * chunks.shape[0]))
    g = torch.Generator().manual_seed(0)
    chunks = chunks[torch.randperm(chunks.shape[0], generator=g)]
    val, train = chunks[:n_val], chunks[n_val:]
    print(f"Chunks: {chunks.shape[0]:,} -> train {train.shape[0]:,} / val {val.shape[0]:,}\n", flush=True)

    results = {}
    for name, c in cfgs.items():
        print(f"=== Training {name} ({c.ffn_type}, {c.n_layers} layers) ===", flush=True)
        results[name] = train_one(name, c, train, val, vocab_size)
        print("", flush=True)

    # --- table ---
    print("=" * 92)
    print(f"{'model':12s} {'ffn':7s} {'lyr':>3} {'params':>11} {'val_loss':>9} "
          f"{'train':>8} {'tok/s':>9} {'wall_s':>8} {'mem_gb':>7} {'moved':>6}")
    print("-" * 92)
    for name in ("small_dense", "sparse", "big_dense", "ple"):
        r = results[name]
        print(f"{name:12s} {r['ffn_type']:7s} {r['n_layers']:>3} {r['n_params']:>11,} "
              f"{r['final_val_loss']:>9.4f} {r['final_train_loss']:>8.4f} {r['tok_s']:>9,.0f} "
              f"{r['wall_s']:>8.1f} {r['peak_mem_gb']:>7.2f} {r['blocks_moved']:>6}")
    print("=" * 92)
    sd, sp, bd = (results[k]["final_val_loss"] for k in ("small_dense", "sparse", "big_dense"))
    print(f"\nval loss: small_dense={sd:.4f}  sparse(deep)={sp:.4f}  big_dense(ceiling)={bd:.4f}")
    print(f"  deep-sparse vs shallow-dense (equal params): "
          f"{'sparse WINS' if sp < sd - 0.01 else 'dense WINS' if sd < sp - 0.01 else 'TIE'} "
          f"(Δ={sp - sd:+.4f})")
    print(f"  deep-sparse gap to dense ceiling: {sp - bd:+.4f} (ceiling has "
          f"{100*(results['sparse']['n_params'])/results['big_dense']['n_params']:.0f}% of its params)")

    with open(OUT_JSON, "w") as f:
        json.dump({"source": source, "steps": STEPS, "batch": BATCH, "seq": SEQ,
                   "device": DEVICE, "P": P, "results": results}, f, indent=2)
    print(f"\nSaved {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
