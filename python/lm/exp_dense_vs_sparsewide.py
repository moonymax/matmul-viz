"""Controlled experiment: PARAM-EQUAL dense vs 1%-sparse (100x wider) FFN LM.

Both models are identical TinyGPT except the FFN:
  * dense       : plain bias-free SwiGLU, hidden H_d = 512  (dense nn.Linear)
  * sparse_wide : bias-free SwiGLU, hidden H_s = 100*H_d = 51200 at density 0.01,
                  gate/up/down = fused-Metal SparseLinearMetal, grad-checkpointed.
By construction (k = round(density * in * out)) the two models have the SAME
total parameter count. Attention / embeddings / norms are dense & identical.

Short TinyStories run, same seed / batches / steps / optimizer for both.
Logs train (and small val) loss curves; writes JSON + prints a table.

Run (ninja MUST be on PATH for the Metal extension to compile):
  PATH="$PWD/.venv/bin:$PATH" ./.venv/bin/python python/lm/exp_dense_vs_sparsewide.py
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

from model import TinyGPT, GPTConfig  # noqa: E402

# ---------------------------------------------------------------- config ----
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 1234

# short run knobs (override via env)
STEPS = int(os.environ.get("STEPS", "1200"))
BATCH = int(os.environ.get("BATCH", "16"))
SEQ = int(os.environ.get("SEQ", "256"))
LOG_EVERY = int(os.environ.get("LOG_EVERY", "25"))
N_STORIES = int(os.environ.get("N_STORIES", "20000"))
LR = 3e-4
BETAS = (0.9, 0.95)
WD = 0.01
WARMUP = 100
GRAD_CLIP = 1.0
VAL_BATCHES = 20

TOKENIZER_PATH = os.path.join(HERE, "tokenizer_v3.json")
OUT_JSON = os.path.join(HERE, "exp_dense_vs_sparsewide_results.json")

BASE_CFG = dict(vocab_size=4096, embed_dim=160, n_layers=6, head_dim=32,
                uniform_heads=6, block_size=SEQ)


# ------------------------------------------------------------------ data ----
def load_tokenizer():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    return tok


def get_stories():
    """Return (list[str], source_str). Fall back to a tiny built-in corpus."""
    try:
        from datasets import load_dataset
        # streaming: fetch only the first N rows instead of downloading full shards
        ds = load_dataset("karpathy/tinystories-gpt4-clean", split="train",
                          streaming=True)
        stories = []
        for ex in ds:
            stories.append(ex["text"])
            if len(stories) >= N_STORIES:
                break
        if not stories:
            raise RuntimeError("no stories streamed")
        return stories, f"karpathy/tinystories-gpt4-clean[:{len(stories)}] (streaming)"
    except Exception as e:  # pragma: no cover  (no network / cache)
        print(f"[data] HF load failed ({type(e).__name__}: {e}); using fallback corpus")
        base = (
            "Once upon a time there was a little girl named Lily. She loved to "
            "play in the park with her dog. One day the sun was shining and the "
            "birds were singing. Lily and her dog ran and played all day long. "
            "They were very happy. At the end of the day they went home and slept."
        )
        return [base] * 4000, "fallback-builtin-corpus"


def build_chunks(stories, tok, block_size):
    eos = tok.token_to_id("<eos>") or 2
    ids = []
    for t in stories:
        ids.extend(tok.encode(t).ids)
        ids.append(eos)
    stride = block_size + 1
    chunks = []
    for i in range(0, len(ids) - block_size, stride):
        c = ids[i:i + stride]
        if len(c) == stride:
            chunks.append(c)
    return torch.tensor(chunks, dtype=torch.long)


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
    groups = [
        {"params": decay, "weight_decay": WD},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=LR, betas=BETAS)


def make_scheduler(opt, total):
    def f(step):
        if step < WARMUP:
            return step / max(1, WARMUP)
        prog = (step - WARMUP) / max(1, total - WARMUP)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


# --------------------------------------------------------------- training ---
def make_config(ffn_type):
    if ffn_type == "dense":
        return GPTConfig(ffn_type="dense", dense_hidden=512, **BASE_CFG).validate()
    return GPTConfig(ffn_type="sparse_wide", wide_hidden=51200, wide_density=0.01,
                     **BASE_CFG).validate()


def batch_iter(train, batch, seed):
    """Deterministic shuffled index stream (identical for both models)."""
    g = torch.Generator().manual_seed(seed)
    n = train.shape[0]
    while True:
        perm = torch.randperm(n, generator=g)
        for i in range(0, n - batch + 1, batch):
            yield perm[i:i + batch]


def train_one(ffn_type, train, val, vocab_size):
    torch.manual_seed(SEED)
    cfg = make_config(ffn_type)
    model = TinyGPT(cfg).to(DEVICE)
    n_params = model.count_parameters()
    opt = build_optimizer(model)
    sched = make_scheduler(opt, STEPS)

    it = batch_iter(train, BATCH, seed=42)  # same seed => same batch order
    curve = []
    model.train()
    t0 = time.time()
    total_tokens = 0
    peak_mem = 0.0

    for step in range(1, STEPS + 1):
        idx = next(it)
        chunk = train[idx].to(DEVICE)
        x, y = chunk[:, :-1], chunk[:, 1:]
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()
        total_tokens += x.numel()
        if DEVICE == "mps":
            peak_mem = max(peak_mem, torch.mps.current_allocated_memory() / 1e9)

        if step % LOG_EVERY == 0 or step == 1:
            lv = eval_val(model, val, vocab_size)
            model.train()
            elapsed = time.time() - t0
            tps = total_tokens / elapsed if elapsed > 0 else 0.0
            curve.append({"step": step, "train_loss": float(loss.item()),
                          "val_loss": lv, "tok_s": tps})
            print(f"  [{ffn_type:11s}] step {step:4d}/{STEPS}  "
                  f"train {loss.item():.4f}  val {lv:.4f}  {tps:,.0f} tok/s")

    elapsed = time.time() - t0
    return {
        "ffn_type": ffn_type,
        "n_params": n_params,
        "config": cfg.to_dict(),
        "curve": curve,
        "final_train_loss": curve[-1]["train_loss"],
        "final_val_loss": curve[-1]["val_loss"],
        "wall_s": elapsed,
        "tok_s": total_tokens / elapsed if elapsed > 0 else 0.0,
        "peak_mem_gb": peak_mem,
    }


@torch.no_grad()
def eval_val(model, val, vocab_size):
    model.eval()
    losses = []
    for i in range(min(VAL_BATCHES, val.shape[0] // BATCH)):
        chunk = val[i * BATCH:(i + 1) * BATCH].to(DEVICE)
        x, y = chunk[:, :-1], chunk[:, 1:]
        logits, _ = model(x)
        losses.append(F.cross_entropy(logits.reshape(-1, vocab_size),
                                      y.reshape(-1)).item())
    return float(sum(losses) / max(len(losses), 1))


def main():
    print(f"Device: {DEVICE}  seed={SEED}  steps={STEPS} batch={BATCH} seq={SEQ}")
    tok = load_tokenizer()
    vocab_size = tok.get_vocab_size()
    print(f"Tokenizer vocab={vocab_size} (reused tokenizer_v3.json)")

    stories, source = get_stories()
    print(f"Data source: {source}  ({len(stories):,} stories)")
    chunks = build_chunks(stories, tok, SEQ)
    n_val = max(BATCH * VAL_BATCHES, int(0.02 * chunks.shape[0]))
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(chunks.shape[0], generator=g)
    chunks = chunks[perm]
    val, train = chunks[:n_val], chunks[n_val:]
    print(f"Chunks: {chunks.shape[0]:,} total -> train {train.shape[0]:,} / val {val.shape[0]:,} "
          f"(block_size={SEQ})")

    # verify param equality up front
    md, ms = make_config("dense"), make_config("sparse_wide")
    from model import TinyGPT as _T
    pd = _T(md).count_parameters()
    ps = _T(ms).count_parameters()
    print(f"\nPARAM CHECK  dense={pd:,}  sparse_wide={ps:,}  "
          f"diff={abs(pd-ps):,} ({100*abs(pd-ps)/pd:.4f}%)  attn_dim={md.attn_dim}")

    results = {}
    for ffn in ("dense", "sparse_wide"):
        print(f"\n=== Training {ffn} ===")
        results[ffn] = train_one(ffn, train, val, vocab_size)

    # ---- table ----
    print("\n" + "=" * 72)
    print(f"{'step':>6} | {'dense_train':>11} {'dense_val':>9} | "
          f"{'sparse_train':>12} {'sparse_val':>10}")
    print("-" * 72)
    dc, sc = results["dense"]["curve"], results["sparse_wide"]["curve"]
    for a, b in zip(dc, sc):
        print(f"{a['step']:>6} | {a['train_loss']:>11.4f} {a['val_loss']:>9.4f} | "
              f"{b['train_loss']:>12.4f} {b['val_loss']:>10.4f}")
    print("=" * 72)
    for ffn in ("dense", "sparse_wide"):
        r = results[ffn]
        print(f"{ffn:11s}: params={r['n_params']:,}  final_train={r['final_train_loss']:.4f}  "
              f"final_val={r['final_val_loss']:.4f}  {r['tok_s']:,.0f} tok/s  "
              f"wall={r['wall_s']:.1f}s  peak_mem={r['peak_mem_gb']:.2f}GB")

    df = results["dense"]["final_val_loss"]
    sf = results["sparse_wide"]["final_val_loss"]
    verdict = ("MATCH (within 0.02)" if abs(df - sf) < 0.02
               else "sparse_wide BEATS dense" if sf < df
               else "sparse_wide LAGS dense")
    print(f"\nVERDICT (final val loss): dense={df:.4f} vs sparse_wide={sf:.4f}  -> {verdict}")

    with open(OUT_JSON, "w") as f:
        json.dump({"source": source, "steps": STEPS, "batch": BATCH, "seq": SEQ,
                   "device": DEVICE, "results": results, "verdict": verdict}, f, indent=2)
    print(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
