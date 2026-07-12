"""Inference for TinyGPT — loads ANY checkpoint (incl. intermediate ones) and
generates text with the KV-cache. Rebuilds the exact model from the config saved
in the checkpoint (so GQA / block-sparse / ple / dense / MTP all just work).

The first argument selects the checkpoint; it can be:
  * a checkpoint FILE      -> loaded directly
        inference.py checkpoints/sparse_step400.pt
  * a DIRECTORY            -> the newest *.pt in it (or --step N to pick one)
        inference.py checkpoints --step 200
  * an ffn_type string     -> the train.py naming scheme (backward compatible)
        inference.py ple

Options:
  --prompt "..."      one-shot generation then exit (default: interactive REPL)
  --max-tokens N      (default 200)
  --temperature T     (default 0.8)
  --top-k K           (default 40)
  --step N            with a directory: pick the checkpoint at step N
  --list              with a directory: list available checkpoints and exit

Examples:
  .venv/bin/python python/lm/inference.py checkpoints/sparse_step400.pt --prompt "Once upon a time"
  .venv/bin/python python/lm/inference.py checkpoints --list
  .venv/bin/python python/lm/inference.py ple            # interactive, train.py checkpoints
"""

import argparse
import glob
import os
import re
import sys

import torch
import torch.nn.functional as F

try:
    from .model import TinyGPT, GPTConfig
except ImportError:
    from model import TinyGPT, GPTConfig


def _ckpt_step(path):
    m = re.search(r"step(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def list_dir_checkpoints(d):
    return sorted(glob.glob(os.path.join(d, "*.pt")), key=_ckpt_step)


def resolve_checkpoint(target, step=None):
    """Return a checkpoint file path from a file / directory / ffn_type."""
    if os.path.isfile(target):
        return target
    if os.path.isdir(target):
        ckpts = list_dir_checkpoints(target)
        if not ckpts:
            print(f"No .pt checkpoints in {target}")
            sys.exit(1)
        if step is not None:
            for c in ckpts:
                if _ckpt_step(c) == step:
                    return c
            print(f"No checkpoint at step {step} in {target}. Available: "
                  f"{[_ckpt_step(c) for c in ckpts]}")
            sys.exit(1)
        return ckpts[-1]  # newest by step
    # else: treat as an ffn_type (train.py naming)
    candidates = [f"checkpoint_{target}.pt", f"checkpoint_{target}_resume.pt"] + \
                 [f"checkpoint_{target}_epoch{i}.pt" for i in range(10, 0, -1)]
    for p in candidates:
        if os.path.exists(p):
            return p
    print(f"Could not resolve checkpoint from {target!r} (not a file, dir, or "
          f"known ffn_type checkpoint).")
    sys.exit(1)


def load_checkpoint(path):
    from tokenizers import Tokenizer

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = GPTConfig.from_dict(ckpt["config"])
    model = TinyGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tok_path = ckpt.get("tokenizer_path", "tokenizer_v3.json")
    if not os.path.isabs(tok_path) and not os.path.exists(tok_path):
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(tok_path))
        if os.path.exists(alt):
            tok_path = alt
    if not os.path.exists(tok_path):
        print(f"Tokenizer not found at {tok_path}")
        sys.exit(1)
    tokenizer = Tokenizer.from_file(tok_path)

    step = ckpt.get("global_step", "?")
    return model, tokenizer, cfg, step


@torch.no_grad()
def generate(model, tokenizer, prompt_text, max_tokens=200, temperature=0.8, top_k=40):
    eos_id = tokenizer.token_to_id("<eos>")
    tokens = tokenizer.encode(prompt_text).ids or [tokenizer.token_to_id("<unk>")]
    idx = torch.tensor([tokens], dtype=torch.long)

    logits, kv_cache = model(idx)             # prefill
    out_ids = []
    for _ in range(max_tokens):
        next_logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k > 0:
            vals, ids = next_logits.topk(min(top_k, next_logits.shape[-1]))
            next_logits = torch.full_like(next_logits, float("-inf")).scatter_(1, ids, vals)
        probs = F.softmax(next_logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        tid = nxt.item()
        if tid == eos_id:
            break
        out_ids.append(tid)
        logits, kv_cache = model(nxt, kv_cache=kv_cache)   # decode with KV-cache
    return tokenizer.decode(out_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default="ple",
                    help="checkpoint file, directory, or ffn_type")
    ap.add_argument("--prompt", default=None, help="one-shot prompt (skip REPL)")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--step", type=int, default=None, help="with a directory: pick this step")
    ap.add_argument("--list", action="store_true", help="with a directory: list checkpoints")
    args = ap.parse_args()

    if args.list and os.path.isdir(args.target):
        for c in list_dir_checkpoints(args.target):
            print(f"  step {_ckpt_step(c):>6}  {c}")
        return

    path = resolve_checkpoint(args.target, args.step)
    print(f"Loading {path} ...")
    model, tokenizer, cfg, step = load_checkpoint(path)
    print(f"Ready. ffn_type={cfg.ffn_type} vocab={cfg.vocab_size} dim={cfg.embed_dim} "
          f"layers={cfg.n_layers} heads={cfg.uniform_heads} kv_heads={cfg.n_kv_heads} step={step}")

    gen = lambda p: generate(model, tokenizer, p, args.max_tokens, args.temperature, args.top_k)

    if args.prompt is not None:
        print(args.prompt, end="", flush=True)
        print(gen(args.prompt))
        return

    print("Interactive. Ctrl+C to exit.\n")
    while True:
        try:
            prompt = input(">>> ")
            if not prompt.strip():
                continue
            print(prompt, end="")
            print(gen(prompt))
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            return


if __name__ == "__main__":
    main()
