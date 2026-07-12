"""Inference for TinyGPT (adapted from tiny-lm/inference_v3.py).

Loads a checkpoint (with its saved config, incl. ffn_type), rebuilds the exact
model, and generates text using the KV-cache. Checkpoint format is the one
saved by train.py.

Usage:
    .venv/bin/python python/lm/inference.py [ffn_type]
where ffn_type in {"ple","sparse","moe"} (default "ple") selects which set of
checkpoints to look for.
"""

import os
import shutil
import sys

import torch
import torch.nn.functional as F

try:
    from .model import TinyGPT, GPTConfig
except ImportError:
    from model import TinyGPT, GPTConfig


def candidate_paths(ffn_type):
    base = [
        f"checkpoint_{ffn_type}.pt",
        f"checkpoint_{ffn_type}_resume.pt",
    ] + [f"checkpoint_{ffn_type}_epoch{i}.pt" for i in range(5, 0, -1)]
    return base


def find_and_copy_checkpoint(ffn_type):
    inference_copy = f"checkpoint_{ffn_type}_inference.pt"
    for path in candidate_paths(ffn_type):
        if os.path.exists(path):
            print(f"Copying {path} -> {inference_copy}")
            shutil.copy2(path, inference_copy)
            return inference_copy
    return None


def load_checkpoint(ffn_type):
    from tokenizers import Tokenizer

    path = find_and_copy_checkpoint(ffn_type)
    if path is None:
        print(f"No checkpoint found for ffn_type={ffn_type!r}. Train first with train.py")
        sys.exit(1)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = GPTConfig.from_dict(ckpt["config"])
    model = TinyGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tok_path = ckpt.get("tokenizer_path", f"tokenizer_{ffn_type}.json")
    if not os.path.exists(tok_path):
        print(f"Tokenizer not found at {tok_path}")
        sys.exit(1)
    tokenizer = Tokenizer.from_file(tok_path)

    step = ckpt.get("global_step", "?")
    epoch = ckpt.get("epoch", "?")
    return model, tokenizer, cfg, step, epoch


@torch.no_grad()
def generate(model, tokenizer, prompt_text, max_tokens=200, temperature=0.8, top_k=40):
    eos_id = tokenizer.token_to_id("<eos>")
    encoding = tokenizer.encode(prompt_text)
    tokens = encoding.ids
    if not tokens:
        tokens = [tokenizer.token_to_id("<unk>")]

    idx = torch.tensor([tokens], dtype=torch.long)

    # Prefill
    logits, kv_cache = model(idx)

    generated_ids = []
    for _ in range(max_tokens):
        next_logits = logits[:, -1, :] / temperature
        if top_k > 0:
            topk_vals, topk_idx = next_logits.topk(top_k)
            next_logits = torch.full_like(next_logits, float("-inf"))
            next_logits.scatter_(1, topk_idx, topk_vals)
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        tok_id = next_token.item()
        if tok_id == eos_id:
            break
        generated_ids.append(tok_id)

        # Decode step (KV-cache)
        logits, kv_cache = model(next_token, kv_cache=kv_cache)

    text = tokenizer.decode(generated_ids)
    print(text)


def main():
    ffn_type = sys.argv[1] if len(sys.argv) > 1 else "ple"
    print(f"Loading checkpoint (ffn_type={ffn_type})...")
    model, tokenizer, cfg, step, epoch = load_checkpoint(ffn_type)
    print(f"Ready. ffn_type={cfg.ffn_type}, vocab={cfg.vocab_size}, dim={cfg.embed_dim}, "
          f"layers={cfg.n_layers}, heads/layer={cfg.uniform_heads}, step={step}, epoch={epoch}")
    print("Type 'reload' to reload the latest checkpoint. Ctrl+C to exit.\n")

    while True:
        try:
            prompt = input(">>> ")
            if not prompt.strip():
                continue
            if prompt.strip().lower() == "reload":
                print("Reloading...")
                model, tokenizer, cfg, step, epoch = load_checkpoint(ffn_type)
                print(f"Loaded step={step}, epoch={epoch}")
                continue
            print(prompt, end="")
            generate(model, tokenizer, prompt)
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            sys.exit(0)


if __name__ == "__main__":
    main()
