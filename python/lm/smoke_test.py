"""Smoke test: build each FFN variant, run forward + loss + backward on random
token ids, assert finite loss / grads and correct logits shape.

Downloads nothing and needs neither `datasets` nor `tokenizers`.

Run:  .venv/bin/python python/lm/smoke_test.py
"""

import torch
import torch.nn.functional as F

try:
    from .model import TinyGPT, GPTConfig, UNIFORM_HEADS
except ImportError:
    from model import TinyGPT, GPTConfig, UNIFORM_HEADS

DEVICE = "cpu"   # deterministic + avoids large-tensor MPS quirks for the sparse variant
B, T = 2, 64


def run_variant(ffn_type: str) -> int:
    torch.manual_seed(0)
    # Real config; only block_size is shrunk for speed (vocab kept at 4096).
    cfg = GPTConfig(ffn_type=ffn_type, block_size=128).validate()
    model = TinyGPT(cfg).to(DEVICE)
    model.train()
    n_params = model.count_parameters()

    idx = torch.randint(0, cfg.vocab_size, (B, T), device=DEVICE)
    targets = torch.randint(0, cfg.vocab_size, (B, T), device=DEVICE)

    logits, _ = model(idx)
    assert logits.shape == (B, T, cfg.vocab_size), \
        f"[{ffn_type}] bad logits shape {tuple(logits.shape)}"

    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
    aux = model.collect_aux_loss()
    total = loss + aux
    assert torch.isfinite(total), f"[{ffn_type}] non-finite loss {total.item()}"

    total.backward()

    n_grad = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"[{ffn_type}] non-finite grad in {name}"
            n_grad += 1
    assert n_grad > 0, f"[{ffn_type}] no gradients produced"

    print(f"  [{ffn_type:6s}] OK  loss={loss.item():.4f}  aux={float(aux.detach()):.6f}  "
          f"logits={tuple(logits.shape)}  grads_ok={n_grad}  params={n_params:,}")
    return n_params


def main():
    print(f"Smoke test on {DEVICE}  (B={B}, T={T}, uniform_heads={UNIFORM_HEADS})")
    print(f"  attn_dim = {UNIFORM_HEADS}*32 = {UNIFORM_HEADS*32} (!= embed_dim 160)\n")
    results = {}
    for ffn in ("ple", "sparse", "moe"):
        results[ffn] = run_variant(ffn)
    print("\nParameter counts (real config, block_size 128):")
    for ffn, n in results.items():
        print(f"  {ffn:6s}: {n:,}")
    print("\nAll variants passed.")


if __name__ == "__main__":
    main()
