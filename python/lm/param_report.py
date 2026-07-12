"""Put the sparse layers into the LM and report parameter counts:
what the model WOULD have if those weight matrices were dense, vs what it
ACTUALLY has with the O(k) SparseLinear layers.

We sparsify every big matmul weight — attention qkv/proj AND the FFN
gate/up/down — and leave lookups/norms/biases (token embedding, LayerNorms,
AttnRes vectors) dense in both, since our sparse layer only replaces matmuls.
The token embedding is tied with the LM head (counted once).

The model here uses the reference SparseLinear; SparseLinearMetal (the fused
Metal kernel) is a drop-in with the *identical* parameter layout — same counts,
just a faster forward/backward on MPS.
"""

from __future__ import annotations

import torch

try:
    from .model import GPTConfig, TinyGPT
    from .sparse_linear import SparseLinear
except ImportError:
    from model import GPTConfig, TinyGPT
    from sparse_linear import SparseLinear

FFN_HIDDEN = 640          # standard 4x expansion (vs embed_dim 160)
DENSITIES = [0.25, 0.10, 0.05]
HEADLINE = 0.10


def analyze(density: float):
    cfg = GPTConfig(
        ffn_type="sparse",
        sparse_attn=True,
        sparse_hidden=FFN_HIDDEN,
        sparse_density=density,
    ).validate()
    model = TinyGPT(cfg)

    sparse_ids = set()
    attn_de = attn_ac = ffn_de = ffn_ac = 0
    for name, m in model.named_modules():
        if isinstance(m, SparseLinear):
            de, ac = m.in_features * m.out_features, m.values.numel()
            sparse_ids.add(id(m.values))
            if ".attn." in name:
                attn_de += de; attn_ac += ac
            else:
                ffn_de += de; ffn_ac += ac

    other = sum(p.numel() for p in model.parameters() if id(p) not in sparse_ids)
    dense_equiv = attn_de + ffn_de
    sparse_actual = attn_ac + ffn_ac
    return dict(cfg=cfg, other=other, attn_de=attn_de, attn_ac=attn_ac,
                ffn_de=ffn_de, ffn_ac=ffn_ac, dense_equiv=dense_equiv,
                sparse_actual=sparse_actual,
                dense_total=other + dense_equiv,
                sparse_total=other + sparse_actual, model=model)


def fmt(n):
    return f"{n:,}"


def main():
    r = analyze(HEADLINE)
    cfg = r["cfg"]
    print(f"Fully-sparse LM  (embed={cfg.embed_dim}, layers={cfg.n_layers}, "
          f"heads={cfg.uniform_heads}x head_dim={cfg.head_dim} -> attn_dim={cfg.attn_dim}, "
          f"vocab={cfg.vocab_size}, ffn_hidden={FFN_HIDDEN})")
    print(f"Sparsified: attention qkv/proj + FFN gate/up/down.  Headline density = {HEADLINE:.0%}\n")

    print(f"  Shared / non-sparsifiable (tok_emb+lm_head tied, LayerNorms, AttnRes): {fmt(r['other'])}")
    print(f"  Sparsifiable matmul weights, if DENSE:")
    print(f"      attention  : {fmt(r['attn_de'])}")
    print(f"      FFN        : {fmt(r['ffn_de'])}")
    print(f"      total dense: {fmt(r['dense_equiv'])}")
    print()
    print(f"  ==> DENSE model would have  : {fmt(r['dense_total'])} params")
    print(f"  ==> SPARSE model actually has: {fmt(r['sparse_total'])} params  "
          f"(at {HEADLINE:.0%} density)")
    print(f"      compression: {r['dense_total'] / r['sparse_total']:.2f}x fewer params; "
          f"sparsifiable weights {r['dense_equiv']:,} -> {r['sparse_actual']:,} "
          f"({r['sparse_actual']/r['dense_equiv']:.1%})")

    print("\n  density sweep:")
    print(f"    {'density':>8} {'sparse total':>14} {'vs dense':>10}")
    dense_total = r["dense_total"]
    print(f"    {'1.00':>8} {fmt(dense_total):>14} {'1.00x':>10}   (dense baseline)")
    for d in DENSITIES:
        rr = analyze(d)
        print(f"    {d:>8.2f} {fmt(rr['sparse_total']):>14} "
              f"{dense_total / rr['sparse_total']:>9.2f}x")

    # sanity: the fully-sparse model actually runs (CPU, reference SparseLinear)
    m = r["model"]
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, _ = m(idx)
    print(f"\n  forward sanity: idx {tuple(idx.shape)} -> logits {tuple(logits.shape)}, "
          f"finite={bool(torch.isfinite(logits).all())}")


if __name__ == "__main__":
    main()
