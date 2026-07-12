"""Tiny GPT language model package (ported from tiny-lm main_v3.py).

Keeps the v3 features (RoPE, SwiGLU, PLE, AttnRes, tied embeddings, KV-cache)
but uses a uniform (even) head schedule, AdamW instead of Muon, and offers three
selectable FFN variants: "ple", "sparse", "moe".
"""
