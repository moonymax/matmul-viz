"""TinyGPT model: RoPE + SwiGLU + AttnRes + tied embeddings + KV-cache.

Ported from tiny-lm/main_v3.py with two structural changes:

  * EVEN head schedule (uniform heads per layer) instead of the v3 "diamond".
    We keep the design constraint  n_heads * head_dim != embed_dim :
    with embed_dim=160 and head_dim=32, uniform 5 heads would give exactly 160
    (equal -> not allowed), so we use 6 heads/layer -> attn_dim = 192 != 160.

  * Three interchangeable FFN variants, selected by GPTConfig.ffn_type:
        "ple"    - v3 baseline SwiGLU + per-layer token embedding
        "sparse" - SwiGLU whose 3 projections are O(k) SparseLinear layers
        "moe"    - DeepSeek-V3-style MoE: fine-grained routed experts + shared
                   expert(s), top-k routing, aux-loss-free bias balancing.

This module imports ONLY torch (no datasets / tokenizers) so it can be
constructed and unit-tested without any dataset dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # works both as a package (lm.model) and as a plain script on sys.path
    from .sparse_linear import SparseLinear
except ImportError:  # pragma: no cover
    from sparse_linear import SparseLinear

# BlockSparseLinear lives in python/ (one level above lm/); add it to sys.path.
try:
    import os as _os
    import sys as _sys
    _PY = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _PY not in _sys.path:
        _sys.path.insert(0, _PY)
    from block_sparse import BlockSparseLinear
except ImportError:  # pragma: no cover
    BlockSparseLinear = None


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
# Even (uniform) head schedule.  Trivially changeable: set UNIFORM_HEADS.
# Constraint kept from v3: n_heads * head_dim != embed_dim.
#   embed_dim=160, head_dim=32  ->  5 heads == 160 (NOT allowed),
#                                    6 heads -> attn_dim = 192 != 160  (used).
UNIFORM_HEADS = 6


@dataclass
class GPTConfig:
    # --- shared (identical across all FFN variants) ---
    vocab_size: int = 4096          # BPE target vocab
    embed_dim: int = 160
    n_layers: int = 11
    head_dim: int = 32
    uniform_heads: int = UNIFORM_HEADS   # HEADS_PER_LAYER = [uniform_heads] * n_layers
    n_kv_heads: int = UNIFORM_HEADS      # GQA: K/V head count (< uniform_heads groups queries)
    block_size: int = 512
    attn_res: bool = True

    # --- multi-token prediction (Meta/Gloeckle parallel heads) ---
    n_mtp: int = 0                  # extra future-token heads (0 disables MTP)
    mtp_weight: float = 0.3         # weight of each auxiliary MTP head loss

    # --- FFN selection ---
    ffn_type: str = "ple"           # {"ple","sparse","moe","moe_ple","dense","sparse_wide","block"}

    # --- expansion knobs (H=embed_dim => "no expansion"; H=k*embed_dim => "expansion") ---
    ple_hidden: int = 160           # SwiGLUPLE hidden (gate/layer_emb/down width)
    moe_ple_hidden: int = 160       # MoE-PLE per-expert hidden (shared emb width)

    # --- "block" variant (moveable dense b x b blocks; grow-and-prune) ---
    block_hidden: int = 768         # SwiGLU hidden dim (wide)
    block_b: int = 16               # block edge
    block_density: float = 0.25     # fraction of b x b tiles active (solved for param parity)

    # --- "sparse" variant ---
    sparse_hidden: int = 6144
    sparse_density: float = 0.25
    sparse_attn: bool = False   # also make attention qkv/proj SparseLinear

    # --- "dense" variant (plain bias-free SwiGLU, param-equal baseline) ---
    dense_hidden: int = 512

    # --- "sparse_wide" variant (very wide FFN, fused Metal SparseLinear) ---
    #   param-equal to "dense" by construction:
    #   k(gate)=k(up)=k(down)=round(density * embed_dim * wide_hidden)
    wide_hidden: int = 51200
    wide_density: float = 0.01

    # --- "moe" variant (DeepSeek-V3 style, fine-grained) ---
    moe_n_routed: int = 16          # number of fine-grained routed experts
    moe_n_shared: int = 1           # always-on shared expert(s)
    moe_top_k: int = 4              # routed experts activated per token
    moe_expert_hidden: int = 96     # SwiGLU hidden dim of each (small) expert
    moe_aux_coef: float = 1e-3      # weight of the (optional) aux load-balance loss
    moe_bias_update: float = 1e-3   # aux-loss-free routing-bias update speed

    @property
    def heads_per_layer(self):
        return [self.uniform_heads] * self.n_layers

    @property
    def attn_dim(self):
        return self.uniform_heads * self.head_dim

    @property
    def kv_dim(self):
        return self.n_kv_heads * self.head_dim

    def validate(self):
        assert self.ffn_type in ("ple", "sparse", "moe", "moe_ple", "dense",
                                 "sparse_wide", "block"), \
            f"bad ffn_type {self.ffn_type}"
        assert self.attn_dim != self.embed_dim, (
            f"design constraint violated: n_heads*head_dim ({self.attn_dim}) "
            f"must differ from embed_dim ({self.embed_dim})"
        )
        assert self.uniform_heads % self.n_kv_heads == 0, (
            f"GQA requires uniform_heads ({self.uniform_heads}) divisible by "
            f"n_kv_heads ({self.n_kv_heads})"
        )
        if self.ffn_type == "block":
            assert self.embed_dim % self.block_b == 0 and self.block_hidden % self.block_b == 0, (
                f"block_b ({self.block_b}) must divide embed_dim ({self.embed_dim}) "
                f"and block_hidden ({self.block_hidden})"
            )
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        """Rebuild from a (possibly older / extra-key) checkpoint dict."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# --------------------------------------------------------------------------- #
# RoPE
# --------------------------------------------------------------------------- #
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
        self.register_buffer("cos", angles.cos())
        self.register_buffer("sin", angles.sin())

    def forward(self, q, k, pos_ids):
        cos = self.cos[pos_ids].unsqueeze(1)
        sin = self.sin[pos_ids].unsqueeze(1)

        def rotate(x):
            x1, x2 = x[..., ::2], x[..., 1::2]
            return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)

        return rotate(q), rotate(k)


# --------------------------------------------------------------------------- #
# Attention (supports n_heads * head_dim != embed_dim) + KV-cache
# --------------------------------------------------------------------------- #
class CausalSelfAttention(nn.Module):
    """GQA-capable causal self-attention.

    Query has `n_heads` heads; keys/values have `n_kv_heads` heads (n_kv_heads
    divides n_heads). Each KV head is shared by `n_heads // n_kv_heads` query
    heads (repeat_interleave before the score matmul). n_kv_heads == n_heads
    recovers plain multi-head attention.
    """

    def __init__(self, embed_dim, n_heads, head_dim, block_size,
                 n_kv_heads=None, sparse=False, density=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        assert n_heads % self.n_kv_heads == 0
        self.n_rep = n_heads // self.n_kv_heads
        self.head_dim = head_dim
        self.attn_dim = attn_dim = n_heads * head_dim
        self.kv_dim = kv_dim = self.n_kv_heads * head_dim
        self.qkv_out = attn_dim + 2 * kv_dim          # q (+ k, v with fewer heads)
        self.sparse = sparse
        if sparse:
            k_qkv = max(1, round(density * embed_dim * self.qkv_out))
            k_proj = max(1, round(density * attn_dim * embed_dim))
            self.qkv = SparseLinear(embed_dim, self.qkv_out, k_qkv)
            self.proj = SparseLinear(attn_dim, embed_dim, k_proj)
        else:
            self.qkv = nn.Linear(embed_dim, self.qkv_out)
            self.proj = nn.Linear(attn_dim, embed_dim)
        self.rope = RotaryEmbedding(head_dim, block_size)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size)).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, x, pos_ids, kv_cache=None):
        B, T, C = x.shape
        # SparseLinear expects 2-D (N, in); nn.Linear handles 3-D natively.
        qkv = self.qkv(x.reshape(B * T, C)) if self.sparse else self.qkv(x)
        qkv = qkv.reshape(B, T, self.qkv_out)
        q, k, v = qkv.split([self.attn_dim, self.kv_dim, self.kv_dim], dim=-1)
        q = q.reshape(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        q, k = self.rope(q, k, pos_ids)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)

        new_cache = (k, v)                            # cache the compact KV (pre-repeat)
        S = k.shape[2]

        # expand KV heads to match the query-head count (GQA)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if T > 1:
            mask = self.causal_mask[:, :, :T, :S]
            att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        out = (att @ v).transpose(1, 2).reshape(B, T, self.attn_dim)
        out = self.proj(out.reshape(B * T, self.attn_dim)).reshape(B, T, C) if self.sparse \
            else self.proj(out)
        return out, new_cache


# --------------------------------------------------------------------------- #
# FFN variant 1: SwiGLU + per-layer embedding (v3 baseline)
# --------------------------------------------------------------------------- #
class SwiGLUPLE(nn.Module):
    """SwiGLU whose up/value path is a per-layer token embedding. Hidden dim H =
    cfg.ple_hidden (H=embed_dim => no expansion; H=k*embed_dim => expansion)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        C = cfg.embed_dim
        H = cfg.ple_hidden
        self.gate = nn.Linear(C, H)
        self.layer_emb = nn.Embedding(cfg.vocab_size, H)
        self.down = nn.Linear(H, C)

    def forward(self, x, token_ids):
        return self.down(F.silu(self.gate(x)) * self.layer_emb(token_ids))


# --------------------------------------------------------------------------- #
# FFN variant 2: SwiGLU with O(k) SparseLinear projections (no PLE)
# --------------------------------------------------------------------------- #
class SparseSwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        C = cfg.embed_dim
        H = cfg.sparse_hidden
        d = cfg.sparse_density
        k_in = max(1, int(round(d * C * H)))    # dim -> hidden
        k_out = max(1, int(round(d * H * C)))   # hidden -> dim
        self.gate = SparseLinear(C, H, k_in)
        self.up = SparseLinear(C, H, k_in)
        self.down = SparseLinear(H, C, k_out)

    def forward(self, x, token_ids=None):
        B, T, C = x.shape
        xf = x.reshape(-1, C)
        h = F.silu(self.gate(xf)) * self.up(xf)
        out = self.down(h)
        return out.reshape(B, T, C)


# --------------------------------------------------------------------------- #
# FFN variant 3: MoE (DeepSeek-V3 style) — BATCHED grouped-GEMM dispatch
# --------------------------------------------------------------------------- #
def _moe_group(topi, gate_w, N, n_routed):
    """Group the (N*top_k) token->expert assignments for a batched (bmm) dispatch.

    Returns tok_sorted (which token each assignment belongs to, grouped by expert),
    slot (its position in a padded [n_routed, cap] grid), cap (= max tokens routed
    to any expert -> DROPLESS: every assignment fits, none dropped), w_sorted (the
    gate weight per assignment) and counts (per-expert load). Padding slots that no
    assignment maps to stay zero (silu(0)*0 @ W = 0) and are never scattered back."""
    k = topi.shape[1]
    flat_e = topi.reshape(-1)
    flat_tok = torch.arange(N, device=topi.device).repeat_interleave(k)
    counts = torch.bincount(flat_e, minlength=n_routed)
    cap = int(counts.max().item())
    order = torch.argsort(flat_e)                     # groups equal experts contiguously
    e_sorted = flat_e[order]
    tok_sorted = flat_tok[order]
    offsets = counts.cumsum(0) - counts               # start index of each expert's block
    rank = torch.arange(flat_e.numel(), device=topi.device) - offsets[e_sorted]
    slot = e_sorted * cap + rank                      # position in the [n_routed*cap] grid
    w_sorted = gate_w.reshape(-1)[order]
    return tok_sorted, slot, cap, w_sorted, counts


class MoEFFN(nn.Module):
    """Fine-grained MoE FFN (DeepSeek-V3 / Qwen3 / GLM / Kimi recipe):
      * many small routed experts + always-on shared expert(s)
      * top-k routing, sigmoid gate, top-k weights normalized to sum 1
      * aux-loss-free load balancing via a per-expert routing bias buffer
      * DROPLESS (no capacity factor); dispatch is a BATCHED grouped GEMM — expert
        weights are stacked (n_routed, ., .) and applied with bmm over a padded
        [n_routed, cap] token grid, so it runs on the matmul units instead of a
        Python per-expert loop. cap = max per-expert load => nothing is dropped.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        C = cfg.embed_dim
        self.n_routed = cfg.moe_n_routed
        self.n_shared = cfg.moe_n_shared
        self.top_k = cfg.moe_top_k
        self.aux_coef = cfg.moe_aux_coef
        self.bias_update = cfg.moe_bias_update
        he = cfg.moe_expert_hidden

        self.router = nn.Linear(C, self.n_routed, bias=False)
        self.Wg = nn.Parameter(torch.empty(self.n_routed, C, he))   # gate
        self.Wu = nn.Parameter(torch.empty(self.n_routed, C, he))   # up
        self.Wd = nn.Parameter(torch.empty(self.n_routed, he, C))   # down
        self.Wsg = nn.Parameter(torch.empty(self.n_shared, C, he))  # shared gate
        self.Wsu = nn.Parameter(torch.empty(self.n_shared, C, he))
        self.Wsd = nn.Parameter(torch.empty(self.n_shared, he, C))
        for w in (self.Wg, self.Wu, self.Wd, self.Wsg, self.Wsu, self.Wsd):
            nn.init.normal_(w, mean=0.0, std=0.02)
        self.register_buffer("expert_bias", torch.zeros(self.n_routed))
        self.last_aux_loss = None

    def forward(self, x, token_ids=None):
        B, T, C = x.shape
        xf = x.reshape(-1, C)
        N = xf.shape[0]

        scores = torch.sigmoid(self.router(xf))                 # (N, n_routed)
        routing = scores + self.expert_bias                     # bias: selection only
        _, topi = routing.topk(self.top_k, dim=-1)
        gate_w = torch.gather(scores, 1, topi)
        gate_w = gate_w / (gate_w.sum(-1, keepdim=True) + 1e-9)

        tok_sorted, slot, cap, w_sorted, counts = _moe_group(topi, gate_w, N, self.n_routed)
        Xpad = xf.new_zeros(self.n_routed * cap, C)
        Xpad[slot] = xf[tok_sorted]
        Xpad = Xpad.view(self.n_routed, cap, C)
        h = F.silu(torch.bmm(Xpad, self.Wg)) * torch.bmm(Xpad, self.Wu)   # (n_routed, cap, he)
        o = torch.bmm(h, self.Wd).reshape(self.n_routed * cap, C)[slot]   # (M, C)
        out = xf.new_zeros(N, C)
        out.index_add_(0, tok_sorted, o * w_sorted.unsqueeze(-1))

        for i in range(self.n_shared):                          # shared expert(s), all tokens
            out = out + (F.silu(xf @ self.Wsg[i]) * (xf @ self.Wsu[i])) @ self.Wsd[i]

        if self.training:
            with torch.no_grad():
                load = counts.float() / counts.sum().clamp(min=1.0)
                self.expert_bias += self.bias_update * torch.sign(1.0 / self.n_routed - load)
            importance = scores.mean(0)
            fraction = counts.float() / max(N, 1)
            self.last_aux_loss = self.aux_coef * self.n_routed * (importance * fraction).sum()
        else:
            self.last_aux_loss = None

        return out.reshape(B, T, C)


# --------------------------------------------------------------------------- #
# FFN variant 3b: MoE-PLE — DeepSeek-style routing (fine-grained routed experts +
# always-on shared expert(s), normalized sigmoid gating, aux-loss-free bias load
# balancing) but the experts SHARE one per-layer token embedding as the up/value
# path; each expert has its OWN gate and OWN down projection. So an expert is
#   down_e( silu(gate_e(x)) * layer_emb(token) )
# with layer_emb shared across all experts (routed + shared).
# --------------------------------------------------------------------------- #
class MoEPLEFFN(nn.Module):
    """MoE-PLE, batched grouped-GEMM (see MoEFFN). Experts SHARE one per-layer
    token embedding as the up/value path; each has its own gate + down."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        C = cfg.embed_dim
        H = cfg.moe_ple_hidden
        self.n_routed = cfg.moe_n_routed
        self.n_shared = cfg.moe_n_shared
        self.top_k = cfg.moe_top_k
        self.aux_coef = cfg.moe_aux_coef
        self.bias_update = cfg.moe_bias_update

        self.router = nn.Linear(C, self.n_routed, bias=False)
        self.layer_emb = nn.Embedding(cfg.vocab_size, H)            # shared value path
        self.Wg = nn.Parameter(torch.empty(self.n_routed, C, H))    # per-expert gate
        self.Wd = nn.Parameter(torch.empty(self.n_routed, H, C))    # per-expert down
        self.Wsg = nn.Parameter(torch.empty(self.n_shared, C, H))
        self.Wsd = nn.Parameter(torch.empty(self.n_shared, H, C))
        for w in (self.Wg, self.Wd, self.Wsg, self.Wsd):
            nn.init.normal_(w, mean=0.0, std=0.02)
        self.register_buffer("expert_bias", torch.zeros(self.n_routed))
        self.last_aux_loss = None

    def forward(self, x, token_ids=None):
        B, T, C = x.shape
        xf = x.reshape(-1, C)
        emb = self.layer_emb(token_ids.reshape(-1))                 # (N, H) shared value
        N = xf.shape[0]

        scores = torch.sigmoid(self.router(xf))
        routing = scores + self.expert_bias
        _, topi = routing.topk(self.top_k, dim=-1)
        gate_w = torch.gather(scores, 1, topi)
        gate_w = gate_w / (gate_w.sum(-1, keepdim=True) + 1e-9)

        tok_sorted, slot, cap, w_sorted, counts = _moe_group(topi, gate_w, N, self.n_routed)
        H = emb.shape[1]
        Xpad = xf.new_zeros(self.n_routed * cap, C)
        Epad = emb.new_zeros(self.n_routed * cap, H)
        Xpad[slot] = xf[tok_sorted]
        Epad[slot] = emb[tok_sorted]
        Xpad = Xpad.view(self.n_routed, cap, C)
        Epad = Epad.view(self.n_routed, cap, H)
        h = F.silu(torch.bmm(Xpad, self.Wg)) * Epad                 # (n_routed, cap, H)
        o = torch.bmm(h, self.Wd).reshape(self.n_routed * cap, C)[slot]
        out = xf.new_zeros(N, C)
        out.index_add_(0, tok_sorted, o * w_sorted.unsqueeze(-1))

        for i in range(self.n_shared):
            out = out + (F.silu(xf @ self.Wsg[i]) * emb) @ self.Wsd[i]

        if self.training:
            with torch.no_grad():
                load = counts.float() / counts.sum().clamp(min=1.0)
                self.expert_bias += self.bias_update * torch.sign(1.0 / self.n_routed - load)
            importance = scores.mean(0)
            fraction = counts.float() / max(N, 1)
            self.last_aux_loss = self.aux_coef * self.n_routed * (importance * fraction).sum()
        else:
            self.last_aux_loss = None

        return out.reshape(B, T, C)


# --------------------------------------------------------------------------- #
# FFN variant 4: plain bias-free SwiGLU (dense baseline for the param-equal
# dense-vs-sparse-wide experiment)
# --------------------------------------------------------------------------- #
class DenseSwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        C = cfg.embed_dim
        H = cfg.dense_hidden
        self.gate = nn.Linear(C, H, bias=False)
        self.up = nn.Linear(C, H, bias=False)
        self.down = nn.Linear(H, C, bias=False)

    def forward(self, x, token_ids=None):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# --------------------------------------------------------------------------- #
# FFN variant 5: VERY-WIDE bias-free SwiGLU with fused-Metal SparseLinear
# projections at low density -> param-equal to DenseSwiGLU by construction.
# Gradient-checkpointed so the (tokens, wide_hidden) activation is recomputed
# in backward rather than stored across all layers.  SparseLinearMetal is
# imported LAZILY so this module stays importable without the Metal toolchain.
# --------------------------------------------------------------------------- #
class SparseWideSwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        try:
            import sys, os
            _metal = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metal")
            if _metal not in sys.path:
                sys.path.insert(0, _metal)
            from sparse_metal import SparseLinearMetal
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "SparseWideSwiGLU needs python/metal/sparse_metal.py (Metal/MPS)"
            ) from e

        C = cfg.embed_dim
        H = cfg.wide_hidden
        d = cfg.wide_density
        k_in = max(1, int(round(d * C * H)))    # dim -> hidden
        k_out = max(1, int(round(d * H * C)))   # hidden -> dim
        self.gate = SparseLinearMetal(C, H, k_in)
        self.up = SparseLinearMetal(C, H, k_in)
        self.down = SparseLinearMetal(H, C, k_out)

    def _inner(self, xf):
        h = F.silu(self.gate(xf)) * self.up(xf)
        return self.down(h)

    def forward(self, x, token_ids=None):
        B, T, C = x.shape
        xf = x.reshape(-1, C)
        if self.training and xf.requires_grad:
            from torch.utils.checkpoint import checkpoint
            out = checkpoint(self._inner, xf, use_reentrant=False)
        else:
            out = self._inner(xf)
        return out.reshape(B, T, C)


# --------------------------------------------------------------------------- #
# FFN variant 6: BLOCK-SPARSE bias-free SwiGLU. gate/up/down are BlockSparseLinear
# (moveable dense b x b tiles, grow-and-prune). Param-equal to DenseSwiGLU by
# construction (n_active chosen from block_density). The blocks run as batched
# dense matmuls (GPU matmul units). During training the layer captures its input
# and output-gradient (forward/backward hooks) so a periodic grow-and-prune can
# migrate whole blocks toward the highest-gradient-energy regions.
# --------------------------------------------------------------------------- #
class BlockSparseSwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        if BlockSparseLinear is None:
            raise ImportError("BlockSparseSwiGLU needs python/block_sparse.py")
        C = cfg.embed_dim
        H = cfg.block_hidden
        b = cfg.block_b
        d = cfg.block_density
        n_in = max(1, round(d * (C // b) * (H // b)))    # blocks for C->H (gate, up)
        n_out = max(1, round(d * (H // b) * (C // b)))   # blocks for H->C (down)
        self.gate = BlockSparseLinear(C, H, b, n_in)
        self.up = BlockSparseLinear(C, H, b, n_in)
        self.down = BlockSparseLinear(H, C, b, n_out)
        self._blocks = [self.gate, self.up, self.down]
        self._cap = {}   # id(layer) -> [input, grad_out] captured during train step
        for m in self._blocks:
            m.register_forward_hook(self._fhook)
            m.register_full_backward_hook(self._bhook)

    def _fhook(self, mod, inp, out):
        if mod.training:
            self._cap.setdefault(id(mod), [None, None])[0] = inp[0].detach()

    def _bhook(self, mod, gin, gout):
        if mod.training:
            self._cap.setdefault(id(mod), [None, None])[1] = gout[0].detach()

    def forward(self, x, token_ids=None):
        B, T, C = x.shape
        xf = x.reshape(-1, C)
        h = F.silu(self.gate(xf)) * self.up(xf)
        out = self.down(h)
        return out.reshape(B, T, C)

    @torch.no_grad()
    def grow_prune(self, optimizer, n_swap):
        moved = 0
        for m in self._blocks:
            cx, cg = self._cap.get(id(m), [None, None])
            if cx is not None and cg is not None:
                moved += m.grow_prune(optimizer, cx, cg, n_swap=n_swap)
        return moved


def build_ffn(cfg: GPTConfig) -> nn.Module:
    if cfg.ffn_type == "ple":
        return SwiGLUPLE(cfg)
    if cfg.ffn_type == "sparse":
        return SparseSwiGLU(cfg)
    if cfg.ffn_type == "moe":
        return MoEFFN(cfg)
    if cfg.ffn_type == "moe_ple":
        return MoEPLEFFN(cfg)
    if cfg.ffn_type == "dense":
        return DenseSwiGLU(cfg)
    if cfg.ffn_type == "sparse_wide":
        return SparseWideSwiGLU(cfg)
    if cfg.ffn_type == "block":
        return BlockSparseSwiGLU(cfg)
    raise ValueError(f"unknown ffn_type {cfg.ffn_type!r}")


# --------------------------------------------------------------------------- #
# Transformer block (attention + selectable FFN)
# --------------------------------------------------------------------------- #
class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig, n_heads: int):
        super().__init__()
        C = cfg.embed_dim
        self.ln1 = nn.LayerNorm(C)
        self.attn = CausalSelfAttention(
            C, n_heads, cfg.head_dim, cfg.block_size,
            n_kv_heads=cfg.n_kv_heads,
            sparse=cfg.sparse_attn, density=cfg.sparse_density,
        )
        self.ln2 = nn.LayerNorm(C)
        self.mlp = build_ffn(cfg)

    def forward(self, x, token_ids, pos_ids, kv_cache=None):
        attn_out, new_cache = self.attn(self.ln1(x), pos_ids, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x), token_ids)
        return x, new_cache


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        C = cfg.embed_dim
        heads = cfg.heads_per_layer
        n_layers = len(heads)
        self.block_size = cfg.block_size
        self.attn_res = cfg.attn_res

        self.tok_emb = nn.Embedding(cfg.vocab_size, C)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, h) for h in heads])
        self.ln_f = nn.LayerNorm(C)
        self.lm_head = nn.Linear(C, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied

        # Multi-token prediction: n_mtp parallel heads (Meta/Gloeckle style). Each
        # reads the shared trunk hidden and predicts a further-future token through
        # the (tied) unembedding. Head d predicts token t+1+d. Off when n_mtp == 0.
        self.n_mtp = cfg.n_mtp
        self.last_mtp_logits = None
        if cfg.n_mtp > 0:
            self.mtp_heads = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(C), nn.Linear(C, C, bias=False))
                for _ in range(cfg.n_mtp)
            ])

        # AttnRes: learned query vector per layer + final output
        if cfg.attn_res:
            self.ar_proj = nn.Parameter(torch.randn(n_layers + 1, C) * 0.02)

        # GPT-2-style init: N(0, 0.02) on Linear/Embedding weights, zero biases.
        # Without this the tied N(0,1) embedding/head makes the initial logits
        # enormous (loss ~150) and wastes ~100 steps recovering. Custom sparse
        # params (SparseLinear.values, BlockSparseLinear.blocks) keep their own
        # fan-in init — _init_weights only touches nn.Linear/nn.Embedding, and it
        # runs in-place so the tied lm_head/tok_emb weight stays shared.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, kv_cache=None):
        B, T = idx.shape
        if kv_cache is not None and kv_cache[0] is not None:
            past_len = kv_cache[0][0].shape[2]
        else:
            past_len = 0
        pos_ids = torch.arange(past_len, past_len + T, device=idx.device).unsqueeze(0).expand(B, -1)

        x = self.tok_emb(idx)

        new_caches = []
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)

        if self.attn_res:
            layer_outs = [x]
            for i, block in enumerate(self.blocks):
                x = self._attn_res_combine(layer_outs, self.ar_proj[i])
                x, cache_i = block(x, idx, pos_ids, kv_cache=kv_cache[i])
                new_caches.append(cache_i)
                layer_outs.append(x)
            x = self._attn_res_combine(layer_outs, self.ar_proj[-1])
        else:
            for i, block in enumerate(self.blocks):
                x, cache_i = block(x, idx, pos_ids, kv_cache=kv_cache[i])
                new_caches.append(cache_i)

        h_trunk = x                                 # shared trunk hidden
        x = self.ln_f(h_trunk)
        if self.n_mtp and self.training:
            self.last_mtp_logits = [
                F.linear(head(h_trunk), self.tok_emb.weight) for head in self.mtp_heads
            ]
        else:
            self.last_mtp_logits = None
        return self.lm_head(x), new_caches

    def mtp_loss(self, y, ignore_index=-100):
        """Auxiliary multi-token-prediction loss over the extra heads.

        Main head predicts y[t] (= token t+1); MTP head d predicts y[t+d]
        (= token t+1+d), so its logits at position t are matched against y
        shifted left by d (the last d positions are dropped)."""
        if not self.n_mtp or self.last_mtp_logits is None:
            return torch.zeros((), device=self.tok_emb.weight.device)
        V = self.cfg.vocab_size
        T = y.shape[1]
        total = torch.zeros((), device=y.device)
        for d, lg in enumerate(self.last_mtp_logits, start=1):
            if T - d <= 0:
                continue
            pred = lg[:, :T - d].reshape(-1, V)
            tgt = y[:, d:].reshape(-1)
            total = total + F.cross_entropy(pred, tgt, ignore_index=ignore_index)
        return self.cfg.mtp_weight * total

    @torch.no_grad()
    def sparse_update(self, optimizer, n_swap):
        """Grow-and-prune all block-sparse FFNs (no-op for other variants)."""
        moved = 0
        for m in self.modules():
            if isinstance(m, BlockSparseSwiGLU):
                moved += m.grow_prune(optimizer, n_swap)
        return moved

    def _attn_res_combine(self, layer_outs, proj):
        """Softmax-weighted sum of previous layer outputs (no big stacks)."""
        scores = torch.stack([(proj * out).sum(-1) for out in layer_outs])  # (n, B, T)
        weights = torch.softmax(scores, dim=0)
        x = weights[0].unsqueeze(-1) * layer_outs[0]
        for j in range(1, len(layer_outs)):
            x = x + weights[j].unsqueeze(-1) * layer_outs[j]
        return x

    def collect_aux_loss(self):
        """Sum MoE auxiliary load-balance losses across blocks (0 if none)."""
        total = None
        for m in self.modules():
            aux = getattr(m, "last_aux_loss", None)
            if aux is not None:
                total = aux if total is None else total + aux
        if total is None:
            return torch.zeros((), device=self.tok_emb.weight.device)
        return total

    def count_parameters(self):
        seen, total = set(), 0
        for _, p in self.named_parameters():
            if p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                total += p.numel()
        return total
