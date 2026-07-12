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
    block_size: int = 512
    attn_res: bool = True

    # --- FFN selection ---
    ffn_type: str = "ple"           # {"ple", "sparse", "moe"}

    # --- "sparse" variant ---
    sparse_hidden: int = 6144
    sparse_density: float = 0.25
    sparse_attn: bool = False   # also make attention qkv/proj SparseLinear

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

    def validate(self):
        assert self.ffn_type in ("ple", "sparse", "moe"), f"bad ffn_type {self.ffn_type}"
        assert self.attn_dim != self.embed_dim, (
            f"design constraint violated: n_heads*head_dim ({self.attn_dim}) "
            f"must differ from embed_dim ({self.embed_dim})"
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

    def apply(self, q, k, pos_ids):
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
    def __init__(self, embed_dim, n_heads, head_dim, block_size, sparse=False, density=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.attn_dim = attn_dim = n_heads * head_dim
        self.sparse = sparse
        if sparse:
            k_qkv = max(1, round(density * embed_dim * 3 * attn_dim))
            k_proj = max(1, round(density * attn_dim * embed_dim))
            self.qkv = SparseLinear(embed_dim, 3 * attn_dim, k_qkv)
            self.proj = SparseLinear(attn_dim, embed_dim, k_proj)
        else:
            self.qkv = nn.Linear(embed_dim, 3 * attn_dim)
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
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q, k = self.rope.apply(q, k, pos_ids)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)

        new_cache = (k, v)
        S = k.shape[2]

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
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        C = cfg.embed_dim
        self.gate = nn.Linear(C, C)
        self.layer_emb = nn.Embedding(cfg.vocab_size, C)
        self.down = nn.Linear(C, C)

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
# FFN variant 3: MoE (DeepSeek-V3 style)
# --------------------------------------------------------------------------- #
class Expert(nn.Module):
    """Small bias-free SwiGLU expert."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoEFFN(nn.Module):
    """Fine-grained MoE FFN.

    Recipe (DeepSeek-V3 / Qwen3-MoE / GLM-4.x consensus):
      * many small routed experts + one (or more) always-on shared expert(s)
      * top-k routing with a sigmoid gate; the top-k gate values are normalized
        to sum to 1 and used as the mixing weights
      * aux-loss-free load balancing: a per-expert routing *bias* (a buffer, no
        gradient) is added to the gate scores *only for top-k selection* and is
        nudged each training step toward balancing per-expert load
        (bias += u * sign(mean_load - load_e)).
      * a small classic switch-transformer aux load-balance loss is also exposed
        via `self.last_aux_loss` (collected by the model, added by the trainer).
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
        self.experts = nn.ModuleList([Expert(C, he) for _ in range(self.n_routed)])
        self.shared = nn.ModuleList([Expert(C, he) for _ in range(self.n_shared)])
        # aux-loss-free routing bias (persistent, not optimized)
        self.register_buffer("expert_bias", torch.zeros(self.n_routed))
        self.last_aux_loss = None

    def forward(self, x, token_ids=None):
        B, T, C = x.shape
        xf = x.reshape(-1, C)          # (N, C)
        N = xf.shape[0]

        logits = self.router(xf)               # (N, n_routed)
        scores = torch.sigmoid(logits)         # gate scores in (0, 1)

        # --- top-k selection uses scores + bias; weights use raw scores ---
        routing = scores + self.expert_bias    # buffer -> no grad through bias
        _, topi = routing.topk(self.top_k, dim=-1)      # (N, k)
        gate_w = torch.gather(scores, 1, topi)          # (N, k)
        gate_w = gate_w / (gate_w.sum(-1, keepdim=True) + 1e-9)

        out = torch.zeros_like(xf)
        flat_i = topi.reshape(-1)                       # (N*k,)
        flat_w = gate_w.reshape(-1)                     # (N*k,)
        token_idx = torch.arange(N, device=xf.device).repeat_interleave(self.top_k)

        counts = torch.bincount(flat_i, minlength=self.n_routed).float()
        for e in range(self.n_routed):
            mask = flat_i == e
            if bool(mask.any()):
                sel = token_idx[mask]
                eo = self.experts[e](xf[sel])
                out.index_add_(0, sel, eo * flat_w[mask].unsqueeze(-1))

        for sh in self.shared:
            out = out + sh(xf)

        if self.training:
            # aux-loss-free balancing: nudge bias toward equal load
            with torch.no_grad():
                load = counts / counts.sum().clamp(min=1.0)
                mean_load = 1.0 / self.n_routed
                self.expert_bias += self.bias_update * torch.sign(mean_load - load)
            # classic (differentiable) switch aux loss, small coefficient
            importance = scores.mean(0)                 # (n_routed,), differentiable
            fraction = counts / max(N, 1)               # (n_routed,), constant
            self.last_aux_loss = self.aux_coef * self.n_routed * (importance * fraction).sum()
        else:
            self.last_aux_loss = None

        return out.reshape(B, T, C)


def build_ffn(cfg: GPTConfig) -> nn.Module:
    if cfg.ffn_type == "ple":
        return SwiGLUPLE(cfg)
    if cfg.ffn_type == "sparse":
        return SparseSwiGLU(cfg)
    if cfg.ffn_type == "moe":
        return MoEFFN(cfg)
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

        # AttnRes: learned query vector per layer + final output
        if cfg.attn_res:
            self.ar_proj = nn.Parameter(torch.randn(n_layers + 1, C) * 0.02)

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

        x = self.ln_f(x)
        return self.lm_head(x), new_caches

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
