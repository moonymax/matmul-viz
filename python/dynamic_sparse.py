"""RigL-style dynamic sparse linear layers: grow-and-prune during training.

Both layers keep a fixed number `k` of active weights whose *positions* migrate:
periodically we prune the "deadest" active weights (small |value| AND stable —
small momentum AND small gradient) and grow new ones from zero at the positions
with the largest gradient.

Storage is genuinely sparse (values / rows / cols) and the forward + backward are
the O(B*k) gather-mul-scatter (`SparseLinearFunction`) — NO dense out x in matrix
is ever formed for the forward/backward. The only difference between the two
layers is how they find the *grow* positions (the gradient at inactive cells):

  * `DynamicSparseLinear` (exact): on a probe step, reconstruct the full dense
    weight gradient as the outer product  G = da^T @ x  (from the captured
    output-gradient `da` and layer input `x`) and grow the top-|G| inactive
    cells. This IS an (out x in) materialization — but it is computed SEPARATELY
    from the forward and only every few steps, so it is not paid on every
    forward/backward the way a masked-dense layer would.
  * `ApproxDynamicSparseLinear`: never forms G at all — approximates it from the
    top-p error rows x top-q activation columns (a tiny p x q matmul).

Both capture `x` and `da` (via a tensor hook) only on probe steps.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from sparse_layer import SparseLinearFunction


# --------------------------------------------------------------------------- #
# penalties (operate on the k active weights of every dynamic-sparse layer)
# --------------------------------------------------------------------------- #
def l1_penalty(model: "nn.Module"):
    """Sum |active weights| over sparse layers — add lambda*this to the loss to
    push active weights toward 0 (giving grow-and-prune near-zero candidates)."""
    total = None
    for m in model.modules():
        if isinstance(m, _DynamicSparseBase):
            t = m.values.abs().sum()
            total = t if total is None else total + t
    return total


def l2_penalty(model: "nn.Module"):
    """Sum of squared active weights over sparse layers (for an L2 loss term)."""
    total = None
    for m in model.modules():
        if isinstance(m, _DynamicSparseBase):
            t = m.values.pow(2).sum()
            total = t if total is None else total + t
    return total


def _momentum(optimizer, param, idx):
    """|Adam first moment| at flat positions `idx` (zeros if no state yet)."""
    st = optimizer.state.get(param, None)
    if st and "exp_avg" in st:
        return st["exp_avg"].reshape(-1)[idx].abs()
    return torch.zeros(len(idx), device=param.device)


def _deadness(w_abs, m_abs, g_abs):
    """Low = safe to prune: near-zero weight AND stable (low momentum, low grad).
    Each signal normalized by its max so all three contribute comparably."""
    def norm(t):
        mx = t.max()
        return t / mx if mx > 0 else t
    return norm(w_abs) + norm(m_abs) + norm(g_abs)


# --------------------------------------------------------------------------- #
# shared base: sparse storage + O(k) forward + probe capture + deadness prune
# --------------------------------------------------------------------------- #
class _DynamicSparseBase(nn.Module):
    def __init__(self, in_features, out_features, k, *, init_scale=None, generator=None):
        super().__init__()
        n = in_features * out_features
        assert 1 <= k <= n
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        scale = init_scale if init_scale is not None else math.sqrt(out_features / k)
        flat = torch.randperm(n, generator=generator)[:k]
        self.register_buffer("rows", (flat // in_features).long())   # output index
        self.register_buffer("cols", (flat % in_features).long())    # input index
        self.values = nn.Parameter(torch.randn(k, generator=generator) * scale)
        self._capture = False
        self._x = None
        self._da = None

    def forward(self, x):
        if self._capture:
            self._x = x.detach()
        out = SparseLinearFunction.apply(x, self.values, self.rows, self.cols, self.out_features)
        if self._capture and out.requires_grad:
            out.register_hook(self._save_da)
        return out

    def _save_da(self, g):
        self._da = g.detach()
        return None

    def _active_flat(self):
        return self.rows.long() * self.in_features + self.cols.long()

    @torch.no_grad()
    def _commit_swap(self, grow_flat, optimizer) -> int:
        """Prune `len(grow_flat)` deadest active slots (small |value| AND stable:
        small momentum AND small |grad|), move them to the new flat positions,
        and reset value + Adam moments there. grad at active cells is
        `self.values.grad` (already computed by the O(k) backward)."""
        m = len(grow_flat)
        if m == 0:
            return 0
        gval = self.values.grad.abs() if self.values.grad is not None \
            else torch.zeros_like(self.values)
        slots = torch.arange(self.k, device=self.values.device)
        dead = _deadness(self.values.abs(), _momentum(optimizer, self.values, slots), gval)
        prune_slots = torch.topk(dead, m, largest=False).indices
        grow_t = torch.as_tensor(grow_flat, dtype=torch.long, device=self.values.device)
        self.rows[prune_slots] = grow_t // self.in_features
        self.cols[prune_slots] = grow_t % self.in_features
        self.values.data[prune_slots] = 0.0            # grow from zero -> output unchanged
        st = optimizer.state.get(self.values, None)
        if st:
            for key in ("exp_avg", "exp_avg_sq"):
                if key in st:
                    st[key][prune_slots] = 0.0
        self._x = self._da = None
        return m


class DynamicSparseLinear(_DynamicSparseBase):
    """Exact grow signal via a periodic dense-gradient probe: on probe steps the
    full weight gradient G = da^T @ x is reconstructed from the captured error and
    input (computed separately from the forward, NOT via a dense forward)."""

    @torch.no_grad()
    def grow_prune(self, optimizer, n_swap: int, topk=None) -> int:
        if self._x is None or self._da is None or n_swap <= 0:
            return 0
        n = self.in_features * self.out_features
        n_swap = min(n_swap, self.k, n - self.k)
        if n_swap <= 0:
            return 0
        # exact dense weight gradient, materialized only here (once per probe):
        #   G[i, j] = sum_b da[b, i] * x[b, j]   -> flat idx  i*in + j
        G = (self._da.t() @ self._x).abs().reshape(-1)   # (out*in,)
        G[self._active_flat()] = -1.0                    # exclude active cells
        grow = torch.topk(G, n_swap, largest=True).indices.tolist()
        return self._commit_swap(grow, optimizer)


class ApproxDynamicSparseLinear(_DynamicSparseBase):
    """Approximate grow signal — never forms the (out x in) gradient. The batch
    gradient is large where an output row (error) AND an input column
    (activation) are both large, so we score only the top-p error rows x top-q
    input columns (a tiny p*q matmul) and grow the best inactive cell(s)."""

    @torch.no_grad()
    def grow_prune(self, optimizer, n_swap: int, topk: int) -> int:
        if self._x is None or self._da is None or n_swap <= 0:
            return 0
        x, da = self._x, self._da                        # (B, in), (B, out)
        in_f, out_f = self.in_features, self.out_features
        p, q = min(topk, out_f), min(topk, in_f)
        topR = torch.topk(da.pow(2).sum(0), p).indices           # top error rows
        topC = torch.topk(x.pow(2).sum(0), q).indices            # top activation cols
        scores = (da[:, topR].t() @ x[:, topC]).abs()            # (p, q) approx |gW|
        cand_flat = (topR.view(-1, 1) * in_f + topC.view(1, -1)).reshape(-1)
        cand_score = scores.reshape(-1)

        active = set(self._active_flat().tolist())
        grow, seen = [], set()
        for ci in torch.argsort(cand_score, descending=True).tolist():
            f = int(cand_flat[ci])
            if f in active or f in seen:
                continue
            seen.add(f)
            grow.append(f)
            if len(grow) >= n_swap:
                break
        return self._commit_swap(grow, optimizer)


# --------------------------------------------------------------------------- #
# training-loop helpers
# --------------------------------------------------------------------------- #
def set_capture(model: nn.Module, on: bool):
    """Turn probe-step capture (of x and da) on/off for every dynamic layer."""
    for m in model.modules():
        if isinstance(m, _DynamicSparseBase):
            m._capture = on
            if not on:
                m._x = m._da = None


def rigl_update(model: nn.Module, optimizer, drop_fraction: float, topk: int = 48) -> int:
    """Grow-and-prune every dynamic layer (exact or approx). Requires that this
    step's forward/backward ran under set_capture(model, True). n_swap is scaled
    per layer by its own k."""
    total = 0
    for m in model.modules():
        if isinstance(m, _DynamicSparseBase):
            n_swap = max(1, round(drop_fraction * m.k))
            total += m.grow_prune(optimizer, n_swap, topk=topk)
    return total


def approx_grow_prune(model: nn.Module, optimizer, n_swap: int, topk: int) -> int:
    """Grow-and-prune with a fixed n_swap across layers (back-compat helper)."""
    total = 0
    for m in model.modules():
        if isinstance(m, _DynamicSparseBase):
            total += m.grow_prune(optimizer, n_swap, topk=topk)
    return total


def enforce_masks(model: nn.Module):
    """No-op: sparse storage has no dense weight to re-pin (kept for API compat)."""
    return None


def cosine_drop_fraction(step: int, total_steps: int, alpha: float = 0.3,
                         t_end_frac: float = 0.8) -> float:
    """RigL schedule: start at alpha, cosine-decay to 0, stop growing at t_end."""
    t_end = total_steps * t_end_frac
    if step >= t_end:
        return 0.0
    return alpha * 0.5 * (1.0 + math.cos(math.pi * step / t_end))
