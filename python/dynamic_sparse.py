"""RigL-style dynamic sparse linear layer: grow-and-prune during training.

Keeps a fixed number `k` of active weights, but the *positions* migrate:
periodically we prune the smallest-magnitude active weights and grow the
largest-gradient inactive ones (from zero). This is the PyTorch analogue of the
grow-and-prune we did in the JS demo.

Implementation note: this is "masked dense" — we store a full dense weight with
inactive entries pinned to 0. That makes the forward identical to a true sparse
op (zeros contribute nothing), while `weight.grad` stays *dense* so we can read
the gradient at inactive positions — which is exactly the grow signal RigL needs
(you can't get it from an O(k) sparse kernel without a periodic dense probe).
For real efficiency you'd pay that dense-gradient probe only every ΔT steps and
run the O(k) kernel otherwise; here (small synthetic models) we keep it simple.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def l1_penalty(model: "nn.Module"):
    """Sum |active weights| over sparse layers — add lambda*this to the training
    loss so active weights are pushed toward 0, giving grow-and-prune genuinely
    near-zero (safe-to-prune) candidates."""
    total = None
    for m in model.modules():
        if isinstance(m, DynamicSparseLinear):
            t = (m.weight * m.mask).abs().sum()
        elif isinstance(m, ApproxDynamicSparseLinear):
            t = m.values.abs().sum()
        else:
            continue
        total = t if total is None else total + t
    return total


def l2_penalty(model: "nn.Module"):
    """Sum of squared active weights over sparse layers (for an L2 loss term)."""
    total = None
    for m in model.modules():
        if isinstance(m, DynamicSparseLinear):
            t = (m.weight * m.mask).pow(2).sum()
        elif isinstance(m, ApproxDynamicSparseLinear):
            t = m.values.pow(2).sum()
        else:
            continue
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


class DynamicSparseLinear(nn.Module):
    def __init__(self, in_features, out_features, k, *, init_scale=None, generator=None):
        super().__init__()
        n = in_features * out_features
        assert 1 <= k <= n
        self.in_features = in_features
        self.out_features = out_features
        self.k = k

        scale = init_scale if init_scale is not None else math.sqrt(out_features / k)
        flat_mask = torch.zeros(n)
        flat_mask[torch.randperm(n, generator=generator)[:k]] = 1.0
        mask = flat_mask.view(out_features, in_features)
        w = torch.randn(out_features, in_features, generator=generator) * scale * mask
        self.weight = nn.Parameter(w)
        self.register_buffer("mask", mask)

    def forward(self, x):
        return F.linear(x, self.weight)  # inactive entries are 0

    @torch.no_grad()
    def enforce_mask(self):
        """Re-pin inactive weights to 0 (Adam nudges them; keep them off)."""
        self.weight.mul_(self.mask)

    @torch.no_grad()
    def rigl_step(self, optimizer, drop_fraction: float) -> int:
        """Prune smallest-|w| active weights, grow largest-|grad| inactive ones.
        Uses the dense weight.grad from the most recent backward. Keeps k fixed
        and resets Adam moments at the swapped positions. Returns #swapped."""
        g = self.weight.grad
        if g is None:
            return 0
        mask_flat = self.mask.view(-1)
        w_flat = self.weight.data.view(-1)
        g_flat = g.reshape(-1)

        active = mask_flat.nonzero(as_tuple=False).view(-1)
        inactive = (mask_flat == 0).nonzero(as_tuple=False).view(-1)
        n_swap = max(1, int(round(drop_fraction * self.k)))
        n_swap = min(n_swap, active.numel(), inactive.numel())
        if n_swap == 0:
            return 0

        # prune the most "dead" active weights: small |w| AND stable (small
        # momentum |m| AND small gradient |g|). Never prune an actively-moving one.
        dead = _deadness(w_flat[active].abs(),
                         _momentum(optimizer, self.weight, active),
                         g_flat[active].abs())
        drop = active[torch.topk(dead, n_swap, largest=False).indices]
        grow = inactive[torch.topk(g_flat[inactive].abs(), n_swap, largest=True).indices]

        mask_flat[drop] = 0.0
        w_flat[drop] = 0.0
        mask_flat[grow] = 1.0
        w_flat[grow] = 0.0  # grow from zero -> output unchanged at the swap

        st = optimizer.state.get(self.weight, None)
        if st:
            for key in ("exp_avg", "exp_avg_sq"):
                if key in st:
                    b = st[key].view(-1)
                    b[drop] = 0.0
                    b[grow] = 0.0
        return n_swap


class ApproxDynamicSparseLinear(nn.Module):
    """Dynamic sparse layer that NEVER forms the n*n gradient.

    Stores only k weights (`values`) at positions (`rows`,`cols`) and runs the
    O(k) gather forward/backward. Grow-and-prune uses the POC approximation:
    the batch gradient gW = sum_t da_t (x) inp_t is large where an output row
    (error) AND an input column (activation) are both large, so we score only
    the top-p error rows x top-q input columns (a tiny p*q matmul) and grow the
    best inactive cell(s). Prune = smallest-|value| active weights. The layer's
    input `x` and its output-gradient `da` are captured (only on probe steps)
    via a tensor hook; no full gradient is ever materialized.
    """

    def __init__(self, in_features, out_features, k, *, init_scale=None, generator=None):
        super().__init__()
        from sparse_layer import SparseLinearFunction
        self._fn = SparseLinearFunction
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
        out = self._fn.apply(x, self.values, self.rows, self.cols, self.out_features)
        if self._capture and out.requires_grad:
            out.register_hook(self._save_da)
        return out

    def _save_da(self, g):
        self._da = g.detach()
        return None

    @torch.no_grad()
    def grow_prune(self, optimizer, n_swap: int, topk: int) -> int:
        if self._x is None or self._da is None or n_swap <= 0:
            return 0
        x, da = self._x, self._da            # (tokens, in), (tokens, out)
        in_f, out_f = self.in_features, self.out_features
        p, q = min(topk, out_f), min(topk, in_f)
        topR = torch.topk(da.pow(2).sum(0), p).indices          # top error rows (outputs)
        topC = torch.topk(x.pow(2).sum(0), q).indices           # top activation cols (inputs)
        scores = (da[:, topR].t() @ x[:, topC]).abs()           # (p, q) approx |gW| at candidates
        cand_flat = (topR.view(-1, 1) * in_f + topC.view(1, -1)).reshape(-1)
        cand_score = scores.reshape(-1)

        active_set = set((self.rows * in_f + self.cols).tolist())
        grow, seen = [], set()
        for ci in torch.argsort(cand_score, descending=True).tolist():
            f = int(cand_flat[ci])
            if f in active_set or f in seen:
                continue
            seen.add(f)
            grow.append(f)
            if len(grow) >= n_swap:
                break
        if not grow:
            return 0
        m = len(grow)
        # prune the most "dead" active weights: small |value| AND stable (small
        # momentum AND small gradient). grad at active cells is self.values.grad
        # (the O(k) sparse backward already computed it) — no full gradient.
        gval = self.values.grad.abs() if self.values.grad is not None \
            else torch.zeros_like(self.values)
        slots = torch.arange(self.k, device=self.values.device)
        dead = _deadness(self.values.abs(), _momentum(optimizer, self.values, slots), gval)
        prune_slots = torch.topk(dead, m, largest=False).indices
        grow_t = torch.tensor(grow, dtype=torch.long)
        self.rows[prune_slots] = grow_t // in_f
        self.cols[prune_slots] = grow_t % in_f
        self.values.data[prune_slots] = 0.0
        st = optimizer.state.get(self.values, None)
        if st:
            for key in ("exp_avg", "exp_avg_sq"):
                if key in st:
                    st[key][prune_slots] = 0.0
        self._x = self._da = None
        return m


# --------------------------------------------------------------------------- #
# training-loop helpers
# --------------------------------------------------------------------------- #
def set_capture(model: nn.Module, on: bool):
    for m in model.modules():
        if isinstance(m, ApproxDynamicSparseLinear):
            m._capture = on
            if not on:
                m._x = m._da = None


def approx_grow_prune(model: nn.Module, optimizer, n_swap: int, topk: int) -> int:
    total = 0
    for m in model.modules():
        if isinstance(m, ApproxDynamicSparseLinear):
            total += m.grow_prune(optimizer, n_swap, topk)
    return total

def enforce_masks(model: nn.Module):
    for m in model.modules():
        if isinstance(m, DynamicSparseLinear):
            m.enforce_mask()


def rigl_update(model: nn.Module, optimizer, drop_fraction: float) -> int:
    total = 0
    for m in model.modules():
        if isinstance(m, DynamicSparseLinear):
            total += m.rigl_step(optimizer, drop_fraction)
    return total


def cosine_drop_fraction(step: int, total_steps: int, alpha: float = 0.3,
                         t_end_frac: float = 0.8) -> float:
    """RigL schedule: start at alpha, cosine-decay to 0, stop growing at t_end."""
    t_end = total_steps * t_end_frac
    if step >= t_end:
        return 0.0
    return alpha * 0.5 * (1.0 + math.cos(math.pi * step / t_end))
