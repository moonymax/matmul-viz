"""A sparse linear layer with explicit forward and backward.

The weight matrix is logically `out_features x in_features` but only `k` cells
are stored/trained. We keep three 1-D tensors:

    values : (k,)   the trainable weights          -> nn.Parameter
    rows   : (k,)   output index of each weight     -> buffer (long)
    cols   : (k,)   input index of each weight       -> buffer (long)

Forward  :  out[b, i] = sum_{t : rows[t] == i}  values[t] * x[b, cols[t]]
Backward :  dvalues[t] = sum_b grad_out[b, rows[t]] * x[b, cols[t]]
            dx[b, j]   = sum_{t : cols[t] == j}  values[t] * grad_out[b, rows[t]]

Everything is O(B * k) in time and O(B * n + k) in memory: the dense n x n
matrix is never materialized. The forward/backward are written by hand (rather
than leaning on autograd over the gather/scatter) so this is a drop-in place to
later swap in a fused gather-mul-scatter kernel (Triton / CUDA / Metal).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SparseLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,          # (B, in_features)
        values: Tensor,     # (k,)
        rows: Tensor,       # (k,) long
        cols: Tensor,       # (k,) long
        out_features: int,
    ) -> Tensor:
        # gather the input coordinate feeding each active weight, scale, scatter
        contrib = values.unsqueeze(0) * x.index_select(1, cols)   # (B, k)
        out = x.new_zeros(x.shape[0], out_features)
        out.index_add_(1, rows, contrib)                          # (B, out_features)
        ctx.save_for_backward(x, values, rows, cols)
        ctx.in_features = x.shape[1]
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x, values, rows, cols = ctx.saved_tensors
        go_rows = grad_out.index_select(1, rows)                  # (B, k)

        grad_values = grad_x = None
        if ctx.needs_input_grad[1]:
            grad_values = (go_rows * x.index_select(1, cols)).sum(0)   # (k,)
        if ctx.needs_input_grad[0]:
            grad_x = x.new_zeros(x.shape[0], ctx.in_features)
            grad_x.index_add_(1, cols, values.unsqueeze(0) * go_rows)  # (B, in_features)

        # grads line up with forward args: x, values, rows, cols, out_features
        return grad_x, grad_values, None, None, None


class SparseLinear(nn.Module):
    """Linear layer whose weight has only `k` nonzero entries (fixed positions)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k: int,
        *,
        init_scale: float = 0.1,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        if k > in_features * out_features:
            raise ValueError("k exceeds the number of matrix cells")
        self.in_features = in_features
        self.out_features = out_features

        flat = torch.randperm(out_features * in_features, generator=generator)[:k]
        # positions are fixed structure -> buffers (a grow/prune controller could
        # rewrite these between steps without touching the autograd graph)
        self.register_buffer("rows", (flat // in_features).long())
        self.register_buffer("cols", (flat % in_features).long())
        self.values = nn.Parameter(torch.randn(k, generator=generator) * init_scale)

    @property
    def k(self) -> int:
        return self.values.numel()

    def forward(self, x: Tensor) -> Tensor:
        return SparseLinearFunction.apply(x, self.values, self.rows, self.cols, self.out_features)

    def to_dense(self) -> Tensor:
        W = self.values.new_zeros(self.out_features, self.in_features)
        W[self.rows, self.cols] = self.values
        return W

    def extra_repr(self) -> str:
        n = self.in_features * self.out_features
        return f"in={self.in_features}, out={self.out_features}, k={self.k} ({100 * self.k / n:.2f}% dense)"


# --------------------------------------------------------------------------- #
# self-test: gradcheck + agreement with a dense reference
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)

    in_f, out_f, k, B = 6, 5, 7, 4

    # ---- gradcheck (double precision, CPU) against the hand-written backward ----
    g = torch.Generator().manual_seed(1)
    flat = torch.randperm(in_f * out_f, generator=g)[:k]
    rows = (flat // in_f).long()
    cols = (flat % in_f).long()
    x = torch.randn(B, in_f, dtype=torch.double, requires_grad=True)
    values = torch.randn(k, dtype=torch.double, requires_grad=True)

    ok = torch.autograd.gradcheck(
        lambda xx, vv: SparseLinearFunction.apply(xx, vv, rows, cols, out_f),
        (x, values),
    )
    print("gradcheck passed:", ok)

    # ---- forward + backward agree with a dense matmul reference ----
    # sparse path
    v_ref = values.detach().clone().requires_grad_(True)
    x_s = x.detach().clone().requires_grad_(True)
    out_sparse = SparseLinearFunction.apply(x_s, v_ref, rows, cols, out_f)
    out_sparse.pow(2).sum().backward()

    # dense reference sharing the same weights
    W = torch.zeros(out_f, in_f, dtype=torch.double)
    W[rows, cols] = values.detach()
    W.requires_grad_(True)
    x_d = x.detach().clone().requires_grad_(True)
    out_dense = x_d @ W.t()
    out_dense.pow(2).sum().backward()

    print("forward max abs diff vs dense:", (out_sparse - out_dense).abs().max().item())
    print("dvalues max abs diff:", (v_ref.grad - W.grad[rows, cols]).abs().max().item())
    print("dx      max abs diff:", (x_s.grad - x_d.grad).abs().max().item())

    # ---- module smoke test on the available device ----
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    layer = SparseLinear(64, 64, k=64, generator=torch.Generator().manual_seed(2)).to(device)
    print("layer:", layer)
    xb = torch.randn(8, 64, device=device)
    yb = layer(xb)
    yb.pow(2).sum().backward()
    print(f"module forward {tuple(xb.shape)} -> {tuple(yb.shape)} on {device}; "
          f"grad nonzero: {int((layer.values.grad != 0).sum())}/{layer.k}")
