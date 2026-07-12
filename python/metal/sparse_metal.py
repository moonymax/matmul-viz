"""Fused Metal (MPS) sparse linear layer.

Drop-in for `sparse_layer.SparseLinear` but the forward/backward run in
hand-written Metal kernels (CSR forward, grad_val over k, CSC grad_x) instead of
gather + index_add. `values` is the trainable parameter and is stored in CSR
order (sorted by (row, col)).
"""

from __future__ import annotations

import os

import torch
from torch import Tensor, nn
from torch.utils.cpp_extension import load

_MODULE = None


def _ext():
    """Compile + cache the Metal extension."""
    global _MODULE
    if _MODULE is None:
        here = os.path.dirname(os.path.abspath(__file__))
        _MODULE = load(
            name="sparse_metal_ext",
            sources=[os.path.join(here, "sparse_metal.mm")],
            extra_cflags=["-std=c++20", "-ObjC++"],
            extra_ldflags=[
                "-framework", "Metal",
                "-framework", "Foundation",
                "-framework", "MetalPerformanceShaders",
            ],
            verbose=False,
        )
    return _MODULE


def build_index_structures(rows: Tensor, cols: Tensor, in_features: int, out_features: int):
    """From nonzero positions (rows, cols) build CSR + CSC index buffers.

    Returns a dict of int64 CPU tensors plus `csr_perm`, the permutation that
    sorts arbitrary-order (rows, cols) into CSR order. Callers reorder their
    `values` by `csr_perm` so that `values` lines up with the CSR structure.
    """
    rows = rows.long()
    cols = cols.long()
    k = rows.numel()

    # CSR order: sort by (row, col) via a single composite key.
    csr_key = rows * in_features + cols
    csr_perm = torch.argsort(csr_key, stable=True)
    csr_rowidx = rows[csr_perm]
    csr_col = cols[csr_perm]
    crow = torch.zeros(out_features + 1, dtype=torch.long)
    crow[1:] = torch.bincount(csr_rowidx, minlength=out_features).cumsum(0)

    # CSC order: sort the CSR-ordered nonzeros by (col, row). csc_valperm[s] is
    # the index into the CSR-ordered `values` of the s-th (col,row)-ordered nz.
    csc_key = csr_col * out_features + csr_rowidx
    csc_valperm = torch.argsort(csc_key, stable=True)
    csc_rowidx = csr_rowidx[csc_valperm]
    csc_col = csr_col[csc_valperm]
    ccol = torch.zeros(in_features + 1, dtype=torch.long)
    ccol[1:] = torch.bincount(csc_col, minlength=in_features).cumsum(0)

    return {
        "csr_perm": csr_perm,
        "crow": crow,
        "csr_col": csr_col,
        "csr_rowidx": csr_rowidx,
        "ccol": ccol,
        "csc_rowidx": csc_rowidx,
        "csc_valperm": csc_valperm,
        "k": k,
    }


# Toggles (mostly for benchmarking each optimization's contribution):
#   SPARSE_METAL_FUSED_BWD=0  -> two separate command-buffer commits in backward
#                                (isolates WIN 1: fused single-commit backward)
#   SPARSE_METAL_TG_GRADVAL=1 -> threadgroup-reduction grad_val kernel (WIN 2)
#
# WIN 2 default is OFF: with this layer's large k (thousands of nonzeros) the
# one-thread-per-nonzero serial grad_val already saturates the GPU, so the
# threadgroup tree-reduction only adds barrier overhead and is measurably slower.
# The tg kernel is kept for the small-k / very-large-batch regime where it wins.
def _fused_bwd() -> bool:
    return os.environ.get("SPARSE_METAL_FUSED_BWD", "1") != "0"


def _tg_gradval() -> bool:
    return os.environ.get("SPARSE_METAL_TG_GRADVAL", "0") != "0"


class SparseLinearMetalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, values, crow, csr_col, csr_rowidx,
                ccol, csc_rowidx, csc_valperm, in_features, out_features):
        ext = _ext()
        out = ext.sparse_forward(x, values, crow, csr_col, out_features)
        ctx.save_for_backward(x, values, crow, csr_col, csr_rowidx,
                              ccol, csc_rowidx, csc_valperm)
        ctx.in_features = in_features
        ctx.out_features = out_features
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x, values, crow, csr_col, csr_rowidx,
         ccol, csc_rowidx, csc_valperm) = ctx.saved_tensors
        ext = _ext()
        grad_out = grad_out.contiguous()
        need_gv = ctx.needs_input_grad[1]
        need_gx = ctx.needs_input_grad[0]
        tg = _tg_gradval()

        grad_x = grad_values = None
        if _fused_bwd():
            # WIN 1: both kernels in one command buffer, one commit, no wait.
            gv, gx = ext.sparse_backward(
                grad_out, x, values, csr_rowidx, csr_col,
                ccol, csc_rowidx, csc_valperm, need_gv, need_gx, tg,
            )
            if need_gv:
                grad_values = gv
            if need_gx:
                grad_x = gx
        else:
            if need_gv:
                grad_values = ext.sparse_grad_val(grad_out, x, csr_rowidx, csr_col, tg)
            if need_gx:
                grad_x = ext.sparse_grad_x(grad_out, values, ccol, csc_rowidx,
                                           csc_valperm, ctx.in_features)

        # line up with forward args
        return (grad_x, grad_values, None, None, None, None, None, None, None, None)


class SparseLinearMetal(nn.Module):
    """Fused-Metal sparse linear layer. Same constructor as SparseLinear."""

    def __init__(self, in_features, out_features, k, *, init_scale=0.1, generator=None):
        super().__init__()
        if k > in_features * out_features:
            raise ValueError("k exceeds the number of matrix cells")
        self.in_features = in_features
        self.out_features = out_features

        flat = torch.randperm(out_features * in_features, generator=generator)[:k]
        rows = (flat // in_features).long()
        cols = (flat % in_features).long()
        vals = torch.randn(k, generator=generator) * init_scale

        self._init_from_positions(rows, cols, vals)

    def _init_from_positions(self, rows, cols, vals):
        idx = build_index_structures(rows, cols, self.in_features, self.out_features)
        # values must be stored in CSR order (sorted by (row, col))
        self.values = nn.Parameter(vals[idx["csr_perm"]].contiguous())
        for name in ("crow", "csr_col", "csr_rowidx", "ccol", "csc_rowidx", "csc_valperm"):
            self.register_buffer(name, idx[name].to(torch.int32))

    @classmethod
    def from_positions(cls, in_features, out_features, rows, cols, values):
        """Build a layer from explicit positions/values (values in the given
        (rows, cols) order; they get reordered to CSR internally)."""
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.in_features = in_features
        obj.out_features = out_features
        obj._init_from_positions(rows.long(), cols.long(), values.detach().clone())
        return obj

    @property
    def k(self) -> int:
        return self.values.numel()

    def forward(self, x: Tensor) -> Tensor:
        return SparseLinearMetalFunction.apply(
            x, self.values, self.crow, self.csr_col, self.csr_rowidx,
            self.ccol, self.csc_rowidx, self.csc_valperm,
            self.in_features, self.out_features,
        )

    def to_dense(self) -> Tensor:
        W = self.values.new_zeros(self.out_features, self.in_features)
        W[self.csr_rowidx.long(), self.csr_col.long()] = self.values
        return W

    def extra_repr(self) -> str:
        n = self.in_features * self.out_features
        return f"in={self.in_features}, out={self.out_features}, k={self.k} ({100 * self.k / n:.2f}% dense)"
