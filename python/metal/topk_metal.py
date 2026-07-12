"""Fast approximate top-K selection on MPS (histogram + threshold compaction).

approx_topk(v, K) returns the flat indices of ~K largest-|v| elements (order
arbitrary, count approximate to within one histogram bin). Built for the rowexact
grow-gradient selection, where torch.topk over ~200K elements is the bottleneck.
"""
from __future__ import annotations

import os
import torch
from torch.utils.cpp_extension import load

_MODULE = None


def _ext():
    global _MODULE
    if _MODULE is None:
        here = os.path.dirname(os.path.abspath(__file__))
        _MODULE = load(
            name="topk_metal_ext",
            sources=[os.path.join(here, "topk_metal.mm")],
            extra_cflags=["-std=c++20", "-ObjC++"],
            extra_ldflags=["-framework", "Metal", "-framework", "Foundation",
                           "-framework", "MetalPerformanceShaders"],
            verbose=False,
        )
    return _MODULE


def approx_topk(v: torch.Tensor, K: int) -> torch.Tensor:
    """Flat indices of ~K largest-|v| elements of an MPS float tensor (unsorted,
    count approximate to within one magnitude bin). One fused Metal command
    buffer, no host round-trips."""
    n = v.numel()
    if K >= n:
        return torch.arange(n, device=v.device, dtype=torch.int32)
    return _ext().approx_topk(v.reshape(-1).contiguous(), int(K))
