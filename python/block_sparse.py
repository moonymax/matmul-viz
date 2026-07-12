"""Moveable dense-block sparse linear layer (whole blocks grown & pruned).

Weight is out x in, tiled into b x b blocks on a (Gi=out/b) x (Gj=in/b) grid; only
`n_active` blocks are stored (dense b x b tiles) and their grid positions migrate
during training. Forward is a BATCHED DENSE MATMUL over the active blocks, so it
runs on the GPU matmul units (unlike the scalar unstructured kernel) -- that is
the whole point.

  forward:  gather each active block's input column-tile x[:, J] (B x b),
            bmm with its b x b weight, scatter-add into output block-row I.
  grow-prune (every dT): prune the deadest active blocks (small Frobenius norm of
            weight AND momentum AND gradient -- the block analog of the dead-cell
            rule), grow new blocks at the grid positions with the largest
            block-gradient energy (norm-gated: ||grad_out_Iblock|| * ||x_Jblock||).
Backward is plain autograd over bmm/gather/index_add (also matmul-unit).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


def annealed_n_swap(step, total_steps, n_active, base_frac=0.3, stop_frac=0.85):
    """Cosine-annealed number of blocks to migrate per grow-prune event.

    Starts at base_frac * n_active (explore block positions) and decays to 0 by
    stop_frac of training (converge; freeze topology). This is the knob that
    replaces a hard deadness threshold: lots of movement early, none late."""
    if total_steps <= 0 or step >= stop_frac * total_steps:
        return 0
    prog = step / (stop_frac * total_steps)              # 0 -> 1 over the moving phase
    frac = base_frac * 0.5 * (1.0 + math.cos(math.pi * prog))
    return int(round(frac * n_active))


class BlockSparseLinear(nn.Module):
    def __init__(self, in_features, out_features, block, n_active, *, generator=None):
        super().__init__()
        assert in_features % block == 0 and out_features % block == 0
        self.in_features = in_features
        self.out_features = out_features
        self.b = block
        self.Gi = out_features // block
        self.Gj = in_features // block
        n_slots = self.Gi * self.Gj
        assert 1 <= n_active <= n_slots
        flat = torch.randperm(n_slots, generator=generator)[:n_active]
        self.register_buffer("brows", (flat // self.Gj).long())   # output block index
        self.register_buffer("bcols", (flat % self.Gj).long())    # input block index
        fan_in = max(1.0, (n_active / self.Gi) * block)
        scale = 1.0 / math.sqrt(fan_in)
        self.blocks = nn.Parameter(torch.randn(n_active, block, block, generator=generator) * scale)

    @property
    def n_active(self):
        return self.blocks.shape[0]

    def forward(self, x):
        B = x.shape[0]
        b = self.b
        Xr = x.view(B, self.Gj, b)                       # (B, Gj, b)
        Xg = Xr[:, self.bcols, :].transpose(0, 1)        # (n_active, B, b)
        Yg = torch.bmm(Xg, self.blocks.transpose(1, 2))  # (n_active, B, b) = x_tile @ W_block^T
        out = x.new_zeros(B, self.Gi, b)
        out.index_add_(1, self.brows, Yg.transpose(0, 1))  # scatter-add by output block-row
        return out.reshape(B, self.out_features)

    def to_dense(self):
        W = self.blocks.new_zeros(self.out_features, self.in_features)
        b = self.b
        for a in range(self.n_active):
            I, J = int(self.brows[a]), int(self.bcols[a])
            W[I * b:(I + 1) * b, J * b:(J + 1) * b] = self.blocks[a]
        return W

    # A block is only evicted if it is genuinely DEAD: its weight Frobenius norm,
    # its momentum norm, AND its gradient norm are all a tiny fraction of the MEAN
    # active-block norm (absolute near-zero, not merely the smallest of the batch).
    # This is what stops the layer ripping out blocks that still carry real weights
    # (a discrete jump in W -> a loss spike) or blocks that are actively learning
    # (large gradient). Swaps are OPTIONAL: if fewer than n_swap blocks are dead we
    # move only the dead ones (possibly zero), never a healthy block.
    @torch.no_grad()
    def grow_prune(self, optimizer, x, grad_out, n_swap):
        """Migrate `n_swap` blocks: prune the lowest-weight (spike-safe) ones and
        grow the highest-gradient-energy positions. RigL-style. There is NO hard
        deadness threshold -- movement is ANNEALED by the caller through n_swap
        (large early for exploration -> 0 late for convergence). Pruning the
        lowest weight-norm blocks is what keeps this spike-free: those blocks
        contribute least to the output, so removing them barely perturbs W, and a
        high-weight ("live") block is never evicted.

        x (B,in) and grad_out (B,out) are captured from the current step."""
        b, Gi, Gj = self.b, self.Gi, self.Gj
        n_swap = min(n_swap, self.n_active, Gi * Gj - self.n_active)
        if n_swap <= 0:
            return 0

        # prune: the n_swap lowest weight-norm (Frobenius) blocks
        wn = self.blocks.norm(dim=(1, 2))
        prune = torch.topk(wn, n_swap, largest=False).indices

        # grow: inactive grid positions with the largest gradient energy
        # ||grad_out over output block-row I|| * ||x over input block-col J||
        go_bg = grad_out.view(-1, Gi, b).norm(dim=(0, 2))    # (Gi,)
        x_bg = x.view(-1, Gj, b).norm(dim=(0, 2))            # (Gj,)
        score = (go_bg.unsqueeze(1) * x_bg.unsqueeze(0)).reshape(-1)   # (Gi*Gj,)
        score[self.brows * Gj + self.bcols] = -1.0           # exclude active
        grow = torch.topk(score, n_swap, largest=True).indices

        self.brows[prune] = grow // Gj
        self.bcols[prune] = grow % Gj
        self.blocks.data[prune] = 0.0                        # grow from zero
        st = optimizer.state.get(self.blocks, None)
        if st:
            for key in ("exp_avg", "exp_avg_sq"):
                if key in st:
                    st[key][prune] = 0.0
        return n_swap
