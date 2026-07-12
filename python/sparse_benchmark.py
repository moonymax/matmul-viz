"""Approximate a big randomly-initialized dense teacher with three student
architectures, and benchmark wall-clock training time.

Task (same as the Next.js demo, scaled up):
  teacher(x) = tanh(W2t @ tanh(W1t @ x)),   W1t, W2t : n x n  random, frozen
  students learn W1, W2 to match the teacher on random inputs.

Students (all 2-layer n->n with tanh between and at the output):
  - dense    : nn.Linear                     (n^2 params / layer)
  - sparse   : SparseLinear (our O(k) layer)  (k  params / layer)
  - lowrank  : W = A @ B, rank r              (2nr params / layer)

For each size n we report: parameter count, final MSE (did it train?), and the
wall-clock time to run a fixed number of Adam steps (fwd + bwd + step).
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparse_layer import SparseLinear

# ---------------- config ----------------
import os

DIMS = [256, 512, 1024]
RANK = 16              # low-rank rank; sparse is param-matched to it (k = 2*n*r)
BATCH = 128
N_SAMPLES = 4096
WARMUP = 15            # untimed warmup steps (exclude first-call / alloc overhead)
MAX_STEPS = 1500       # give up (DNF) after this many steps
EVAL_EVERY = 25        # check full-dataset loss this often (eval time not counted)
THRESHOLD = float(os.environ.get("BENCH_THRESHOLD", "0.1"))  # target full MSE
LR = 1e-3
SEED = 0


def pick_device() -> str:
    import os
    forced = os.environ.get("BENCH_DEVICE")
    if forced:
        return forced
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sync(device: str):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


# ---------------- layers / models ----------------
class LowRankLinear(nn.Module):
    def __init__(self, in_f: int, out_f: int, r: int):
        super().__init__()
        s = (1.0 / (in_f * r)) ** 0.25  # so W = A@B has ~1/sqrt(in) scale
        self.A = nn.Parameter(torch.randn(out_f, r) * s)
        self.B = nn.Parameter(torch.randn(r, in_f) * s)

    def forward(self, x):
        return (x @ self.B.t()) @ self.A.t()


class TwoLayer(nn.Module):
    def __init__(self, l1: nn.Module, l2: nn.Module):
        super().__init__()
        self.l1, self.l2 = l1, l2

    def forward(self, x):
        return torch.tanh(self.l2(torch.tanh(self.l1(x))))


def make_student(arch: str, n: int, r: int, gen: torch.Generator) -> nn.Module:
    if arch == "dense":
        return TwoLayer(nn.Linear(n, n, bias=False), nn.Linear(n, n, bias=False))
    if arch == "lowrank":
        return TwoLayer(LowRankLinear(n, n, r), LowRankLinear(n, n, r))
    if arch == "sparse":
        k = 2 * n * r  # param-match the low-rank model
        scale = math.sqrt(n / k)  # ~unit output variance for k/n active per row
        return TwoLayer(
            SparseLinear(n, n, k, init_scale=scale, generator=gen),
            SparseLinear(n, n, k, init_scale=scale, generator=gen),
        )
    raise ValueError(arch)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def make_teacher(n: int, device: str) -> nn.Module:
    g = torch.Generator().manual_seed(SEED + 999)
    s = 1.0 / math.sqrt(n)
    l1 = nn.Linear(n, n, bias=False)
    l2 = nn.Linear(n, n, bias=False)
    with torch.no_grad():
        l1.weight.copy_(torch.randn(n, n, generator=g) * s)
        l2.weight.copy_(torch.randn(n, n, generator=g) * s)
    t = TwoLayer(l1, l2).to(device)
    for p in t.parameters():
        p.requires_grad_(False)
    return t


def train_to_threshold(model: nn.Module, X: torch.Tensor, Y: torch.Tensor, device: str):
    """Speed run: train until full-dataset MSE < THRESHOLD; return the accumulated
    *training* wall-clock (eval time excluded) and step count to get there."""
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    N = X.shape[0]

    def one_step(idx):
        xb, yb = X[idx], Y[idx]
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(xb), yb)
        loss.backward()
        opt.step()
        return loss

    @torch.no_grad()
    def full_loss():
        return F.mse_loss(model(X), Y).item()

    g = torch.Generator().manual_seed(SEED + 7)
    for _ in range(WARMUP):
        one_step(torch.randint(0, N, (BATCH,), generator=g))
    sync(device)

    train_time = 0.0
    reached_step = None
    best = float("inf")
    step = 0
    while step < MAX_STEPS:
        # time a block of EVAL_EVERY steps with a single device sync at each end
        sync(device)
        t0 = time.perf_counter()
        for _ in range(EVAL_EVERY):
            one_step(torch.randint(0, N, (BATCH,), generator=g))
            step += 1
        sync(device)
        train_time += time.perf_counter() - t0

        fl = full_loss()
        best = min(best, fl)
        if fl < THRESHOLD:
            reached_step = step
            break

    return reached_step, train_time, best


def main():
    torch.manual_seed(SEED)
    device = pick_device()
    print(f"device: {device}  |  SPEED RUN to full-MSE < {THRESHOLD}  |  rank={RANK}, "
          f"batch={BATCH}, max_steps={MAX_STEPS}, samples={N_SAMPLES}\n")

    header = (f"{'dim':>5} {'arch':>8} {'params':>12} {'density':>9} "
              f"{'reached':>9} {'steps':>7} {'train s':>9} {'ms/step':>9} {'best MSE':>10}")
    print(header)
    print("-" * len(header))

    for n in DIMS:
        teacher = make_teacher(n, device)
        gx = torch.Generator().manual_seed(SEED + 1)
        X = torch.randn(N_SAMPLES, n, generator=gx).to(device)
        with torch.no_grad():
            Y = teacher(X).detach()

        for arch in ("dense", "sparse", "lowrank"):
            gen = torch.Generator().manual_seed(SEED + 2)
            model = make_student(arch, n, RANK, gen).to(device)
            params = n_params(model)
            density = params / (2 * n * n)  # fraction of the two dense n×n layers
            step, train_time, best = train_to_threshold(model, X, Y, device)
            steps_done = step if step is not None else MAX_STEPS
            reached = "yes" if step is not None else "DNF"
            print(f"{n:>5} {arch:>8} {params:>12,} {density:>8.2%} "
                  f"{reached:>9} {steps_done:>7} {train_time:>9.3f} "
                  f"{1000 * train_time / steps_done:>9.3f} {best:>10.5f}")
        print()


if __name__ == "__main__":
    main()
