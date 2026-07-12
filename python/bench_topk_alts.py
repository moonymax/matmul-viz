"""Is torch.topk the bottleneck we think, and can a threshold beat it?

Selecting ~K largest |values| from a p x D strip (the rowexact bottleneck).
Compare: sorted topk, unsorted topk, and threshold-select (one kthvalue to find a
cutoff, then boolean mask -> ~K indices, no full sort). For grow we only need the
~K largest cells, not them sorted, and approximate K is fine.
"""
from __future__ import annotations

import time
import torch

D = 2048
P = 96
K = 9216
WARMUP, REPS = 30, 200
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def sync():
    if DEV.type == "mps":
        torch.mps.synchronize()


def timed(fn, v):
    for _ in range(WARMUP):
        fn(v)
    sync(); t0 = time.perf_counter()
    for _ in range(REPS):
        r = fn(v)
    sync()
    return (time.perf_counter() - t0) / REPS, r


def main():
    torch.manual_seed(0)
    strip = torch.randn(P, D, device=DEV).abs()
    v = strip.reshape(-1)
    n = v.numel()
    print(f"device={DEV}  strip={P}x{D}={n}  K={K}  reps={REPS}\n")

    t_sorted, idx_true = timed(lambda v: torch.topk(v, K, sorted=True).indices, v)
    t_unsorted, _ = timed(lambda v: torch.topk(v, K, sorted=False).indices, v)

    # threshold-select: find the K-th largest value as a cutoff, then mask.
    def thresh(v):
        tau = torch.kthvalue(v, n - K + 1).values      # K-th largest
        return (v >= tau).nonzero(as_tuple=False).view(-1)

    t_thresh, idx_thr = timed(thresh, v)

    # approximate threshold from a quantile on a subsample (cheap cutoff estimate)
    def thresh_approx(v):
        tau = torch.quantile(v[::8].float(), 1.0 - K / n)   # cutoff from 1/8 sample
        return (v >= tau).nonzero(as_tuple=False).view(-1)

    t_ta, idx_ta = timed(thresh_approx, v)

    true_set = set(idx_true.tolist())
    ov_thr = len(true_set & set(idx_thr.tolist())) / K
    ov_ta = len(true_set & set(idx_ta.tolist())) / len(true_set)
    print(f"  {'method':>20} {'time(us)':>10} {'count':>8} {'overlap@K':>10}")
    print(f"  {'topk sorted':>20} {1e6*t_sorted:>10.1f} {K:>8} {1.0:>10.3f}")
    print(f"  {'topk unsorted':>20} {1e6*t_unsorted:>10.1f} {K:>8} {'-':>10}")
    print(f"  {'threshold (kthvalue)':>20} {1e6*t_thresh:>10.1f} {idx_thr.numel():>8} {ov_thr:>10.3f}")
    print(f"  {'threshold (approx)':>20} {1e6*t_ta:>10.1f} {idx_ta.numel():>8} {ov_ta:>10.3f}")


if __name__ == "__main__":
    main()
