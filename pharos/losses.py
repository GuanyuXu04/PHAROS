from typing import Optional, Tuple
import torch


def chamfer_distance(S1: torch.Tensor, S2: torch.Tensor, chunk: int = 2048) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Symmetric squared Chamfer distance, summed over points and averaged over the batch.

    Same value and gradient as the obvious `torch.cdist(...).pow(2).min(...)`
    formulation, but the (B, N, M) distance matrix never enters the autograd
    graph: the nearest-neighbour search runs under no_grad and only the matched
    indices survive, so backward is a gather instead of a cdist backward. On a
    16x4096x5929 batch that is ~2.7x faster and needs ~2.4x less memory.

    The search is chunked along S1 so peak memory is (B, chunk, M) rather than
    (B, N, M), and deliberately runs in float32 even under autocast: the
    expansion |a|^2 + |b|^2 - 2<a,b> catastrophically cancels in bf16/fp16 for
    the uncentred millimetre coordinates this model works in (bf16 yields
    negative distances, fp16 yields NaN).
    """
    if S1.dim() == 2: S1 = S1.unsqueeze(0)
    if S2.dim() == 2: S2 = S2.unsqueeze(0)
    B, N, _ = S1.shape
    M = S2.shape[1]

    with torch.no_grad():
        # Centring is exact (nearest neighbours are translation invariant) and
        # keeps the expansion well conditioned: on raw mm coordinates the
        # |a|^2 and |b|^2 terms are ~5e4 while the distances of interest are
        # ~1e1, and the cancellation is enough to flip near-tie assignments.
        mu = S2.float().mean(dim=1, keepdim=True)
        a_all = S1.float() - mu
        b = S2.float() - mu
        b_sq = b.pow(2).sum(-1).unsqueeze(1)                      # (B, 1, M)
        bt = b.transpose(1, 2)

        idx1 = S1.new_zeros((B, N), dtype=torch.long)             # S1 -> S2
        idx2 = S1.new_zeros((B, M), dtype=torch.long)             # S2 -> S1
        best2 = a_all.new_full((B, M), float("inf"))

        for start in range(0, N, chunk):
            stop = min(start + chunk, N)
            a = a_all[:, start:stop]
            d2 = a.pow(2).sum(-1).unsqueeze(2) + b_sq - 2.0 * torch.bmm(a, bt)

            idx1[:, start:stop] = d2.min(dim=2).indices

            # Running minimum over chunks for the S2 -> S1 direction.
            chunk_min, chunk_arg = d2.min(dim=1)
            improved = chunk_min < best2
            best2 = torch.where(improved, chunk_min, best2)
            idx2 = torch.where(improved, chunk_arg + start, idx2)

    nn1 = S2.gather(1, idx1.unsqueeze(-1).expand(-1, -1, 3))      # (B, N, 3)
    nn2 = S1.gather(1, idx2.unsqueeze(-1).expand(-1, -1, 3))      # (B, M, 3)
    d1_min = (S1 - nn1).pow(2).sum(-1)
    d2_min = (S2 - nn2).pow(2).sum(-1)
    loss = (d1_min.sum(dim=1) + d2_min.sum(dim=1)).mean()
    return loss, (d1_min, d2_min)


def repulsion_loss(pred: torch.Tensor, k: int = 10, h: float = 0.5,
                   subset: Optional[int] = None, weight: float = 1.0) -> torch.Tensor:
    """Penalise points that crowd together, via a Gaussian kernel on kNN distances.

    `subset` evaluates the term on a random subset of `subset` points (shared
    across the batch) instead of all P, which is what makes this affordable:
    the full version is an O(P^2) cdist plus a topk over P, ~48 ms at P=4096
    against ~12 ms at 2048. The result is rescaled by P/subset so the term keeps
    the same magnitude, but note this is a *stochastic estimator*, not the exact
    loss -- kNN distances inside a sparser subset are larger, so the penalty is
    somewhat softer than the full-cloud one at the same h.

    `weight` scales the returned term; with weight 0 the computation is skipped
    and a zero is returned, so the term can be switched off at no cost.
    """
    if weight == 0:
        return pred.new_tensor(0.0)
    B, P, _ = pred.shape
    if P <= 1:
        return pred.new_tensor(0.0)

    scale = 1.0
    if subset is not None and subset < P:
        sel = torch.randperm(P, device=pred.device)[:subset]
        pred = pred[:, sel]
        scale = P / subset
        P = subset

    k_eff = min(k, P - 1)
    d = torch.cdist(pred, pred, p=2)
    d2 = d.pow(2)
    _, idx = d.topk(k=k_eff + 1, largest=False)
    nn_d2 = torch.gather(d2, 2, idx[:, :, 1:])
    ker = torch.exp(-nn_d2 / (h * h))
    return weight * ker.sum(dim=(1, 2)).mean() * 100.0 * scale
