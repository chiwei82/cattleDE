"""Losses for region-restricted multiple-instance contact learning.

Only the pair label is supervised. Everything that shapes the heatmap is either
structural (the region mask, applied in the model) or a prior expressed here:

  L_mil  BCE on the pooled logit. For a negative pair, pushing the pooled logit
         down pushes EVERY pixel in R down, because LSE upper-bounds the max —
         so negatives supervise the heatmap densely with no extra term. That is
         why hard negatives (cows overlapping but not interacting, which the
         IoU>0.1 pair filter guarantees) carry most of the localisation signal.
  L_sp   area-normalised L1. Without it the heatmap saturates over all of R,
         since the pooling is indifferent to how much of R is lit.
  L_tv   total variation, so the response forms one connected blob rather than
         scattered high-frequency pixels.

Both priors are divided by |R| so their magnitude does not track the size of the
candidate region.
"""

import torch
import torch.nn.functional as F


def contact_losses(z, region, pooled_logit, label, pos_weight=None,
                   lambda_sparsity=0.01, lambda_tv=0.3):
    """Total loss plus a dict of the detached components for logging."""
    if pos_weight is not None and not torch.is_tensor(pos_weight):
        pos_weight = torch.tensor(float(pos_weight), device=z.device)

    mil = F.binary_cross_entropy_with_logits(pooled_logit, label, pos_weight=pos_weight)

    b = z.shape[0]
    heat = torch.sigmoid(z) * region
    area = region.reshape(b, -1).sum(dim=1).clamp(min=1.0)

    sparsity = (heat.reshape(b, -1).sum(dim=1) / area).mean()

    dx = (heat[..., :, 1:] - heat[..., :, :-1]).abs().reshape(b, -1).sum(dim=1)
    dy = (heat[..., 1:, :] - heat[..., :-1, :]).abs().reshape(b, -1).sum(dim=1)
    tv = ((dx + dy) / area).mean()

    total = mil + lambda_sparsity * sparsity + lambda_tv * tv
    return total, {
        "loss": float(total.detach()),
        "mil": float(mil.detach()),
        "sparsity": float(sparsity.detach()),
        "tv": float(tv.detach()),
    }


def anneal_tau(epoch, epochs, tau_start, tau_end):
    """Linear temperature schedule for the MIL pooling (mean-like -> max-like)."""
    if epochs <= 1:
        return tau_end
    return tau_start + (tau_end - tau_start) * epoch / (epochs - 1)
