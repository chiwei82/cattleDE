
import torch
import torch.nn.functional as F


def prior_loss(z, region, prior, prior_weight):
    active = prior_weight > 1e-6
    if not bool(active.any()):
        return z.new_zeros(()), 0.0

    b = z.shape[0]
    g = (prior * region).reshape(b, -1)
    mass = g.sum(dim=1).clamp(min=1e-6)
    inside = (z.reshape(b, -1) * g).sum(dim=1) / mass

    target = torch.ones_like(inside)
    per_sample = F.binary_cross_entropy_with_logits(inside, target, reduction="none")
    loss = (per_sample * prior_weight * active).sum() / active.sum().clamp(min=1)
    return loss, float(active.float().mean())


def contact_losses(z, region, pooled_logit, label, pos_weight=None,
                   lambda_sparsity=0.01, lambda_tv=0.3,
                   prior=None, prior_weight=None, lambda_prior=0.0):
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
    parts = {
        "loss": 0.0,
        "mil": float(mil.detach()),
        "sparsity": float(sparsity.detach()),
        "tv": float(tv.detach()),
        "prior": 0.0,
        "prior_frac": 0.0,
    }
    if lambda_prior > 0.0 and prior is not None and prior_weight is not None:
        pl, frac = prior_loss(z, region, prior, prior_weight)
        total = total + lambda_prior * pl
        parts["prior"] = float(pl.detach()) if torch.is_tensor(pl) else float(pl)
        parts["prior_frac"] = frac

    parts["loss"] = float(total.detach())
    return total, parts


def anneal_tau(epoch, epochs, tau_start, tau_end):
    if epochs <= 1:
        return tau_end
    return tau_start + (tau_end - tau_start) * epoch / (epochs - 1)
