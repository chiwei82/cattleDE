
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def keypoint_pool(z, kp_xy, kp_valid, radius, topk):
    b, _, h, w = z.shape
    kernel = 2 * int(radius) + 1
    smoothed = F.avg_pool2d(z, kernel_size=kernel, stride=1, padding=int(radius))

    gx = kp_xy[..., 0] / max(w - 1, 1) * 2.0 - 1.0
    gy = kp_xy[..., 1] / max(h - 1, 1) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)
    sampled = F.grid_sample(smoothed, grid, mode="bilinear",
                            padding_mode="border", align_corners=True)
    scores = sampled[:, 0, :, 0]

    neg_inf = torch.finfo(scores.dtype).min / 4
    masked = scores.masked_fill(~kp_valid, neg_inf)
    n_valid = kp_valid.sum(dim=1)

    k = torch.clamp(torch.minimum(n_valid, torch.full_like(n_valid, int(topk))), min=1)
    k_max = int(k.max().item())
    values, _ = masked.topk(k_max, dim=1)
    keep = torch.arange(k_max, device=z.device)[None, :] < k[:, None]
    values = values.masked_fill(~keep, 0.0)
    pooled = values.sum(dim=1) / k.to(values.dtype)

    empty = n_valid == 0
    if bool(empty.any()):
        fallback = z.reshape(b, -1).max(dim=1).values
        pooled = torch.where(empty, fallback, pooled)
    return pooled, scores


def masked_topk_mean(z, region, frac):
    b = z.shape[0]
    z_flat = z.reshape(b, -1)
    r_flat = region.reshape(b, -1)

    n = r_flat.sum(dim=1)
    k = torch.clamp(torch.ceil(n * frac), min=1.0).long()
    masked = z_flat.masked_fill(r_flat < 0.5, float("-inf"))

    k_max = int(k.max().item())
    values, _ = masked.topk(k_max, dim=1)
    keep = torch.arange(k_max, device=z.device)[None, :] < k[:, None]
    values = values.masked_fill(~keep, 0.0)
    return values.sum(dim=1) / k.to(values.dtype)


def masked_lse(z, region, tau):
    b = z.shape[0]
    z_flat = z.reshape(b, -1)
    r_flat = region.reshape(b, -1)
    count = r_flat.sum(dim=1).clamp(min=1.0)
    scaled = (z_flat * tau).masked_fill(r_flat < 0.5, float("-inf"))
    return (torch.logsumexp(scaled, dim=1) - torch.log(count)) / tau


def _widen_patch_embed(vit, in_chans):
    old = vit.patch_embed.proj
    new = nn.Conv2d(in_chans, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding,
                    bias=old.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :old.in_channels] = old.weight
        if old.bias is not None:
            new.bias.copy_(old.bias)
    vit.patch_embed.proj = new
    vit.patch_embed.in_chans = in_chans
    print(f"[model] patch embedding widened 3 -> {in_chans} channels "
          "(new planes zero-initialised)")


def _unfreeze_widened_embed(vit, in_chans):
    if in_chans != 3:
        vit.patch_embed.proj.requires_grad_(True)


class _TimmViTFeatures(nn.Module):

    def __init__(self, name, image_size, freeze_blocks, in_chans=3):
        super().__init__()
        import timm

        self.vit = timm.create_model(name, pretrained=True, num_classes=0)
        if in_chans != 3:
            _widen_patch_embed(self.vit, in_chans)
        self.patch = self.vit.patch_embed.patch_size[0]
        self.grid = image_size // self.patch
        self.dim = self.vit.embed_dim
        self.num_prefix = getattr(self.vit, "num_prefix_tokens", 1)

        self.vit.patch_embed.requires_grad_(False)
        for block in self.vit.blocks[:freeze_blocks]:
            block.requires_grad_(False)
        _unfreeze_widened_embed(self.vit, in_chans)

    def load_action_encoder(self, ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        state = state.get("state_dict", state)
        own = self.vit.state_dict()
        copied = {}
        for key, value in state.items():
            for prefix in ("backbone_vit.", "backbone.", "model.", "encoder."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    break
            if key in own and own[key].shape == value.shape:
                copied[key] = value
        self.vit.load_state_dict(copied, strict=False)
        print(f"[model] warm-started {len(copied)}/{len(own)} ViT tensors from {ckpt_path}")

    def forward(self, x):
        tokens = self.vit.forward_features(x)[:, self.num_prefix:]
        b, n, d = tokens.shape
        return tokens.transpose(1, 2).reshape(b, d, self.grid, self.grid)


class _DINOv2Features(nn.Module):

    def __init__(self, name, image_size, freeze_blocks, in_chans=3):
        super().__init__()
        self.vit = torch.hub.load("facebookresearch/dinov2", name)
        if in_chans != 3:
            _widen_patch_embed(self.vit, in_chans)
        self.patch = 14
        self.grid = image_size // self.patch
        self.dim = self.vit.embed_dim

        self.vit.patch_embed.requires_grad_(False)
        for block in self.vit.blocks[:freeze_blocks]:
            block.requires_grad_(False)
        _unfreeze_widened_embed(self.vit, in_chans)

    def forward(self, x):
        tokens = self.vit.forward_features(x)["x_norm_patchtokens"]
        b, n, d = tokens.shape
        return tokens.transpose(1, 2).reshape(b, d, self.grid, self.grid)


class ContactMIL(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        mcfg = cfg["model"]
        image_size = int(mcfg["image_size"])
        freeze_blocks = int(mcfg["freeze_blocks"])
        backbone = str(mcfg["backbone"]).lower()

        in_chans = 6 if cfg["data"].get("mask_channels", False) else 3

        if backbone == "timm":
            self.features = _TimmViTFeatures(mcfg["timm_name"], image_size,
                                             freeze_blocks, in_chans)
            ckpt = mcfg.get("action_ckpt")
            if ckpt:
                from .data import REPO_ROOT

                ckpt_path = os.path.join(REPO_ROOT, ckpt)
                if os.path.exists(ckpt_path):
                    self.features.load_action_encoder(ckpt_path)
                else:
                    print(f"[model] action checkpoint not found at {ckpt_path} — "
                          "using ImageNet initialisation")
        elif backbone == "dinov2":
            self.features = _DINOv2Features(mcfg["dinov2_name"], image_size,
                                            freeze_blocks, in_chans)
        else:
            raise ValueError("model.backbone must be 'timm' or 'dinov2'")

        dim = self.features.dim
        self.image_size = image_size
        self.tau = float(mcfg["tau_start"])
        self.pooling = str(mcfg.get("pooling", "topk")).lower()
        self.topk_frac = float(mcfg.get("topk_frac", 0.05))
        if self.pooling not in ("topk", "lse", "keypoint"):
            raise ValueError("model.pooling must be 'topk', 'lse' or 'keypoint'")
        pcfg = cfg.get("pose", {})
        self.kp_radius = int(pcfg.get("radius_px", 18))
        self.kp_topk = int(pcfg.get("pool_topk", 3))

        self.head = nn.Sequential(
            nn.Conv2d(dim, 256, 3, padding=1), nn.GroupNorm(32, 256), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1), nn.GroupNorm(32, 128), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.GroupNorm(16, 64), nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, -4.0)

    def set_tau(self, tau):
        self.tau = float(tau)

    def forward(self, image, region, kp_xy=None, kp_valid=None):
        feat = self.features(image)
        z = self.head(feat)
        z = F.interpolate(z, size=image.shape[-2:], mode="bilinear", align_corners=False)
        z = z.masked_fill(region < 0.5, -1e4)

        if self.pooling == "keypoint":
            if kp_xy is None or kp_valid is None:
                raise ValueError("pooling='keypoint' needs kp_xy and kp_valid; "
                                 "run precompute_pose.py and set pose.use_in_model")
            pooled, joint_scores = keypoint_pool(z, kp_xy, kp_valid,
                                                 self.kp_radius, self.kp_topk)
            return z, pooled, joint_scores
        if self.pooling == "topk":
            return z, masked_topk_mean(z, region, self.topk_frac), None
        return z, masked_lse(z, region, self.tau), None

    def parameter_groups(self, lr_backbone, lr_head):
        backbone_params = [p for p in self.features.parameters() if p.requires_grad]
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": self.head.parameters(), "lr": lr_head},
        ]
