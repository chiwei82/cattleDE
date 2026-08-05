"""Contact-localisation model: a ViT backbone, a dense head, and MIL pooling.

The design constraint that makes weak supervision work here is that the ONLY
path from pixels to the classification loss runs through the heatmap:

    patch tokens -> dense logits z -> mask to R -> masked LSE -> BCE(label)

There is deliberately no CLS-token classifier and no global pooling branch. If
one existed the network could satisfy the interaction label from scene-level
cues (herd density, background, crop geometry) and leave the heatmap arbitrary.
Forcing the score through a spatial soft-max inside R means the only way to
raise it is to raise the response at some location where the two cows meet.

The pooling is normalised by |R| so that a pair with a large overlap region does
not score higher simply for having more pixels to pool over.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_lse(z, region, tau):
    """Region-restricted, area-normalised log-sum-exp pooling.

        s = (1 / tau) * log( (1 / |R|) * sum_{u in R} exp(tau * z_u) )

    tau -> 0 behaves like the mean over R, tau -> inf like the max. Annealing tau
    upwards lets early training spread gradient over the whole candidate region
    and later training concentrate it on the peak.

    Returns pooled logits of shape (B,).
    """
    b = z.shape[0]
    z_flat = z.reshape(b, -1)
    r_flat = region.reshape(b, -1)
    count = r_flat.sum(dim=1).clamp(min=1.0)
    scaled = (z_flat * tau).masked_fill(r_flat < 0.5, float("-inf"))
    return (torch.logsumexp(scaled, dim=1) - torch.log(count)) / tau


class _TimmViTFeatures(nn.Module):
    """timm ViT wrapper exposing patch tokens as a spatial feature map.

    timm is already a dependency of train/action_with_image.py, so this path adds
    nothing new to the environment and can be warm-started from the stage-1
    action encoder.
    """

    def __init__(self, name, image_size, freeze_blocks):
        super().__init__()
        import timm

        self.vit = timm.create_model(name, pretrained=True, num_classes=0)
        self.patch = self.vit.patch_embed.patch_size[0]
        self.grid = image_size // self.patch
        self.dim = self.vit.embed_dim
        self.num_prefix = getattr(self.vit, "num_prefix_tokens", 1)

        self.vit.patch_embed.requires_grad_(False)
        for block in self.vit.blocks[:freeze_blocks]:
            block.requires_grad_(False)

    def load_action_encoder(self, ckpt_path):
        """Warm-start from checkpoints/action.ckpt (read-only).

        The Lightning checkpoint stores the encoder under 'backbone'/'backbone_vit'
        prefixes; only matching keys are copied, so a mismatch degrades to
        ImageNet initialisation instead of failing the run.
        """
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
    """DINOv2 wrapper; sharper spatial features but requires torch.hub download."""

    def __init__(self, name, image_size, freeze_blocks):
        super().__init__()
        self.vit = torch.hub.load("facebookresearch/dinov2", name)
        self.patch = 14
        self.grid = image_size // self.patch
        self.dim = self.vit.embed_dim

        self.vit.patch_embed.requires_grad_(False)
        for block in self.vit.blocks[:freeze_blocks]:
            block.requires_grad_(False)

    def forward(self, x):
        tokens = self.vit.forward_features(x)["x_norm_patchtokens"]
        b, n, d = tokens.shape
        return tokens.transpose(1, 2).reshape(b, d, self.grid, self.grid)


class ContactMIL(nn.Module):
    """Dense contact head over a frozen-ish ViT, trained by region-restricted MIL."""

    def __init__(self, cfg):
        super().__init__()
        mcfg = cfg["model"]
        image_size = int(mcfg["image_size"])
        freeze_blocks = int(mcfg["freeze_blocks"])
        backbone = str(mcfg["backbone"]).lower()

        if backbone == "timm":
            self.features = _TimmViTFeatures(mcfg["timm_name"], image_size, freeze_blocks)
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
            self.features = _DINOv2Features(mcfg["dinov2_name"], image_size, freeze_blocks)
        else:
            raise ValueError("model.backbone must be 'timm' or 'dinov2'")

        dim = self.features.dim
        self.image_size = image_size
        self.tau = float(mcfg["tau_start"])

        # Two bilinear upsampling stages take the patch grid to 4x resolution;
        # the final interpolation to input size is done in forward(). Contact
        # regions are blob-shaped, so decoding at 4x the patch grid and
        # interpolating is enough and far cheaper than a full-resolution decoder.
        self.head = nn.Sequential(
            nn.Conv2d(dim, 256, 3, padding=1), nn.GroupNorm(32, 256), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1), nn.GroupNorm(32, 128), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.GroupNorm(16, 64), nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )
        # Start with a near-empty heatmap so the sparsity prior is satisfied at
        # initialisation and evidence has to be earned.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, -4.0)

    def set_tau(self, tau):
        self.tau = float(tau)

    def forward(self, image, region):
        """Return dense logits masked to R, and the pooled pair logit."""
        feat = self.features(image)
        z = self.head(feat)
        z = F.interpolate(z, size=image.shape[-2:], mode="bilinear", align_corners=False)
        # -1e4 rather than -inf keeps autocast and the TV term finite.
        z = z.masked_fill(region < 0.5, -1e4)
        return z, masked_lse(z, region, self.tau)

    def parameter_groups(self, lr_backbone, lr_head):
        backbone_params = [p for p in self.features.parameters() if p.requires_grad]
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": self.head.parameters(), "lr": lr_head},
        ]
