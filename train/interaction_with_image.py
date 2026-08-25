import argparse
import os
import shutil
import sys
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, WeightedRandomSampler, Sampler
from torchmetrics import Accuracy, F1Score, MetricCollection, MatthewsCorrCoef

# add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import CattleCroppedInteractionDataset
from src.loss_utils import InfoNCE, LDAMLoss, FocalLoss
from src.interaction_eval import PredictionCollector
from src.augmentation import ImageMaskingFromSkeletonForInteraction, AK_JOINT_MAP

# ── Config (see global_config.yaml at the repository root) ────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
with open(os.path.join(_REPO_ROOT, "global_config.yaml")) as _f:
    _CFG = yaml.safe_load(_f)


class ShallowCNNforContext(nn.Module):
    """
    Shallow CNN feature extractor for the context image.
    """

    def __init__(self, in_channels: int = 3, num_features_out: int = 512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=32, stride=4, padding=14),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, num_features_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features_out),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.num_features = num_features_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x


class LitHybridStreamFusion(pl.LightningModule):
    """LightningModule that learns interaction classification with a hybrid-stream design."""

    def __init__(
        self,
        num_classes: int,
        fusion_type: str,
        learning_rate: float,
        vit_ckpt_path: Optional[str],
        freeze_vit: bool,
        cls_num_list: list,
        main_loss_cfg: DictConfig,
        pre_fusion_loss_cfg: DictConfig,
        pooling_type: str,
        pre_fusion_loss_weight: float,
        imagenet_pretrained: bool = True,
    ):
        super().__init__()

        valid_fusion_types = {"attention", "mlp"}
        if fusion_type not in valid_fusion_types:
            raise ValueError("fusion_type must be 'attention' or 'mlp'")

        pre_fusion_loss_name = (
            pre_fusion_loss_cfg.name if pre_fusion_loss_cfg is not None else "none"
        )
        valid_pre_fusion_losses = {"triplet", "infonce", "none"}
        if pre_fusion_loss_name not in valid_pre_fusion_losses:
            raise ValueError(
                "pre_fusion_loss.name must be 'triplet', 'infonce', or 'none'"
            )

        valid_pooling_types = {"flatten", "gmp", "gap", "gap_gmp"}
        if pooling_type not in valid_pooling_types:
            raise ValueError(
                "pooling_type must be one of 'flatten', 'gmp', 'gap', or 'gap_gmp'"
            )

        if pre_fusion_loss_name == "none" and pre_fusion_loss_weight != 0.0:
            raise ValueError(
                "pre_fusion_loss_weight must be 0.0 when pre_fusion_loss.name is 'none'"
            )
        if pre_fusion_loss_name != "none" and pre_fusion_loss_weight <= 0.0:
            raise ValueError(
                "pre_fusion_loss_weight must be positive when pre_fusion_loss.name is not 'none'"
            )

        self.pre_fusion_loss_name = pre_fusion_loss_name
        self.pre_fusion_loss_weight = float(pre_fusion_loss_weight)

        self.save_hyperparameters(ignore=["main_loss_cfg", "pre_fusion_loss_cfg"])

        is_load_pretrained = bool(vit_ckpt_path) and os.path.exists(vit_ckpt_path)
        # The action checkpoint (if loaded) fully overrides the backbone, so the
        # ImageNet-vs-random choice only affects the no-action-space runs.
        self.backbone_vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=(imagenet_pretrained and not is_load_pretrained),
            num_classes=0,
        )

        if is_load_pretrained:
            self.load_vit_backbone_from_checkpoint(vit_ckpt_path)
        elif vit_ckpt_path:
            print(
                f"Warning: ViT checkpoint path provided but not found at {vit_ckpt_path}."
            )
        else:
            src = "ImageNet" if imagenet_pretrained else "random"
            print(f"Info: No action checkpoint — backbone uses {src} weights.")

        if self.hparams.freeze_vit and is_load_pretrained:
            print("Freezing ViT backbone weights.")
            for param in self.backbone_vit.parameters():
                param.requires_grad = False

        num_ftrs_vit = self.backbone_vit.num_features
        projection_dim = num_ftrs_vit

        self.vit_feature_projector = nn.Sequential(
            nn.Linear(num_ftrs_vit, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.cnn_feature_projector = nn.Sequential(
            nn.Linear(num_ftrs_vit, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.context_infonce_projector = nn.Sequential(
            nn.Linear(num_ftrs_vit, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.backbone_cnn = ShallowCNNforContext(num_features_out=num_ftrs_vit)

        if self.hparams.fusion_type == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=projection_dim, num_heads=8, batch_first=True
            )

            if self.hparams.pooling_type == "flatten":
                fusion_out_dim = projection_dim * 3
            elif self.hparams.pooling_type == "gap_gmp":
                fusion_out_dim = projection_dim * 2
            else:
                fusion_out_dim = projection_dim

            self.classifier = nn.Linear(fusion_out_dim, num_classes)
        else:
            fused_feature_dim = projection_dim * 3
            self.classifier = nn.Sequential(
                nn.Linear(fused_feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes),
            )

        if main_loss_cfg.name == "focal":
            if "gamma" not in main_loss_cfg:
                raise ValueError("main_loss.focal.gamma must be defined in the configuration.")
            gamma = float(main_loss_cfg.gamma)
            alpha = main_loss_cfg.alpha if "alpha" in main_loss_cfg else None
            self.classification_criterion = FocalLoss(
                gamma=gamma,
                alpha=alpha,
            )
        elif main_loss_cfg.name == "ldam":
            if not self.hparams.cls_num_list:
                raise ValueError("cls_num_list must be provided when using LDAMLoss.")
            if "max_m" not in main_loss_cfg or "s" not in main_loss_cfg:
                raise ValueError("main_loss.ldam must define both max_m and s in the configuration.")
            self.classification_criterion = LDAMLoss(
                cls_num_list=self.hparams.cls_num_list,
                max_m=float(main_loss_cfg.max_m),
                s=float(main_loss_cfg.s),
            )
        elif main_loss_cfg.name == "cross_entropy":
            self.classification_criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported main_loss: {main_loss_cfg.name}")

        if self.pre_fusion_loss_name == "triplet":
            if "margin" not in pre_fusion_loss_cfg:
                raise ValueError("pre_fusion_loss.triplet.margin must be defined in the configuration.")
            margin = float(pre_fusion_loss_cfg.margin)
            self.pre_fusion_criterion = nn.TripletMarginLoss(margin=margin)
        elif self.pre_fusion_loss_name == "infonce":
            if "temperature" not in pre_fusion_loss_cfg:
                raise ValueError("pre_fusion_loss.infonce.temperature must be defined in the configuration.")
            temperature = float(pre_fusion_loss_cfg.temperature)
            self.pre_fusion_criterion = InfoNCE(
                negative_mode="unpaired", temperature=temperature
            )
        else:
            self.pre_fusion_criterion = None

        # Imbalance-aware metrics: a majority-only classifier scores acc~=0.95 but
        # bal_acc=0.5, macro-F1~=0.49, and mcc=0 — so these expose "just guess the
        # majority" instead of rewarding it.
        metrics = MetricCollection(
            {
                "acc": Accuracy(task="multiclass", num_classes=num_classes),
                "bal_acc": Accuracy(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
                "f1score": F1Score(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
                "mcc": MatthewsCorrCoef(task="multiclass", num_classes=num_classes),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        # Optional per-sample prediction logger for the test split (set by main()).
        # When present, test_step records image_path / predicted / truth for the
        # confusion matrix and false-case CSVs.
        self.test_collector = None

    def load_vit_backbone_from_checkpoint(self, ckpt_path: str) -> None:
        print(f"Loading ViT backbone weights from checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        prefix = "model."
        backbone_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix) and not k.startswith(prefix + "head"):
                new_key = k[len(prefix) :]
                backbone_state_dict[new_key] = v
        if not backbone_state_dict:
            raise ValueError(
                f"No backbone weights with prefix '{prefix}' found in checkpoint."
            )
        self.backbone_vit.load_state_dict(backbone_state_dict, strict=False)

    def forward(self, images1, images2, images_context):
        features1 = self.backbone_vit(images1)
        features2 = self.backbone_vit(images2)
        raw_context = self.backbone_cnn(images_context)

        features1 = self.vit_feature_projector(features1)
        features2 = self.vit_feature_projector(features2)
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)

        features_context_fusion = self.cnn_feature_projector(raw_context)
        features_context_fusion = F.normalize(features_context_fusion, p=2, dim=1)

        features_context_infonce = self.context_infonce_projector(raw_context)
        features_context_infonce = F.normalize(features_context_infonce, p=2, dim=1)

        if self.hparams.fusion_type == "attention":
            stacked_features = torch.stack(
                (features1, features2, features_context_fusion), dim=1
            )
            attn_output, _ = self.attention(
                stacked_features, stacked_features, stacked_features
            )

            if self.hparams.pooling_type == "flatten":
                combined_features = attn_output.flatten(start_dim=1)
            elif self.hparams.pooling_type == "gmp":
                combined_features, _ = torch.max(attn_output, dim=1)
            elif self.hparams.pooling_type == "gap":
                combined_features = torch.mean(attn_output, dim=1)
            else:  # gap_gmp
                avg_pool = torch.mean(attn_output, dim=1)
                max_pool, _ = torch.max(attn_output, dim=1)
                combined_features = torch.cat((avg_pool, max_pool), dim=1)
        else:
            combined_features = torch.cat(
                (features1, features2, features_context_fusion), dim=1
            )

        logits = self.classifier(combined_features)
        return logits, (features1, features2, features_context_fusion), features_context_infonce

    def _compute_pre_fusion_loss(
        self,
        labels: torch.Tensor,
        features1: torch.Tensor,
        features2: torch.Tensor,
        features_context_infonce: torch.Tensor,
    ) -> torch.Tensor:
        if self.pre_fusion_loss_name == "none" or self.pre_fusion_criterion is None:
            return features1.new_zeros(())

        interaction_mask = labels != 0
        no_interaction_mask = labels == 0

        if not torch.any(interaction_mask) or not torch.any(no_interaction_mask):
            return features1.new_zeros(())

        interaction_indices = torch.where(interaction_mask)[0]
        no_interaction_indices = torch.where(no_interaction_mask)[0]

        anchors = []
        positives = []
        negatives = []

        for idx_tensor in interaction_indices:
            idx = idx_tensor.item()
            anchor_feat = features_context_infonce[idx]
            positive_feat1 = features1[idx]
            positive_feat2 = features2[idx]

            rand_idx = torch.randint(
                low=0,
                high=len(no_interaction_indices),
                size=(1,),
                device=labels.device,
            ).item()
            negative_index = no_interaction_indices[rand_idx]
            negative_feat = features_context_infonce[negative_index]

            anchors.extend([anchor_feat, anchor_feat])
            positives.extend([positive_feat1, positive_feat2])
            negatives.extend([negative_feat, negative_feat])

        if not anchors:
            return features1.new_zeros(())

        anchor_embs = torch.stack(anchors)
        positive_embs = torch.stack(positives)
        negative_embs = torch.stack(negatives)
        return self.pre_fusion_criterion(anchor_embs, positive_embs, negative_embs)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        images1, images2, images_context, labels, supp = batch
        logits, pre_fusion_features, features_context_infonce = self(
            images1, images2, images_context
        )
        features1, features2, _ = pre_fusion_features

        # Record per-sample predictions on the test split for confusion-matrix
        # and false-case analysis (image_path comes from the batch supplement).
        if stage == "test" and self.test_collector is not None:
            preds = logits.argmax(dim=1)
            self.test_collector.add(
                supp["image_path"],
                preds.detach().cpu().tolist(),
                labels.detach().cpu().tolist(),
            )

        main_loss = self.classification_criterion(logits, labels)
        pre_fusion_loss = self._compute_pre_fusion_loss(
            labels, features1, features2, features_context_infonce
        )

        weight = self.pre_fusion_loss_weight if self.pre_fusion_loss_name != "none" else 0.0
        total_loss = main_loss + weight * pre_fusion_loss

        metrics = getattr(self, f"{stage}_metrics", None)
        if metrics is not None:
            metrics.update(logits, labels)

        on_step = stage == "train"
        prog_bar = stage in {"train", "val"}
        total_loss_name = "train_total_loss" if stage == "train" else f"{stage}_loss"

        self.log(
            total_loss_name,
            total_loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=prog_bar,
            sync_dist=True,
        )
        self.log(
            f"{stage}_main_loss",
            main_loss,
            on_step=on_step,
            on_epoch=True,
            sync_dist=True,
        )
        if self.pre_fusion_loss_name != "none":
            self.log(
                f"{stage}_pre_fusion_loss",
                pre_fusion_loss,
                on_step=on_step,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}_pre_fusion_weight",
                torch.tensor(weight, device=self.device),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        self.test_metrics.reset()

        if self.test_collector is not None and self.test_collector.rows:
            pred_csv, false_csv = self.test_collector.write()
            print(self.test_collector.report())
            print(f"[interaction] per-sample predictions -> {pred_csv}")
            print(f"[interaction] false cases (FP+FN)    -> {false_csv}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.1, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1score",
                "interval": "epoch",
                "frequency": 1,
            },
        }


class OversampleMinoritySampler(Sampler):
    """Keep every negative once and repeat each positive `factor` times, reshuffled
    each epoch. Unlike WeightedRandomSampler (which balances to 50/50 within a fixed
    budget and drops ~half the negatives), this keeps ALL negatives and only
    oversamples the positives. factor=None -> repeat positives to match the negative
    count (round(n0/n1))."""

    def __init__(self, labels, factor=None):
        labels = np.asarray(labels)
        self.neg_idx = np.where(labels == 0)[0]
        self.pos_idx = np.where(labels == 1)[0]
        n0, n1 = len(self.neg_idx), len(self.pos_idx)
        if n1 == 0:
            self.k = 1
        elif factor is None:
            self.k = max(1, round(n0 / n1))       # match the negative count
        else:
            self.k = max(1, int(factor))
        self.base = np.concatenate([self.neg_idx, np.tile(self.pos_idx, self.k)])
        print(f"[interaction] oversample_pos sampler: {n0} negatives kept, "
              f"{n1} positives x{self.k} = {len(self.base)} samples/epoch")

    def __iter__(self):
        idx = self.base.copy()
        np.random.shuffle(idx)
        return iter(idx.tolist())

    def __len__(self):
        return len(self.base)


class CattleInteractionDataModule(pl.LightningDataModule):
    """Image-only, BINARY interaction DataModule reading annotated_interaction.csv.
    The split is read from the CSV 'split' column (6:2:2 from interaction_prep)."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.cls_num_list = None
        self.train_sampler_weights = None
        image_size = cfg["image_size"]
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_train = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=12),
            T.ToTensor(), normalize,
        ])
        self.transform_val = T.Compose([
            T.Resize((image_size, image_size)), T.ToTensor(), normalize,
        ])

        # Skeleton-aware Cutout on the merged two-cow crop (needs HRNet poses).
        # When enabled, the train dataset loads poses (use_pose=True) and masks
        # around the interaction-defining joints. Falls back to no masking on any
        # sample whose poses are missing/invalid.
        sk = cfg.get("skeleton_aug", {}) or {}
        self.use_skeleton_aug = bool(sk.get("use", False))
        if self.use_skeleton_aug:
            self.skeleton_aware_transform = ImageMaskingFromSkeletonForInteraction(
                joint_map=AK_JOINT_MAP,
                cutout_prob=sk.get("cutout_prob", 0.5),
                n_holes=sk.get("n_holes", 3),
                scale=tuple(sk.get("scale", [0.02, 0.2])),
                ratio=tuple(sk.get("ratio", [0.3, 3.3])),
                margin=sk.get("margin", 10),
            )
        else:
            self.skeleton_aware_transform = None

    def _binary_label(self, row):
        """Map a CSV row to a binary label: 0 = no interaction, 1 = interaction.
        Prefers an explicit has_interaction/label column, else derives from
        label_v1: values in no_interaction_names -> 0, values in exclude_names
        (e.g. 'not well-cropped') or blank -> None (dropped), else -> 1."""
        no_names = {str(x).strip().lower() for x in self.cfg["no_interaction_names"]}
        exclude = {str(x).strip().lower() for x in self.cfg.get("exclude_names", [])}
        for col in ("has_interaction", "label"):
            v = row.get(col, "")
            if v not in (None, ""):
                try:
                    return 1 if int(float(v)) != 0 else 0
                except ValueError:
                    pass
        v1 = (row.get("label_v1") or "").strip().lower()
        if not v1 or v1 in exclude:
            return None            # unlabelled or excluded -> dropped from training
        return 0 if v1 in no_names else 1

    def _resolve(self, p):
        """Join a repo-relative path onto repo_root; empty stays empty."""
        p = (p or "").strip()
        return os.path.join(self.cfg["repo_root"], p) if p else ""

    def _load_entries(self):
        import csv
        entries = []
        with open(self.cfg["interaction_csv"], newline="") as f:
            for row in csv.DictReader(f):
                label = self._binary_label(row)
                if label is None:
                    continue
                entries.append({
                    "image_path": os.path.join(self.cfg["repo_root"], row["image_path"]),
                    "bbox1_xyxy": row.get("bbox1_xyxy", "[0 0 0 0]"),
                    "bbox2_xyxy": row.get("bbox2_xyxy", "[0 0 0 0]"),
                    "merged_bbox_xyxy": row.get("merged_bbox_xyxy", "[0 0 0 0]"),
                    # HRNet pose .npy for each cow (needed by the skeleton-aware
                    # transform; empty if interaction_prep ran with use_pose=false).
                    "pose_path_1": self._resolve(row.get("pose_path_1", "")),
                    "pose_path_2": self._resolve(row.get("pose_path_2", "")),
                    "split": (row.get("split") or "").strip(),
                    "label": label,
                })
        return entries

    def setup(self, stage=None):
        entries = self._load_entries()
        buckets = {"train": [], "val": [], "test": []}
        for e in entries:
            sp = e["split"] if e["split"] in buckets else "train"
            buckets[sp].append(e)
        n0 = {sp: sum(1 for e in buckets[sp] if e["label"] == 0) for sp in buckets}
        n1 = {sp: sum(1 for e in buckets[sp] if e["label"] == 1) for sp in buckets}
        for sp in ("train", "val", "test"):
            print(f"[interaction] {sp:5s} total {len(buckets[sp]):5d} | "
                  f"no_interaction:{n0[sp]}  interaction:{n1[sp]}")
        if not buckets["train"]:
            raise RuntimeError(
                f"No labelled interaction rows in {self.cfg['interaction_csv']}. "
                "Annotate label_v1/label_v2 (or add a has_interaction column) "
                "before training the interaction model.")
        # Empty val/test tolerated: val-based checkpointing/early-stop falls back
        # to the last epoch, and test is skipped.
        self._val_empty = not buckets["val"]
        self._test_empty = not buckets["test"]
        if self._val_empty:
            print("[interaction] [WARN] val split empty — checkpointing on the last epoch.")

        # Train applies skeleton-aware Cutout (needs poses); val/test stay clean.
        self.train_dataset = CattleCroppedInteractionDataset(
            entries=buckets["train"], transform=self.transform_train,
            use_pose=self.use_skeleton_aug,
            skeleton_aware_transform=self.skeleton_aware_transform,
            is_aware_skeleton=self.use_skeleton_aug)
        self.val_dataset = CattleCroppedInteractionDataset(
            entries=buckets["val"], transform=self.transform_val, use_pose=False)
        self.test_dataset = CattleCroppedInteractionDataset(
            entries=buckets["test"], transform=self.transform_val, use_pose=False)

        counts = torch.zeros(2, dtype=torch.float)
        for e in buckets["train"]:
            counts[e["label"]] += 1
        self.cls_num_list = [int(c) for c in counts]
        self.train_labels = [e["label"] for e in buckets["train"]]
        weights = 1.0 / torch.where(counts > 0, counts, torch.tensor(float("inf")))
        self.train_sampler_weights = torch.tensor(
            [weights[e["label"]] for e in buckets["train"]])
        print(f"[interaction] class counts (binary): {self.cls_num_list}")

    def train_dataloader(self):
        mode = self.cfg.get("sampler", "weighted")
        if mode == "oversample_pos":
            # Keep all negatives, only repeat positives (see OversampleMinoritySampler).
            sampler = OversampleMinoritySampler(
                self.train_labels, factor=self.cfg.get("oversample_factor"))
        else:
            # Default: 50/50 balance within a fixed budget (drops ~half the negatives).
            sampler = WeightedRandomSampler(
                weights=self.train_sampler_weights,
                num_samples=len(self.train_sampler_weights), replacement=True)
        return DataLoader(self.train_dataset, batch_size=self.cfg["batch_size"],
                          sampler=sampler, num_workers=self.cfg["num_workers"],
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg["batch_size"],
                          num_workers=self.cfg["num_workers"], pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg["batch_size"],
                          num_workers=self.cfg["num_workers"], pin_memory=True)


def main() -> None:
    """Binary, image-only interaction training (config-driven, no hydra)."""
    ap = argparse.ArgumentParser(description="Train the binary interaction model.")
    ap.add_argument("--csv", default="annotated_interaction.csv",
                    help="CSV filename under paths.annotated_dir "
                         "(e.g. annotated_interaction_test.csv).")
    ap.add_argument("--backbone_init", choices=["action", "imagenet", "random"],
                    default=None,
                    help="ViT backbone init for the ablation. 'action' = pre-train "
                         "with the individual action space (loads paths.action_ckpt); "
                         "'random' = no pre-training (random weights); 'imagenet' = "
                         "ImageNet weights. Default: config pretrained_backbone "
                         "(True->action, False->random). Non-action runs write to "
                         "log/checkpoint paths suffixed with the init name so they "
                         "don't overwrite the action run.")
    ap.add_argument("--skeleton_aug", choices=["on", "off"], default=None,
                    help="Override skeleton_aug.use (pose-aware Cutout). "
                         "Default: config value.")
    ap.add_argument("--tag", default=None,
                    help="Explicit suffix for the run_dir/checkpoint (e.g. _full). "
                         "Overrides the backbone_init auto-suffix so several "
                         "ablations don't overwrite each other.")
    args = ap.parse_args()

    pl.seed_everything(_CFG["random_seed"], workers=True)

    icfg = dict(_CFG["interaction_train"])
    icfg["repo_root"] = _REPO_ROOT
    # CLI override of the pose-aware Cutout switch (copy the nested dict so we
    # don't mutate the shared config object).
    if args.skeleton_aug is not None:
        icfg["skeleton_aug"] = dict(icfg.get("skeleton_aug", {}) or {})
        icfg["skeleton_aug"]["use"] = (args.skeleton_aug == "on")
    icfg["interaction_csv"] = os.path.join(
        _REPO_ROOT, _CFG["paths"]["annotated_dir"], args.csv)

    data_module = CattleInteractionDataModule(icfg)
    data_module.setup(stage="fit")

    # Backbone init for the "with vs without action-space pre-training" ablation.
    # Default falls back to the config's pretrained_backbone flag.
    backbone_init = args.backbone_init or (
        "action" if icfg["pretrained_backbone"] else "random")
    if backbone_init == "action":
        vit_ckpt = os.path.join(_REPO_ROOT, _CFG["paths"]["action_ckpt"])
        imagenet_pretrained = True          # irrelevant: action ckpt overrides it
    elif backbone_init == "imagenet":
        vit_ckpt = None
        imagenet_pretrained = True
    else:                                    # random = no pre-training at all
        vit_ckpt = None
        imagenet_pretrained = False
    print(f"[interaction] backbone_init={backbone_init}  (vit_ckpt={vit_ckpt})")

    # Build the loss configs (output stays binary; these only shape the objective).
    ml = icfg.get("main_loss", "ldam")
    if ml == "ldam":
        main_loss_cfg = OmegaConf.create(
            {"name": "ldam", "max_m": icfg["ldam_max_m"], "s": icfg["ldam_s"]})
    elif ml == "focal":
        main_loss_cfg = OmegaConf.create(
            {"name": "focal", "gamma": icfg.get("focal_gamma", 2.0)})
    else:
        main_loss_cfg = OmegaConf.create({"name": "cross_entropy"})

    pfl = icfg.get("pre_fusion_loss", "infonce")
    if pfl == "infonce":
        pre_fusion_loss_cfg = OmegaConf.create(
            {"name": "infonce", "temperature": icfg["pre_fusion_temperature"]})
        pre_fusion_weight = icfg["pre_fusion_weight"]
    elif pfl == "triplet":
        pre_fusion_loss_cfg = OmegaConf.create(
            {"name": "triplet", "margin": icfg["pre_fusion_margin"]})
        pre_fusion_weight = icfg["pre_fusion_weight"]
    else:
        pre_fusion_loss_cfg = OmegaConf.create({"name": "none"})
        pre_fusion_weight = 0.0
    print(f"[interaction] main_loss={ml}, pre_fusion_loss={pfl}, "
          f"pre_fusion_weight={pre_fusion_weight}")

    model = LitHybridStreamFusion(
        num_classes=2,                              # binary: no_interaction / interaction
        learning_rate=icfg["learning_rate"],
        vit_ckpt_path=vit_ckpt,
        freeze_vit=icfg["freeze_backbone"],
        fusion_type=icfg["fusion_type"],
        cls_num_list=data_module.cls_num_list,
        main_loss_cfg=main_loss_cfg,
        pre_fusion_loss_cfg=pre_fusion_loss_cfg,
        pooling_type=icfg["pooling_type"],
        pre_fusion_loss_weight=pre_fusion_weight,
        imagenet_pretrained=imagenet_pretrained,
    )

    # Output suffix: explicit --tag wins; otherwise non-action runs get the
    # backbone_init name so ablations don't overwrite each other.
    if args.tag:
        tag = args.tag if args.tag.startswith("_") else "_" + args.tag
    else:
        tag = "" if backbone_init == "action" else f"_{backbone_init}"
    run_dir = os.path.join(_REPO_ROOT, icfg["run_dir"] + tag)
    if data_module._val_empty:
        checkpoint_callback = ModelCheckpoint(
            save_last=True, dirpath=os.path.join(run_dir, "ckpt"),
            filename="interaction_last")
        callbacks = [checkpoint_callback]
        limit_val = 0.0
    else:
        checkpoint_callback = ModelCheckpoint(
            monitor="val_f1score", mode="max", save_top_k=1, save_last=True,
            dirpath=os.path.join(run_dir, "ckpt"), filename="interaction_best")
        callbacks = [checkpoint_callback,
                     EarlyStopping(monitor="val_f1score", mode="max",
                                   patience=10, verbose=True)]
        limit_val = 1.0

    trainer = pl.Trainer(
        max_epochs=icfg["epochs"], accelerator="auto", devices=1,
        default_root_dir=run_dir, logger=CSVLogger(run_dir, name="csv"),
        num_sanity_val_steps=0, limit_val_batches=limit_val,
        callbacks=callbacks)

    print("--- Starting interaction (binary) training ---")
    trainer.fit(model, datamodule=data_module)

    best = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    if not getattr(data_module, "_test_empty", False):
        # Log per-sample test predictions (confusion matrix + false-case CSVs).
        model.test_collector = PredictionCollector(out_dir=run_dir, split="test")
        trainer.test(datamodule=data_module, ckpt_path=best)
    else:
        print("[interaction] test split empty — skipping test.")

    out = os.path.join(_REPO_ROOT, _CFG["paths"]["interaction_ckpt"])
    if tag:
        base, ext = os.path.splitext(out)
        out = base + tag + ext
    os.makedirs(os.path.dirname(out), exist_ok=True)
    shutil.copy(best, out)
    print(f"[interaction] checkpoint -> {out}  (from {best})")


if __name__ == "__main__":
    main()