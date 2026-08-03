"""
Action metric-learning training — image only (no pose), config-driven.

Follows reproduction/CattleAct/metric/action_with_image.py: a timm ViT-B/16 is
fine-tuned as an EMBEDDING encoder with a triplet loss (batch-all mining) plus a
zero-mean latent regularization that keeps the embedding distribution centered
and compact. This yields a discriminative action latent space; the interaction
model reuses this checkpoint's ViT backbone.

Evaluation is k-NN (1-NN) in the embedding space (knn accuracy + macro-F1).
Reads data/annotated/annotated_action.csv; the 6:2:2 split is taken from the
train/val/test folder in each image path.

Run from the repo root:
  python -m train.action_with_image
"""

import os
import shutil
import sys
from collections import defaultdict

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, BatchSampler
from torchmetrics import Accuracy, F1Score
from torchvision import transforms as T

# Add the parent directory to the system path to allow for package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import CattleActionDataset

# ── Config (see global_config.yaml at the repository root) ────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
with open(os.path.join(_REPO_ROOT, "global_config.yaml")) as _f:
    _CFG = yaml.safe_load(_f)

# 7-class action labels (must match prep/action_prep.py action_prep.labels order)
ACTION_MAP_LABEL = {name: i for i, name in enumerate(_CFG["action_prep"]["labels"])}


class LitVisionTransformer(pl.LightningModule):
    """ViT-B/16 (timm) action encoder trained with triplet loss + zero-mean
    latent regularization. forward() returns the raw embedding."""

    def __init__(self, embedding_size, learning_rate=1e-5, margin=0.5,
                 reg_loss_weight=0.1, sigma_s_squared=10.0):
        super().__init__()
        self.save_hyperparameters()
        self.map_label = dict(ACTION_MAP_LABEL)
        self.learning_rate = learning_rate
        self.margin = margin

        # timm ViT-B/16 whose head projects to the embedding dim. Same backbone
        # as the interaction model so its weights transfer.
        self.model = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=embedding_size)

        self.criterion = nn.TripletMarginLoss(margin=self.margin, p=2)

        n = len(self.map_label)
        self.val_knn_accuracy = Accuracy(task="multiclass", num_classes=n)
        self.test_knn_accuracy = Accuracy(task="multiclass", num_classes=n)
        self.val_f1score = F1Score(task="multiclass", num_classes=n, average="weighted")
        self.test_f1score = F1Score(task="multiclass", num_classes=n, average="weighted")

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)                       # raw embedding

    def training_step(self, batch, batch_idx):
        imgs, _, labels, _ = batch
        labels = labels.view(-1)

        raw_embeddings = self.forward(imgs)
        # zero-mean latent regularization (keeps embeddings centered/compact)
        reg_loss = torch.sum(raw_embeddings.pow(2)) / (2 * self.hparams.sigma_s_squared)

        embeddings = F.normalize(raw_embeddings, p=2, dim=1)
        anchor, positive, negative = self._get_all_valid_triplets(embeddings, labels)
        if anchor is None:
            triplet_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            triplet_loss = self.criterion(anchor, positive, negative)

        total_loss = triplet_loss + self.hparams.reg_loss_weight * reg_loss
        self.log("train_triplet_loss", triplet_loss, on_step=True, on_epoch=True)
        self.log("train_reg_loss", reg_loss, on_step=True, on_epoch=True)
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def _get_all_valid_triplets(self, embeddings, labels):
        """Batch-All mining: every valid triplet with positive margin loss."""
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        n = labels.size(0)
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        identity = torch.eye(n, dtype=torch.bool, device=self.device)
        mask_positive = labels_eq & ~identity
        mask_negative = ~labels_eq
        anchor_positive_dist = dist_matrix.unsqueeze(2)
        anchor_negative_dist = dist_matrix.unsqueeze(1)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin
        mask_valid = mask_positive.unsqueeze(2) & mask_negative.unsqueeze(1)
        mask_loss = (triplet_loss > 0) & mask_valid
        if not mask_loss.any():
            return None, None, None
        a_idx, p_idx, ng_idx = torch.where(mask_loss)
        return embeddings[a_idx], embeddings[p_idx], embeddings[ng_idx]

    def validation_step(self, batch, batch_idx):
        imgs, _, labels, _ = batch
        emb = F.normalize(self.forward(imgs), p=2, dim=1)
        self.validation_step_outputs.append(
            {"embeddings": emb.cpu(), "labels": labels.view(-1).cpu()})

    def _knn_eval(self, outputs, knn_metric, f1_metric, prefix):
        if not outputs:
            return
        emb = torch.cat([o["embeddings"] for o in outputs], dim=0)
        labels = torch.cat([o["labels"] for o in outputs], dim=0)
        if emb.size(0) < 2:
            return
        dist = torch.cdist(emb, emb, p=2)
        dist.fill_diagonal_(float("inf"))
        preds = labels[torch.argmin(dist, dim=1)]
        preds, labels = preds.to(self.device), labels.to(self.device)
        knn_metric.update(preds, labels)
        f1_metric.update(preds, labels)
        self.log(f"{prefix}_knn_acc", knn_metric.compute(), on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_f1score", f1_metric.compute(), on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self._knn_eval(self.validation_step_outputs,
                       self.val_knn_accuracy, self.val_f1score, "val")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        imgs, _, labels, _ = batch
        emb = F.normalize(self.forward(imgs), p=2, dim=1)
        self.test_step_outputs.append(
            {"embeddings": emb.cpu(), "labels": labels.view(-1).cpu()})

    def on_test_epoch_end(self):
        self._knn_eval(self.test_step_outputs,
                       self.test_knn_accuracy, self.test_f1score, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class BalancedBatchSampler(BatchSampler):
    """Each batch contains n_classes classes with n_samples samples each, so the
    triplet miner always has positives and negatives."""

    def __init__(self, labels, n_classes, n_samples):
        self.labels = np.array(labels)
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_batches = len(self.labels) // (n_classes * n_samples)
        if self.n_batches == 0:
            raise ValueError(
                f"Not enough samples for one batch: have {len(self.labels)}, "
                f"need {n_classes * n_samples}.")
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)

    def __iter__(self):
        for _ in range(self.n_batches):
            available = [l for l, idx in self.label_to_indices.items() if idx]
            if len(available) < self.n_classes:
                raise ValueError(
                    f"Not enough classes with samples: need {self.n_classes}, "
                    f"have {len(available)}.")
            batch_classes = np.random.choice(available, self.n_classes, replace=False)
            batch = []
            for c in batch_classes:
                idxs = self.label_to_indices[c]
                batch.extend(np.random.choice(
                    idxs, self.n_samples, replace=len(idxs) < self.n_samples))
            yield batch

    def __len__(self):
        return self.n_batches


class CattleActionDataModule(pl.LightningDataModule):
    """Image-only action DataModule reading annotated_action.csv (config-driven).
    The split is taken from the train/val/test folder in each image path."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.label_map = dict(ACTION_MAP_LABEL)
        image_size = cfg["image_size"]
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # RandAugment + N passes of RandomErasing (image-only stand-in for the
        # reference's cutout/skeleton masking). Values from config.
        erasers = [
            T.RandomErasing(p=cfg["cutout_prob"],
                            scale=tuple(cfg["cutout_scale"]),
                            ratio=tuple(cfg["cutout_ratio"]))
            for _ in range(cfg["cutout_n_holes"])
        ]
        self.transform_train = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=cfg["randaug_num_ops"],
                          magnitude=cfg["randaug_magnitude"]),
            T.ToTensor(), normalize,
        ] + erasers)
        self.transform_val = T.Compose([
            T.Resize(image_size), T.CenterCrop(image_size),
            T.ToTensor(), normalize,
        ])
        self.train_batch_sampler = None
        self._val_empty = self._test_empty = True

    def _load_entries(self):
        import csv
        entries = []
        with open(self.cfg["action_csv"], newline="") as f:
            for row in csv.DictReader(f):
                label = self.label_map.get(row["Label"], -1)
                if label == -1:
                    continue
                pose = row.get("pose_path", "") or ""
                entries.append({
                    "image_path": os.path.join(self.cfg["repo_root"], row["image_path"]),
                    "pose_path": os.path.join(self.cfg["repo_root"], pose) if pose else "",
                    "label": label,
                })
        return entries

    def setup(self, stage=None):
        entries = self._load_entries()
        buckets = {"train": [], "val": [], "test": []}
        for e in entries:
            for sp in ("train", "val", "test"):
                if f"{os.sep}{sp}{os.sep}" in e["image_path"]:
                    buckets[sp].append(e)
                    break

        id2name = {v: k for k, v in self.label_map.items()}
        print("[action] samples per split / class:")
        for sp in ("train", "val", "test"):
            counts = torch.bincount(
                torch.tensor([e["label"] for e in buckets[sp]] or [0]),
                minlength=len(self.label_map))
            if not buckets[sp]:
                counts = torch.zeros(len(self.label_map), dtype=torch.long)
            per_cls = "  ".join(f"{id2name[i]}:{int(counts[i])}"
                                for i in range(len(self.label_map)))
            print(f"  {sp:5s} total {len(buckets[sp]):5d} | {per_cls}")

        if not buckets["train"]:
            raise RuntimeError("[action] train split is empty — run prep/action_prep.py.")
        self._val_empty = not buckets["val"]
        self._test_empty = not buckets["test"]
        if self._val_empty:
            print("[action] [WARN] val split empty — checkpointing on the last epoch.")

        self.train_dataset = CattleActionDataset(
            buckets["train"], self.label_map,
            image_transform=self.transform_train, custom_image_transform=None)
        self.val_dataset = CattleActionDataset(
            buckets["val"], self.label_map, image_transform=self.transform_val)
        self.test_dataset = CattleActionDataset(
            buckets["test"], self.label_map, image_transform=self.transform_val)

        # BalancedBatchSampler — cap classes_per_batch at the classes present.
        labels = [e["label"] for e in buckets["train"]]
        present = len(set(labels))
        n_classes = min(self.cfg["classes_per_batch"], present)
        if n_classes < 2:
            raise RuntimeError(
                "[action] need >= 2 action classes in train for triplet mining.")
        self._n_classes = n_classes
        self.train_batch_sampler = BalancedBatchSampler(
            labels, n_classes=n_classes, n_samples=self.cfg["samples_per_class"])

    def _eval_bs(self):
        return self._n_classes * self.cfg["samples_per_class"]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_sampler=self.train_batch_sampler,
                          num_workers=self.cfg["num_workers"], pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self._eval_bs(),
                          num_workers=self.cfg["num_workers"], pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self._eval_bs(),
                          num_workers=self.cfg["num_workers"], pin_memory=True)


def main():
    pl.seed_everything(_CFG["random_seed"], workers=True)

    acfg = dict(_CFG["action_train"])
    acfg["repo_root"] = _REPO_ROOT
    acfg["action_csv"] = os.path.join(
        _REPO_ROOT, _CFG["paths"]["annotated_dir"], "annotated_action.csv")

    data_module = CattleActionDataModule(acfg)
    data_module.setup()  # know val/test emptiness before building callbacks

    model = LitVisionTransformer(
        embedding_size=acfg["embedding_size"],
        learning_rate=acfg["learning_rate"],
        margin=acfg["margin"],
        reg_loss_weight=acfg["reg_loss_weight"],
        sigma_s_squared=acfg["sigma_s_squared"])

    run_dir = os.path.join(_REPO_ROOT, acfg["run_dir"])
    if data_module._val_empty:
        ckpt_cb = ModelCheckpoint(save_last=True, dirpath=os.path.join(run_dir, "ckpt"),
                                  filename="action_last")
        limit_val = 0.0
    else:
        ckpt_cb = ModelCheckpoint(monitor="val_knn_acc", mode="max", save_top_k=1,
                                  save_last=True, dirpath=os.path.join(run_dir, "ckpt"),
                                  filename="action_best")
        limit_val = 1.0

    trainer = pl.Trainer(
        max_epochs=acfg["epochs"], accelerator="auto", devices=1,
        default_root_dir=run_dir, logger=CSVLogger(run_dir, name="csv"),
        num_sanity_val_steps=0, limit_val_batches=limit_val,
        use_distributed_sampler=False, callbacks=[ckpt_cb])

    trainer.fit(model, datamodule=data_module)
    if not data_module._test_empty:
        trainer.test(datamodule=data_module, ckpt_path="best")
    else:
        print("[action] test split empty — skipping test.")

    best = ckpt_cb.best_model_path or ckpt_cb.last_model_path
    out = os.path.join(_REPO_ROOT, _CFG["paths"]["action_ckpt"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    shutil.copy(best, out)
    print(f"[action] checkpoint -> {out}  (from {best})")


if __name__ == "__main__":
    main()
