"""
Action classification training — image only (no pose), config-driven.

Reads data/annotated/annotated_action.csv produced by prep/action_prep.py; the
6:2:2 train/val/test split is taken from the {split} folder in each image path.
The backbone is a timm ViT-B/16 so the learned latent space can be reused by the
interaction model (train/interaction_with_image.py loads this checkpoint's ViT).

Run from the repo root:
  python -m train.action_with_image
"""

import os
import shutil
import sys

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import yaml
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
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
    """ViT-B/16 (timm) fine-tuned for per-cow action classification, image only."""

    def __init__(self, num_classes, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.map_label = dict(ACTION_MAP_LABEL)
        self.learning_rate = learning_rate

        # timm ViT-B/16 (matches the interaction model's backbone so its weights
        # can be transferred). num_classes sets the classification head.
        self.model = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=num_classes
        )

        self.criterion = nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1score = F1Score(task="multiclass", num_classes=num_classes, average="weighted")
        self.test_f1score = F1Score(task="multiclass", num_classes=num_classes, average="weighted")

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch):
        imgs, _, labels, _ = batch
        labels = labels.view(-1)
        logits = self.forward(imgs)
        loss = self.criterion(logits, labels)
        return logits, loss, labels

    def training_step(self, batch, batch_idx):
        logits, loss, labels = self._common_step(batch)
        self.train_accuracy(logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, loss, labels = self._common_step(batch)
        self.val_accuracy(logits, labels)
        self.val_f1score(logits, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_accuracy, on_epoch=True, prog_bar=True)
        self.log("val_f1score", self.val_f1score, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        logits, loss, labels = self._common_step(batch)
        preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
        self.test_accuracy.update(preds, labels)
        self.test_f1score.update(preds, labels)
        self.log("test_loss", loss, on_epoch=True, sync_dist=True)
        self.log("test_acc", self.test_accuracy, on_epoch=True, sync_dist=True)
        self.log("test_f1score", self.test_f1score, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class CattleActionDataModule(pl.LightningDataModule):
    """Image-only action DataModule reading annotated_action.csv (config-driven).
    The split is taken from the train/val/test folder in each image path."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.label_map = dict(ACTION_MAP_LABEL)
        image_size = cfg["image_size"]
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_train = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(), normalize,
        ])
        self.transform_val = T.Compose([
            T.Resize(image_size), T.CenterCrop(image_size),
            T.ToTensor(), normalize,
        ])
        self.train_sampler = None

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

        # Report per-split, per-class counts so missing classes are visible.
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
            missing = [id2name[i] for i in range(len(self.label_map))
                       if int(counts[i]) == 0]
            if missing:
                print(f"    [WARN] '{sp}' has NO samples for: {missing}")

        if not buckets["train"]:
            raise RuntimeError("[action] train split is empty — run prep/action_prep.py.")
        # Rare classes may be absent from val/test — this is accepted; the macro
        # metrics simply ignore classes with no samples. An entirely empty val
        # (or test) is also tolerated: val-based checkpointing then falls back to
        # the last epoch, and test is skipped.
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

        labels = [e["label"] for e in buckets["train"]]
        counts = torch.bincount(torch.tensor(labels),
                                minlength=len(self.label_map)).float()
        weights = 1.0 / (counts + 1e-6)
        sample_weights = torch.tensor([weights[l] for l in labels])
        self.train_sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg["batch_size"],
                          num_workers=self.cfg["num_workers"], pin_memory=True,
                          sampler=self.train_sampler)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg["batch_size"],
                          num_workers=self.cfg["num_workers"], pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg["batch_size"],
                          num_workers=self.cfg["num_workers"], pin_memory=True)


def main():
    ap = argparse.ArgumentParser(description="Train / evaluate the action classifier.")
    ap.add_argument("--eval_only", default=None, help="Path to a checkpoint; run val+test only, no training.")
    args = ap.parse_args()

    pl.seed_everything(_CFG["random_seed"], workers=True)

    acfg = dict(_CFG["action_train"])
    acfg["repo_root"] = _REPO_ROOT
    acfg["action_csv"] = os.path.join(
        _REPO_ROOT, _CFG["paths"]["annotated_dir"], "annotated_action.csv")

    data_module = CattleActionDataModule(acfg)
    data_module.setup()  # know whether val/test are empty before building callbacks
    model = LitVisionTransformer(
        num_classes=len(ACTION_MAP_LABEL), learning_rate=acfg["learning_rate"])

    run_dir = os.path.join(_REPO_ROOT, acfg["run_dir"])

    if args.eval_only:
        model = LitVisionTransformer.load_from_checkpoint(args.eval_only)
        trainer = pl.Trainer(accelerator="auto", devices=1,
                            logger=CSVLogger(run_dir, name="csv_eval"))
        trainer.validate(model, datamodule=data_module)
        trainer.test(model, datamodule=data_module)
        return

    if data_module._val_empty:
        # No val to monitor: keep the last epoch's checkpoint.
        ckpt_cb = ModelCheckpoint(
            save_last=True, dirpath=os.path.join(run_dir, "ckpt"), filename="action_last")
        limit_val = 0.0
    else:
        ckpt_cb = ModelCheckpoint(
            monitor="val_f1score", mode="max", save_top_k=1, save_last=True,
            dirpath=os.path.join(run_dir, "ckpt"), filename="action_best")
        limit_val = 1.0

    trainer = pl.Trainer(
        max_epochs=acfg["epochs"], accelerator="auto", devices=1,
        default_root_dir=run_dir, logger=CSVLogger(run_dir, name="csv"),
        num_sanity_val_steps=0,  # skip sanity check (avoids stalls on odd val sets)
        limit_val_batches=limit_val,  # 0 disables validation when val is empty
        callbacks=[ckpt_cb])

    trainer.fit(model, datamodule=data_module)
    if not getattr(data_module, "_test_empty", False):
        trainer.test(datamodule=data_module, ckpt_path="best")
    else:
        print("[action] test split empty — skipping test.")

    # Publish best (fall back to last if val never improved / was skipped).
    best = ckpt_cb.best_model_path or ckpt_cb.last_model_path
    out = os.path.join(_REPO_ROOT, _CFG["paths"]["action_ckpt"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    shutil.copy(best, out)
    print(f"[action] checkpoint -> {out}  (from {best})")


if __name__ == "__main__":
    main()
