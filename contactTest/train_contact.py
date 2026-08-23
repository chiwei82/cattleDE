
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contactTest.src.data import ContactPairDataset, describe, load_records, split_records
from contactTest.src.losses import anneal_tau, contact_losses
from contactTest.src.model import ContactMIL
from contactTest.src.utils import load_config, roc_auc

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))


def out_path(cfg, key, *parts):
    path = os.path.join(CONTACT_ROOT, cfg["output"][key], *parts)
    os.makedirs(os.path.dirname(path) if os.path.splitext(path)[1] else path, exist_ok=True)
    return path


def model_forward(model, batch, device):
    image = batch["image"].to(device, non_blocking=True)
    region = batch["region"].to(device, non_blocking=True)
    if getattr(model, "pooling", "") == "keypoint":
        return model(image, region,
                     batch["kp_xy"].to(device, non_blocking=True),
                     batch["kp_valid"].to(device, non_blocking=True))
    return model(image, region)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    labels, scores = [], []
    for batch in loader:
        _, pooled, _ = model_forward(model, batch, device)
        scores.extend(torch.sigmoid(pooled.float()).cpu().tolist())
        labels.extend(batch["label"].tolist())
    return roc_auc(labels, scores), labels, scores


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(CONTACT_ROOT, "config.yaml"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--backbone", choices=["timm", "dinov2"], default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.backbone is not None:
        cfg["model"]["backbone"] = args.backbone
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers

    seed = int(cfg["random_seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    records = load_records(cfg)
    buckets = split_records(records)
    describe(buckets)
    if not buckets["train"]:
        raise RuntimeError("no labelled training rows — check data.csv and labels.* in the config")

    train_set = ContactPairDataset(buckets["train"], cfg, train=True)
    val_set = ContactPairDataset(buckets["val"], cfg, train=False)
    test_set = ContactPairDataset(buckets["test"], cfg, train=False)

    tcfg = cfg["train"]
    loader_kwargs = dict(batch_size=tcfg["batch_size"], num_workers=tcfg["num_workers"],
                         pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContactMIL(cfg).to(device)

    pos_weight = None
    if cfg["loss"].get("auto_pos_weight", True):
        n_pos = sum(1 for r in buckets["train"] if r["label"] == 1)
        n_neg = len(buckets["train"]) - n_pos
        pos_weight = n_neg / max(n_pos, 1)
        print(f"[train] class imbalance {n_neg}:{n_pos} -> BCE pos_weight {pos_weight:.2f}")

    optimizer = torch.optim.AdamW(
        model.parameter_groups(tcfg["lr_backbone"], tcfg["lr_head"]),
        weight_decay=tcfg["weight_decay"])
    epochs = int(tcfg["epochs"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[tcfg["lr_backbone"], tcfg["lr_head"]],
        total_steps=epochs * max(1, len(train_loader)), pct_start=0.2)

    prior_cfg = (cfg.get("pose", {}) or {}).get("prior", {}) or {}
    lambda_prior = float(prior_cfg.get("weight", 0.0)) if prior_cfg.get("enabled") else 0.0
    if lambda_prior > 0:
        print(f"[train] pose prior ON, lambda {lambda_prior} — a soft hint towards "
              "the head-to-body contact site on POSITIVES only")

    use_amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_dir = out_path(cfg, "run_dir")
    ckpt_path = os.path.join(run_dir, cfg["output"]["ckpt_name"])
    log_path = os.path.join(run_dir, "train_log.csv")
    with open(os.path.join(run_dir, "config_snapshot.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "tau", "loss", "mil", "sparsity", "tv",
                         "prior", "prior_frac", "val_auc"])

    best_auc = -1.0
    for epoch in range(epochs):
        tau = anneal_tau(epoch, epochs, cfg["model"]["tau_start"], cfg["model"]["tau_end"])
        model.set_tau(tau)
        model.train()
        running = {"loss": 0.0, "mil": 0.0, "sparsity": 0.0, "tv": 0.0,
                   "prior": 0.0, "prior_frac": 0.0}
        seen = 0

        for batch in train_loader:
            region = batch["region"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                z, pooled, _ = model_forward(model, batch, device)
                loss, parts = contact_losses(
                    z.float(), region, pooled.float(), label,
                    pos_weight=pos_weight,
                    lambda_sparsity=cfg["loss"]["lambda_sparsity"],
                    lambda_tv=cfg["loss"]["lambda_tv"],
                    prior=batch["prior"].to(device, non_blocking=True)
                    if lambda_prior > 0 else None,
                    prior_weight=batch["prior_weight"].to(device, non_blocking=True)
                    if lambda_prior > 0 else None,
                    lambda_prior=lambda_prior)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            for k in running:
                running[k] += parts[k]
            seen += 1

        for k in running:
            running[k] /= max(seen, 1)
        val_auc, _, _ = evaluate(model, val_loader, device) if len(val_set) else (float("nan"), [], [])

        print(f"[train] epoch {epoch + 1:02d}/{epochs}  tau {tau:5.2f}  "
              f"loss {running['loss']:.4f}  mil {running['mil']:.4f}  "
              f"sp {running['sparsity']:.4f}  tv {running['tv']:.4f}"
              + (f"  prior {running['prior']:.4f} ({running['prior_frac']:.0%})"
                 if lambda_prior > 0 else "")
              + f"  val_auc {val_auc:.4f}")
        log_writer.writerow([epoch + 1, f"{tau:.3f}"] +
                            [f"{running[k]:.6f}" for k in
                             ("loss", "mil", "sparsity", "tv", "prior", "prior_frac")] +
                            [f"{val_auc:.6f}"])
        log_file.flush()

        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model": model.state_dict(), "config": cfg,
                        "epoch": epoch + 1, "val_auc": val_auc}, ckpt_path)

    if best_auc < 0:
        torch.save({"model": model.state_dict(), "config": cfg,
                    "epoch": epochs, "val_auc": float("nan")}, ckpt_path)
    log_file.close()

    summary = {"best_val_auc": best_auc, "checkpoint": ckpt_path}
    if len(test_set):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        model.set_tau(cfg["model"]["tau_end"])
        test_auc, labels, scores = evaluate(model, test_loader, device)
        summary["test_auc"] = test_auc
        print(f"[train] test pair-classification AUC {test_auc:.4f} "
              f"({int(sum(labels))} positives / {len(labels)} pairs)")
        with open(os.path.join(run_dir, "test_scores.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rel_image", "label", "score"])
            for record, y, s in zip(buckets["test"], labels, scores):
                w.writerow([record["rel_image"], int(y), f"{s:.6f}"])

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[train] checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
