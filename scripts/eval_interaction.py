"""Evaluate trained interaction checkpoints on val/test.

Loads every checkpoints/<loss>/interaction_<model>.ckpt, runs inference on the
val and test splits, and reports overall Accuracy, Balanced Accuracy, Macro-F1,
MCC, plus precision/recall/F1 on the interaction (positive) class.

The criterion is irrelevant for inference (LDAM/Focal/InfoNCE hold no state), so
every checkpoint is rebuilt with a cross_entropy head; only the network weights
matter.

    python scripts/eval_interaction.py \
        --ckpt_root /user/work/yx25778/cattleDE/checkpoints \
        --csv annotated_interaction_test_resplit.csv

IMPORTANT: use the SAME --csv the models were trained on, so the val/test split
column matches training (otherwise test sessions may overlap training -> invalid).
"""

import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train.interaction_with_image import (
    LitHybridStreamFusion, CattleInteractionDataModule, _CFG, _REPO_ROOT,
)

LOSSES = ["cross_entropy", "focal", "ldam"]
MODELS = ["full", "without_action_random", "without_both", "without_pose"]


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hp = ckpt.get("hyper_parameters", {})
    w = float(hp.get("pre_fusion_loss_weight", 0.0) or 0.0)
    main_cfg = OmegaConf.create({"name": "cross_entropy"})
    pre_cfg = (OmegaConf.create({"name": "infonce", "temperature": 0.07})
               if w > 0 else OmegaConf.create({"name": "none"}))
    model = LitHybridStreamFusion.load_from_checkpoint(
        ckpt_path, map_location=device,
        main_loss_cfg=main_cfg, pre_fusion_loss_cfg=pre_cfg)
    model.eval().to(device)
    return model


@torch.no_grad()
def infer(model, loader, device):
    preds, truths = [], []
    for batch in loader:
        i1, i2, ic, labels, _ = batch
        logits = model(i1.to(device), i2.to(device), ic.to(device))[0]
        preds.extend(logits.argmax(1).cpu().tolist())
        truths.extend(labels.tolist())
    return np.array(preds), np.array(truths)


def metrics(preds, truths):
    tp = int(((preds == 1) & (truths == 1)).sum())
    tn = int(((preds == 0) & (truths == 0)).sum())
    fp = int(((preds == 1) & (truths == 0)).sum())
    fn = int(((preds == 0) & (truths == 1)).sum())
    n = tp + tn + fp + fn

    def f1(p, r):
        return 2 * p * r / (p + r) if (p + r) else 0.0
    # positive = interaction
    p_pos = tp / (tp + fp) if (tp + fp) else 0.0
    r_pos = tp / (tp + fn) if (tp + fn) else 0.0
    f1_pos = f1(p_pos, r_pos)
    # negative = no_interaction
    p_neg = tn / (tn + fn) if (tn + fn) else 0.0
    r_neg = tn / (tn + fp) if (tn + fp) else 0.0
    f1_neg = f1(p_neg, r_neg)

    acc = (tp + tn) / n if n else 0.0
    bal_acc = (r_pos + r_neg) / 2          # mean per-class recall
    macro_f1 = (f1_pos + f1_neg) / 2
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0
    return dict(acc=acc, bal_acc=bal_acc, macro_f1=macro_f1, mcc=mcc,
                int_p=p_pos, int_r=r_pos, int_f1=f1_pos, n=n,
                n_pos=int((truths == 1).sum()))


def main():
    ap = argparse.ArgumentParser(description="Evaluate interaction checkpoints.")
    ap.add_argument("--ckpt_root",
                    default=os.path.join(_REPO_ROOT, "checkpoints"),
                    help="Dir holding <loss>/interaction_<model>.ckpt.")
    ap.add_argument("--csv", default="annotated_interaction_test_resplit.csv",
                    help="CSV under paths.annotated_dir (must match training split).")
    ap.add_argument("--losses", nargs="+", default=LOSSES)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    icfg = dict(_CFG["interaction_train"])
    icfg["repo_root"] = _REPO_ROOT
    icfg["interaction_csv"] = os.path.join(
        _REPO_ROOT, _CFG["paths"]["annotated_dir"], args.csv)
    dm = CattleInteractionDataModule(icfg)
    dm.setup(stage="fit")
    loaders = {"val": dm.val_dataloader, "test": dm.test_dataloader}

    hdr = (f"{'loss':<13} {'model':<24} {'split':<5} {'n':>5} {'nPos':>4} "
           f"{'Acc':>6} {'BalAcc':>7} {'MacroF1':>8} {'MCC':>7} "
           f"{'Int_P':>6} {'Int_R':>6} {'Int_F1':>7}")
    print(hdr)
    print("-" * len(hdr))
    for loss in args.losses:
        for mdl in args.models:
            path = os.path.join(args.ckpt_root, loss, f"interaction_{mdl}.ckpt")
            if not os.path.exists(path):
                print(f"{loss:<13} {mdl:<24} [missing] {path}")
                continue
            model = load_model(path, device)
            for sp in args.splits:
                loader = loaders[sp]()
                if len(loader.dataset) == 0:
                    print(f"{loss:<13} {mdl:<24} {sp:<5} [empty split]")
                    continue
                preds, truths = infer(model, loader, device)
                m = metrics(preds, truths)
                print(f"{loss:<13} {mdl:<24} {sp:<5} {m['n']:>5} {m['n_pos']:>4} "
                      f"{m['acc']:>6.3f} {m['bal_acc']:>7.3f} {m['macro_f1']:>8.3f} "
                      f"{m['mcc']:>7.3f} {m['int_p']:>6.3f} {m['int_r']:>6.3f} "
                      f"{m['int_f1']:>7.3f}")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
