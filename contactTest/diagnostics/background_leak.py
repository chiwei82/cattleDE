"""Diagnostic B — is the interaction label predictable from background alone?

Usage (from the repository root):

    python -m contactTest.diagnostics.background_leak

Blanks both cow boxes out of every crop, leaving only the surrounding pen, and
fits a linear probe on frozen ViT features of what remains. If that alone
separates the classes, then positives and negatives differ systematically in
where or when they were filmed, and any heatmap trained on this data will be
free to key on the background instead of the animals.

Interpretation of the reported AUC:

    ~0.50        no leakage; the background is uninformative as it should be
    0.55-0.65    mild leakage; acceptable but worth noting
    > 0.65       positives and negatives come from systematically different
                 conditions (camera, pen area, time of day). Rebalance the
                 sampling before trusting any localisation result.

Boxes are a superset of the animals, so blanking them removes some background
too. That makes this test conservative: it can under-report leakage, never
invent it.
"""

import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from contactTest.src.data import (IMAGENET_MEAN, IMAGENET_STD, PAD_VALUE, letterbox,
                                  load_records, relative_boxes, split_records)
from contactTest.src.utils import load_config, roc_auc

CONTACT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def blanked_tensor(record, size):
    """Load a crop, blank both cow boxes, letterbox, and normalise."""
    bgr = cv2.imread(record["image_path"])
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    for (x1, y1, x2, y2) in relative_boxes(record, h, w):
        rgb[y1:y2, x1:x2] = PAD_VALUE
    canvas, _, _, _ = letterbox(rgb, size, PAD_VALUE)
    x = (canvas.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(2, 0, 1)


@torch.no_grad()
def embed(records, size, device, batch_size=32):
    """Mean-pooled frozen ViT features of the background-only crops."""
    import timm

    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    model.eval().to(device)

    feats, labels = [], []
    batch, batch_labels = [], []

    def flush():
        if not batch:
            return
        x = torch.stack(batch).to(device)
        feats.append(model(x).cpu().numpy())
        labels.extend(batch_labels)
        batch.clear()
        batch_labels.clear()

    for i, record in enumerate(records):
        tensor = blanked_tensor(record, size)
        if tensor is None:
            continue
        batch.append(tensor)
        batch_labels.append(record["label"])
        if len(batch) == batch_size:
            flush()
        if (i + 1) % 500 == 0:
            print(f"[diag-B] embedded {i + 1}/{len(records)}")
    flush()
    return np.concatenate(feats, axis=0), np.array(labels)


def main():
    cfg = load_config(os.path.join(CONTACT_ROOT, "config.yaml"))
    size = int(cfg["model"]["image_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buckets = split_records(load_records(cfg))

    eval_split = "test" if buckets["test"] else "val"
    print(f"[diag-B] embedding train ({len(buckets['train'])} rows)")
    x_train, y_train = embed(buckets["train"], size, device)
    print(f"[diag-B] embedding {eval_split} ({len(buckets[eval_split])} rows)")
    x_test, y_test = embed(buckets[eval_split], size, device)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x_train)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                             random_state=int(cfg["random_seed"]))
    clf.fit(scaler.transform(x_train), y_train)
    auc = roc_auc(y_test, clf.predict_proba(scaler.transform(x_test))[:, 1])

    if auc < 0.55:
        verdict = "OK - background carries no label information"
    elif auc < 0.65:
        verdict = "CAUTION - mild background leakage; note it in the write-up"
    else:
        verdict = "WARNING - positives and negatives differ by filming condition"

    print(f"[diag-B] background-only AUC = {auc:.4f}")
    print(f"[diag-B] verdict: {verdict}")

    out_dir = os.path.join(CONTACT_ROOT, cfg["output"]["diag_dir"])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "background_leak.json"), "w") as f:
        json.dump({"auc": auc, "verdict": verdict, "eval_split": eval_split,
                   "n_train": int(len(y_train)), "n_eval": int(len(y_test))}, f, indent=2)
    print(f"[diag-B] wrote {os.path.join(out_dir, 'background_leak.json')}")


if __name__ == "__main__":
    main()
