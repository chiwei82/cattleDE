"""Diagnostic A — can the interaction label be predicted from box geometry alone?

Usage (from the repository root):

    python -m contactTest.diagnostics.geometry_baseline

Fits a gradient-boosted classifier on nothing but the two detector boxes and the
crop dimensions. Crops are stored un-resized, so the crop height and width are
themselves informative features and are included deliberately.

Interpretation of the reported test AUC:

    < 0.70   appearance is genuinely required; proceed
    0.70-0.85 geometry carries real signal; keep this baseline in the write-up
              as the number the contact model has to beat
    > 0.90   the task is solvable from box layout alone. A heatmap model will
              have no incentive to look at the pixels, so localisation will be
              meaningless. Fix the sampling of positives/negatives first.

This is a gate, not a metric. Run it before spending GPU time on training.
"""

import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from contactTest.src.data import load_records, relative_boxes, split_records
from contactTest.src.utils import load_config, roc_auc

CONTACT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FEATURE_NAMES = [
    "dx_norm", "dy_norm", "centre_dist_norm",
    "aspect_1", "aspect_2", "size_ratio",
    "iou", "inter_over_crop",
    "crop_h", "crop_w", "crop_aspect", "log_crop_area",
]


def features(record):
    """Geometry-only feature vector, all derived from the CSV (no image read)."""
    merged = record["merged"]
    ox, oy = max(0, int(merged[0])), max(0, int(merged[1]))
    crop_w = max(1, int(merged[2]) - ox)
    crop_h = max(1, int(merged[3]) - oy)
    (x1, y1, x2, y2), (u1, v1, u2, v2) = relative_boxes(record, crop_h, crop_w)

    w1, h1, w2, h2 = x2 - x1, y2 - y1, u2 - u1, v2 - v1
    s1 = np.sqrt(max(w1 * h1, 1))
    s2 = np.sqrt(max(w2 * h2, 1))
    dx = ((x1 + x2) - (u1 + u2)) / 2.0 / s1
    dy = ((y1 + y2) - (v1 + v2)) / 2.0 / s1

    iw = max(0, min(x2, u2) - max(x1, u1))
    ih = max(0, min(y2, v2) - max(y1, v1))
    inter = iw * ih
    union = w1 * h1 + w2 * h2 - inter

    return [
        dx, dy, float(np.hypot(dx, dy)),
        w1 / max(h1, 1), w2 / max(h2, 1), s1 / max(s2, 1e-6),
        inter / max(union, 1e-6), inter / float(crop_h * crop_w),
        float(crop_h), float(crop_w), crop_w / float(crop_h),
        float(np.log(crop_h * crop_w)),
    ]


def build(records):
    x = np.array([features(r) for r in records], dtype=np.float64)
    y = np.array([r["label"] for r in records], dtype=np.int64)
    return x, y


def main():
    cfg = load_config(os.path.join(CONTACT_ROOT, "config.yaml"))
    buckets = split_records(load_records(cfg))

    x_train, y_train = build(buckets["train"])
    x_test, y_test = build(buckets["test"] or buckets["val"])
    split_used = "test" if buckets["test"] else "val"
    print(f"[diag-A] train {len(y_train)} rows ({y_train.sum()} pos) | "
          f"{split_used} {len(y_test)} rows ({y_test.sum()} pos)")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance

    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05,
        class_weight="balanced", random_state=int(cfg["random_seed"]))
    clf.fit(x_train, y_train)
    scores = clf.predict_proba(x_test)[:, 1]
    auc = roc_auc(y_test, scores)

    if auc < 0.70:
        verdict = "OK - appearance is required; proceed to training"
    elif auc < 0.85:
        verdict = "CAUTION - geometry carries real signal; report this as a baseline"
    elif auc < 0.90:
        verdict = "WARNING - geometry explains most of the label"
    else:
        verdict = "STOP - the label is solvable from box layout alone; fix sampling first"

    print(f"[diag-A] geometry-only AUC = {auc:.4f}")
    print(f"[diag-A] verdict: {verdict}")

    imp = permutation_importance(clf, x_test, y_test, n_repeats=10,
                                 random_state=int(cfg["random_seed"]), scoring="roc_auc")
    order = np.argsort(-imp.importances_mean)
    print("[diag-A] most informative geometry features:")
    for i in order[:6]:
        print(f"           {FEATURE_NAMES[i]:<18s} {imp.importances_mean[i]:+.4f}")

    out_dir = os.path.join(CONTACT_ROOT, cfg["output"]["diag_dir"])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "geometry_baseline.json"), "w") as f:
        json.dump({
            "auc": auc, "verdict": verdict, "eval_split": split_used,
            "n_train": int(len(y_train)), "n_eval": int(len(y_test)),
            "importance": {FEATURE_NAMES[i]: float(imp.importances_mean[i])
                           for i in range(len(FEATURE_NAMES))},
        }, f, indent=2)
    print(f"[diag-A] wrote {os.path.join(out_dir, 'geometry_baseline.json')}")


if __name__ == "__main__":
    main()
