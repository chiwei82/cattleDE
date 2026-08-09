"""Visualise and measure where SAM cannot decide which animal owns a pixel.

Usage (from the repository root):

    python -m contactTest.visualize_sam_confusion --limit 12
    python -m contactTest.visualize_sam_confusion --split val --limit 30 --no-images

Writes to contactTest/log/sam_confusion/<split>/ only.

The idea being tested: when two animals are genuinely touching, a segmenter has
no evidence for where one ends and the other begins, so its decision boundary
there is uncertain. That uncertainty is a signal for contact, and unlike the
pose route it needs no anatomical model.

SAM's mask decoder emits a per-pixel logit; thresholding at 0 gives the binary
mask. A logit near 0 means "unsure". Taken alone that fires on every boundary
including animal-against-floor, so this script prompts SAM once per animal and
multiplies the two uncertainties:

    u_x(p)       = 4 * sigmoid(logit_x(p)) * (1 - sigmoid(logit_x(p)))
    confusion(p) = u_i(p) * u_j(p)

At an animal/floor edge only one of the two is unsure — the other confidently
says "not mine" — so the product stays low. It is high only where BOTH
segmentations are undecided, which is where the two bodies meet.

Also reported is the plain mask overlap (both masks claiming the same pixel),
which is the same idea with the logits thrown away.

Per-pixel logits need the reference segment-anything package (return_logits).
Ultralytics returns binary masks only, so under it the confusion map is
unavailable and just the overlap is reported.
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contactTest.src.data import load_records, relative_boxes, split_records
from contactTest.src.utils import load_config, roc_auc

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))
C_I, C_J = (214, 120, 42), (52, 104, 235)          # RGB, cow i / cow j


class SamLogits:
    """Box-prompted SAM returning per-pixel logits, not just binary masks."""

    def __init__(self, weights, model_type="vit_b"):
        import torch
        from segment_anything import SamPredictor, sam_model_registry

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[model_type](checkpoint=weights).to(device)
        self.predictor = SamPredictor(sam)
        print(f"[sam] segment-anything {model_type} on {device}")

    def __call__(self, bgr, boxes):
        """Returns a list of full-resolution logit maps, one per box."""
        self.predictor.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        out = []
        for b in boxes:
            logits, _, _ = self.predictor.predict(
                box=np.asarray(b, dtype=np.float32)[None],
                multimask_output=False, return_logits=True)
            out.append(logits[0].astype(np.float32))
        return out


def uncertainty(logit):
    """Bernoulli variance, rescaled to peak at 1 where the logit crosses 0."""
    p = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
    return (4.0 * p * (1.0 - p)).astype(np.float32)


def confusion_maps(logit_i, logit_j):
    """Two readings of the same uncertainty, at different strictness.

    strict = u_i * u_j
        Fires only where BOTH segmentations are undecided about the same pixel,
        so an animal/floor edge — where one is unsure and the other confidently
        says "not mine" — is suppressed. Precise, but it can also suppress a
        genuine contact whenever one of the two masks happens to be confident.

    loose  = max(u_i, u_j)
        Fires wherever EITHER is undecided, so it also lights up floor edges and
        occlusions. Use when recall matters more than precision: the question is
        whether anything fires at a real contact at all.
    """
    ui, uj = uncertainty(logit_i), uncertainty(logit_j)
    return ui * uj, np.maximum(ui, uj)


def colourise(gray01, rgb):
    """Tint a [0,1] map with a single hue, for overlaying."""
    return (gray01[..., None] * np.asarray(rgb, np.float32)).astype(np.uint8)


def _heat(img_rgb, m):
    heat = cv2.cvtColor(cv2.applyColorMap((np.clip(m, 0, 1) * 255).astype(np.uint8),
                                          cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    w = np.clip(m, 0, 1)[..., None] * 0.85
    return (img_rgb * (1 - w) + heat * w).astype(np.uint8)


def panel(img_rgb, mi, mj, strict, loose, overlap, boxes, band):
    """One row: boxes | masks | overlap | strict confusion | loose confusion."""
    h, w = img_rgb.shape[:2]
    tiles = []

    a = img_rgb.copy()
    for b, c in zip(boxes, (C_I, C_J)):
        cv2.rectangle(a, (b[0], b[1]), (b[2], b[3]), c, 2)
    tiles.append(a)

    b_ = img_rgb.astype(np.float32).copy()
    for m, c in ((mi, C_I), (mj, C_J)):
        sel = m > 0
        b_[sel] = b_[sel] * 0.45 + np.asarray(c, np.float32) * 0.55
    tiles.append(b_.astype(np.uint8))

    c_ = (img_rgb.astype(np.float32) * 0.35).astype(np.uint8)
    c_[overlap > 0] = (255, 255, 255)
    tiles.append(c_)

    for m in (strict, loose):
        if m is None:
            tiles.append(np.full_like(img_rgb, 235))
            continue
        tile = _heat(img_rgb, m)
        # Outline the band the statistics are computed in, and mark the peak
        # inside it, so a fired location can be judged at a glance.
        cnt, _ = cv2.findContours(band.astype(np.uint8), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(tile, cnt, -1, (120, 255, 120), 1)
        if band.any():
            masked = np.where(band, m, -1)
            py, px = np.unravel_index(int(masked.argmax()), m.shape)
            cv2.circle(tile, (px, py), 9, (0, 0, 0), 3, lineType=cv2.LINE_AA)
            cv2.circle(tile, (px, py), 9, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        tiles.append(tile)

    gap = np.full((h, 6, 3), 250, np.uint8)
    return np.hstack([t for pair in zip(tiles, [gap] * len(tiles))
                      for t in pair][:-1])


def contact_band(mi, mj, dilate=15):
    """Where the two dilated masks meet — the only place contact can be."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1,) * 2)
    return (cv2.dilate(mi, k) > 0) & (cv2.dilate(mj, k) > 0)


def band_stats(strict, loose, overlap, band):
    """Summarise both readings inside the contact band."""
    n = int(band.sum())
    out = {"band_px": n,
           "overlap_px": int(overlap.sum()),
           "overlap_frac_of_band": float(overlap.sum() / n) if n else 0.0}
    for name, m in (("strict", strict), ("loose", loose)):
        if m is None or not n:
            out.update({f"{name}_mean": 0.0, f"{name}_max": 0.0, f"{name}_area": 0.0})
            continue
        v = m[band]
        out[f"{name}_mean"] = float(v.mean())
        out[f"{name}_max"] = float(v.max())
        out[f"{name}_area"] = float((v > 0.5).sum() / n)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(CONTACT_ROOT, "config.yaml"))
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=12, help="positives to process")
    ap.add_argument("--neg-per-pos", type=float, default=2.0)
    ap.add_argument("--model-type", default="vit_b")
    ap.add_argument("--weights", default=None, help="overrides data.sam_weights")
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    weights = args.weights or cfg["data"].get("sam_weights", "sam_b.pt")

    rows = split_records(load_records(cfg))[args.split]
    pos = [r for r in rows if r["label"] == 1][:args.limit]
    neg = [r for r in rows if r["label"] == 0]
    rng = np.random.default_rng(int(cfg["random_seed"]))
    n_neg = min(len(neg), int(round(args.neg_per_pos * len(pos))))
    records = pos + [neg[i] for i in sorted(rng.choice(len(neg), n_neg, replace=False))]
    if not records:
        raise SystemExit(f"no rows in split '{args.split}'")
    print(f"[sam] {len(records)} pairs ({len(pos)} positive, {n_neg} negative)")

    try:
        sam = SamLogits(weights, args.model_type)
        have_logits = True
    except Exception as err:                       # noqa: BLE001
        raise SystemExit(
            f"could not load segment-anything ({err}).\n"
            "Per-pixel logits need the reference package:\n"
            "  pip install git+https://github.com/facebookresearch/segment-anything.git\n"
            "  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth\n"
            "then pass --weights sam_vit_b_01ec64.pth")

    out_dir = os.path.join(CONTACT_ROOT, "log", "sam_confusion", args.split)
    os.makedirs(out_dir, exist_ok=True)
    report = []

    for i, record in enumerate(records):
        bgr = cv2.imread(record["image_path"])
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        boxes = relative_boxes(record, h, w)
        try:
            li, lj = sam(bgr, boxes)
        except Exception as err:                   # noqa: BLE001
            print(f"[sam] failed on {record['rel_image']}: {err}")
            continue

        mi, mj = (li > 0).astype(np.uint8), (lj > 0).astype(np.uint8)
        overlap = (mi & mj).astype(np.uint8)
        strict, loose = confusion_maps(li, lj) if have_logits else (None, None)
        band = contact_band(mi, mj)

        stats = band_stats(strict, loose, overlap, band)
        stats.update(rel_image=record["rel_image"], label=record["label"],
                     label_v2=record["label_v2"],
                     source_video=record["source_video"])
        report.append(stats)

        if not args.no_images:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            row = panel(rgb, mi, mj, strict, loose, overlap, boxes, band)
            tag = record["label_v2"] or ("interaction" if record["label"] else "no_interaction")
            name = (f"{'POS' if record['label'] else 'NEG'}_{i:03d}_"
                    f"{tag.replace(' ', '-')}_{os.path.basename(record['rel_image'])}")
            cv2.imwrite(os.path.join(out_dir, name), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
        if (i + 1) % 10 == 0:
            print(f"[sam] processed {i + 1}/{len(records)}")

    if not report:
        raise SystemExit("nothing processed")

    labels = np.array([r["label"] for r in report])
    print("\n[sam] 面板順序：原圖+框 | 兩個 mask | overlap(白) | "
          "strict confusion | loose confusion")
    print("[sam] 綠框 = 統計用的接觸帶，白圈 = 帶內最高點\n")

    # Recall first: on positives, does anything fire in the band at all?
    pos_rows = [r for r in report if r["label"] == 1]
    if pos_rows:
        print("在正樣本的接觸帶內，有東西亮起來的比例：")
        print(f"{'':<10}{'>0.3':>9}{'>0.5':>9}{'>0.7':>9}{'>0.9':>9}")
        for name in ("strict", "loose", "overlap_frac_of_band"):
            key = name if name.endswith("band") else f"{name}_max"
            v = np.array([r[key] for r in pos_rows], dtype=float)
            hits = "".join(f"{np.mean(v > t):>9.0%}" for t in (.3, .5, .7, .9))
            print(f"{name.replace('_frac_of_band',''):<10}{hits}")
        print()

    print(f"{'metric':<26}{'positives':>12}{'negatives':>12}{'AUC':>8}")
    summary = {}
    for key in ("overlap_frac_of_band", "strict_mean", "strict_max", "strict_area",
                "loose_mean", "loose_max", "loose_area"):
        v = np.array([r[key] for r in report], dtype=float)
        if labels.min() == labels.max():
            continue
        auc = roc_auc(labels, v)
        summary[key] = {"pos_median": float(np.median(v[labels == 1])),
                        "neg_median": float(np.median(v[labels == 0])),
                        "auc": auc}
        print(f"{key:<26}{np.median(v[labels == 1]):>12.4f}"
              f"{np.median(v[labels == 0]):>12.4f}{auc:>8.3f}")

    best = max((s["auc"] for s in summary.values()), default=float("nan"))
    if best >= 0.75:
        verdict = ("STRONG - segmentation ambiguity separates contact from proximity "
                   "well; worth feeding to the model as an input plane.")
    elif best >= 0.65:
        verdict = ("USABLE - comparable to the pose prior but needs no anatomical "
                   "model; worth an input plane and an ablation.")
    elif best >= 0.55:
        verdict = "WEAK - present but thin; do not build on it alone."
    else:
        verdict = ("NONE - SAM is equally undecided whether or not the animals "
                   "touch, so this carries no contact information.")
    print(f"\n[sam] verdict: {verdict}")

    with open(os.path.join(out_dir, "confusion_report.csv"), "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        wtr.writeheader()
        wtr.writerows(report)
    with open(os.path.join(out_dir, "confusion_summary.json"), "w") as f:
        json.dump({"split": args.split, "n": len(report),
                   "n_pos": int(labels.sum()), "metrics": summary,
                   "verdict": verdict}, f, indent=2)
    print(f"[sam] wrote {out_dir}")


if __name__ == "__main__":
    main()
