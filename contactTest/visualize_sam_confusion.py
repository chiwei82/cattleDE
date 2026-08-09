"""Locate, per image, the region where SAM cannot decide which animal owns it.

Usage (from the repository root):

    python -m contactTest.visualize_sam_confusion --limit 24
    python -m contactTest.visualize_sam_confusion --split val --limit 60 --no-images

Writes to contactTest/log/sam_confusion/<split>/ only.

This is an UNSUPERVISED image measurement, not a predictor. SAM knows nothing
about interaction; prompted with a box it answers "which pixels belong to this
object", and it becomes uncertain wherever the evidence for the boundary is
weak. That uncertainty is a property of the picture alone, so the interaction
label takes no part in computing it and no part in judging it — the label rides
along only as a caption on the panels.

SAM's mask decoder emits a per-pixel logit; thresholding at 0 gives the binary
mask, and a logit near 0 means "unsure". One animal's uncertainty alone marks
every edge it has, including against the floor, so the map is formed from both
prompts at once:

    u_x(p) = 4 * sigmoid(logit_x(p)) * (1 - sigmoid(logit_x(p)))

    strict(p) = u_i(p) * u_j(p)      both segmentations undecided here
    loose(p)  = max(u_i(p), u_j(p))  either one undecided here

`strict` keeps only pixels neither prompt can claim — mutual ambiguity, which is
what an interface between two bodies looks like to a segmenter. `loose` keeps
single-object edges too (floor, occlusion, railing); it is the reading to use
when a false alarm on the floor is acceptable and coverage matters more.

What the report answers: whether SAM yields a usable confusion region on these
images at all — is it non-empty, is it one coherent blob rather than scattered
speckle, how large is it, and where does it sit. Whether such a region coincides
with contact is a separate question this script deliberately does not ask.
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
from contactTest.src.utils import load_config

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
        occlusions. Use when a false alarm on the floor is acceptable and
        coverage of the real interface matters more.
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


def region_stats(maps, band, thresh=0.5):
    """Describe each confusion map as a REGION, without reference to any label.

    Recorded per reading: whether it produced anything at all, how big it is,
    how fragmented (a coherent interface is one or two connected components,
    speckle is many), its peak, and how much of it falls in the band where the
    two dilated masks meet — the only place an interface between the animals
    can physically be.
    """
    n_band = int(band.sum())
    out = {"band_px": n_band}
    for name, m in maps.items():
        if m is None:
            out.update({f"{name}_nonempty": 0, f"{name}_area_px": 0,
                        f"{name}_components": 0, f"{name}_max": 0.0,
                        f"{name}_frac_in_band": 0.0})
            continue
        binary = (m > thresh).astype(np.uint8)
        area = int(binary.sum())
        n_comp, _ = cv2.connectedComponents(binary)
        out[f"{name}_nonempty"] = int(area > 0)
        out[f"{name}_area_px"] = area
        out[f"{name}_components"] = max(n_comp - 1, 0)      # label 0 is background
        out[f"{name}_max"] = float(m.max())
        out[f"{name}_frac_in_band"] = float((binary & band).sum() / area) if area else 0.0
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

        stats = region_stats({"strict": strict, "loose": loose,
                              "overlap": overlap.astype(np.float32)}, band)
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

    print("\n[sam] 面板順序：原圖+框 | 兩個 mask | overlap(白) | "
          "strict confusion | loose confusion")
    print("[sam] 綠框 = 兩個膨脹 mask 的交會帶，白圈 = 帶內最高點\n")

    # Everything below describes the maps themselves. The interaction label is
    # NOT used: SAM's uncertainty is a property of the image, so asking whether
    # it separates interacting from non-interacting pairs would impose on SAM a
    # semantics it does not have.
    summary = {}
    print(f"{'reading':<10}{'非空比例':>12}{'區域大小':>13}{'連通塊數':>12}{'峰值':>10}")
    for name in ("strict", "loose", "overlap"):
        nonempty = np.array([r[f"{name}_nonempty"] for r in report], float)
        area = np.array([r[f"{name}_area_px"] for r in report], float)
        comps = np.array([r[f"{name}_components"] for r in report], float)
        peak = np.array([r[f"{name}_max"] for r in report], float)
        inband = np.array([r[f"{name}_frac_in_band"] for r in report], float)
        summary[name] = {"nonempty_frac": float(nonempty.mean()),
                         "area_px_median": float(np.median(area)),
                         "components_median": float(np.median(comps)),
                         "peak_median": float(np.median(peak)),
                         "frac_in_band_median": float(np.median(inband))}
        print(f"{name:<10}{nonempty.mean():>11.0%}{np.median(area):>10.0f} px"
              f"{np.median(comps):>12.1f}{np.median(peak):>10.3f}")
    print("\n（中位數；區域 = 該讀法 > 0.5 的像素）")

    print(f"\n{'reading':<10}{'區域落在交會帶內的比例':>24}")
    for name in ("strict", "loose", "overlap"):
        print(f"{name:<10}{summary[name]['frac_in_band_median']:>22.0%}")

    s_ok, s_clean = summary["strict"]["nonempty_frac"], summary["strict"]["components_median"]
    if s_ok >= 0.7 and s_clean <= 3:
        verdict = ("USABLE - SAM yields a non-empty, coherent mutual-ambiguity "
                   "region on most images; cache it as an input plane.")
    elif s_ok >= 0.7:
        verdict = ("NOISY - regions exist but are fragmented; smooth them or keep "
                   "the largest component before use.")
    elif summary["loose"]["nonempty_frac"] >= 0.7:
        verdict = ("STRICT TOO TIGHT - mutual ambiguity is rare, so SAM is "
                   "confident about the boundary between the two animals. Use the "
                   "loose reading, or prompt with points to force a split.")
    else:
        verdict = ("NO SIGNAL - SAM is confident almost everywhere, most likely "
                   "because it returned one animal, or both as a single object, "
                   "rather than an uncertain boundary. Check the mask panel.")
    print(f"\n[sam] verdict: {verdict}")

    with open(os.path.join(out_dir, "confusion_report.csv"), "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        wtr.writeheader()
        wtr.writerows(report)
    with open(os.path.join(out_dir, "confusion_summary.json"), "w") as f:
        json.dump({"split": args.split, "n": len(report),
                   "readings": summary, "verdict": verdict}, f, indent=2)
    print(f"[sam] wrote {out_dir}")


if __name__ == "__main__":
    main()
