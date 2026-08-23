
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contactTest.roi_sweep import (band, disc_mask, ellipse_mask, rect_mask,
                                   scaled_rect, FAM_ORDER, MAPPING)
from contactTest.sam_contact_region import load_masks
from contactTest.score_contact import read_gt
from contactTest.src.data import load_records, records_for, relative_boxes
from contactTest.src.utils import load_config

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))

C_HIT, C_MISS = (80, 240, 110), (245, 85, 75)

COLOUR = {
    "merged box":        (220, 50, 50),
    "box intersection":  (217, 107, 41),
    "inscribed ellipse": (230, 158, 51),
    "midpoint disc":     (140, 140, 140),
    "SAM 3 overlap":     (51, 120, 191),
    "SAM 3 dilated":     (38, 102, 178),
}
PARAM_UNIT = {"box intersection": "s", "inscribed ellipse": "s",
              "midpoint disc": "r", "SAM 3 dilated": "r"}


def read_sweep(path, coverage_key=None):
    import csv
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    key = coverage_key or ("point_coverage" if "point_coverage" in rows[0]
                           else "recall")
    out = {}
    for r in rows:
        out.setdefault(r["family"], []).append(
            (float(r["param"]), float(r[key])))
    for v in out.values():
        v.sort()
    return {k: (np.array([p for p, _ in v]), np.array([c for _, c in v]))
            for k, v in out.items()}


def param_for(curve, target):
    p, c = curve
    if c.max() < target:
        return float(p[-1]), False
    k = int(np.argmax(c >= target))
    if k == 0:
        return float(p[0]), True
    lo, hi = c[k - 1], c[k]
    frac = 0.0 if hi == lo else (target - lo) / (hi - lo)
    return float(p[k - 1] + frac * (p[k] - p[k - 1])), True


def overlay(bgr, region, points, colour):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    sel = region.astype(bool)
    rgb[sel] = rgb[sel] * 0.55 + np.asarray(colour, np.float32) * 0.45
    out = rgb.astype(np.uint8)
    cnt, _ = cv2.findContours(sel.astype(np.uint8), cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnt, -1, colour, 2, cv2.LINE_AA)
    for (x, y) in points:
        inside = sel[int(np.clip(y, 0, sel.shape[0] - 1)),
                     int(np.clip(x, 0, sel.shape[1] - 1))]
        cv2.circle(out, (int(x), int(y)), 8, (15, 15, 15), -1, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), 7, C_HIT if inside else C_MISS,
                   -1 if inside else 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(CONTACT_ROOT, "config.yaml"))
    ap.add_argument("--split", default="known_interact",
                    choices=["train", "val", "test", "all", "known_interact"])
    ap.add_argument("--image", required=True)
    ap.add_argument("--target", type=float, default=0.8)
    ap.add_argument("--sam3-r", type=float, default=22.0)
    ap.add_argument("--sweep-csv", default=None)
    ap.add_argument("--source", default="sam3_text", choices=["sam3_text", "cache"])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--text", default="cow")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.conf is None:
        args.conf = float(cfg["data"].get("sam3_conf", 0.6))

    candidates = ([args.sweep_csv] if args.sweep_csv else [
        os.path.join(CONTACT_ROOT, "log", "roi_sweep", args.split,
                     "roi_sweep_dense.csv"),
        os.path.join(CONTACT_ROOT, "log", "roi_sweep", args.split,
                     "roi_sweep.csv"),
        os.path.join(CONTACT_ROOT, f"roi_sweep_{args.split}.csv")])
    found = [c for c in candidates if c and os.path.exists(c)]
    if not found:
        raise SystemExit("no sweep curve found; run roi_sweep.py first — the "
                         "parameters are read off it. Looked at:\n    "
                         + "\n    ".join(candidates))
    full = [c for c in found if "SAM 3 dilated" in read_sweep(c)]
    sweep_path = (full or found)[0]
    curves = read_sweep(sweep_path)
    print(f"[fam] parameters read from {sweep_path}")
    if "SAM 3 dilated" not in curves:
        print("[fam] WARNING: that curve has no SAM 3 family — it was a "
              "--baselines-only sweep, so the SAM 3 dilated panel is missing")

    gt = read_gt(os.path.join(CONTACT_ROOT, "log", "annotate", args.split,
                              "contact_gt.csv"))
    hits = [k for k in gt if args.image in k]
    if len(hits) != 1:
        raise SystemExit(f"--image {args.image!r} matched {len(hits)} crops"
                         + ("" if hits else "") + "".join(f"\n    {h}" for h in hits[:8]))
    rel = hits[0]
    ann = gt[rel]
    rec = {r["rel_image"]: r for r in
           records_for(load_records(cfg, require_label=False), args.split)}.get(rel)
    if rec is None:
        raise SystemExit(f"{rel} is not in split '{args.split}'")

    bgr = cv2.imread(rec["image_path"])
    if bgr is None:
        raise SystemExit(f"cannot read {rec['image_path']}")
    h, w = bgr.shape[:2]
    shape = (h, w)
    b1, b2 = relative_boxes(rec, h, w)
    pts = ann["points"]
    print(f"[fam] {rel}\n[fam] crop {w}x{h}, {len(pts)} clicked point(s), "
          f"status {ann['status'] or 'contact'!r}")

    masks = None
    if args.source == "cache":
        masks = load_masks(rec, shape)
    else:
        from contactTest.sam3 import Sam3
        masks = Sam3(args.weights, args.text, args.conf).assign_to_boxes(bgr, [b1, b2])
    if masks is None or len(masks) < 2:
        print("[fam] WARNING: no usable mask pair; the SAM 3 panels will be empty")
        masks = [np.zeros(shape, np.uint8)] * 2
    masks = [np.asarray(m).astype(np.uint8) for m in masks]

    it = (max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3]))
    mid = ((b1[0] + b1[2] + b2[0] + b2[2]) / 4.0,
           (b1[1] + b1[3] + b2[1] + b2[3]) / 4.0)

    panels = []
    for fam in FAM_ORDER:
        note = ""
        if fam == "merged box":
            region, label = np.ones(shape, bool), "the crop itself"
        elif fam == "SAM 3 overlap":
            region = (masks[0] > 0) & (masks[1] > 0)
            label = "r = 0"
        else:
            if fam not in curves:
                print(f"[fam] {fam}: not in the sweep CSV, skipped")
                continue
            if fam == "SAM 3 dilated":
                p, reached = args.sam3_r, True
            else:
                p, reached = param_for(curves[fam], args.target)
            unit = PARAM_UNIT[fam]
            if fam in ("box intersection", "inscribed ellipse"):
                region = (rect_mask if fam == "box intersection" else ellipse_mask)(
                    shape, scaled_rect(it, p, h, w))
                label = f"{unit} = {p:.3f}"
            elif fam == "midpoint disc":
                region = disc_mask(shape, mid, int(round(p)))
                label = f"{unit} = {int(round(p))} px"
            else:
                region = band(masks[0], masks[1], max(1, int(round(p))))
                label = f"{unit} = {int(round(p))} px"
            if not reached:
                note = " — never reaches the target; drawn at its maximum"
        cov = sum(1 for (x, y) in pts
                  if region[int(np.clip(y, 0, h - 1)), int(np.clip(x, 0, w - 1))])
        panels.append({
            "family": fam, "label": label, "note": note, "region": region,
            "a_i": float(region.sum()) / float(h * w),
            "cov": cov, "n": len(pts),
        })

    print(f"\n{'family':<20}{'parameter':>16}{'a_i here':>10}"
          f"{'covered here':>14}")
    for p in panels:
        frac = f"{p['cov']}/{p['n']}"
        print(f"{p['family']:<20}{p['label']:>16}{p['a_i']:>10.4f}{frac:>14}"
              + p["note"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, p in zip(axes, panels):
        ax.imshow(overlay(bgr, p["region"], pts, COLOUR[p["family"]]))
        ax.set_xlabel(f"{p['family']}"
                      ,fontsize=10.5, labelpad=8
                      )
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    out = args.out or os.path.join(CONTACT_ROOT, "log", "roi_families", args.split,
                                   f"{os.path.splitext(os.path.basename(rel))[0]}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\n[fam] wrote {out}")


if __name__ == "__main__":
    main()
