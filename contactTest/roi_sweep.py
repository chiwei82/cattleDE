"""Recall against area for every candidate ROI, as curves rather than one point.

Usage (from the repository root):

    python -m contactTest.roi_sweep --split known_interact
    python -m contactTest.roi_sweep --split all --source cache
    python -m contactTest.roi_sweep --split all --baselines-only    # no GPU

Writes contactTest/log/roi_sweep/<split>/ : one CSV of every (family, parameter)
point and one PNG of the curves.

WHY A CURVE

This is a PROPOSAL task, not a prediction task. The question is whether a region
can be proposed that carries the contact in less area than simply taking the
detector's boxes. Recall and area are then two ends of one trade-off, not two
independent scores, and a single operating point cannot express a trade-off —
`dilate_px = 22` picks one point on a curve nobody has drawn.

Drawing it also settles what r = 22 is worth. If the SAM 3 curve lies above the
rectangle curves along its whole length, the conclusion does not depend on that
constant; if the curves cross, the constant is doing the work and has to be
justified.

THE FAMILIES, AND WHAT EACH ONE WOULD PROVE

    merged box        the whole crop. What the pipeline hands downstream today,
                      so it is the incumbent: recall 1.0 at area 1.0 by
                      definition, and every other row is measured against it.

    box intersection  where the two detector boxes overlap, scaled about its
                      centre. Uses NO model: if this matches SAM 3 at equal
                      area, the segmentation is buying nothing.

    inscribed ellipse the same rectangle's inscribed ellipse. A rectangle has
                      corners the contact interface never reaches, so this
                      separates "a better SHAPE" from "a better place".

    midpoint disc     a disc on the line between the two box centres. The
                      crudest possible localisation.

    SAM 3 overlap     mask_i AND mask_j, no dilation. One point, no parameter.

    SAM 3 dilated     dilate(mask_i, r) AND dilate(mask_j, r), r swept. The
                      method under test.

RECALL IS OVER POINTS, AREA IS OVER IMAGES

Matching evaluate_contact, so the numbers here are readable against it:
recall = covered clicks / all clicks, a-bar = mean over images of area/crop.

Images marked "no contact" are excluded: recall is undefined without clicks, so
they can only dilute the area axis. Their cost is real and is printed
separately, not folded into the curve.

An image where the region comes out EMPTY stays in, contributing 0 recall and 0
area. In a proposal task "proposed nothing" is a failure, not a perfectly small
region, and dropping those images would flatter whichever family produced them.
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contactTest.sam_contact_region import load_masks
from contactTest.score_contact import read_gt
from contactTest.src.data import load_records, records_for, relative_boxes
from contactTest.src.utils import load_config

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))

RADII = [4, 8, 12, 16, 22, 30, 40, 55]
SCALES = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]


def scaled_rect(b, factor, h, w):
    """`b` scaled about its centre so its area is `factor` x the original."""
    cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    s = float(np.sqrt(max(factor, 0.0)))
    hw, hh = (b[2] - b[0]) / 2.0 * s, (b[3] - b[1]) / 2.0 * s
    return (max(0.0, cx - hw), max(0.0, cy - hh),
            min(float(w), cx + hw), min(float(h), cy + hh))


def rect_mask(shape, r):
    m = np.zeros(shape, bool)
    x1, y1 = int(round(r[0])), int(round(r[1]))
    x2, y2 = int(round(r[2])), int(round(r[3]))
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = True
    return m


def ellipse_mask(shape, r):
    m = np.zeros(shape, np.uint8)
    cx, cy = (r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0
    ax, ay = (r[2] - r[0]) / 2.0, (r[3] - r[1]) / 2.0
    if ax >= 1 and ay >= 1:
        cv2.ellipse(m, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, 1, -1)
    return m.astype(bool)


def disc_mask(shape, c, radius):
    m = np.zeros(shape, np.uint8)
    if radius >= 1:
        cv2.circle(m, (int(c[0]), int(c[1])), int(radius), 1, -1)
    return m.astype(bool)


def band(mi, mj, r):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
    return (cv2.dilate(mi, k) > 0) & (cv2.dilate(mj, k) > 0)


def covered(region, points):
    h, w = region.shape
    n = 0
    for (x, y) in points:
        xx = int(np.clip(x, 0, w - 1)); yy = int(np.clip(y, 0, h - 1))
        n += bool(region[yy, xx])
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(CONTACT_ROOT, "config.yaml"))
    ap.add_argument("--split", default="all",
                    choices=["train", "val", "test", "all", "known_interact"])
    ap.add_argument("--source", default="sam3_text", choices=["sam3_text", "cache"])
    ap.add_argument("--baselines-only", action="store_true",
                    help="skip every SAM 3 family; needs no model and no GPU")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--text", default="cow")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--radii", default=",".join(str(r) for r in RADII))
    ap.add_argument("--scales", default=",".join(str(s) for s in SCALES))
    ap.add_argument("--gt", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    radii = [int(v) for v in args.radii.split(",") if v.strip()]
    scales = [float(v) for v in args.scales.split(",") if v.strip()]

    cfg = load_config(args.config)
    if args.conf is None:
        args.conf = float(cfg["data"].get("sam3_conf", 0.6))
    gt_path = args.gt or os.path.join(CONTACT_ROOT, "log", "annotate", args.split,
                                      "contact_gt.csv")
    if not os.path.exists(gt_path):
        raise SystemExit(f"no ground truth at {gt_path}")
    gt = read_gt(gt_path)
    by_rel = {r["rel_image"]: r for r in
              records_for(load_records(cfg, require_label=False), args.split)}

    seg = None
    if not args.baselines_only and args.source == "sam3_text":
        from contactTest.sam3 import Sam3
        seg = Sam3(args.weights, args.text, args.conf)

    # (family, param) -> [covered points, total points, sum of area fractions]
    acc, n_img = {}, 0
    no_contact_area, n_no_contact, no_mask = [], 0, 0

    for rel, ann in gt.items():
        if ann["status"] == "skip":
            continue
        rec = by_rel.get(rel)
        if rec is None:
            continue
        bgr = cv2.imread(rec["image_path"])
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        shape = (h, w)
        crop_px = float(h * w)
        b1, b2 = relative_boxes(rec, h, w)
        pts = ann["points"]

        masks = None
        if not args.baselines_only:
            try:
                masks = (load_masks(rec, shape) if seg is None
                         else seg.assign_to_boxes(bgr, [b1, b2]))
            except Exception as err:                 # noqa: BLE001
                print(f"[roi] SAM 3 failed on {rel}: {err}")
                masks = None
            if masks is None or len(masks) < 2:
                no_mask += 1
                # Not skipped: a proposal method that produces nothing on an
                # image has failed there, and excluding it would hide that.
                masks = [np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)]
            masks = [np.asarray(m).astype(np.uint8) for m in masks]

        regions = {}
        it = (max(b1[0], b2[0]), max(b1[1], b2[1]),
              min(b1[2], b2[2]), min(b1[3], b2[3]))
        mid = ((b1[0] + b1[2] + b2[0] + b2[2]) / 4.0,
               (b1[1] + b1[3] + b2[1] + b2[3]) / 4.0)
        regions[("merged box", 1.0)] = np.ones(shape, bool)
        for s in scales:
            r = scaled_rect(it, s, h, w)
            regions[("box intersection", s)] = rect_mask(shape, r)
            regions[("inscribed ellipse", s)] = ellipse_mask(shape, r)
        for r in radii:
            regions[("midpoint disc", float(r))] = disc_mask(shape, mid, r)
        if masks is not None:
            regions[("SAM 3 overlap", 0.0)] = (masks[0] > 0) & (masks[1] > 0)
            for r in radii:
                regions[("SAM 3 dilated", float(r))] = band(masks[0], masks[1], r)

        if not pts:
            n_no_contact += 1
            no_contact_area.append(
                {k: float(v.sum()) / crop_px for k, v in regions.items()})
            continue

        n_img += 1
        for k, reg in regions.items():
            c, t, a = acc.get(k, (0, 0, 0.0))
            acc[k] = (c + covered(reg, pts), t + len(pts),
                      a + float(reg.sum()) / crop_px)
        if args.limit and n_img >= args.limit:
            break

    if not acc:
        raise SystemExit("nothing scored")

    rows = []
    for (fam, param), (c, t, a) in acc.items():
        a_bar = a / n_img
        rows.append({"family": fam, "param": param, "recall": c / t,
                     "a_bar": a_bar, "recall_per_area": (c / t) / a_bar
                     if a_bar > 0 else float("nan"),
                     "covered": c, "points": t, "n_images": n_img})
    rows.sort(key=lambda r: (r["family"], r["param"]))

    print(f"\n[roi] {n_img} images with clicks, {rows[0]['points']} clicks, "
          f"{n_no_contact} images marked 'no contact' (excluded from the curve)")
    if no_mask:
        print(f"[roi] SAM 3 produced no usable pair on {no_mask} image(s); "
              "scored as an empty region, not skipped")
    fam_order = ["merged box", "box intersection", "inscribed ellipse",
                 "midpoint disc", "SAM 3 overlap", "SAM 3 dilated"]
    print(f"\n{'family':<20}{'param':>7}{'a-bar':>9}{'recall':>9}{'recall/area':>13}")
    for fam in fam_order:
        for r in [x for x in rows if x["family"] == fam]:
            star = "  <-" if fam == "SAM 3 dilated" and r["param"] == 22.0 else ""
            print(f"{fam:<20}{r['param']:>7.2f}{r['a_bar']:>9.4f}"
                  f"{r['recall']:>9.3f}{r['recall_per_area']:>13.1f}{star}")

    if no_contact_area:
        print(f"\n[roi] area proposed on the {n_no_contact} 'no contact' images, "
              "which the curve above excludes:")
        for fam in fam_order:
            v = [d[k] for d in no_contact_area for k in d if k[0] == fam]
            if v:
                print(f"    {fam:<20} mean a_i {float(np.mean(v)):.4f}")

    out_dir = os.path.join(CONTACT_ROOT, "log", "roi_sweep", args.split)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "roi_sweep.csv")
    with open(path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"\n[roi] wrote {path}")

    if args.no_plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    style = {"merged box": ("0.35", "s"), "box intersection": ((0.85,0.42,0.16), "o"),
             "inscribed ellipse": ((0.90,0.62,0.20), "^"),
             "midpoint disc": ((0.55,0.55,0.55), "v"),
             "SAM 3 overlap": ((0.20,0.47,0.75), "D"),
             "SAM 3 dilated": ((0.15,0.40,0.70), "o")}
    for fam in fam_order:
        r = sorted([x for x in rows if x["family"] == fam], key=lambda x: x["a_bar"])
        if not r:
            continue
        col, mk = style[fam]
        ax.plot([x["a_bar"] for x in r], [x["recall"] for x in r],
                marker=mk, color=col, lw=1.6 if len(r) > 1 else 0,
                ms=6, label=fam, alpha=0.9)
    star = [x for x in rows if x["family"] == "SAM 3 dilated" and x["param"] == 22.0]
    if star:
        ax.scatter([star[0]["a_bar"]], [star[0]["recall"]], s=160,
                   facecolors="none", edgecolors="black", linewidths=1.6, zorder=5)
        ax.annotate("r = 22", (star[0]["a_bar"], star[0]["recall"]),
                    textcoords="offset points", xytext=(10, -12), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("a-bar — mean proposed area as a share of the crop (log)")
    ax.set_ylabel("recall — share of clicked contact points inside the region")
    ax.set_title(f"ROI proposal: recall against area — {args.split} "
                 f"({n_img} images, {rows[0]['points']} clicks)", fontsize=11)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8.5, loc="lower right")
    png = os.path.join(out_dir, "roi_sweep.png")
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    print(f"[roi] wrote {png}")
    print("[roi] a family whose curve lies ABOVE another reaches the same recall "
          "in less area; where curves cross, the choice of parameter decides "
          "which is better and has to be argued")


if __name__ == "__main__":
    main()
