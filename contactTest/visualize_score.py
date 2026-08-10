"""Draw the scored band together with the clicked ground truth.

Usage (from the repository root):

    python -m contactTest.visualize_score --split train --dilate-px 22
    python -m contactTest.visualize_score --split train --dilate-px 22 --reading gap

Writes to contactTest/log/score_vis/<split>/ only.

The numbers from score_contact say 77% of clicks land inside the band at
dilate_px 22; this shows which 77%. Files are named by hit rate, so the crops
the band gets wrong sort to the top and can be looked at rather than guessed at.

The region is drawn FILLED, not only outlined. Panel 3 of visualize_sam_confusion
outlines it with RETR_EXTERNAL, which does not draw holes — a pixel inside a hole
looks enclosed but scores as a miss. Here what is shaded is exactly the array the
scorer indexes, so a point that reads as a miss can be seen to be one.

Every miss is drawn with a line to its nearest band pixel: that segment is the
"miss dist" column, and it separates a click a few pixels outside the edge from
one on the wrong animal entirely.
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contactTest.sam_contact_region import contact_readings, load_masks
from contactTest.score_contact import read_gt
from contactTest.src.data import load_records, relative_boxes, split_records
from contactTest.src.utils import load_config

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))

C_HIT = (90, 220, 110)      # RGB
C_MISS = (235, 90, 80)
C_BAND = (120, 235, 130)
C_I, C_J = (214, 120, 42), (52, 104, 235)


def render(bgr, boxes, region, points, mi, mj):
    """Crop, the two masks, the scored region, and each click marked hit or miss."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    for m, c in ((mi, C_I), (mj, C_J)):
        sel = m > 0
        rgb[sel] = rgb[sel] * 0.82 + np.asarray(c, np.float32) * 0.18
    rgb[region] = rgb[region] * 0.55 + np.asarray(C_BAND, np.float32) * 0.45
    out = rgb.astype(np.uint8)

    cnt, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_CCOMP,
                              cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnt, -1, C_BAND, 2, lineType=cv2.LINE_AA)

    dt = (cv2.distanceTransform((~region).astype(np.uint8), cv2.DIST_L2, 3)
          if region.any() else None)
    h, w = region.shape
    hits = 0
    for (x, y) in points:
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        inside = bool(region[y, x])
        hits += inside
        if not inside and dt is not None:
            # Straight to the nearest band pixel, found by walking the distance
            # transform's gradient - the drawn segment is the scored distance.
            near, best = None, 1e9
            ys, xs = np.nonzero(region)
            if len(xs):
                k = np.argmin((xs - x) ** 2 + (ys - y) ** 2)
                near = (int(xs[k]), int(ys[k]))
            if near:
                cv2.line(out, (x, y), near, (25, 25, 25), 3, cv2.LINE_AA)
                cv2.line(out, (x, y), near, C_MISS, 1, cv2.LINE_AA)
        c = C_HIT if inside else C_MISS
        cv2.circle(out, (x, y), 6, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), 5, c, -1, cv2.LINE_AA)
    return out, hits


def banner(img, lines, height=44):
    bar = np.full((height, img.shape[1], 3), 245, np.uint8)
    for i, (text, col) in enumerate(lines):
        cv2.putText(bar, text, (8, 18 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    col, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(CONTACT_ROOT, "config.yaml"))
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--reading", default="dilated",
                    choices=["overlap", "gap", "surface", "dilated"])
    ap.add_argument("--dilate-px", type=int, default=22)
    ap.add_argument("--touch-px", type=int, default=10)
    ap.add_argument("--strip-px", type=int, default=6)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--gt", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    gt_path = args.gt or os.path.join(CONTACT_ROOT, "log", "annotate", args.split,
                                      "contact_gt.csv")
    if not os.path.exists(gt_path):
        raise SystemExit(f"no ground truth at {gt_path}")
    gt = read_gt(gt_path)
    by_rel = {r["rel_image"]: r
              for r in split_records(load_records(cfg, require_label=False))[args.split]}

    out_dir = os.path.join(CONTACT_ROOT, "log", "score_vis", args.split)
    os.makedirs(out_dir, exist_ok=True)

    made, no_mask, total_hits, total_pts = [], 0, 0, 0
    none_fired = none_total = 0

    for rel, ann in gt.items():
        if ann["status"] == "skip":
            continue
        record = by_rel.get(rel)
        if record is None:
            continue
        bgr = cv2.imread(record["image_path"])
        if bgr is None:
            continue
        masks = load_masks(record, bgr.shape[:2])
        if masks is None:
            no_mask += 1
            continue
        mi, mj = masks
        region = contact_readings(mi, mj, args.touch_px, args.dilate_px,
                                  args.strip_px)[args.reading]

        if ann["status"] == "none":
            none_total += 1
            none_fired += int(region.any())
            img, _ = render(bgr, relative_boxes(record, *bgr.shape[:2]),
                            region, [], mi, mj)
            frac = region.sum() / float(bgr.shape[0] * bgr.shape[1])
            made.append((-1.0, banner(img, [
                ("MARKED 'NO CONTACT' - control", (150, 30, 30)),
                (f"band still covers {region.sum()} px ({frac:.1%} of the crop)",
                 (80, 80, 80))]), f"none_{os.path.basename(rel)}"))
            continue

        if not ann["points"]:
            continue
        img, hits = render(bgr, relative_boxes(record, *bgr.shape[:2]),
                           region, ann["points"], mi, mj)
        n = len(ann["points"])
        total_hits += hits
        total_pts += n
        rate = hits / n
        frac = region.sum() / float(bgr.shape[0] * bgr.shape[1])
        img = banner(img, [
            (f"{hits}/{n} clicks inside  ({rate:.0%})", (25, 110, 40) if rate >= .8
             else ((170, 100, 20) if rate >= .4 else (170, 40, 30))),
            (f"{args.reading} @ dilate={args.dilate_px}   "
             f"{region.sum()} px ({frac:.1%} of the crop)", (80, 80, 80))])
        made.append((rate, img, os.path.basename(rel)))

    if not made:
        raise SystemExit("nothing rendered - run precompute_masks.py first")

    # Worst first: the point of looking is to see what is being got wrong.
    made.sort(key=lambda t: t[0])
    for i, (rate, img, name) in enumerate(made[:args.limit]):
        tag = "none" if rate < 0 else f"{int(round(rate * 100)):03d}"
        cv2.imwrite(os.path.join(out_dir, f"{tag}_{i:03d}_{name}"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"\n[vis] {args.reading} at dilate_px={args.dilate_px}")
    if no_mask:
        print(f"[vis] {no_mask} crops skipped for want of a cached mask")
    print(f"[vis] {total_hits}/{total_pts} clicks inside the region "
          f"({total_hits / max(total_pts, 1):.0%})")
    if none_total:
        print(f"[vis] control: the region is non-empty in {none_fired}/{none_total} "
              "crops marked 'no contact'")
    print(f"[vis] wrote {min(len(made), args.limit)} images to {out_dir}")
    print("[vis] green dot = click inside, red dot = outside with a line to the")
    print("      nearest band pixel. Files sort worst-first; 'none_*' are the")
    print("      control crops, which carry no clicks.")


if __name__ == "__main__":
    main()
