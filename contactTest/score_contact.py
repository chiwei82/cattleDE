"""Score the contact-region definitions against clicked ground truth.

Usage (from the repository root):

    python -m contactTest.score_contact --split train
    python -m contactTest.score_contact --split train --touch-px 16

Reads contactTest/log/annotate/<split>/contact_gt.csv (from annotate_contact.py)
and the SAM masks in the mask cache, then scores every reading in
sam_contact_region.py on the same rows.

WHAT IS REPORTED, AND WHY IT IS THESE

  hit rate     the share of clicked points that land inside the region
  size         the region's median area, and its share of the crop
  fire rate    the share of crops annotated "no contact" where the region is
               non-empty anyway

  Hit rate alone rewards a region that covers everything, and size alone rewards
  a region that covers nothing, so neither is reported without the other. A
  reading is good when it has a high hit rate at a small size — and the "no
  contact" control catches anything that achieves the hit rate by always firing.

  Precision, recall and IoU are reported only for polygon ground truth, which
  clicked points cannot support: a point has no area, so |band ∩ GT| / |band| is
  undefined in any useful sense. Clicks answer "is it in the right place"; area
  metrics need an area to compare against.

DISTANCE, NOT JUST HIT/MISS

  Also reported is the distance from each clicked point to the nearest pixel of
  the region. A miss by 5 px and a miss by 200 px are different failures, and a
  binary hit rate hides that. The distance is normalised by the animal's size so
  that crops taken at different ranges from the camera stay comparable.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contactTest.sam_contact_region import (DEPTH_STATS, contact_readings,
                                            depth_stats, load_depth, load_masks)
from contactTest.src.data import load_records, records_for, relative_boxes
from contactTest.src.utils import load_config

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))
ORDER = ["overlap", "gap", "surface", "dilated"]


def read_gt(path):
    """rel_image -> {"points": [(x, y), ...], "status": str}."""
    gt = defaultdict(lambda: {"points": [], "status": ""})
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rel = row["rel_image"]
            if row["status"] == "point":
                gt[rel]["points"].append((int(row["x"]), int(row["y"])))
            else:
                gt[rel]["status"] = row["status"]
    return gt


def check_gt(gt, by_rel):
    """Report what an annotation pass actually contains, before anything is run.

    Two failure modes are worth separating. Coordinates outside the crop mean the
    page recorded screen pixels rather than image pixels, which would invalidate
    every point. Coordinates inside the crop but far from either animal mean the
    clicks landed on the pen, which is a labelling problem rather than a software
    one. Neither is visible from the point count alone.
    """
    n_img = n_pts = n_none = n_skip = 0
    oob = off_animal = unknown = 0
    per_img, dists = [], []

    for rel, ann in gt.items():
        if ann["status"] == "skip":
            n_skip += 1
            continue
        if ann["status"] == "none":
            n_none += 1
            continue
        if not ann["points"]:
            continue
        n_img += 1
        n_pts += len(ann["points"])
        per_img.append(len(ann["points"]))

        record = by_rel.get(rel)
        if record is None:
            unknown += 1
            continue
        bgr = cv2.imread(record["image_path"])
        if bgr is None:
            unknown += 1
            continue
        h, w = bgr.shape[:2]
        boxes = relative_boxes(record, h, w)
        for (x, y) in ann["points"]:
            if not (0 <= x < w and 0 <= y < h):
                oob += 1
                continue
            # Distance to the nearer box, 0 when inside one of them.
            d = min(max(bx1 - x, 0, x - bx2) + max(by1 - y, 0, y - by2)
                    for (bx1, by1, bx2, by2) in boxes)
            dists.append(d)
            if d > 0:
                off_animal += 1

    print(f"\n[check] {n_img} crops with points, {n_pts} points in total")
    if per_img:
        p = np.array(per_img)
        print(f"[check] points per crop: median {np.median(p):.0f}  "
              f"max {p.max()}  (crops with exactly 1: {np.mean(p == 1):.0%})")
    print(f"[check] {n_none} marked 'no contact', {n_skip} skipped")
    if unknown:
        print(f"[check] {unknown} rows are not in this split or unreadable")

    print(f"\n[check] points outside the crop: {oob}"
          + ("   <- coordinates are wrong, the CSV cannot be used"
             if oob else "   (good - coordinates are in image pixels)"))
    if dists:
        d = np.array(dists)
        print(f"[check] points outside BOTH boxes: {off_animal} of {len(d)}"
              f"   median distance to the nearer box {np.median(d):.0f} px")
        if off_animal / len(d) > 0.2:
            print("[check] a fifth or more of the clicks are off both animals - "
                  "worth looking at before scoring against them")
    print("\n[check] no masks were needed for this; run without --check-only "
          "once precompute_masks has been run")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(CONTACT_ROOT, "config.yaml"))
    ap.add_argument("--split", default="train",
                    choices=["train", "val", "test", "all"],
                    help="all covers every session, which is what "
                         "annotate_contact now samples over. Stage 2 "
                         "fits nothing, so a held-out split protects "
                         "against nothing here")
    ap.add_argument("--touch-px", type=int, default=10)
    ap.add_argument("--dilate-px", type=int, default=15)
    ap.add_argument("--strip-px", type=int, default=6)
    ap.add_argument("--gt", default=None,
                    help="defaults to log/annotate/<split>/contact_gt.csv")
    ap.add_argument("--depth-tol", default="0.05,0.10",
                    help="comma-separated depth tolerances, as a fraction of the "
                         "crop's depth spread. Empty string disables the depth "
                         "section. Requires precompute_depth.py to have run")
    ap.add_argument("--depth-reading", default="dilated", choices=ORDER,
                    help="which contact region the depth gates are applied to")
    ap.add_argument("--check-only", action="store_true",
                    help="inspect the ground truth and stop. Needs no masks and "
                         "no torch, so an annotation pass can be verified before "
                         "any of the expensive steps are run")
    args = ap.parse_args()

    cfg = load_config(args.config)
    gt_path = args.gt or os.path.join(CONTACT_ROOT, "log", "annotate", args.split,
                                      "contact_gt.csv")
    if not os.path.exists(gt_path):
        raise SystemExit(f"no ground truth at {gt_path}\n"
                         "run annotate_contact.py, click through the page, and "
                         "save the exported CSV there")
    gt = read_gt(gt_path)

    by_rel = {r["rel_image"]: r
              for r in records_for(load_records(cfg, require_label=False), args.split)}

    if args.check_only:
        check_gt(gt, by_rel)
        return

    hits = {n: [] for n in ORDER}          # per clicked point: inside or not
    dists = {n: [] for n in ORDER}         # per clicked point: normalised distance
    dists_px = {n: [] for n in ORDER}      # the same distance in raw pixels
    sizes = {n: [] for n in ORDER}         # per crop with points: area
    fracs = {n: [] for n in ORDER}
    fires = {n: [] for n in ORDER}         # per "no contact" crop: non-empty or not
    n_pts = n_none = missing_mask = missing_row = 0

    tols = [float(t) for t in args.depth_tol.split(",") if t.strip()]
    # Keyed by (stat, tolerance) for the single region named by --depth-reading.
    # ("none", None) is the ungated band that every gated row is compared
    # against; "all" is every gate applied together.
    combos = [("none", None)] + [(st, t) for st in DEPTH_STATS + ["all"] for t in tols]
    d_hits = {c: [] for c in combos}
    d_area = {c: [] for c in combos}
    d_frac = {c: [] for c in combos}
    d_fires = {c: [] for c in combos}
    n_depth = n_no_depth = n_no_stat = 0

    for rel, ann in gt.items():
        if ann["status"] == "skip":
            continue
        record = by_rel.get(rel)
        if record is None:
            missing_row += 1
            continue
        bgr = cv2.imread(record["image_path"])
        if bgr is None:
            missing_row += 1
            continue
        masks = load_masks(record, bgr.shape[:2])
        if masks is None:
            missing_mask += 1
            continue
        mi, mj = masks
        readings = contact_readings(mi, mj, args.touch_px, args.dilate_px,
                                    args.strip_px)

        # The same rows scored again under each depth gate. The statistics do
        # not depend on the tolerance, so they are computed once and
        # thresholded rather than rebuilding the dilations for every setting.
        dep = load_depth(record, bgr.shape[:2]) if tols else None
        if tols:
            n_depth += dep is not None
            n_no_depth += dep is None
        if dep is not None:
            st = depth_stats(mi, mj, dep[0], dep[1], dep[2],
                             relative_boxes(record, *bgr.shape[:2]))
            n_no_stat += sum(v is None for v in st.values())
            base = readings[args.depth_reading]
            variants = {("none", None): base}
            for tol in tols:
                keep_all = None
                for sname in DEPTH_STATS:
                    if st.get(sname) is None:
                        continue
                    k = st[sname] <= tol
                    variants[(sname, tol)] = base & k
                    keep_all = k if keep_all is None else (keep_all & k)
                if keep_all is not None:
                    variants[("all", tol)] = base & keep_all
            for key, reg in variants.items():
                if ann["status"] == "none":
                    d_fires[key].append(int(reg.any()))
                    continue
                if not ann["points"]:
                    continue
                d_area[key].append(int(reg.sum()))
                d_frac[key].append(reg.sum() / float(bgr.shape[0] * bgr.shape[1]))
                for (px, py) in ann["points"]:
                    xx = int(np.clip(px, 0, bgr.shape[1] - 1))
                    yy = int(np.clip(py, 0, bgr.shape[0] - 1))
                    d_hits[key].append(int(bool(reg[yy, xx])))

        # Normaliser for the distances: the geometric mean of the two box
        # diagonals, so "20 px away" means the same thing on a near and a far
        # animal.
        boxes = relative_boxes(record, *bgr.shape[:2])
        scale = float(np.sqrt(np.prod([np.hypot(b[2] - b[0], b[3] - b[1])
                                       for b in boxes])))

        if ann["status"] == "none":
            n_none += 1
            for n in ORDER:
                fires[n].append(int(readings[n].any()))
            continue

        if not ann["points"]:
            continue
        n_pts += len(ann["points"])
        for n in ORDER:
            region = readings[n]
            sizes[n].append(int(region.sum()))
            fracs[n].append(region.sum() / float(bgr.shape[0] * bgr.shape[1]))
            # Distance from every pixel to the region; 0 inside it.
            dt = cv2.distanceTransform((~region).astype(np.uint8), cv2.DIST_L2, 3) \
                if region.any() else None
            for (x, y) in ann["points"]:
                x = int(np.clip(x, 0, bgr.shape[1] - 1))
                y = int(np.clip(y, 0, bgr.shape[0] - 1))
                inside = bool(region[y, x])
                hits[n].append(int(inside))
                raw = 0.0 if inside else (float(dt[y, x]) if dt is not None
                                          else float("inf"))
                dists_px[n].append(raw)
                dists[n].append(raw / max(scale, 1e-6))

    if not n_pts and not n_none:
        raise SystemExit("nothing scored - the ground truth file has no usable rows")
    if missing_mask:
        print(f"[score] {missing_mask} crops have no cached mask; run "
              "precompute_masks.py to cover them")
    if missing_row:
        print(f"[score] {missing_row} annotated rows are not in split "
              f"'{args.split}' or could not be read")

    print(f"\n[score] {n_pts} clicked points over "
          f"{len(sizes[ORDER[0]])} crops, plus {n_none} marked 'no contact'")
    print(f"[score] touch_px={args.touch_px} dilate_px={args.dilate_px} "
          f"strip_px={args.strip_px}\n")

    print(f"{'reading':<10}{'hit rate':>10}{'size':>10}{'of crop':>9}"
          f"{'miss dist':>11}{'fires on none':>15}")
    summary = {}
    for n in ORDER:
        hr = float(np.mean(hits[n])) if hits[n] else float("nan")
        sz = float(np.median(sizes[n])) if sizes[n] else float("nan")
        fr = float(np.median(fracs[n])) if fracs[n] else float("nan")
        miss = [d for d, h in zip(dists[n], hits[n]) if not h and np.isfinite(d)]
        md = float(np.median(miss)) if miss else 0.0
        fo = float(np.mean(fires[n])) if fires[n] else float("nan")
        summary[n] = {"hit_rate": hr, "size_px_median": sz, "frac_of_crop": fr,
                      "miss_dist_median_norm": md, "fires_on_no_contact": fo}
        print(f"{n:<10}{hr:>9.0%}{sz:>9.0f}px{fr:>9.1%}{md:>11.3f}{fo:>14.0%}")

    # A click carries human imprecision of perhaps five or ten pixels, which a
    # strict per-pixel test charges against small regions and not against large
    # ones. The distances are already computed, so the same measurement can be
    # read at several tolerances rather than committing to one radius.
    print(f"\n{'reading':<10}{'hit @0px':>10}{'@5px':>8}{'@10px':>8}{'@20px':>8}"
          f"{'  (tolerance around the click)':<32}")
    for n in ORDER:
        d = np.array(dists_px[n], float)
        summary[n]["hit_at_px"] = {str(t): float(np.mean(d <= t)) for t in (0, 5, 10, 20)}
        print(f"{n:<10}" + "".join(f"{np.mean(d <= t):>{w}.0%}"
                                   for t, w in zip((0, 5, 10, 20), (10, 8, 8, 8))))
    print("A reading whose hit rate climbs steeply from 0 to 10 px was being")
    print("charged for click precision, not for being in the wrong place.")

    print("\nhit rate  = clicked points falling inside the region")
    print("size      = median area; a high hit rate at a large size means little")
    print("miss dist = for misses only, distance to the region divided by the")
    print("            animal's scale, so near misses and wild ones separate")
    print("fires on none = crops marked 'no contact' where the region is not empty")

    depth_summary = {}
    if tols and n_depth:
        if n_no_depth:
            print(f"\n[depth] {n_no_depth} of {n_depth + n_no_depth} crops have no "
                  "cached depth map and sit out the section below; run "
                  "precompute_depth.py to cover them")
        if n_no_stat:
            print(f"[depth] {n_no_stat} statistics could not be formed (one "
                  "silhouette lies wholly inside the other, leaving the hidden "
                  "animal with no visible pixel to read a depth from)")

        print(f"\nDEPTH GATES on '{args.depth_reading}' — {n_depth} crops")
        print("  body  drop pixels far from the animals' body depth  -> FLOOR, FEET")
        print("  step  drop pixels on a depth discontinuity          -> OCCLUSION EDGE")
        print("  pair  drop pixels where the two animals are at different range")
        print("  all   every gate at once\n")
        print(f"{'gate':<7}{'tol':>7}{'hit rate':>10}{'area':>10}{'of crop':>9}"
              f"{'lift':>8}{'selectivity':>13}{'fires on none':>15}")

        h0 = float(np.mean(d_hits[("none", None)])) if d_hits[("none", None)] else 0.0
        a0 = float(np.mean(d_area[("none", None)])) if d_area[("none", None)] else 0.0
        for key in combos:
            if not d_hits[key] and not d_fires[key]:
                continue
            hr = float(np.mean(d_hits[key])) if d_hits[key] else 0.0
            ar = float(np.mean(d_area[key])) if d_area[key] else 0.0
            # Mean, not median: a gate strong enough to empty the band on more
            # than half the crops drives a median area to zero, which would
            # report a region that is not empty as having infinite lift.
            fr = float(np.mean(d_frac[key])) if d_frac[key] else 0.0
            fo = float(np.mean(d_fires[key])) if d_fires[key] else float("nan")
            # A gate deleting pixels at random would keep hits in the same
            # proportion as area, so selectivity is 1 for a gate that carries no
            # information about WHERE contact is.
            sel = ((hr / h0) / (ar / a0)) if (h0 > 0 and a0 > 0 and ar > 0) \
                else float("nan")
            lift = (hr / fr) if fr > 0 else float("nan")
            sname, tol = key
            label = "-" if tol is None else f"{tol:.2f}"
            depth_summary[f"{sname}@{label}"] = {
                "hit_rate": hr, "area_px_mean": ar, "frac_of_crop": fr,
                "lift": lift, "selectivity": sel, "fires_on_no_contact": fo}
            liftxt = f"{lift:>7.0f}x" if np.isfinite(lift) else f"{'-':>8}"
            selxt = f"{sel:>13.2f}" if np.isfinite(sel) else f"{'-':>13}"
            print(f"{sname:<7}{label:>7}{hr:>9.0%}{ar:>9.0f}px{fr:>9.1%}"
                  f"{liftxt}{selxt}{fo:>14.0%}")

        print("\nselectivity = (share of hits kept) / (share of area kept).")
        print("  1.00  the gate removed contact points at exactly the rate it")
        print("        removed area — no better than deleting random pixels, so")
        print("        depth added nothing and the smaller region is not a gain.")
        print("  >1    the pixels it discarded were disproportionately NOT")
        print("        contact. This is the only column that justifies a gate.")
        print("  <1    worse than random: the gate is cutting away the right")
        print("        pixels. For 'body' that means low contacts are being lost")
        print("        along with the floor, which no depth map can separate.")
        print("\n'fires on none' is the control. The ungated band is non-empty on")
        print("100% of them, which is why it is a candidate region and not a")
        print("detector. A gate that drives that down while holding selectivity")
        print("above 1 has turned it into one.")

    best = max(ORDER, key=lambda n: (summary[n]["hit_rate"] or 0)
               - (summary[n]["frac_of_crop"] or 0))
    print(f"\n[score] highest hit rate for its size: {best}")
    print("[score] this ranks the definitions against YOUR annotation only; it "
          "does not establish that any of them is correct in general")

    out = os.path.join(CONTACT_ROOT, "log", "annotate", args.split,
                       "contact_score.json")
    with open(out, "w") as f:
        json.dump({"split": args.split, "n_points": n_pts, "n_no_contact": n_none,
                   "touch_px": args.touch_px, "dilate_px": args.dilate_px,
                   "readings": summary, "depth_reading": args.depth_reading,
                   "n_with_depth": n_depth, "depth_gates": depth_summary},
                  f, indent=2)
    print(f"[score] wrote {out}")


if __name__ == "__main__":
    main()
