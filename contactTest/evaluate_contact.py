"""Cluster-level evaluation of the SAM 3 contact region against clicked GT.

Usage (from the repository root):

    python -m contactTest.evaluate_contact --split train --weights sam3.pt
    python -m contactTest.evaluate_contact --split train --weights sam3.pt \\
        --depth-tol 0.10 --depth-gate pair
    python -m contactTest.evaluate_contact --split train --source cache \\
        --mask-dir log/mask_cache          # a prebuilt cache, without re-running

Runs the SAM 3 pipeline itself by default — concept prompt `--text cow`, then
the returned instances assigned to the two detector boxes — so no mask cache has
to exist first and there is no way to measure one segmenter while believing it
is another. `--source cache` reads a prebuilt cache instead — whatever
precompute_masks wrote — which avoids re-running the model when only the
downstream settings are being varied.

Writes to contactTest/log/evaluate/<split>/ only: one image per crop, a
per-image CSV, and a summary JSON. The aggregate numbers are printed.

The predicted region is split into CONNECTED COMPONENTS, called clusters here.
That is the difference from score_contact.py, which treats the region as one
undifferentiated set of pixels: a cluster is a candidate contact site, so a
region made of one blob on the right animals and four scattered on the railing
is not the same result as a single blob, even though both have the same hit rate.

THE SIX METRICS

  1  Sensitivity = GT points covered by any cluster / all GT points
     Pooled over points, not averaged over images, so a crop with 20 clicks
     counts for more than one with 2. Alone it is maximised by covering
     everything, which is what 3 and 2 are for.

  2  FPPI = clusters covering no GT point / number of images
     False positives per image. Counts CLUSTERS, not pixels, so one large
     wrong blob costs the same as one small wrong blob — this measures how
     many wrong places are proposed, and metric 3 measures how much they cost.
     Images marked "no contact" contribute every cluster they have, since
     none of them can cover a GT point. They are counted in with the rest:
     a "no contact" mark is the annotator's finding about that image, not a
     designed negative arm, so there is nothing to report separately.

  3  a-bar = mean over images of (predicted pixels / image pixels)
     Mean over images as specified, so every crop counts once regardless of
     size. Alone it is minimised by predicting nothing.

  4  Lift = Sensitivity / a-bar
     How much more concentrated contact is inside the region than it would be
     under a region of the same size placed at random. 1.0 is chance.

  5  Hit quality = mean over covered GT points of
         area(the cluster containing that point) / area(the GT discs)
     Near 1 means the cluster that found the contact is about the size of the
     ground truth it found. Large means it was right but grossly oversized —
     a hit that carries no localisation. This is what separates "in the right
     place" from "in the right place and no bigger than it needs to be", which
     Sensitivity alone cannot express.

     The GT discs are each clicked point dilated to a radius of
     --gt-dilate-scale x dilate_px (0.5 x by default), and the denominator is
     the area of their UNION within that image, so overlapping clicks are not
     counted twice. Set --gt-dilate-px to give the radius in pixels instead.

  6  Blind area = pixels in clusters covering no GT point / all predicted pixels
     The pixel-weighted counterpart of FPPI. FPPI counts how many places are
     wrongly proposed; this is how much of the prediction they consume. One
     huge blind blob and ten blind specks score the same FPPI and very
     different blind area, and they are different problems: the first wastes
     the budget, the second is noise.

     Pooled over all pixels in all images, matching "total proposed area".
     The per-image ratio is in the CSV as blind_frac. Images marked "no
     contact" contribute all of their area, since none of it can cover a
     point.

Everything is reported per image as well, in evaluation.csv.
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contactTest.sam3 import Sam3
from contactTest.sam_contact_region import (contact_readings, depth_stats,
                                            load_depth, load_masks)
from contactTest.score_contact import read_gt
from contactTest.src.data import load_records, records_for, relative_boxes
from contactTest.src.utils import load_config

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))

C_HIT = (90, 220, 110)
C_MISS = (235, 90, 80)
C_GT_DISC = (250, 220, 90)
# Distinct hues so neighbouring clusters stay tellable apart; a cluster that
# covers no GT point is drawn in the last colour regardless.
C_CLUSTER = [(120, 235, 130), (90, 190, 240), (245, 175, 90), (200, 140, 235),
             (140, 220, 210), (240, 210, 120)]
C_FP = (225, 120, 120)


def gt_disc_mask(shape, points, radius):
    """Union of discs of `radius` centred on the clicked points."""
    m = np.zeros(shape, np.uint8)
    for (x, y) in points:
        cv2.circle(m, (int(x), int(y)), int(max(radius, 1)), 1, -1)
    return m


def evaluate_one(region, points, shape, gt_radius, denom_px=None):
    """Cluster-level numbers for a single image.

    Returns the per-image record. `points` may be empty — the image was marked
    "no contact" — and then every cluster is a false positive and the
    point-based metrics are UNDEFINED rather than zero.

    `denom_px` is the area a_i is a share OF. It defaults to the whole image,
    which is right when the whole image could have been predicted. It must be
    overridden when the prediction was restricted to part of the image, because
    otherwise a_i is diluted by territory the method was never allowed to use —
    and Lift, being Sensitivity / a_i, is inflated by exactly that factor.
    """
    h, w = shape
    n_lab, lab = cv2.connectedComponents(region.astype(np.uint8))
    n_clusters = max(n_lab - 1, 0)

    # Which cluster, if any, contains each clicked point.
    hit_of = []
    for (x, y) in points:
        xx = int(np.clip(x, 0, w - 1))
        yy = int(np.clip(y, 0, h - 1))
        hit_of.append(int(lab[yy, xx]))          # 0 = background = miss

    covered = [c for c in hit_of if c > 0]
    hitting = set(covered)
    # A cluster is a false positive when it covers no GT point at all.
    fp_clusters = n_clusters - len(hitting)

    area = float(region.sum())
    a_i = area / float(denom_px if denom_px else (h * w))

    # Metric 6. Area of the clusters covering no GT point at all. FPPI counts
    # how MANY places are wrongly proposed; this is how much of the prediction
    # they consume, which is a different failure: one huge blind blob and ten
    # specks give the same FPPI and very different blind area.
    sizes_all = np.bincount(lab.ravel(), minlength=n_lab).astype(float)
    blind = float(sum(sizes_all[c] for c in range(1, n_lab)
                      if c not in hitting))

    rec = {
        "n_points": len(points),
        "n_covered": len(covered),
        "sensitivity": (len(covered) / len(points)) if points else float("nan"),
        "n_clusters": n_clusters,
        "n_fp_clusters": fp_clusters,
        "area_px": int(area),
        "a_i": a_i,
        "lift": ((len(covered) / len(points)) / a_i)
                if points and a_i > 0 else float("nan"),
        "blind_area_px": int(blind),
        "blind_frac": (blind / area) if area > 0 else float("nan"),
    }

    # Metric 5. Denominator is the union of the GT discs in this image, so two
    # clicks a few pixels apart do not inflate it.
    if covered:
        gt_area = float(gt_disc_mask((h, w), points, gt_radius).sum())
        q = [sizes_all[c] / gt_area for c in covered] if gt_area > 0 else []
        rec["hit_quality"] = float(np.mean(q)) if q else float("nan")
        rec["gt_disc_px"] = int(gt_area)
    else:
        rec["hit_quality"] = float("nan")
        rec["gt_disc_px"] = 0
    return rec, lab, hitting


def render(bgr, region, lab, hitting, points, gt_radius, rec):
    """Crop with clusters coloured, GT discs outlined and each click marked."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Every cluster gets its own colour; the ones covering no GT point share a
    # single warning colour, because their COUNT is the metric, not their hue.
    for c in range(1, int(lab.max()) + 1):
        sel = lab == c
        if not sel.any():
            continue
        col = (C_CLUSTER[(c - 1) % len(C_CLUSTER)] if c in hitting else C_FP)
        rgb[sel] = rgb[sel] * 0.5 + np.asarray(col, np.float32) * 0.5
    out = rgb.astype(np.uint8)

    for c in range(1, int(lab.max()) + 1):
        cnt, _ = cv2.findContours((lab == c).astype(np.uint8), cv2.RETR_CCOMP,
                                  cv2.CHAIN_APPROX_SIMPLE)
        col = (C_CLUSTER[(c - 1) % len(C_CLUSTER)] if c in hitting else C_FP)
        cv2.drawContours(out, cnt, -1, col, 2, lineType=cv2.LINE_AA)

    for (x, y) in points:
        cv2.circle(out, (int(x), int(y)), int(max(gt_radius, 1)), C_GT_DISC, 1,
                   cv2.LINE_AA)
    h, w = region.shape
    for (x, y) in points:
        xx = int(np.clip(x, 0, w - 1)); yy = int(np.clip(y, 0, h - 1))
        col = C_HIT if region[yy, xx] else C_MISS
        cv2.circle(out, (xx, yy), 6, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(out, (xx, yy), 5, col, -1, cv2.LINE_AA)
    return out


def banner(img, lines, height=58):
    bar = np.full((height, img.shape[1], 3), 245, np.uint8)
    longest = max((len(t) for t, _ in lines), default=1)
    scale = float(np.clip((img.shape[1] - 16) / (longest * 19.0), 0.28, 0.46))
    for i, (text, col) in enumerate(lines):
        cv2.putText(bar, text, (8, 17 + i * 17), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    col, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


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
    ap.add_argument("--reading", default="dilated",
                    choices=["overlap", "gap", "surface", "dilated"])
    ap.add_argument("--dilate-px", type=int, default=22)
    ap.add_argument("--touch-px", type=int, default=10)
    ap.add_argument("--strip-px", type=int, default=6)
    ap.add_argument("--gt-dilate-scale", type=float, default=0.5,
                    help="GT disc radius as a multiple of --dilate-px")
    ap.add_argument("--gt-dilate-px", type=int, default=None,
                    help="GT disc radius in pixels; overrides --gt-dilate-scale")
    ap.add_argument("--depth-tol", type=float, default=None,
                    help="apply depth gates to the region before evaluating")
    ap.add_argument("--depth-gate", default="pair",
                    help="comma-separated: pair, body, step")
    ap.add_argument("--min-cluster-px", type=int, default=0,
                    help="drop clusters smaller than this before evaluating. 0 "
                         "keeps every speck, which is the honest default: "
                         "raising it improves FPPI by definition, so any value "
                         "above 0 must be reported alongside the number")
    ap.add_argument("--limit", type=int, default=0, help="0 = every annotated crop")
    ap.add_argument("--source", default="sam3_text",
                    choices=["sam3_text", "cache"],
                    help="sam3_text (default) runs SAM 3 concept segmentation "
                         "here and now; cache reads a prebuilt mask cache. This "
                         "is what decides WHICH segmenter is measured, so it is "
                         "echoed in the output")
    ap.add_argument("--weights", default=None,
                    help="Hugging Face id or a local snapshot DIRECTORY")
    ap.add_argument("--text", default="cow",
                    help="noun phrase for concept segmentation")
    ap.add_argument("--conf", type=float, default=None,
                    help="SAM 3 score floor; default is data.sam3_conf "
                         "from config.yaml, which mirrors the value the "
                         "detector stage used")
    ap.add_argument("--mask-dir", default=None,
                    help="cache to read when --source cache, overriding "
                         "data.mask_dir")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.conf is None:
        args.conf = float(cfg["data"].get("sam3_conf", 0.6))
    if args.mask_dir:
        cfg["data"]["mask_dir"] = args.mask_dir
    mask_dir = cfg["data"].get("mask_dir")

    seg = None
    if args.source == "sam3_text":
        seg = Sam3(args.weights, args.text, args.conf)
        source_label = f"SAM 3 concept prompt, text={args.text!r}"
    else:
        source_label = f"cached masks from {mask_dir}"
    gt_path = args.gt or os.path.join(CONTACT_ROOT, "log", "annotate", args.split,
                                      "contact_gt.csv")
    if not os.path.exists(gt_path):
        raise SystemExit(f"no ground truth at {gt_path}")
    gt = read_gt(gt_path)
    by_rel = {r["rel_image"]: r
              for r in records_for(load_records(cfg, require_label=False), args.split)}

    gt_radius = (args.gt_dilate_px if args.gt_dilate_px is not None
                 else max(1, int(round(args.gt_dilate_scale * args.dilate_px))))

    out_dir = os.path.join(CONTACT_ROOT, "log", "evaluate", args.split)
    os.makedirs(out_dir, exist_ok=True)

    rows, no_mask, no_depth = [], 0, 0
    for rel, ann in gt.items():
        if ann["status"] == "skip":
            continue
        if args.limit and len(rows) >= args.limit:
            break
        record = by_rel.get(rel)
        if record is None:
            continue
        bgr = cv2.imread(record["image_path"])
        if bgr is None:
            continue
        boxes = relative_boxes(record, *bgr.shape[:2])
        if seg is None:
            masks = load_masks(record, bgr.shape[:2])
        else:
            try:
                got = seg.assign_to_boxes(bgr, boxes)
                masks = ([(np.asarray(g) > 0.5).astype(np.uint8) for g in got]
                         if got is not None else None)
            except Exception as err:                   # noqa: BLE001
                print(f"[eval] SAM 3 failed on {rel}: {err}")
                masks = None
        if masks is None or len(masks) < 2 or masks[0].sum() == 0 \
                or masks[1].sum() == 0:
            no_mask += 1
            continue
        mi, mj = masks[0], masks[1]
        region = contact_readings(mi, mj, args.touch_px, args.dilate_px,
                                  args.strip_px)[args.reading]

        if args.depth_tol is not None:
            dep = load_depth(record, bgr.shape[:2])
            if dep is None:
                no_depth += 1
            else:
                st = depth_stats(mi, mj, dep[0], dep[1], dep[2], boxes)
                keep = None
                for g in args.depth_gate.split(","):
                    s_map = st.get(g.strip())
                    if s_map is None:
                        continue
                    k_ = s_map <= args.depth_tol
                    keep = k_ if keep is None else (keep & k_)
                if keep is not None:
                    region = region & keep

        if args.min_cluster_px > 0 and region.any():
            n_lab, lab0 = cv2.connectedComponents(region.astype(np.uint8))
            sizes = np.bincount(lab0.ravel(), minlength=n_lab)
            small = np.isin(lab0, np.where(sizes < args.min_cluster_px)[0])
            region = region & ~small

        points = ann["points"] if ann["status"] != "none" else []
        rec, lab, hitting = evaluate_one(region, points, bgr.shape[:2], gt_radius)
        rec["rel_image"] = rel
        rec["no_contact"] = int(ann["status"] == "none")
        rows.append(rec)

        if not args.no_images:
            img = render(bgr, region, lab, hitting, points, gt_radius, rec)
            sens = rec["sensitivity"]
            head = ("marked 'no contact'" if rec["no_contact"]
                    else f"sensitivity {sens:.0%}  ({rec['n_covered']}/{rec['n_points']})")
            img = banner(img, [
                (f"{os.path.basename(rel)}   {head}",
                 (150, 30, 30) if rec["no_contact"] else
                 ((25, 110, 40) if sens >= .8 else (170, 60, 25))),
                (f"clusters {rec['n_clusters']}  false-positive {rec['n_fp_clusters']}"
                 f"   a_i {rec['a_i']:.2%}"
                 + ("" if rec["no_contact"] else f"   lift {rec['lift']:.0f}x"),
                 (80, 80, 80)),
                (f"hit quality {rec['hit_quality']:.2f}"
                 f"   (GT discs r={gt_radius}px, {rec['gt_disc_px']}px total)"
                 if np.isfinite(rec["hit_quality"]) else
                 "hit quality: no covered points", (80, 80, 80))])
            tag = "none" if rec["no_contact"] else f"{int(round(sens * 100)):03d}"
            cv2.imwrite(os.path.join(out_dir, f"{tag}_{os.path.basename(rel)}"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    if not rows:
        raise SystemExit("nothing evaluated - run precompute_masks.py first")

    # ---- aggregate ----------------------------------------------------------
    # Every non-skip crop is in `rows` and every metric below is computed over
    # all of them. Crops marked "no contact" are not a separate arm: they carry
    # no GT points, so they contribute nothing to the point-weighted metrics by
    # arithmetic rather than by exclusion, and their area and clusters count
    # towards a-bar, FPPI and blind area like any other crop's.
    n_img = len(rows)
    n_none = sum(r["no_contact"] for r in rows)

    tot_pts = sum(r["n_points"] for r in rows)
    tot_cov = sum(r["n_covered"] for r in rows)
    sensitivity = tot_cov / tot_pts if tot_pts else float("nan")

    tot_fp = sum(r["n_fp_clusters"] for r in rows)
    fppi = tot_fp / n_img

    a_bar = float(np.mean([r["a_i"] for r in rows]))
    lift = sensitivity / a_bar if a_bar > 0 else float("nan")

    tot_area = sum(r["area_px"] for r in rows)
    tot_blind = sum(r["blind_area_px"] for r in rows)
    blind_frac = tot_blind / tot_area if tot_area else float("nan")

    q = [r["hit_quality"] for r in rows if np.isfinite(r["hit_quality"])]
    # Weighted by covered points so the mean is over MEASUREMENTS, matching the
    # per-point definition, rather than over images.
    wq = [(r["hit_quality"], r["n_covered"]) for r in rows
          if np.isfinite(r["hit_quality"])]
    hit_q = (sum(v * n for v, n in wq) / sum(n for _, n in wq)) if wq else float("nan")

    if no_mask:
        what = ("produced no usable pair of instances" if seg is not None
                else "have no cached mask")
        print(f"[eval] {no_mask} crops skipped: {what}. They are EXCLUDED from\n"
              "       every number below, so a large count makes the rest\n"
              "       optimistic - it is a success rate, not a neutral filter.")
    if no_depth:
        print(f"[eval] {no_depth} crops evaluated ungated - no cached depth map")

    print(f"\n{'=' * 62}")
    print(f"  {args.reading} at dilate_px={args.dilate_px}"
          + (f", depth gate {args.depth_gate} <= {args.depth_tol}"
             if args.depth_tol is not None else ", no depth gate"))
    print(f"  {n_img} images ({n_none} marked 'no contact'), {tot_pts} GT points")
    print(f"  masks: {source_label}")
    print(f"{'=' * 62}\n")
    print(f"  1  Sensitivity   {sensitivity:>9.3f}   {tot_cov}/{tot_pts} points covered")
    print(f"  2  FPPI          {fppi:>9.3f}   {tot_fp} empty clusters / {n_img} images")
    print(f"  3  a-bar         {a_bar:>9.4f}   mean predicted area fraction")
    print(f"  4  Lift          {lift:>9.1f}   Sensitivity / a-bar")
    print(f"  5  Hit quality   {hit_q:>9.2f}   cluster area / GT disc area "
          f"(r={gt_radius}px)")
    print(f"  6  Blind area    {blind_frac:>9.3f}   {tot_blind}/{tot_area} px in "
          "clusters covering no GT")

    if np.isfinite(blind_frac):
        print(f"\n  Blind area {blind_frac:.1%} of everything predicted sits in "
              "clusters that")
        print("  cover no GT point. Read it with FPPI: the same FPPI with a low "
              "blind")
        print("  area means the wrong proposals are small, with a high one means "
              "they are")
        print("  eating the budget that a-bar charges for.")

    if np.isfinite(hit_q):
        print(f"\n  Hit quality {hit_q:.2f} means the cluster that found a contact "
              f"is on average")
        print(f"  {hit_q:.1f}x the area of the ground truth in that image. 1.0 would "
              "be exact;")
        print("  a large value is a hit that carries little localisation.")
    if args.min_cluster_px > 0:
        print(f"\n  NOTE: clusters under {args.min_cluster_px}px were discarded, "
              "which improves FPPI by construction. Report that alongside it.")

    per = os.path.join(out_dir, "evaluation.csv")
    keys = ["rel_image", "no_contact", "n_points", "n_covered", "sensitivity",
            "n_clusters", "n_fp_clusters", "area_px", "a_i", "lift",
            "hit_quality", "gt_disc_px", "blind_area_px", "blind_frac"]
    with open(per, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=keys)
        wtr.writeheader()
        for r in rows:
            wtr.writerow({k: r.get(k, "") for k in keys})

    summary = {"split": args.split, "reading": args.reading,
               "source": args.source, "source_label": source_label,
               "n_skipped": no_mask,
               "dilate_px": args.dilate_px, "gt_disc_radius_px": gt_radius,
               "depth_tol": args.depth_tol,
               "depth_gate": args.depth_gate if args.depth_tol is not None else None,
               "min_cluster_px": args.min_cluster_px,
               "n_images": n_img, "n_no_contact": n_none, "n_points": tot_pts,
               "sensitivity": sensitivity, "fppi": fppi, "a_bar": a_bar,
               "lift": lift, "hit_quality": hit_q,
               "blind_area_frac": blind_frac,
               "blind_area_px": tot_blind, "total_area_px": tot_area}
    with open(os.path.join(out_dir, "evaluation.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[eval] per-image numbers -> {per}")
    if not args.no_images:
        print(f"[eval] images -> {out_dir} (worst sensitivity first)")
        print("[eval] each cluster has its own colour; RED clusters cover no GT")
        print("[eval] point. Yellow circles are the GT discs metric 5 divides by.")
        print("[eval] green dot = covered click, red dot = uncovered.")


if __name__ == "__main__":
    main()
