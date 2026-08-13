"""Control experiment: SAM 3 on the whole frame, doing YOLO's pairing itself.

Usage (from the repository root):

    python -m contactTest.evaluate_wholeframe --split train --weights sam3.pt \\
        --video-root /user/work/sf24225/data/Full_behav/Marco
    python -m contactTest.evaluate_wholeframe --split train --weights sam3.pt \\
        --scope frame          # unrestricted, and biased - see below
    python -m contactTest.evaluate_wholeframe --split train --weights sam3.pt \\
        --depth-tol 0.10 --depth-gate pair

Writes to contactTest/log/wholeframe/<split>/ only.

THE QUESTION

Every result so far measures SAM 3 on a crop that YOLO already chose: detect
cattle, keep box pairs with 0.1 < IoU < 0.8, cut the merged box out. SAM 3 is
handed a picture containing two animals and little else. This script removes
that scaffolding — the uncropped frame goes to SAM 3, `--text cow` finds every
animal, and the SAME IoU rule is applied to SAM 3's own instances to form pairs.
If the contact regions come out comparable, the detector stage is not carrying
the result; if they come out worse, the crop is doing work worth naming.

THE MEASUREMENT PROBLEM, AND WHAT IS DONE ABOUT IT

The clicked ground truth lives in CROP pixel coordinates. A crop is exactly the
merged box cut from the frame and never resized, so a click maps back by adding
the merged box origin — the exact inverse of relative_boxes, and clamped the
same way because ~12% of merged boxes are clipped at the frame border.

The harder problem is that the ground truth is INCOMPLETE at frame level.
Measured on this dataset: the 99 clicked crops sit in 96 frames, and those
frames contain 351 detected pairs in total. 72% of the pairs in those frames
carry no clicks at all. A contact SAM 3 correctly finds on one of those 252
unannotated pairs is indistinguishable, to any metric here, from a mistake.

So the default `--scope annotated` evaluates only inside the merged boxes of the
pairs that were actually annotated. Both methods are then judged on identical
image territory and the comparison isolates the thing in question: given the
same region, does global context help or hurt? `--scope frame` lifts the
restriction and is reported too, but it is a LOWER BOUND, not a fair number, and
is labelled that way in the output.

WHAT IS ALSO REPORTED

Detector agreement — how many animals SAM 3 finds against how many YOLO boxes
exist in that frame, and how many pairs each rule yields. That is the direct
evidence for whether the detector stage could be dropped, separately from
whether the contact region is any good.
"""

import argparse
import collections
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contactTest.evaluate_contact import evaluate_one, render, banner
from contactTest.sam_contact_region import contact_readings, depth_stats, load_depth
from contactTest.score_contact import read_gt
from contactTest.src.data import load_records, split_records
from contactTest.src.utils import load_config

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Default from global_config.yaml paths.data_root. Duplicated rather than
# imported so contactTest stays self-contained; the videos are only ever read.
DEFAULT_VIDEO_ROOT = "/user/work/sf24225/data/Full_behav/Marco"


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def mask_box(m):
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max()) + 1,
            float(ys.max()) + 1)


class FrameSource:
    """Frames decoded from the source videos, with the capture kept open.

    The crops were cut from video, not from stored images, and `frame_number` is
    the raw frame index the decoder counted, so seeking to it is exact. Frames
    are requested in file order, which is close to video order, so holding one
    capture open and seeking within it avoids reopening a video per frame.
    """

    def __init__(self, root):
        self.root = root
        self.cap = None
        self.open_stem = None
        self.index = {}
        for dirpath, _, files in os.walk(root):
            for f in files:
                if os.path.splitext(f)[1].lower() in (".mp4", ".avi", ".mov", ".mkv"):
                    self.index.setdefault(os.path.splitext(f)[0], os.path.join(dirpath, f))

    def get(self, source_video, frame_number):
        stem = os.path.splitext(source_video)[0]
        path = self.index.get(stem)
        if path is None:
            return None, f"video not found for {stem}"
        if self.open_stem != stem:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(path)
            self.open_stem = stem
            if not self.cap.isOpened():
                return None, f"cannot open {path}"
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ok, frame = self.cap.read()
        return (frame, None) if ok else (None, f"cannot read frame {frame_number}")


def crop_to_frame(points, merged, shape):
    """Clicked crop coordinates back to frame coordinates.

    The crop is the merged box cut out and never resized, so the offset is the
    box origin. It is clamped at zero the same way relative_boxes clamps it,
    because a merged box that ran past the frame border was clipped when the
    crop was written and its stored coordinates can be negative.
    """
    ox, oy = max(0, int(merged[0])), max(0, int(merged[1]))
    h, w = shape
    return [(int(np.clip(x + ox, 0, w - 1)), int(np.clip(y + oy, 0, h - 1)))
            for (x, y) in points]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(CONTACT_ROOT, "config.yaml"))
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT)
    ap.add_argument("--weights", default="sam3.pt")
    ap.add_argument("--text", default="cow")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-low", type=float, default=0.1,
                    help="pairing rule, matching prep/interaction_prep.py")
    ap.add_argument("--iou-high", type=float, default=0.8)
    ap.add_argument("--scope", default="annotated", choices=["annotated", "frame"],
                    help="annotated (default) scores only inside the merged boxes "
                         "of the annotated pairs, which is the only like-for-like "
                         "comparison; frame scores the whole image and is a lower "
                         "bound because 72%% of the pairs in these frames carry "
                         "no ground truth")
    ap.add_argument("--reading", default="dilated",
                    choices=["overlap", "gap", "surface", "dilated"])
    ap.add_argument("--dilate-px", type=int, default=22)
    ap.add_argument("--touch-px", type=int, default=10)
    ap.add_argument("--strip-px", type=int, default=6)
    ap.add_argument("--gt-dilate-scale", type=float, default=0.5)
    ap.add_argument("--gt-dilate-px", type=int, default=None)
    ap.add_argument("--depth-tol", type=float, default=None)
    ap.add_argument("--depth-gate", default="pair")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    gt_path = args.gt or os.path.join(CONTACT_ROOT, "log", "annotate", args.split,
                                      "contact_gt.csv")
    if not os.path.exists(gt_path):
        raise SystemExit(f"no ground truth at {gt_path}")
    gt = read_gt(gt_path)
    by_rel = {r["rel_image"]: r
              for r in split_records(load_records(cfg, require_label=False))[args.split]}

    # Annotated crops grouped by the frame they came from: one SAM 3 pass per
    # frame serves every pair annotated in it.
    frames = collections.defaultdict(list)
    for rel, ann in gt.items():
        if ann["status"] == "skip":
            continue
        rec = by_rel.get(rel)
        if rec is None:
            continue
        frames[(rec["source_video"], int(rec["frame_number"]))].append((rel, ann, rec))
    if not frames:
        raise SystemExit("no annotated crops resolve to a frame")

    gt_radius = (args.gt_dilate_px if args.gt_dilate_px is not None
                 else max(1, int(round(args.gt_dilate_scale * args.dilate_px))))

    try:
        from contactTest.precompute_masks import _SAM3Text
        seg = _SAM3Text(args.weights, args.text, args.conf)
    except Exception as err:                       # noqa: BLE001
        raise SystemExit(f"could not load SAM 3 ({err}). See visualize_sam3_"
                         "confusion.py for the install and access steps.")

    src = FrameSource(args.video_root)
    if not src.index:
        raise SystemExit(f"no videos found under {args.video_root}\n"
                         "pass --video-root; the crops were cut from video, not "
                         "from stored frames, so the frames must be decoded")

    out_dir = os.path.join(CONTACT_ROOT, "log", "wholeframe", args.split)
    os.makedirs(out_dir, exist_ok=True)

    rows, failed = [], 0
    det_yolo, det_sam3, pair_yolo, pair_sam3 = [], [], [], []

    for (video, fno), items in sorted(frames.items()):
        if args.limit and len(rows) >= args.limit:
            break
        frame, err = src.get(video, fno)
        if frame is None:
            print(f"[wf] {video} frame {fno}: {err}")
            failed += 1
            continue
        H, W = frame.shape[:2]

        try:
            inst = seg.instances(frame)
        except Exception as err:                   # noqa: BLE001
            print(f"[wf] SAM 3 failed on {video} frame {fno}: {err}")
            failed += 1
            continue
        if not inst or len(inst) < 2:
            failed += 1
            continue

        boxes3 = [mask_box(m) for m in inst]
        keep = [k for k, b in enumerate(boxes3) if b is not None]
        inst = [inst[k] for k in keep]
        boxes3 = [boxes3[k] for k in keep]

        # SAM 3's own pairing, by exactly the rule the detector stage used.
        pairs = [(i, j) for i in range(len(inst)) for j in range(i + 1, len(inst))
                 if args.iou_low < box_iou(boxes3[i], boxes3[j]) < args.iou_high]

        region = np.zeros((H, W), bool)
        for (i, j) in pairs:
            r = contact_readings(inst[i], inst[j], args.touch_px, args.dilate_px,
                                 args.strip_px)[args.reading]
            region |= r

        # How many animals and pairs each route produced, for the same frame.
        n_yolo_pairs = len(items)
        det_sam3.append(len(inst))
        pair_sam3.append(len(pairs))
        pair_yolo.append(n_yolo_pairs)

        # GT points, mapped out of every annotated crop in this frame.
        points, scope = [], np.zeros((H, W), bool)
        n_ctrl = 0
        for rel, ann, rec in items:
            merged = rec["merged"]
            x1, y1 = max(0, int(merged[0])), max(0, int(merged[1]))
            x2, y2 = min(W, int(merged[2])), min(H, int(merged[3]))
            if x2 > x1 and y2 > y1:
                scope[y1:y2, x1:x2] = True
            if ann["status"] == "none":
                n_ctrl += 1
                continue
            points.extend(crop_to_frame(ann["points"], merged, (H, W)))

        if args.scope == "annotated":
            region = region & scope

        if args.depth_tol is not None:
            # Depth is cached per CROP, not per frame, so a whole-frame gate
            # would need a whole-frame depth pass. Rather than resize a crop's
            # map onto the frame — which would put its values in the wrong
            # place — the gate is applied per annotated pair, inside that pair's
            # own box, and pixels outside every annotated box are left ungated.
            gated = region.copy()
            for rel, ann, rec in items:
                merged = rec["merged"]
                x1, y1 = max(0, int(merged[0])), max(0, int(merged[1]))
                x2, y2 = min(W, int(merged[2])), min(H, int(merged[3]))
                if x2 <= x1 or y2 <= y1:
                    continue
                sub_shape = (y2 - y1, x2 - x1)
                dep = load_depth(rec, sub_shape)
                if dep is None:
                    continue
                sub = region[y1:y2, x1:x2]
                if not sub.any():
                    continue
                # Instance masks restricted to this box, for the pair readings.
                loc = [m[y1:y2, x1:x2] for m in inst]
                areas = sorted(range(len(loc)), key=lambda k: -loc[k].sum())[:2]
                if len(areas) < 2 or loc[areas[1]].sum() == 0:
                    continue
                lb = [(0.0, 0.0, float(sub_shape[1]), float(sub_shape[0]))] * 2
                st = depth_stats(loc[areas[0]], loc[areas[1]], dep[0], dep[1],
                                 dep[2], lb)
                keepm = None
                for g in args.depth_gate.split(","):
                    s_map = st.get(g.strip())
                    if s_map is None:
                        continue
                    k_ = s_map <= args.depth_tol
                    keepm = k_ if keepm is None else (keepm & k_)
                if keepm is not None:
                    gated[y1:y2, x1:x2] = sub & keepm
            region = gated

        rec_m, lab, hitting = evaluate_one(region, points, (H, W), gt_radius)
        rec_m["rel_image"] = f"{video}#{fno}"
        rec_m["is_control"] = int(len(points) == 0)
        rec_m["n_sam3_instances"] = len(inst)
        rec_m["n_sam3_pairs"] = len(pairs)
        rec_m["n_yolo_pairs"] = n_yolo_pairs
        rows.append(rec_m)

        if not args.no_images:
            img = render(frame, region, lab, hitting, points, gt_radius, rec_m)
            if args.scope == "annotated":
                cnt, _ = cv2.findContours(scope.astype(np.uint8), cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cnt, -1, (255, 255, 255), 2, cv2.LINE_AA)
            s = rec_m["sensitivity"]
            img = banner(img, [
                (f"{video} frame {fno}   "
                 + ("CONTROL" if rec_m["is_control"] else
                    f"sensitivity {s:.0%} ({rec_m['n_covered']}/{rec_m['n_points']})"),
                 (150, 30, 30) if rec_m["is_control"] else
                 ((25, 110, 40) if s >= .8 else (170, 60, 25))),
                (f"SAM 3 found {len(inst)} cattle -> {len(pairs)} pairs   "
                 f"YOLO gave {n_yolo_pairs} annotated pair(s)", (80, 80, 80)),
                (f"clusters {rec_m['n_clusters']}  blind {rec_m['blind_frac']:.0%}"
                 f"   a_i {rec_m['a_i']:.2%}   scope={args.scope}"
                 + ("  (white outline = scored area)"
                    if args.scope == "annotated" else ""), (80, 80, 80))])
            tag = "ctrl" if rec_m["is_control"] else f"{int(round(s * 100)):03d}"
            cv2.imwrite(os.path.join(out_dir, f"{tag}_{video}_{fno:08d}.jpg"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    if not rows:
        raise SystemExit("nothing evaluated")

    n_img = len(rows)
    tot_pts = sum(r["n_points"] for r in rows)
    tot_cov = sum(r["n_covered"] for r in rows)
    sensitivity = tot_cov / tot_pts if tot_pts else float("nan")
    tot_fp = sum(r["n_fp_clusters"] for r in rows)
    fppi = tot_fp / n_img
    a_bar = float(np.mean([r["a_i"] for r in rows]))
    lift = sensitivity / a_bar if a_bar > 0 else float("nan")
    wq = [(r["hit_quality"], r["n_covered"]) for r in rows
          if np.isfinite(r["hit_quality"])]
    hit_q = (sum(v * n for v, n in wq) / sum(n for _, n in wq)) if wq else float("nan")
    tot_area = sum(r["area_px"] for r in rows)
    tot_blind = sum(r["blind_area_px"] for r in rows)
    blind = tot_blind / tot_area if tot_area else float("nan")

    print(f"\n{'=' * 66}")
    print(f"  WHOLE FRAME, no YOLO: SAM 3 text={args.text!r}, pairs by "
          f"{args.iou_low} < IoU < {args.iou_high}")
    print(f"  {args.reading} at dilate_px={args.dilate_px}"
          + (f", depth gate {args.depth_gate} <= {args.depth_tol}"
             if args.depth_tol is not None else ", no depth gate"))
    print(f"  {n_img} frames, {tot_pts} GT points, scope={args.scope}")
    print(f"{'=' * 66}\n")
    print(f"  1  Sensitivity   {sensitivity:>9.3f}   {tot_cov}/{tot_pts} points covered")
    print(f"  2  FPPI          {fppi:>9.3f}   {tot_fp} empty clusters / {n_img} frames")
    print(f"  3  a-bar         {a_bar:>9.4f}   mean predicted area fraction")
    print(f"  4  Lift          {lift:>9.1f}")
    print(f"  5  Hit quality   {hit_q:>9.2f}   (GT discs r={gt_radius}px)")
    print(f"  6  Blind area    {blind:>9.3f}   {tot_blind}/{tot_area} px")

    print(f"\n  Detector stage, same frames:")
    print(f"    SAM 3 found  {np.mean(det_sam3):.1f} cattle per frame "
          f"-> {np.mean(pair_sam3):.1f} pairs at {args.iou_low}-{args.iou_high} IoU")
    print(f"    YOLO left    {np.mean(pair_yolo):.1f} annotated pair(s) per frame")
    print("    These are not the same quantity: the YOLO figure is pairs that")
    print("    were ANNOTATED, not pairs it found, so it is a floor. Read the")
    print("    two as 'does SAM 3 propose at least as many candidates', not as")
    print("    a precision comparison.")

    if args.scope == "frame":
        print("\n  SCOPE = frame. 72% of the pairs in these frames carry no")
        print("  clicks, so contacts correctly found there are counted as false")
        print("  positives. FPPI and blind area here are LOWER BOUNDS on the")
        print("  method, not estimates of it. Use --scope annotated to compare")
        print("  against evaluate_contact.py.")
    else:
        print("\n  SCOPE = annotated: only the merged boxes of annotated pairs were")
        print("  scored, so this is directly comparable with evaluate_contact.py")
        print("  on the same crops. The difference between the two is what the")
        print("  crop is worth, with everything else held fixed.")
    if failed:
        print(f"\n  {failed} frames skipped (video or SAM 3 failure); excluded "
              "from every number above")

    keys = ["rel_image", "is_control", "n_points", "n_covered", "sensitivity",
            "n_clusters", "n_fp_clusters", "area_px", "a_i", "lift",
            "hit_quality", "gt_disc_px", "blind_area_px", "blind_frac",
            "n_sam3_instances", "n_sam3_pairs", "n_yolo_pairs"]
    with open(os.path.join(out_dir, "wholeframe.csv"), "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=keys)
        wtr.writeheader()
        for r in rows:
            wtr.writerow({k: r.get(k, "") for k in keys})
    with open(os.path.join(out_dir, "wholeframe.json"), "w") as f:
        json.dump({"split": args.split, "scope": args.scope, "text": args.text,
                   "iou_low": args.iou_low, "iou_high": args.iou_high,
                   "reading": args.reading, "dilate_px": args.dilate_px,
                   "n_frames": n_img, "n_points": tot_pts,
                   "sensitivity": sensitivity, "fppi": fppi, "a_bar": a_bar,
                   "lift": lift, "hit_quality": hit_q, "blind_area_frac": blind,
                   "sam3_instances_mean": float(np.mean(det_sam3)),
                   "sam3_pairs_mean": float(np.mean(pair_sam3)),
                   "annotated_pairs_mean": float(np.mean(pair_yolo))}, f, indent=2)
    print(f"\n[wf] wrote {out_dir}")


if __name__ == "__main__":
    main()
