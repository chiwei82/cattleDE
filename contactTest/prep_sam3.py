"""Rebuild the interaction dataset with SAM 3 in place of YOLO, nothing else changed.

Usage (from the repository root):

    python -m contactTest.prep_sam3 --split train --weights sam3.pt
    python -m contactTest.prep_sam3 --split train --weights sam3.pt --all-frames
    # then, on the dataset it wrote:
    python -m contactTest.evaluate_contact --split train --weights sam3.pt \\
        --config contactTest/log/prep_sam3/config_sam3.yaml \\
        --gt contactTest/log/prep_sam3/contact_gt_sam3.csv

Writes only under contactTest/log/prep_sam3/ — the original data/interaction
tree and its CSV are never touched.

WHAT IS HELD FIXED, AND WHAT IS SWAPPED

Swapped: the detector. YOLO's boxes become the bounding boxes of SAM 3's
concept-segmented instances (`--text cow`).

Held fixed, deliberately identically: the pair rule (iou_low < IoU < iou_high),
the nested-box rejection, the merged box, the border-clipped crop, the file
naming and the CSV schema. compute_iou, union_bbox and is_nested below are
copied verbatim from prep/interaction_prep.py rather than reimplemented, so
"downstream unchanged" is a fact about the code and not a claim about it. If
that file ever changes, these have to be brought back into step.

WHAT THIS SHOWS, AND WHAT IT DOES NOT

It tests whether the DETECTOR is interchangeable. If the same downstream reaches
comparable numbers from SAM 3 boxes, then the result rests on the pipeline
structure — detect, pair by overlap, crop the union — rather than on YOLO
specifically, and the detector is a replaceable component.

It does NOT show the pipeline is necessary. That is a different experiment and
it has already been run: evaluate_wholeframe.py removes the crop stage entirely
and sensitivity falls from 0.877 to 0.055. Necessity comes from that; this
script only says whether the first stage has to be YOLO.

THE GROUND TRUTH HAS TO MOVE WITH THE CROPS

SAM 3's boxes differ from YOLO's, so the merged boxes differ, so the crops are
different images and the clicked points no longer sit where they did. Each click
is therefore carried through crop -> frame -> new crop, and a new
contact_gt_sam3.csv is written in the new crops' coordinates. Pairs are matched
between the two datasets by merged-box IoU, and how often that match exists at
all is reported: an annotated pair with no counterpart is not a scoring failure
but a pairing failure, and the two must not be added together.
"""

import argparse
import collections
import csv
import json
import os
import sys
from itertools import combinations

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contactTest.evaluate_wholeframe import FrameSource, mask_box
from contactTest.score_contact import read_gt
from contactTest.src.data import load_records, split_records
from contactTest.src.utils import load_config

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CONTACT_ROOT, ".."))


# ── Copied verbatim from prep/interaction_prep.py ─────────────────────────────
# Duplicated rather than imported: that module pulls in YOLO, HRNet and torch at
# import time, and importing it to reuse four small pure functions would make
# this script fail for reasons unrelated to what it does. The point of the
# experiment is that these are unchanged, so they are copied character for
# character rather than rewritten.

def compute_iou(b1, b2):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def union_bbox(b1, b2):
    return (min(b1[0], b2[0]), min(b1[1], b2[1]),
            max(b1[2], b2[2]), max(b1[3], b2[3]))


def is_nested(b1, b2, thresh=0.85):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    smaller = min(a1, a2)
    return smaller > 0 and inter / smaller > thresh


def fmt_bbox(b):
    return f"[{b[0]} {b[1]} {b[2]} {b[3]}]"


def safe_crop_bgr(frame, bbox):
    h, w = frame.shape[:2]
    x1 = max(0, bbox[0]); y1 = max(0, bbox[1])
    x2 = min(w, bbox[2]); y2 = min(h, bbox[3])
    return frame[y1:y2, x1:x2]
# ── end verbatim block ────────────────────────────────────────────────────────


CSV_FIELDS = ["image_path", "bbox1_xyxy", "bbox2_xyxy", "merged_bbox_xyxy",
              "bbox_confs", "pose_path_1", "pose_path_2", "label_v1", "label_v2",
              "source_video", "frame_number", "split"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(CONTACT_ROOT, "config.yaml"))
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--video-root", default=None)
    ap.add_argument("--weights", default="sam3.pt")
    ap.add_argument("--text", default="cow")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-low", type=float, default=None)
    ap.add_argument("--iou-high", type=float, default=None)
    ap.add_argument("--nested-thresh", type=float, default=0.85,
                    help="mirrors interaction_prep.nested_thresh")
    ap.add_argument("--out-dir", default=os.path.join("log", "prep_sam3"),
                    help="relative to contactTest/; nothing outside it is written")
    ap.add_argument("--all-frames", action="store_true",
                    help="every sampled frame of the split. The default is the "
                         "frames your clicks live in, which is all the "
                         "comparison needs and far cheaper")
    ap.add_argument("--match-iou", type=float, default=0.3,
                    help="least merged-box IoU before a SAM 3 pair counts as the "
                         "counterpart of an annotated pair")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.video_root is None:
        args.video_root = cfg["data"].get("video_dir")
    if args.iou_low is None:
        args.iou_low = float(cfg["data"].get("pair_iou_low", 0.1))
    if args.iou_high is None:
        args.iou_high = float(cfg["data"].get("pair_iou_high", 0.8))

    gt_path = args.gt or os.path.join(CONTACT_ROOT, "log", "annotate", args.split,
                                      "contact_gt.csv")
    gt = read_gt(gt_path) if os.path.exists(gt_path) else {}
    by_rel = {r["rel_image"]: r
              for r in split_records(load_records(cfg, require_label=False))[args.split]}

    # Frames to process, and the annotated pairs sitting in each.
    wanted = collections.defaultdict(list)
    for rel, ann in gt.items():
        if ann["status"] == "skip":
            continue
        rec = by_rel.get(rel)
        if rec is not None:
            wanted[(rec["source_video"], int(rec["frame_number"]))].append((rel, ann, rec))
    if args.all_frames:
        for rec in by_rel.values():
            wanted.setdefault((rec["source_video"], int(rec["frame_number"])), [])
    if not wanted:
        raise SystemExit("no frames to process")

    src = FrameSource(args.video_root)
    missing = {os.path.splitext(v)[0] for (v, _) in wanted} - set(src.index)
    if missing:
        raise SystemExit(f"{len(missing)} videos not found under {args.video_root}: "
                         + ", ".join(sorted(missing)[:4]))

    out_root = os.path.join(CONTACT_ROOT, args.out_dir)
    os.makedirs(out_root, exist_ok=True)

    from contactTest.precompute_masks import _SAM3Text
    seg = _SAM3Text(args.weights, args.text, args.conf)

    rows, gt_rows = [], []
    n_frames = n_pairs = 0
    matched = unmatched = 0
    pts_carried = pts_lost = 0

    for (video, fno), items in sorted(wanted.items()):
        if args.limit and n_frames >= args.limit:
            break
        frame, err = src.get(video, fno)
        if frame is None:
            print(f"[prep3] {video} {fno}: {err}")
            continue
        H, W = frame.shape[:2]
        video_stem = os.path.splitext(video)[0].replace(" ", "_")

        inst = seg.instances(frame)
        boxes = []
        for m in inst:
            b = mask_box(m)
            if b is None:
                continue
            # Integer xyxy plus a confidence slot, so the tuple layout matches
            # what _extract_boxes produced for YOLO. SAM 3's concept output has
            # no per-instance score exposed here, so it is recorded as 1.0
            # rather than invented.
            boxes.append((int(b[0]), int(b[1]), int(b[2]), int(b[3]), 1.0))
        if len(boxes) < 2:
            continue
        n_frames += 1

        crops_dir = os.path.join(out_root, args.split, "crops", video_stem)
        os.makedirs(crops_dir, exist_ok=True)

        made_here = []
        for pair_idx, (i, j) in enumerate(combinations(range(len(boxes)), 2)):
            bbox1, bbox2 = boxes[i], boxes[j]
            iou = compute_iou(bbox1, bbox2)
            if not (args.iou_low < iou < args.iou_high):
                continue
            if is_nested(bbox1, bbox2, args.nested_thresh):
                continue
            merged = union_bbox(bbox1, bbox2)
            merged_crop = safe_crop_bgr(frame, merged)
            if merged_crop.size == 0:
                continue

            stem = f"frame_{fno:08d}_pair_{pair_idx:02d}"
            crop_abs = os.path.join(crops_dir, f"{stem}.jpg")
            cv2.imwrite(crop_abs, merged_crop)
            rel_new = os.path.relpath(crop_abs, REPO_ROOT)
            rows.append({
                "image_path": rel_new,
                "bbox1_xyxy": fmt_bbox(bbox1), "bbox2_xyxy": fmt_bbox(bbox2),
                "merged_bbox_xyxy": fmt_bbox(merged),
                "bbox_confs": f"{bbox1[4]:.4f} {bbox2[4]:.4f}",
                "pose_path_1": "", "pose_path_2": "",
                "label_v1": "", "label_v2": "",
                "source_video": video, "frame_number": fno, "split": args.split,
            })
            made_here.append((rel_new, merged))
            n_pairs += 1

        # Carry each annotated pair's clicks into whichever new pair corresponds
        # to it, matched on merged-box overlap.
        for rel, ann, rec in items:
            old_m = rec["merged"]
            best, best_iou = None, 0.0
            for rel_new, new_m in made_here:
                v = compute_iou(old_m, new_m)
                if v > best_iou:
                    best, best_iou = (rel_new, new_m), v
            if best is None or best_iou < args.match_iou:
                unmatched += 1
                pts_lost += len(ann["points"])
                continue
            matched += 1
            rel_new, new_m = best
            ox_o, oy_o = max(0, int(old_m[0])), max(0, int(old_m[1]))
            ox_n, oy_n = max(0, int(new_m[0])), max(0, int(new_m[1]))
            ch = min(H, int(new_m[3])) - oy_n
            cw = min(W, int(new_m[2])) - ox_n
            if ann["status"] == "none":
                gt_rows.append({"rel_image": rel_new, "status": "none",
                                "x": "", "y": ""})
                continue
            for (x, y) in ann["points"]:
                fx, fy = x + ox_o, y + oy_o          # crop -> frame
                nx, ny = fx - ox_n, fy - oy_n        # frame -> new crop
                if 0 <= nx < cw and 0 <= ny < ch:
                    gt_rows.append({"rel_image": rel_new, "status": "point",
                                    "x": int(nx), "y": int(ny)})
                    pts_carried += 1
                else:
                    # The click landed outside the new merged box. Dropping it
                    # silently would flatter the new dataset by removing the
                    # very points its boxes failed to enclose.
                    pts_lost += 1

    if not rows:
        raise SystemExit("no pairs produced")

    csv_path = os.path.join(out_root, f"annotated_interaction_sam3_{args.split}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    gt_out = os.path.join(out_root, f"contact_gt_sam3_{args.split}.csv")
    with open(gt_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rel_image", "status", "x", "y"])
        w.writeheader()
        w.writerows(gt_rows)

    # A config pointing at this dataset, so evaluate_contact runs on it unchanged.
    cfg_out = dict(cfg)
    cfg_out["data"] = dict(cfg["data"])
    cfg_out["data"]["csv"] = os.path.relpath(csv_path, REPO_ROOT)
    cfg_path = os.path.join(out_root, f"config_sam3_{args.split}.yaml")
    with open(cfg_path, "w") as f:
        json.dump(cfg_out, f, indent=2)     # YAML is a superset of JSON

    tot_pts = pts_carried + pts_lost
    print(f"\n[prep3] {n_frames} frames -> {n_pairs} pairs "
          f"({n_pairs / max(n_frames, 1):.1f} per frame)")
    print(f"[prep3] rule held fixed: {args.iou_low} < IoU < {args.iou_high}, "
          f"nested > {args.nested_thresh} rejected")
    print(f"\n[prep3] annotated pairs matched to a SAM 3 pair: "
          f"{matched}/{matched + unmatched}"
          f"  ({matched / max(matched + unmatched, 1):.0%})")
    if unmatched:
        print(f"[prep3] {unmatched} had no counterpart at merged-box IoU >= "
              f"{args.match_iou}. Those are PAIRING failures, not scoring")
        print("[prep3] failures, and the metrics below cannot see them - quote "
              "this rate beside any score from this dataset.")
    print(f"[prep3] clicks carried across: {pts_carried}/{tot_pts} "
          f"({pts_carried / max(tot_pts, 1):.0%}); {pts_lost} fell outside the "
          "new crops")
    print(f"\n[prep3] dataset  {csv_path}")
    print(f"[prep3] mapped GT {gt_out}")
    print(f"[prep3] config    {cfg_path}")
    print("\n[prep3] score it with the same six metrics:")
    print(f"  python -m contactTest.evaluate_contact --split {args.split} "
          f"--weights sam3.pt \\\n"
          f"      --config {os.path.relpath(cfg_path, REPO_ROOT)} \\\n"
          f"      --gt {os.path.relpath(gt_out, REPO_ROOT)}")


if __name__ == "__main__":
    main()
