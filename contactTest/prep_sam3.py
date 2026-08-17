"""Build the interaction dataset with SAM 3 as the detector. Same pipeline as prep.

Usage (from the repository root):

    python -m contactTest.prep_sam3 --output-dir data/interaction_sam3
    python -m contactTest.prep_sam3 --output-dir data/interaction_sam3 --limit 2

Mirrors prep/interaction_prep.py step for step:

    1. discover every video under --video-dir
    2. assign splits 6:2:2 per video, same seed, same rule
    3. walk each video at --sample-fps, counting reads
    4. DETECT  <- the only substitution: SAM 3 concept prompting, not YOLO
    5. pair when iou_low < IoU(box1, box2) < iou_high and not nested
    6. save the union crop
    7. write the CSV, same columns, label_v1 / label_v2 blank for annotation

Nothing is read from the existing dataset. No CSV, no ground truth, no split
lookup: the videos are the input, exactly as they are for prep. An earlier
version of this file started from annotated_interaction_test.csv and rebuilt
parts of it, which meant it could only ever visit frames where YOLO had already
found a pair — the one thing a replacement detector must not be limited to.

WHAT IS SUBSTITUTED, AND WHAT IS NOT

Substituted: the detector. `YOLO.predict(...)` and `box.xyxy` become SAM 3
concept segmentation and `pred_boxes`. Both emit an axis-aligned box per animal
plus a confidence, so the stage below consumes the same kind of thing.

Not substituted: compute_iou, union_bbox, is_nested, safe_crop_bgr, the file
naming, the CSV columns. Those are copied verbatim from interaction_prep rather
than reimplemented, so "downstream unchanged" is a fact about the code and not a
claim about it.

Pairing is on BOXES. SAM 3 also returns masks, and they are better localised,
but prep paired on boxes and the point of this file is that only the detector
differs. Masks are not used here at all.

THRESHOLDS

iou_low, iou_high, nested_thresh, sample_fps and the 6:2:2 ratios all come from
config.yaml, mirroring interaction_prep. `conf` is data.sam3_conf, set to the
same value as yolo_conf so neither detector is quietly given the easier
threshold — though the two are different quantities and equal numbers do not
mean equal strictness.
"""

import argparse
import csv
import os
import random
import sys
from itertools import combinations
from pathlib import Path

import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contactTest.sam3 import Sam3
from contactTest.src.utils import load_config

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CONTACT_ROOT, ".."))

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}
CSV_FIELDS = ["image_path", "bbox1_xyxy", "bbox2_xyxy", "merged_bbox_xyxy",
              "bbox_confs", "pose_path_1", "pose_path_2", "label_v1", "label_v2",
              "source_video", "frame_number", "split"]


# ── Copied verbatim from prep/interaction_prep.py ─────────────────────────────
# Duplicated rather than imported: that module pulls in YOLO, HRNet and torch at
# import time, so importing it to reuse five small pure functions would make this
# script fail for reasons unrelated to what it does. The experiment rests on
# these being unchanged, so they are copied character for character.

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


def assign_videos_622(video_names, seed, val_ratio=0.2, test_ratio=0.2):
    """Split source videos 6:2:2. Same rule and same seed as prep, so a video
    lands in the same split in both datasets and the two stay comparable."""
    order = sorted(video_names)
    random.Random(seed).shuffle(order)
    n = len(order)
    if n < 3:
        print(f"[WARN] only {n} video(s); cannot form 6:2:2. All -> train.")
        return {v: "train" for v in order}
    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio))
    n_test = min(n_test, n - 2)
    n_val = min(n_val, n - 1 - n_test)
    assignment = {}
    for i, v in enumerate(order):
        if i < n_test:
            assignment[v] = "test"
        elif i < n_test + n_val:
            assignment[v] = "val"
        else:
            assignment[v] = "train"
    return assignment
# ── end verbatim block ────────────────────────────────────────────────────────


def detect_boxes(seg, frame):
    """SAM 3's boxes for one frame, as (x1, y1, x2, y2, conf) integers.

    The counterpart of prep's _extract_boxes, which read YOLO's box.xyxy and
    box.conf. Same tuple layout, so everything below indexes it identically.

    Masks are discarded here on purpose. prep paired on boxes; pairing on masks
    would be a second substitution and the comparison would no longer isolate
    the detector.
    """
    _, boxes, scores = seg.detect(frame)
    out = []
    for b, sc in zip(boxes, scores):
        if b is None:
            continue
        x1, y1, x2, y2 = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2, y2, float(sc)))
    return out


def process_video(video_path, output_dir, split, seg, sample_fps,
                  iou_low, iou_high, nested_thresh):
    """Every sampled frame of one video, detected, paired and written out.

    The loop is prep's: read straight through, count successful reads, act when
    the counter is a multiple of frame_step. `frame_number` is that counter, so
    it means the same thing in both datasets. Seeking would address the
    decoder's position instead, and on this HEVC footage it also returns frames
    rebuilt from references it never decoded, with cap.read() still true.
    """
    video_name = os.path.basename(video_path)
    video_stem = Path(video_path).stem.replace(" ", "_")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] cannot open: {video_path}")
        return []

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_step = max(1, int(round(src_fps / sample_fps)))

    crops_dir = os.path.join(output_dir, split, "crops", video_stem)
    os.makedirs(crops_dir, exist_ok=True)

    rows = []
    frame_idx = 0
    n_sampled = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            n_sampled += 1
            boxes = detect_boxes(seg, frame)

            for pair_idx, (i, j) in enumerate(combinations(range(len(boxes)), 2)):
                bbox1, bbox2 = boxes[i], boxes[j]
                iou = compute_iou(bbox1, bbox2)
                if not (iou_low < iou < iou_high):
                    continue
                if is_nested(bbox1, bbox2, nested_thresh):
                    continue

                merged = union_bbox(bbox1, bbox2)
                merged_crop = safe_crop_bgr(frame, merged)
                if merged_crop.size == 0:
                    continue

                stem = f"frame_{frame_idx:08d}_pair_{pair_idx:02d}"
                crop_abs = os.path.join(crops_dir, f"{stem}.jpg")
                cv2.imwrite(crop_abs, merged_crop)
                rows.append({
                    "image_path": os.path.relpath(crop_abs, REPO_ROOT),
                    "bbox1_xyxy": fmt_bbox(bbox1),
                    "bbox2_xyxy": fmt_bbox(bbox2),
                    "merged_bbox_xyxy": fmt_bbox(merged),
                    "bbox_confs": f"{bbox1[4]:.4f} {bbox2[4]:.4f}",
                    "pose_path_1": "", "pose_path_2": "",
                    "label_v1": "", "label_v2": "",
                    "source_video": video_name,
                    "frame_number": frame_idx,
                    "split": split,
                })
        frame_idx += 1

    cap.release()
    print(f"  {video_name}: {n_sampled} frames sampled -> {len(rows)} pairs")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(CONTACT_ROOT, "config.yaml"))
    ap.add_argument("--video-dir", default=None,
                    help="default: data.video_dir from config.yaml")
    ap.add_argument("--output-dir", default="data/interaction_sam3",
                    help="dataset root, from the repository root. Refuses to "
                         "write into a non-empty directory")
    ap.add_argument("--weights", default=None,
                    help="Hugging Face id or a local snapshot DIRECTORY")
    ap.add_argument("--text", default="cow")
    ap.add_argument("--conf", type=float, default=None,
                    help="SAM 3 score floor; default is data.sam3_conf")
    ap.add_argument("--sample-fps", type=float, default=None,
                    help="default: data.sample_fps")
    ap.add_argument("--iou-low", type=float, default=None)
    ap.add_argument("--iou-high", type=float, default=None)
    ap.add_argument("--nested-thresh", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N videos, for a trial run")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = cfg["data"]
    if args.video_dir is None:
        args.video_dir = d.get("video_dir")
    if args.conf is None:
        args.conf = float(d.get("sam3_conf", 0.6))
    if args.sample_fps is None:
        args.sample_fps = float(d.get("sample_fps", 1.0))
    if args.iou_low is None:
        args.iou_low = float(d.get("pair_iou_low", 0.1))
    if args.iou_high is None:
        args.iou_high = float(d.get("pair_iou_high", 0.8))

    out_root = (args.output_dir if os.path.isabs(args.output_dir)
                else os.path.join(REPO_ROOT, args.output_dir))
    if os.path.isdir(out_root) and os.listdir(out_root):
        raise SystemExit(
            f"{out_root} exists and is not empty.\n"
            "Refusing to write into it: a half-written dataset is worse than "
            "none, and the crops are what every later measurement reads.\n"
            "Remove it or pass a different --output-dir.")

    video_paths = sorted(p for p in Path(args.video_dir).iterdir()
                         if p.suffix.lower() in VIDEO_EXTS)
    if not video_paths:
        raise SystemExit(f"no videos under {args.video_dir}")

    # Assigned over the FULL video set so the split is stable regardless of
    # --limit, exactly as prep assigns over the full list regardless of which
    # videos it happens to process on a given run.
    assignment = assign_videos_622([p.name for p in video_paths],
                                   int(cfg.get("random_seed", 42)))
    print(f"Found {len(video_paths)} videos under {args.video_dir}")
    print("Video split (6:2:2):")
    for name, sp in sorted(assignment.items(), key=lambda kv: (kv[1], kv[0])):
        print(f"  [{sp}] {name}")

    to_process = video_paths[:args.limit] if args.limit else video_paths
    print(f"\nsample_fps={args.sample_fps}  "
          f"{args.iou_low} < IoU < {args.iou_high}  "
          f"nested>{args.nested_thresh}  conf={args.conf}  text={args.text!r}")

    seg = Sam3(args.weights, args.text, args.conf)

    all_rows = []
    for vp in to_process:
        all_rows.extend(process_video(
            video_path=str(vp), output_dir=out_root,
            split=assignment[vp.name], seg=seg,
            sample_fps=args.sample_fps, iou_low=args.iou_low,
            iou_high=args.iou_high, nested_thresh=args.nested_thresh))

    if not all_rows:
        raise SystemExit("no pairs produced")

    os.makedirs(out_root, exist_ok=True)
    csv_path = os.path.join(out_root, "annotated_interaction_sam3.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(all_rows)

    per_split = {}
    for r in all_rows:
        per_split[r["split"]] = per_split.get(r["split"], 0) + 1
    print(f"\nDone. {len(all_rows)} pairs -> {csv_path}")
    print("  " + "  ".join(f"{k}: {v}" for k, v in sorted(per_split.items())))
    print("\nlabel_v1 / label_v2 are blank, as prep leaves them. The clicked")
    print("contact points are NOT carried over: they live in the old crops'")
    print("coordinates, and these crops come from different boxes, so mapping")
    print("them is a separate step with its own losses rather than something")
    print("to do silently here.")


if __name__ == "__main__":
    main()
