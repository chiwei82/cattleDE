"""SAM 3 on the whole frame: instances, then pairs by BOX IoU, then the figure.

Usage (from the repository root):

    python -m contactTest.wholeframe_pairs --split train --limit 12
    python -m contactTest.wholeframe_pairs --split train --text cow --conf 0.25
    python -m contactTest.wholeframe_pairs --split train --crop-to-pair

Writes to contactTest/log/wholeframe_pairs/<split>/ only.

WHAT IT DOES

    frame  ──►  SAM 3, text prompt  ──►  masks + boxes + scores
                                              │
                    every unordered pair of instances
                                              │
                    iou_low < box_iou(box_i, box_j) < iou_high
                                              │
                    dilate(mask_i, r) AND dilate(mask_j, r)
                                              │
                    the three-panel figure from visualize_sam3_confusion

The frame is uncropped: no detector, no candidate region, nothing chosen in
advance. SAM 3 finds the animals, and the SAME overlap rule the detector stage
used decides which of them are a pair.

PAIRING IS ON BOXES, AND ONLY ON BOXES

`box_iou(boxes[i], boxes[j])` where `boxes` is SAM 3's own `pred_boxes`. Not the
extent of the masks. The rule being reproduced is the one interaction_prep
applied to YOLO's `box.xyxy`, so the substitute detector has to supply the same
kind of quantity; a box measured off a mask is something neither pipeline
produced, and pairing on it would make this a comparison between YOLO and a
post-processing step rather than between two detectors.

The masks are used for the region and for nothing else. That split — boxes
decide WHICH pair, masks decide WHERE the contact is — is the whole reason
Sam3.detect returns both, index-aligned.

WHAT THE FIGURE SHOWS

Per pair, the three panels of visualize_sam3_confusion, built by the same
function so the two are readable against each other:

    1  the frame with the two instances' boxes
    2  the two instance masks
    3  the uncertainty product, with the contact band outlined in green

`--crop-to-pair` renders each pair cut to its own merged box instead of the full
1920x1080, because at frame scale a pair is a few percent of the picture and
nothing about it can be judged. The measurement is identical either way; only
the framing of the image changes.

THIS DRAWS, IT DOES NOT SCORE

No ground truth is read and no metric is computed. Scoring lives in
evaluate_wholeframe.py, which applies the annotation and the six metrics. This
is for looking at what the pairing rule actually selects.
"""

import argparse
import collections
import csv
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contactTest.evaluate_wholeframe import FrameSource
from contactTest.sam3 import Sam3, box_iou
from contactTest.sam_contact_region import contact_readings
from contactTest.src.data import load_records, split_records
from contactTest.src.utils import load_config
from contactTest.visualize_sam3_confusion import panel, uncertainty

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))


def merged_box(a, b, pad, shape):
    h, w = shape
    x1 = max(0, int(min(a[0], b[0])) - pad)
    y1 = max(0, int(min(a[1], b[1])) - pad)
    x2 = min(w, int(max(a[2], b[2])) + pad)
    y2 = min(h, int(max(a[3], b[3])) + pad)
    return (x1, y1, x2, y2) if x2 > x1 + 8 and y2 > y1 + 8 else None


def banner(width, lines, height=42):
    bar = np.full((height, width, 3), 245, np.uint8)
    longest = max((len(t) for t, _ in lines), default=1)
    scale = float(np.clip((width - 16) / (longest * 19.0), 0.28, 0.46))
    for i, (text, col) in enumerate(lines):
        cv2.putText(bar, text, (8, 17 + i * 17), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    col, 1, cv2.LINE_AA)
    return bar


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(CONTACT_ROOT, "config.yaml"))
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--video-root", default=None,
                    help="default: data.video_dir from config.yaml")
    ap.add_argument("--weights", default=None,
                    help="Hugging Face id or a local snapshot DIRECTORY")
    ap.add_argument("--text", default="cow")
    ap.add_argument("--conf", type=float, default=0.85)
    ap.add_argument("--iou-low", type=float, default=None,
                    help="default: data.pair_iou_low, mirroring what prep used")
    ap.add_argument("--iou-high", type=float, default=None)
    ap.add_argument("--dilate-px", type=int, default=22)
    ap.add_argument("--touch-px", type=int, default=10)
    ap.add_argument("--strip-px", type=int, default=6)
    ap.add_argument("--frames", type=int, default=6, help="frames to process")
    ap.add_argument("--limit", type=int, default=24, help="pairs to draw")
    ap.add_argument("--crop-to-pair", action="store_true",
                    help="render each pair cut to its own merged box. At frame "
                         "scale a pair is a few percent of the picture")
    ap.add_argument("--pad", type=int, default=60,
                    help="context round the pair when --crop-to-pair")
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.video_root is None:
        args.video_root = cfg["data"].get("video_dir")
    if args.iou_low is None:
        args.iou_low = float(cfg["data"].get("pair_iou_low", 0.1))
    if args.iou_high is None:
        args.iou_high = float(cfg["data"].get("pair_iou_high", 0.8))

    records = split_records(load_records(cfg, require_label=False))[args.split]
    if not records:
        raise SystemExit(f"no rows in split '{args.split}'")
    by_video = collections.defaultdict(set)
    for r in records:
        by_video[r["source_video"]].add(int(r["frame_number"]))

    src = FrameSource(args.video_root)
    if not src.index:
        raise SystemExit(f"no videos under {args.video_root}")

    # Resolve the videos before loading a 3.45 GB checkpoint.
    missing = {os.path.splitext(v)[0] for v in by_video} - set(src.index)
    if missing == {os.path.splitext(v)[0] for v in by_video}:
        raise SystemExit(f"none of the videos were found under {args.video_root}")

    seg = Sam3(args.weights, args.text, args.conf)

    out_dir = os.path.join(CONTACT_ROOT, "log", "wholeframe_pairs", args.split)
    os.makedirs(out_dir, exist_ok=True)

    rows, drawn, n_frames = [], 0, 0
    for video in sorted(by_video):
        if n_frames >= args.frames:
            break
        wanted = sorted(by_video[video])[:args.frames - n_frames]
        for fno, frame in src.frames_for(video, wanted):
            n_frames += 1
            H, W = frame.shape[:2]

            # Continuous masks: panel 3 is an uncertainty product and needs the
            # per-pixel score, not a threshold of it.
            masks, boxes, scores = seg.detect(frame, binary=False)
            usable = [k for k, b in enumerate(boxes) if b is not None]
            if len(usable) < 2:
                print(f"[wfp] {video} {fno}: fewer than two instances with a "
                      "pred_box; nothing to pair")
                continue
            masks = [masks[k] for k in usable]
            boxes = [boxes[k] for k in usable]
            scores = [scores[k] for k in usable]
            hard = [(m > 0.5).astype(np.uint8) for m in masks]

            # THE pairing step. Boxes only.
            pairs = [(i, j) for i in range(len(boxes))
                     for j in range(i + 1, len(boxes))
                     if args.iou_low < box_iou(boxes[i], boxes[j]) < args.iou_high]
            print(f"[wfp] {video} {fno}: {len(boxes)} instances -> "
                  f"{len(pairs)} pairs at {args.iou_low}-{args.iou_high} box IoU")

            for (i, j) in pairs:
                band = contact_readings(hard[i], hard[j], args.touch_px,
                                        args.dilate_px, args.strip_px)["dilated"]
                strict = uncertainty(masks[i], masks[j])
                rows.append({
                    "source_video": video, "frame_number": fno,
                    "inst_i": i, "inst_j": j,
                    "box_iou": round(box_iou(boxes[i], boxes[j]), 4),
                    "score_i": round(scores[i], 4), "score_j": round(scores[j], 4),
                    "mask_i_px": int(hard[i].sum()), "mask_j_px": int(hard[j].sum()),
                    "band_px": int(band.sum()),
                })
                if args.no_images or drawn >= args.limit:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bi, bj = boxes[i], boxes[j]
                if args.crop_to_pair:
                    mb = merged_box(bi, bj, args.pad, (H, W))
                    if mb is None:
                        continue
                    x1, y1, x2, y2 = mb
                    sl = (slice(y1, y2), slice(x1, x2))
                    shift = lambda b: (b[0] - x1, b[1] - y1, b[2] - x1, b[3] - y1)
                    row = panel(rgb[sl], [shift(bi), shift(bj)],
                                hard[i][sl], hard[j][sl], strict[sl], band[sl])
                else:
                    row = panel(rgb, [bi, bj], hard[i], hard[j], strict, band)

                row = np.vstack([banner(row.shape[1], [
                    (f"{video}  frame {fno}   instances {i} and {j}   "
                     f"box IoU {box_iou(bi, bj):.3f}"
                     f"   scores {scores[i]:.2f} / {scores[j]:.2f}", (60, 60, 60)),
                    (f"paired on SAM 3 pred_boxes; band is "
                     f"dilate({args.dilate_px}) of the MASKS   "
                     f"band {int(band.sum())} px", (110, 110, 110))]), row])
                cv2.imwrite(os.path.join(
                    out_dir, f"{os.path.splitext(video)[0]}_{fno:08d}"
                             f"_{i:02d}_{j:02d}.jpg"),
                    cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
                drawn += 1

    if not rows:
        raise SystemExit("no pairs formed")

    print(f"\n[wfp] {n_frames} frames -> {len(rows)} pairs, {drawn} drawn")
    iou = np.array([r["box_iou"] for r in rows], float)
    print(f"[wfp] box IoU of the formed pairs: median {np.median(iou):.3f}  "
          f"min {iou.min():.3f}  max {iou.max():.3f}")
    band = np.array([r["band_px"] for r in rows], float)
    print(f"[wfp] band size: median {np.median(band):.0f} px, "
          f"empty on {np.mean(band == 0):.0%} of pairs")
    if np.mean(band == 0) > 0.1:
        print("[wfp] a pair whose band is empty passed the box rule while its "
              "MASKS are further apart than 2x the dilation radius — the boxes "
              "overlap but the animals do not")

    path = os.path.join(out_dir, "pairs.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[wfp] wrote {path}")
    if not args.no_images:
        print(f"[wfp] figures -> {out_dir}")
        print("[wfp] panels: frame + the two boxes | the two masks | "
              "uncertainty + band")


if __name__ == "__main__":
    main()
