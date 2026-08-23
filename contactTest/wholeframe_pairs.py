
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
from contactTest.sam_contact_region import dilated_band
from contactTest.src.data import load_records, records_for
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
    ap.add_argument("--split", default="train",
                    choices=["train", "val", "test", "all", "known_interact"])
    ap.add_argument("--video-root", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--text", default="cow")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--iou-low", type=float, default=None)
    ap.add_argument("--iou-high", type=float, default=None)
    ap.add_argument("--dilate-px", type=int, default=22)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--crop-to-pair", action="store_true")
    ap.add_argument("--pad", type=int, default=60)
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.conf is None:
        args.conf = float(cfg["data"].get("sam3_conf", 0.6))
    if args.video_root is None:
        args.video_root = cfg["data"].get("video_dir")
    if args.iou_low is None:
        args.iou_low = float(cfg["data"].get("pair_iou_low", 0.1))
    if args.iou_high is None:
        args.iou_high = float(cfg["data"].get("pair_iou_high", 0.8))

    records = records_for(load_records(cfg, require_label=False), args.split)
    if not records:
        raise SystemExit(f"no rows in split '{args.split}'")
    by_video = collections.defaultdict(set)
    for r in records:
        by_video[r["source_video"]].add(int(r["frame_number"]))

    src = FrameSource(args.video_root)
    if not src.index:
        raise SystemExit(f"no videos under {args.video_root}")

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

            pairs = [(i, j) for i in range(len(boxes))
                     for j in range(i + 1, len(boxes))
                     if args.iou_low < box_iou(boxes[i], boxes[j]) < args.iou_high]
            print(f"[wfp] {video} {fno}: {len(boxes)} instances -> "
                  f"{len(pairs)} pairs at {args.iou_low}-{args.iou_high} box IoU")

            for (i, j) in pairs:
                band = dilated_band(hard[i], hard[j], args.dilate_px)
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

    path = os.path.join(out_dir, "pairs.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[wfp] wrote {path}")
    if not args.no_images:
        print(f"[wfp] figures -> {out_dir}")


if __name__ == "__main__":
    main()
