"""Show what SAM 3 sees in a whole frame: instances first, then the pairs.

Usage (from the repository root):

    python -m contactTest.visualize_wholeframe --split train --weights sam3.pt
    python -m contactTest.visualize_wholeframe --split train --weights sam3.pt \\
        --limit 12 --only-failed
    python -m contactTest.visualize_wholeframe --split train --weights sam3.pt \\
        --no-zoom

Writes to contactTest/log/wholeframe_vis/<split>/ only.

This exists to explain the whole-frame result rather than restate it. On this
footage SAM 3 finds ~35 cattle per frame and forms ~46 pairs, yet sensitivity
inside the annotated region collapses. Only three things can be happening, and
they call for different responses, so the figure is built to tell them apart:

  the two animals were never separated   SAM 3 merged them into one instance,
                                         which is a segmentation failure and
                                         precisely the ability contact detection
                                         needs most
  one of them was not found at all       a detection failure at frame scale
  both found, but not PAIRED             their mask boxes hug the animals and
                                         overlap less than the detector boxes
                                         did, so the 0.1 lower bound - chosen
                                         for detector boxes - rejects them.
                                         A threshold problem, not a model one.

LAYOUT

  top-left      the frame, with the annotated pair's detector boxes and the
                clicked points mapped back from crop coordinates
  top-right     every SAM 3 instance in its own colour
  bottom-left   the pairs: a line between the centroids of every paired
                instance, with the two instances matching the annotated pair
                drawn thick and labelled with what happened to them
  bottom-right  the contact bands those pairs produce, with the GT points
                marked covered or not

  Each panel is repeated zoomed to the annotated pair underneath, because at
  1920x1080 with thirty-odd animals the pair in question is a few percent of
  the picture and nothing can be judged from the full frame alone. --no-zoom
  drops that row.
"""

import argparse
import collections
import colorsys
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contactTest.evaluate_wholeframe import (FrameSource, box_iou, crop_to_frame,
                                             mask_box)
from contactTest.sam_contact_region import contact_readings
from contactTest.score_contact import read_gt
from contactTest.src.data import load_records, records_for
from contactTest.src.utils import load_config

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))

C_BOX1, C_BOX2 = (255, 190, 60), (80, 170, 255)
C_GT = (250, 220, 90)
C_HIT, C_MISS = (90, 235, 110), (240, 80, 70)
C_BAND = (120, 245, 130)


def instance_colour(k):
    """A distinct colour per instance; ~35 per frame have to stay tellable apart."""
    h = (k * 0.61803398875) % 1.0          # golden ratio: consecutive ids differ
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def centroid(mask):
    ys, xs = np.nonzero(mask)
    return (int(xs.mean()), int(ys.mean())) if len(xs) else None


def label(img, text, org, colour, scale=0.6):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4,
                cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1,
                cv2.LINE_AA)


def panel_instances(rgb, inst):
    out = rgb.astype(np.float32).copy()
    for k, m in enumerate(inst):
        sel = m > 0
        out[sel] = out[sel] * 0.45 + np.asarray(instance_colour(k), np.float32) * 0.55
    out = out.astype(np.uint8)
    for k, m in enumerate(inst):
        cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnt, -1, instance_colour(k), 2, cv2.LINE_AA)
    return out


def panel_pairs(rgb, inst, pairs, focus, verdict):
    """Centroid links for every pair, with the annotated one picked out."""
    out = (rgb.astype(np.float32) * 0.55).astype(np.uint8)
    cents = [centroid(m) for m in inst]
    for k, m in enumerate(inst):
        cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnt, -1, instance_colour(k), 1, cv2.LINE_AA)
    for (i, j) in pairs:
        if cents[i] and cents[j]:
            cv2.line(out, cents[i], cents[j], (235, 235, 235), 1, cv2.LINE_AA)
    for (i, j) in pairs:
        if cents[i] and cents[j]:
            for c in (cents[i], cents[j]):
                cv2.circle(out, c, 4, (235, 235, 235), -1, cv2.LINE_AA)

    i1, i2 = focus
    col = (90, 240, 120) if verdict == "formed" else (245, 90, 80)
    for idx in {i1, i2}:
        if idx is None or idx >= len(inst):
            continue
        cnt, _ = cv2.findContours(inst[idx].astype(np.uint8), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnt, -1, col, 4, cv2.LINE_AA)
    if i1 is not None and i2 is not None and i1 != i2 \
            and cents[i1] and cents[i2]:
        cv2.line(out, cents[i1], cents[i2], col, 3, cv2.LINE_AA)
    return out


def panel_bands(rgb, region, points, covered):
    out = rgb.astype(np.float32).copy()
    out[region] = out[region] * 0.45 + np.asarray(C_BAND, np.float32) * 0.55
    out = out.astype(np.uint8)
    cnt, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_CCOMP,
                              cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnt, -1, C_BAND, 2, cv2.LINE_AA)
    for (x, y), ok in zip(points, covered):
        cv2.circle(out, (x, y), 7, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), 6, C_HIT if ok else C_MISS, -1, cv2.LINE_AA)
    return out


def panel_reference(rgb, b1, b2, points):
    out = rgb.copy()
    for b, c in ((b1, C_BOX1), (b2, C_BOX2)):
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), c, 3)
    for (x, y) in points:
        cv2.circle(out, (x, y), 7, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), 6, C_GT, -1, cv2.LINE_AA)
    return out


def grid(tiles, titles, width):
    """Two-by-two, each tile scaled to `width` and captioned."""
    out = []
    for t, cap in zip(tiles, titles):
        h, w = t.shape[:2]
        s = width / float(w)
        # INTER_AREA is a downscaling filter and smears when it enlarges. The
        # zoom row crops a small region and then enlarges it, which is exactly
        # the case that matters most here, so the filter follows the direction.
        t = cv2.resize(t, (width, max(1, int(round(h * s)))),
                       interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC)
        label(t, cap, (10, 26), (255, 255, 255), 0.7)
        out.append(t)
    hgt = max(t.shape[0] for t in out)
    out = [np.vstack([t, np.full((hgt - t.shape[0], t.shape[1], 3), 245, np.uint8)])
           for t in out]
    gap = np.full((hgt, 6, 3), 245, np.uint8)
    rows = [np.hstack([out[0], gap, out[1]]), np.hstack([out[2], gap, out[3]])]
    hgap = np.full((6, rows[0].shape[1], 3), 245, np.uint8)
    return np.vstack([rows[0], hgap, rows[1]])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(CONTACT_ROOT, "config.yaml"))
    ap.add_argument("--split", default="train",
                    choices=["train", "val", "test", "all", "known_interact"],
                    help="which ground-truth set to read. 'all' and "
                         "'known_interact' are the sets annotate_contact "
                         "now builds; they are not CSV splits, so the row "
                         "index spans everything and the GT decides "
                         "membership")
    ap.add_argument("--video-root", default=None)
    ap.add_argument("--weights", default=None,
                    help="Hugging Face id or a local snapshot DIRECTORY")
    ap.add_argument("--text", default="cow")
    ap.add_argument("--conf", type=float, default=None,
                    help="SAM 3 score floor; default is data.sam3_conf "
                         "from config.yaml, which mirrors the value the "
                         "detector stage used")
    ap.add_argument("--iou-low", type=float, default=None)
    ap.add_argument("--iou-high", type=float, default=None)
    ap.add_argument("--match-iou", type=float, default=0.3)
    ap.add_argument("--reading", default="dilated",
                    choices=["overlap", "gap", "surface", "dilated"])
    ap.add_argument("--dilate-px", type=int, default=22)
    ap.add_argument("--touch-px", type=int, default=10)
    ap.add_argument("--strip-px", type=int, default=6)
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--tile-width", type=int, default=760)
    ap.add_argument("--zoom-pad", type=int, default=120,
                    help="context around the annotated pair in the zoom row")
    ap.add_argument("--no-zoom", action="store_true")
    ap.add_argument("--only-failed", action="store_true",
                    help="only frames where the annotated pair was NOT formed, "
                         "which is where the explanation is")
    ap.add_argument("--gt", default=None)
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

    gt_path = args.gt or os.path.join(CONTACT_ROOT, "log", "annotate", args.split,
                                      "contact_gt.csv")
    if not os.path.exists(gt_path):
        raise SystemExit(f"no ground truth at {gt_path}")
    gt = read_gt(gt_path)
    by_rel = {r["rel_image"]: r
              for r in records_for(load_records(cfg, require_label=False), args.split)}

    items = []
    for rel, ann in gt.items():
        if ann["status"] == "skip":
            continue
        rec = by_rel.get(rel)
        if rec is not None:
            items.append((rel, ann, rec))
    if not items:
        raise SystemExit("no annotated crops resolve to a frame")

    src = FrameSource(args.video_root)
    if not src.index:
        raise SystemExit(f"no videos under {args.video_root}")

    from contactTest.sam3 import Sam3
    seg = Sam3(args.weights, args.text, args.conf)

    out_dir = os.path.join(CONTACT_ROOT, "log", "wholeframe_vis", args.split)
    os.makedirs(out_dir, exist_ok=True)

    made = 0
    tally = {"formed": 0, "outrange": 0, "merged": 0, "unsegmented": 0}

    # Sequential decoding, as everywhere else: seeking addresses the decoder's
    # position rather than the read counter prep stored, and on this HEVC
    # footage it also returns pictures rebuilt from references it never decoded.
    by_video = collections.defaultdict(list)
    for t in items:
        by_video[t[2]["source_video"]].append(t)
    decoded = {}
    for video, ts in by_video.items():
        want = sorted({int(t[2]["frame_number"]) for t in ts})
        for fno, fr in src.frames_for(video, want):
            decoded[(video, fno)] = fr

    for rel, ann, rec in sorted(items, key=lambda t: t[0]):
        if made >= args.limit:
            break
        frame = decoded.get((rec["source_video"], int(rec["frame_number"])))
        if frame is None:
            print(f"[wfv] {rel}: frame not reached by a sequential read")
            continue
        H, W = frame.shape[:2]
        inst, boxes3, _ = seg.detect(frame)   # SAM 3's own boxes
        if not inst:
            continue
        keep = [k for k, b in enumerate(boxes3) if b is not None]
        inst = [inst[k] for k in keep]
        boxes3 = [boxes3[k] for k in keep]
        if len(inst) < 2:
            continue

        pairs = [(i, j) for i in range(len(inst)) for j in range(i + 1, len(inst))
                 if args.iou_low < box_iou(boxes3[i], boxes3[j]) < args.iou_high]

        b1, b2 = rec["bbox1"], rec["bbox2"]
        i1 = max(range(len(inst)), key=lambda k: box_iou(boxes3[k], b1))
        i2 = max(range(len(inst)), key=lambda k: box_iou(boxes3[k], b2))
        m1, m2 = box_iou(boxes3[i1], b1), box_iou(boxes3[i2], b2)
        pair_iou = box_iou(boxes3[i1], boxes3[i2]) if i1 != i2 else 1.0
        if min(m1, m2) < args.match_iou:
            verdict, why = "unsegmented", (
                f"an animal was not segmented (best IoU {min(m1, m2):.2f} "
                f"< {args.match_iou})")
        elif i1 == i2:
            verdict, why = "merged", "both detector boxes matched ONE instance"
        elif not (args.iou_low < pair_iou < args.iou_high):
            verdict, why = "outrange", (
                f"both found, but mask-box IoU {pair_iou:.3f} is outside "
                f"{args.iou_low}-{args.iou_high}  (detector boxes had "
                f"{box_iou(b1, b2):.3f})")
        else:
            verdict, why = "formed", f"pair formed, mask-box IoU {pair_iou:.3f}"
        tally[verdict] += 1
        if args.only_failed and verdict == "formed":
            continue

        region = np.zeros((H, W), bool)
        for (i, j) in pairs:
            region |= contact_readings(inst[i], inst[j], args.touch_px,
                                       args.dilate_px, args.strip_px)[args.reading]

        pts = (crop_to_frame(ann["points"], rec["merged"], (H, W))
               if ann["status"] != "none" else [])
        cov = [bool(region[y, x]) for (x, y) in pts]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tiles = [panel_reference(rgb, b1, b2, pts),
                 panel_instances(rgb, inst),
                 panel_pairs(rgb, inst, pairs, (i1, i2), verdict),
                 panel_bands(rgb, region, pts, cov)]
        caps = ["1  frame + annotated pair + GT",
                f"2  SAM 3 instances ({len(inst)})",
                f"3  pairs ({len(pairs)}) - thick = the annotated one",
                f"4  contact bands  ({sum(cov)}/{len(pts)} GT covered)"]
        img = grid(tiles, caps, args.tile_width)

        if not args.no_zoom:
            x1 = max(0, int(min(b1[0], b2[0])) - args.zoom_pad)
            y1 = max(0, int(min(b1[1], b2[1])) - args.zoom_pad)
            x2 = min(W, int(max(b1[2], b2[2])) + args.zoom_pad)
            y2 = min(H, int(max(b1[3], b2[3])) + args.zoom_pad)
            if x2 > x1 + 8 and y2 > y1 + 8:
                zt = [t[y1:y2, x1:x2] for t in tiles]
                zoom = grid(zt, ["1  zoom", "2  zoom", "3  zoom", "4  zoom"],
                            args.tile_width)
                sep = np.full((10, max(img.shape[1], zoom.shape[1]), 3), 245, np.uint8)
                wdt = max(img.shape[1], zoom.shape[1])
                img = np.vstack([
                    np.hstack([img, np.full((img.shape[0], wdt - img.shape[1], 3),
                                            245, np.uint8)]),
                    sep,
                    np.hstack([zoom, np.full((zoom.shape[0], wdt - zoom.shape[1], 3),
                                             245, np.uint8)])])

        bar = np.full((66, img.shape[1], 3), 245, np.uint8)
        col = (25, 110, 40) if verdict == "formed" else (170, 40, 30)
        label(bar, f"{os.path.basename(rel)}   frame {rec['frame_number']}",
              (10, 22), (60, 60, 60), 0.6)
        label(bar, f"VERDICT: {why}", (10, 44), col, 0.62)
        img = np.vstack([bar, img])

        cv2.imwrite(os.path.join(out_dir, f"{verdict}_{os.path.basename(rel)}"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        made += 1

    print(f"\n[wfv] wrote {made} figures to {out_dir}")
    tot = sum(tally.values())
    if tot:
        print(f"[wfv] over the {tot} annotated pairs reached:")
        for k, v in tally.items():
            print(f"      {k:<13}{v:>4}  {v / tot:>5.0%}")
    print("[wfv] filenames start with the verdict, so the failures group together")
    print("[wfv] panel 3: thick outline = the two instances matching your boxes;")
    print("      green means they were paired, red means they were not")


if __name__ == "__main__":
    main()
