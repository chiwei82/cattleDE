"""Overlay AP-10K pose on interaction pairs, and report whether it is usable.

Usage (from the repository root):

    # positives from the training split, with per-joint confidence report
    python -m contactTest.visualize_pose --split train --limit 40

    # negatives too, as a control
    python -m contactTest.visualize_pose --split val --include-negatives --limit 60

    # confidence statistics only, no images written
    python -m contactTest.visualize_pose --split train --limit 400 --no-images

Writes to contactTest/log/pose/<split>/ only.

This is a GATE, not a feature. HRNet-W32/AP-10K was trained on ground-level
animal photography; an overhead pen camera with heavy occlusion is out of
distribution. Everything that pose could buy this project — a canonical output
vocabulary, a tighter candidate region, part-aware auxiliary supervision —
depends on the keypoints actually landing on the right anatomy, and a wrong
prior is worse than the loose box prior currently in use. Read the confidence
table and look at the overlays before building anything on top of pose.

Each overlay also draws the shortest head-to-body link between the two animals.
That link is a training-free contact estimate: 380 of the 384 positive pairs are
head-initiated, so if the pose is sound the segment should already sit on the
contact site. Whether it does is the single most useful thing this script tells
you.
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contactTest.src.data import load_records, relative_boxes, split_records
from contactTest.src.pose import (HEAD_JOINTS, KEYPOINT_NAMES, SKELETON,
                                  build_transform, closest_head_link,
                                  load_pose_model, pose_for_pair)
from contactTest.src.utils import load_config

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))

# BGR. Cow 1 warm, cow 2 cool, so the two skeletons stay separable when they
# overlap - which, in an interacting pair, they always do.
COW_COLOURS = [(60, 160, 255), (255, 180, 60)]
HEAD_COLOUR = (60, 60, 255)
LINK_COLOUR = (80, 255, 80)


def draw_pose(canvas, keypoints, boxes, min_conf, draw_labels=True):
    """Draw both skeletons, their boxes, and the head-to-body link."""
    for i in range(keypoints.shape[0]):
        colour = COW_COLOURS[i % len(COW_COLOURS)]
        x1, y1, x2, y2 = boxes[i]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, f"cow{i + 1}", (x1 + 3, y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

        for a, b in SKELETON:
            if keypoints[i, a, 2] < min_conf or keypoints[i, b, 2] < min_conf:
                continue
            pa = tuple(np.round(keypoints[i, a, :2]).astype(int))
            pb = tuple(np.round(keypoints[i, b, :2]).astype(int))
            cv2.line(canvas, pa, pb, colour, 2, lineType=cv2.LINE_AA)

        for j in range(keypoints.shape[1]):
            if keypoints[i, j, 2] < min_conf:
                continue
            p = tuple(np.round(keypoints[i, j, :2]).astype(int))
            is_head = j in HEAD_JOINTS
            cv2.circle(canvas, p, 4 if is_head else 3, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(canvas, p, 3 if is_head else 2,
                       HEAD_COLOUR if is_head else colour, -1, cv2.LINE_AA)
            if draw_labels and is_head:
                cv2.putText(canvas, KEYPOINT_NAMES[j][:4], (p[0] + 5, p[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, HEAD_COLOUR, 1, cv2.LINE_AA)

    link = closest_head_link(keypoints, min_conf)
    if link is not None:
        pa, pb, dist, na, nb = link
        pa_i = tuple(np.round(pa).astype(int))
        pb_i = tuple(np.round(pb).astype(int))
        cv2.line(canvas, pa_i, pb_i, (20, 20, 20), 4, lineType=cv2.LINE_AA)
        cv2.line(canvas, pa_i, pb_i, LINK_COLOUR, 2, lineType=cv2.LINE_AA)
        mid = ((pa_i[0] + pb_i[0]) // 2, (pa_i[1] + pb_i[1]) // 2)
        cv2.circle(canvas, mid, 10, (20, 20, 20), 3, lineType=cv2.LINE_AA)
        cv2.circle(canvas, mid, 10, LINK_COLOUR, 1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, f"{na} <-> {nb}  {dist:.0f}px", (6, canvas.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(canvas, f"{na} <-> {nb}  {dist:.0f}px", (6, canvas.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, LINK_COLOUR, 1, cv2.LINE_AA)
    return canvas, link


def banner(canvas, text):
    """Prepend a caption strip so each overlay is self-describing."""
    bar = np.full((26, canvas.shape[1], 3), 245, np.uint8)
    cv2.putText(bar, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (25, 25, 25), 1, cv2.LINE_AA)
    return np.vstack([bar, canvas])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(CONTACT_ROOT, "config.yaml"))
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--include-negatives", action="store_true",
                    help="also render non-interacting pairs, as a control")
    ap.add_argument("--min-conf", type=float, default=None,
                    help="overrides pose.min_conf from the config")
    ap.add_argument("--no-images", action="store_true",
                    help="only compute the confidence report")
    ap.add_argument("--no-labels", action="store_true",
                    help="do not print head joint names on the overlay")
    args = ap.parse_args()

    cfg = load_config(args.config)
    min_conf = args.min_conf if args.min_conf is not None else float(cfg["pose"]["min_conf"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = split_records(load_records(cfg))[args.split]
    if not args.include_negatives:
        records = [r for r in records if r["label"] == 1]
    if not records:
        raise SystemExit(f"no matching rows in split '{args.split}'")
    records = records[:args.limit]
    print(f"[pose] {len(records)} pairs from '{args.split}' on {device}")

    model = load_pose_model(cfg, device)
    transform = build_transform(cfg)

    out_dir = os.path.join(CONTACT_ROOT, "log", "pose", args.split)
    os.makedirs(out_dir, exist_ok=True)

    n_joints = int(cfg["pose"]["num_joints"])
    conf_sum = np.zeros(n_joints)
    conf_hit = np.zeros(n_joints)
    rows, link_dists = [], []

    for i, record in enumerate(records):
        crop = cv2.imread(record["image_path"])
        if crop is None:
            continue
        keypoints, boxes = pose_for_pair(record, crop, model, transform, cfg, device)

        for cow in range(keypoints.shape[0]):
            conf_sum += keypoints[cow, :, 2]
            conf_hit += (keypoints[cow, :, 2] >= min_conf)

        canvas = crop.copy()
        canvas, link = draw_pose(canvas, keypoints, boxes, min_conf,
                                 draw_labels=not args.no_labels)

        kind = record["label_v2"] or ("interaction" if record["label"] else "no_interaction")
        name = f"{i:03d}_{kind.replace(' ', '-')}_{os.path.basename(record['rel_image'])}"
        if not args.no_images:
            head = (f"{kind}  |  {record['source_video'].replace('.mp4', '')}"
                    f"  frame {record['frame_number']}")
            cv2.imwrite(os.path.join(out_dir, name), banner(canvas, head))

        head_conf = float(keypoints[:, HEAD_JOINTS, 2].max())
        row = {
            "file": name,
            "rel_image": record["rel_image"],
            "label": record["label"],
            "label_v2": record["label_v2"],
            "source_video": record["source_video"],
            "frame_number": record["frame_number"],
            "mean_conf": f"{float(keypoints[:, :, 2].mean()):.4f}",
            "max_head_conf": f"{head_conf:.4f}",
            "n_joints_above_thresh": int((keypoints[:, :, 2] >= min_conf).sum()),
        }
        if link is not None:
            row.update({"link_a": link[3], "link_b": link[4],
                        "link_dist_px": f"{link[2]:.1f}"})
            link_dists.append(link[2])
        else:
            row.update({"link_a": "", "link_b": "", "link_dist_px": ""})
        rows.append(row)

        if (i + 1) % 20 == 0:
            print(f"[pose] processed {i + 1}/{len(records)}")

    n_cows = 2 * len(rows)
    order = np.argsort(-conf_sum)
    print(f"\n[pose] per-joint confidence over {n_cows} animals "
          f"(threshold {min_conf}):")
    print(f"       {'joint':<18} {'mean conf':>10} {'% above':>9}")
    for j in order:
        print(f"       {KEYPOINT_NAMES[j]:<18} {conf_sum[j] / n_cows:>10.3f} "
              f"{100 * conf_hit[j] / n_cows:>8.1f}%")

    head_mean = float(conf_sum[HEAD_JOINTS].sum() / (len(HEAD_JOINTS) * n_cows))
    head_rate = float(conf_hit[HEAD_JOINTS].sum() / (len(HEAD_JOINTS) * n_cows))
    print(f"\n[pose] head joints: mean conf {head_mean:.3f}, "
          f"{100 * head_rate:.1f}% above threshold")
    if link_dists:
        d = np.array(link_dists)
        print(f"[pose] head-to-body link: median {np.median(d):.0f} px, "
              f"p10 {np.percentile(d, 10):.0f}, p90 {np.percentile(d, 90):.0f} "
              f"({len(d)}/{len(rows)} pairs had one)")

    if head_rate < 0.5:
        verdict = ("STOP - head joints are unreliable on this camera angle. A "
                   "wrong anatomical prior is worse than the loose box prior; do "
                   "not build the region or the keypoint output space on this.")
    elif head_rate < 0.8:
        verdict = ("MIXED - usable as auxiliary supervision, too shaky to define "
                   "the candidate region or the output vocabulary.")
    else:
        verdict = ("OK - head joints are reliable enough to define a keypoint "
                   "output space and a tighter candidate region.")
    print(f"[pose] verdict: {verdict}")
    print("[pose] confidence is the model's own certainty, not correctness — "
          "look at the overlays before trusting it.")

    with open(os.path.join(out_dir, "pose_report.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(out_dir, "pose_summary.json"), "w") as f:
        json.dump({
            "split": args.split, "n_pairs": len(rows), "min_conf": min_conf,
            "per_joint_mean_conf": {KEYPOINT_NAMES[j]: float(conf_sum[j] / n_cows)
                                    for j in range(n_joints)},
            "per_joint_above_thresh": {KEYPOINT_NAMES[j]: float(conf_hit[j] / n_cows)
                                       for j in range(n_joints)},
            "head_mean_conf": head_mean, "head_above_thresh": head_rate,
            "link_dist_median_px": float(np.median(link_dists)) if link_dists else None,
            "verdict": verdict,
        }, f, indent=2)
    print(f"[pose] wrote {out_dir}")


if __name__ == "__main__":
    main()
