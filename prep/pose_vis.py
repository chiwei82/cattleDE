"""
Visualize HRNet pose estimation on random dataset frames.

Randomly samples frames from data/object/{train,val,test}/images/, runs the
same YOLO -> crop -> HRNet pipeline as interaction_prep.py, and draws the
17 AP-10K keypoints + skeleton of every detected cow onto the full frame.
Results are written to simu/pose/.

Usage (from the repo root):
  python prep/pose_vis.py [--num 20] [--conf_kp 0.3] [--device cuda]
"""

import argparse
import os
import random
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interaction_prep import (
    _CFG, _REPO_ROOT, _resolve,
    load_hrnet, run_hrnet, decode_heatmaps, keypoints_to_frame_space,
    _extract_boxes, safe_crop_bgr,
)

DATASET_DIR = _resolve(_CFG["yolo_prep"]["output_dir"])
OUT_DIR     = os.path.join(_REPO_ROOT, "simu", "pose")

# AP-10K joint order (0-indexed):
#  0 L_eye  1 R_eye  2 nose  3 neck  4 tail_root
#  5 L_shoulder  6 L_elbow  7 L_front_paw
#  8 R_shoulder  9 R_elbow 10 R_front_paw
# 11 L_hip      12 L_knee  13 L_back_paw
# 14 R_hip      15 R_knee  16 R_back_paw
SKELETON = [
    (0, 1), (0, 2), (1, 2), (2, 3), (3, 4),
    (3, 5), (5, 6), (6, 7),
    (3, 8), (8, 9), (9, 10),
    (4, 11), (11, 12), (12, 13),
    (4, 14), (14, 15), (15, 16),
]

KP_COLOR   = (0, 0, 255)      # BGR: red joints
LINE_COLOR = (0, 255, 255)    # yellow bones
BOX_COLOR  = (0, 255, 0)      # green detection box


def draw_pose(img, kps, conf_thresh):
    """kps: (17, 3) [x, y, conf] in frame pixels."""
    for a, b in SKELETON:
        if kps[a, 2] >= conf_thresh and kps[b, 2] >= conf_thresh:
            cv2.line(img, (int(kps[a, 0]), int(kps[a, 1])),
                     (int(kps[b, 0]), int(kps[b, 1])), LINE_COLOR, 2)
    for x, y, c in kps:
        if c >= conf_thresh:
            cv2.circle(img, (int(x), int(y)), 4, KP_COLOR, -1)


def main():
    icfg = _CFG["interaction_prep"]
    pcfg = _CFG["pose_vis"]
    parser = argparse.ArgumentParser(
        description="Draw HRNet skeletons on random dataset frames."
    )
    parser.add_argument("--num",       type=int,   default=pcfg["num_samples"],
                        help="Number of frames to sample.")
    parser.add_argument("--conf_kp",   type=float, default=pcfg["conf_kp"],
                        help="Min keypoint confidence to draw.")
    parser.add_argument("--yolo_ckpt",  default=_CFG["paths"]["yolo_ckpt"])
    parser.add_argument("--hrnet_ckpt", default=_CFG["paths"]["hrnet_ckpt"])
    parser.add_argument("--yolo_conf",  type=float, default=icfg["yolo_conf"])
    parser.add_argument("--yolo_imgsz", type=int,   default=icfg["yolo_imgsz"])
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Collect all dataset images across splits
    all_imgs = []
    for split in ("train", "val", "test"):
        d = os.path.join(DATASET_DIR, split, "images")
        if os.path.isdir(d):
            all_imgs += [os.path.join(d, f) for f in os.listdir(d)
                         if f.endswith(".jpg")]
    if not all_imgs:
        print(f"No images found under {DATASET_DIR}. Run prep/yolo_prep.py first.")
        return

    random.seed(_CFG["random_seed"])
    sampled = random.sample(all_imgs, min(args.num, len(all_imgs)))
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device(args.device)
    print("Loading YOLO ...")
    yolo_model = YOLO(_resolve(args.yolo_ckpt))
    print("Loading HRNet ...")
    hrnet_model = load_hrnet(_resolve(args.hrnet_ckpt), device)

    for img_path in sampled:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  [WARN] cannot read {img_path}, skipping.")
            continue

        results = yolo_model.predict(source=frame, conf=args.yolo_conf,
                                     imgsz=args.yolo_imgsz, verbose=False)[0]
        boxes = _extract_boxes(results)

        crops, valid_boxes = [], []
        for bbox in boxes:
            crop = safe_crop_bgr(frame, bbox)
            if crop.size == 0:
                continue
            crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            valid_boxes.append((bbox, crop.shape[1], crop.shape[0]))

        if crops:
            kps_all = decode_heatmaps(run_hrnet(crops, hrnet_model, device))
            for kps_hm, (bbox, cw, ch) in zip(kps_all, valid_boxes):
                kps = keypoints_to_frame_space(kps_hm, cw, ch, bbox[0], bbox[1])
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                              BOX_COLOR, 2)
                draw_pose(frame, kps, args.conf_kp)

        cv2.putText(frame, f"cows: {len(valid_boxes)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        out_path = os.path.join(OUT_DIR, os.path.basename(img_path))
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  {os.path.basename(img_path)}: {len(valid_boxes)} cows -> {out_path}")

    print(f"\nDone. Visualizations in {OUT_DIR}")


if __name__ == "__main__":
    main()
