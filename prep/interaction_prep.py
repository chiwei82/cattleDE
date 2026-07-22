"""
prepare_interaction_dataset.py

Processes videos to generate data/annotated/annotated_interaction.csv for interaction
detection training.

Pipeline per video:
  1. Sample frames at --sample_fps (default 1).
  2. Run YOLO to detect cattle bounding boxes.
  3. Find pairs where iou_low < IoU(B1, B2) < iou_high (candidate interactions).
  4. For each pair:
     a. Save the union (merged) crop as the context image.
     b. Run HRNet on each individual cattle crop → keypoints in frame-space.
     c. Save both poses as .npy.
  5. Write annotated_interaction.csv; label_v1 / label_v2 are left blank for human annotation.

Output layout (split assigned per source video, 6:2:2 train:val:test, so pairs
from the same video never leak across splits):
  data/interaction/{split}/crops/{video_stem}/frame_XXXXXXXX_pair_XX.jpg
  data/interaction/{split}/poses/{video_stem}/frame_XXXXXXXX_pair_XX_{1,2}.npy
  data/annotated/annotated_interaction.csv

Supports incremental runs — already-processed videos are skipped unless
--overwrite is passed.

Usage (from CattleAct root):
  python scripts/prepare_interaction_dataset.py \\
      --video_dir "/user/work/sf24225/data/Full_behav/Videos to process batch 1 - social contacts" \\
      --output_dir data/interaction \\
      --yolo_ckpt  checkpoints/yolo.pt \\
      --hrnet_ckpt checkpoints/hrnet_w32_ap10k_256x256-18aac840_20211029.pth \\
      [--sample_fps 1] [--iou_low 0.2] [--iou_high 0.7] \\
      [--yolo_conf 0.3] [--yolo_imgsz 1280] [--device cuda] \\
      [--nested_thresh 0.85]
"""

import argparse
import csv
import os
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
import yaml

# hrnet.py lives in customization/src/ (distinct from the repo-root src/ package).
# Inserting customization/ at the front of sys.path makes this src resolve first.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.hrnet import HRNetW32

# ── Config (see global_config.yaml at the repository root) ────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_REPO_ROOT, "global_config.yaml")) as _f:
    _CFG = yaml.safe_load(_f)


def _resolve(p):
    """Resolve config paths relative to the repo root, not the CWD."""
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)

NUM_JOINTS  = _CFG["hrnet"]["num_joints"]
INPUT_SIZE  = _CFG["hrnet"]["input_size"]
HMAP_SIZE   = _CFG["hrnet"]["hmap_size"]

_HRNET_TRANSFORM = T.Compose([
    T.Resize((INPUT_SIZE, INPUT_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

CSV_FIELDNAMES = [
    "image_path", "bbox1_xyxy", "bbox2_xyxy", "merged_bbox_xyxy",
    "pose_path_1", "pose_path_2",
    "label_v1", "label_v2",
    "source_video", "frame_number", "split",
]


# ── HRNet ──────────────────────────────────────────────────────────────────────

class KeypointHead(nn.Module):
    def __init__(self, in_channels: int = 32, num_joints: int = NUM_JOINTS):
        super().__init__()
        self.final_layer = nn.Conv2d(in_channels, num_joints, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_layer(x)


class HRNetPoseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = HRNetW32()
        self.keypoint_head = KeypointHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.keypoint_head(self.backbone(x))


def load_hrnet(ckpt_path: str, device: torch.device) -> HRNetPoseModel:
    model = HRNetPoseModel().to(device)
    raw   = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = raw.get("state_dict", raw.get("model", raw))
    state = {
        k.removeprefix("module.").removeprefix("model.").removeprefix("net."): v
        for k, v in state.items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] HRNet: {len(missing)} missing keys (e.g. {missing[0]})")
    if unexpected:
        print(f"  [WARN] HRNet: {len(unexpected)} unexpected keys (e.g. {unexpected[0]})")
    model.eval()
    return model


def decode_heatmaps(heatmaps: np.ndarray) -> np.ndarray:
    """(N, J, H, W) → (N, J, 3) [x, y, conf] with ±0.25px sub-pixel shift."""
    N, J, H, W = heatmaps.shape
    flat  = heatmaps.reshape(N, J, -1)
    idx   = flat.argmax(axis=2)
    conf  = flat[np.arange(N)[:, None], np.arange(J), idx]
    y = (idx // W).astype(np.float32)
    x = (idx %  W).astype(np.float32)
    for n in range(N):
        for j in range(J):
            py, px = int(y[n, j]), int(x[n, j])
            hm = heatmaps[n, j]
            if 1 <= px < W - 1:
                x[n, j] += 0.25 * np.sign(hm[py, px + 1] - hm[py, px - 1])
            if 1 <= py < H - 1:
                y[n, j] += 0.25 * np.sign(hm[py + 1, px] - hm[py - 1, px])
    return np.stack([x, y, conf], axis=2).astype(np.float32)


@torch.no_grad()
def run_hrnet(crops_pil: List[Image.Image], model: HRNetPoseModel,
              device: torch.device) -> np.ndarray:
    """Returns heatmaps (N, J, H, W)."""
    tensor = torch.stack([_HRNET_TRANSFORM(img) for img in crops_pil]).to(device)
    return model(tensor).cpu().numpy()


def keypoints_to_frame_space(kps_hmap: np.ndarray,
                              crop_w: int, crop_h: int,
                              bbox_x1: int, bbox_y1: int) -> np.ndarray:
    """
    Converts keypoints from heatmap space to original frame pixel space.
    kps_hmap : (J, 3) — [x_hm, y_hm, conf] in [0, HMAP_SIZE]
    Returns  : (J, 3) — [x_px, y_px, conf] in frame coordinates
    """
    kps = kps_hmap.copy()
    kps[:, 0] = kps[:, 0] * (crop_w / HMAP_SIZE) + bbox_x1
    kps[:, 1] = kps[:, 1] * (crop_h / HMAP_SIZE) + bbox_y1
    return kps


# ── Geometry ───────────────────────────────────────────────────────────────────

def compute_iou(b1: Tuple, b2: Tuple) -> float:
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def union_bbox(b1: Tuple, b2: Tuple) -> Tuple[int, int, int, int]:
    return (min(b1[0], b2[0]), min(b1[1], b2[1]),
            max(b1[2], b2[2]), max(b1[3], b2[3]))


def is_nested(b1: Tuple, b2: Tuple, thresh: float = 0.85) -> bool:
    """
    True when the smaller box is almost entirely contained inside the larger
    one. IoU alone can't catch this: if bbox2 sits fully inside bbox1,
    IoU = area(bbox2) / area(bbox1), which lands well inside a normal
    (iou_low, iou_high) range even though both boxes are on the *same*
    animal (e.g. a spurious detection on the head/torso of an already-boxed
    cow). A real pair of two separate, merely-adjacent cattle almost never
    has one box this deeply inside the other.
    """
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    smaller = min(a1, a2)
    return smaller > 0 and inter / smaller > thresh


def fmt_bbox(b: Tuple) -> str:
    return f"[{b[0]} {b[1]} {b[2]} {b[3]}]"


def safe_crop_bgr(frame: np.ndarray, bbox: Tuple) -> np.ndarray:
    h, w = frame.shape[:2]
    x1 = max(0, bbox[0]); y1 = max(0, bbox[1])
    x2 = min(w, bbox[2]); y2 = min(h, bbox[3])
    return frame[y1:y2, x1:x2]


# ── YOLO result parsing (handles both standard and OBB models) ────────────────

def _extract_boxes(results) -> List[Tuple[int, int, int, int]]:
    """
    Returns axis-aligned (x1, y1, x2, y2) boxes from a YOLO result object.
    Works for both standard detection models (results.boxes) and OBB models
    (results.obb) by taking the bounding rectangle of the 4 OBB corners.
    """
    boxes = []
    if results.boxes is not None and len(results.boxes):
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    elif results.obb is not None and len(results.obb):
        # xyxyxyxy: (N, 4, 2) — four corner points per box
        corners = results.obb.xyxyxyxy.cpu().numpy()  # (N, 4, 2)
        for pts in corners:
            x1 = int(pts[:, 0].min()); y1 = int(pts[:, 1].min())
            x2 = int(pts[:, 0].max()); y2 = int(pts[:, 1].max())
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes


# ── Per-video processing ───────────────────────────────────────────────────────

def assign_videos_622(video_names: List[str], seed: int) -> dict:
    """Split source videos 6:2:2 (train:val:test). Assigning at video level
    keeps pairs sampled from the same footage out of both train and test.
    Deterministic for a fixed seed and video set."""
    order = sorted(video_names)
    random.Random(seed).shuffle(order)
    n = len(order)
    if n < 3:
        print(f"[WARN] only {n} video(s); cannot form 6:2:2. All -> train.")
        return {v: "train" for v in order}
    n_test = max(1, round(n * _CFG["split"]["test_ratio"]))
    n_val  = max(1, round(n * _CFG["split"]["val_ratio"]))
    n_test = min(n_test, n - 2)
    n_val  = min(n_val, n - 1 - n_test)
    assignment = {}
    for i, v in enumerate(order):
        if i < n_test:
            assignment[v] = "test"
        elif i < n_test + n_val:
            assignment[v] = "val"
        else:
            assignment[v] = "train"
    return assignment


def process_video(
    video_path: str,
    output_dir: str,
    split: str,
    yolo_model: YOLO,
    hrnet_model: HRNetPoseModel,
    device: torch.device,
    sample_fps: float,
    iou_low: float,
    iou_high: float,
    yolo_conf: float,
    yolo_imgsz: int,
    nested_thresh: float,
) -> List[dict]:
    video_name = os.path.basename(video_path)
    video_stem = Path(video_path).stem.replace(" ", "_")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Cannot open: {video_path}")
        return []

    src_fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step  = max(1, int(round(src_fps / sample_fps)))

    crops_dir = os.path.join(output_dir, split, "crops", video_stem)
    poses_dir = os.path.join(output_dir, split, "poses", video_stem)
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(poses_dir, exist_ok=True)

    rows      = []
    frame_idx = 0
    pbar      = tqdm(total=total_frames // frame_step,
                     desc=video_name[:40], leave=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            # ── Detection ─────────────────────────────────────────────────────
            results = yolo_model.predict(
                source=frame, conf=yolo_conf, imgsz=yolo_imgsz, verbose=False,
            )[0]
            boxes = _extract_boxes(results)

            # ── Find interacting pairs ─────────────────────────────────────────
            for pair_idx, (i, j) in enumerate(combinations(range(len(boxes)), 2)):
                bbox1, bbox2 = boxes[i], boxes[j]
                iou = compute_iou(bbox1, bbox2)
                if not (iou_low < iou < iou_high):
                    continue
                if is_nested(bbox1, bbox2, nested_thresh):
                    continue

                merged = union_bbox(bbox1, bbox2)
                merged_crop = safe_crop_bgr(frame, merged)
                crop1       = safe_crop_bgr(frame, bbox1)
                crop2       = safe_crop_bgr(frame, bbox2)
                if merged_crop.size == 0 or crop1.size == 0 or crop2.size == 0:
                    continue

                # ── Save context crop ─────────────────────────────────────────
                # CSV paths are stored relative to the repo root so the CSV in
                # data/annotated/ is unambiguous about where the files live.
                stem     = f"frame_{frame_idx:08d}_pair_{pair_idx:02d}"
                crop_abs = os.path.join(output_dir, split, "crops", video_stem,
                                        f"{stem}.jpg")
                crop_rel = os.path.relpath(crop_abs, _REPO_ROOT)
                cv2.imwrite(crop_abs, merged_crop)

                # ── HRNet pose (both cattle in one forward pass) ───────────────
                pil1 = Image.fromarray(cv2.cvtColor(crop1, cv2.COLOR_BGR2RGB))
                pil2 = Image.fromarray(cv2.cvtColor(crop2, cv2.COLOR_BGR2RGB))
                heatmaps = run_hrnet([pil1, pil2], hrnet_model, device)  # (2,J,64,64)
                kps_all  = decode_heatmaps(heatmaps)                      # (2,J,3)

                # Convert to frame-space coordinates
                kps1 = keypoints_to_frame_space(
                    kps_all[0], crop1.shape[1], crop1.shape[0], bbox1[0], bbox1[1])
                kps2 = keypoints_to_frame_space(
                    kps_all[1], crop2.shape[1], crop2.shape[0], bbox2[0], bbox2[1])

                pose_abs1 = os.path.join(output_dir, split, "poses", video_stem,
                                         f"{stem}_1.npy")
                pose_abs2 = os.path.join(output_dir, split, "poses", video_stem,
                                         f"{stem}_2.npy")
                pose_rel1 = os.path.relpath(pose_abs1, _REPO_ROOT)
                pose_rel2 = os.path.relpath(pose_abs2, _REPO_ROOT)
                np.save(pose_abs1, kps1)
                np.save(pose_abs2, kps2)

                rows.append({
                    "image_path":       crop_rel,
                    "bbox1_xyxy":       fmt_bbox(bbox1),
                    "bbox2_xyxy":       fmt_bbox(bbox2),
                    "merged_bbox_xyxy": fmt_bbox(merged),
                    "pose_path_1":      pose_rel1,
                    "pose_path_2":      pose_rel2,
                    "label_v1":         "",
                    "label_v2":         "",
                    "source_video":     video_name,
                    "frame_number":     frame_idx,
                    "split":            split,
                })

            pbar.update(1)

        frame_idx += 1

    pbar.close()
    cap.release()
    return rows


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build annotated_interaction.csv for interaction detection training."
    )
    icfg = _CFG["interaction_prep"]
    parser.add_argument("--video_dir",  required=True,
                        help="Folder containing input videos.")
    parser.add_argument("--output_dir", default=icfg["output_dir"])
    parser.add_argument("--yolo_ckpt",  default=_CFG["paths"]["yolo_ckpt"])
    parser.add_argument("--hrnet_ckpt", default=_CFG["paths"]["hrnet_ckpt"])
    parser.add_argument("--sample_fps", type=float, default=icfg["sample_fps"],
                        help="Frames per second to sample (default 1).")
    parser.add_argument("--iou_low",    type=float, default=icfg["iou_low"])
    parser.add_argument("--iou_high",   type=float, default=icfg["iou_high"])
    parser.add_argument("--yolo_conf",  type=float, default=icfg["yolo_conf"])
    parser.add_argument("--yolo_imgsz", type=int,   default=icfg["yolo_imgsz"])
    parser.add_argument("--nested_thresh", type=float, default=icfg["nested_thresh"],
                        help="Reject a pair if the smaller box has more than "
                             "this fraction of its area inside the larger box "
                             "(catches two detections on the same animal, "
                             "which would otherwise crop only one cow).")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite",  action="store_true",
                        help="Re-process videos already in annotated_interaction.csv.")
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output_dir = _resolve(args.output_dir)
    args.yolo_ckpt  = _resolve(args.yolo_ckpt)
    args.hrnet_ckpt = _resolve(args.hrnet_ckpt)
    os.makedirs(args.output_dir, exist_ok=True)
    annotated_dir = _resolve(_CFG["paths"]["annotated_dir"])
    os.makedirs(annotated_dir, exist_ok=True)
    csv_path = os.path.join(annotated_dir, "annotated_interaction.csv")

    # Load existing rows so the script can be run incrementally
    existing_rows     = []
    processed_videos  = set()
    if os.path.exists(csv_path) and not args.overwrite:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            existing_rows    = list(reader)
            processed_videos = {r["source_video"] for r in existing_rows}
        print(f"Resuming: {len(existing_rows)} existing rows, "
              f"{len(processed_videos)} videos already done.")

    video_exts = {".avi", ".mp4", ".mov", ".mkv"}
    video_paths = sorted(
        p for p in Path(args.video_dir).iterdir()
        if p.suffix.lower() in video_exts
    )
    to_process = [p for p in video_paths if p.name not in processed_videos]

    print(f"Found {len(video_paths)} videos, {len(to_process)} to process.")
    if not to_process:
        print("Nothing to do. Use --overwrite to reprocess.")
        return

    # Assign splits over the full video set so the assignment stays stable
    # across incremental runs (fixed seed, same video list).
    assignment = assign_videos_622([p.name for p in video_paths],
                                   _CFG["random_seed"])
    print("Video split (6:2:2):")
    for name, sp in sorted(assignment.items(), key=lambda kv: (kv[1], kv[0])):
        print(f"  [{sp}] {name}")

    print("Loading YOLO …")
    yolo_model  = YOLO(args.yolo_ckpt)
    print("Loading HRNet …")
    hrnet_model = load_hrnet(args.hrnet_ckpt, device)

    all_new_rows = []
    for vp in tqdm(to_process, desc="Videos"):
        rows = process_video(
            video_path  = str(vp),
            output_dir  = args.output_dir,
            split       = assignment[vp.name],
            yolo_model  = yolo_model,
            hrnet_model = hrnet_model,
            device      = device,
            sample_fps  = args.sample_fps,
            iou_low     = args.iou_low,
            iou_high    = args.iou_high,
            yolo_conf   = args.yolo_conf,
            yolo_imgsz  = args.yolo_imgsz,
            nested_thresh = args.nested_thresh,
        )
        all_new_rows.extend(rows)
        tqdm.write(f"  {vp.name}: {len(rows)} pairs saved")

    combined = existing_rows + all_new_rows
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(combined)

    print(f"\nDone. {len(all_new_rows)} new rows → total {len(combined)} in {csv_path}")
    print(f"Next: annotate label_v1 / label_v2 in {csv_path}, then run:")
    print(f"  python -m train.interaction_with_image data.num_workers=0")


if __name__ == "__main__":
    main()
