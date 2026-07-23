"""
Prepare action dataset from multi-session behaviour clips.

Expected directory structure (set CLIPS_BASE_DIR below):
  clips/
    {session_id}/
      {behavior}/
        {tracklet_id}_{start_frame}_{end_frame}.mp4

Clips are already per-cow crops — frames are extracted directly.

Label mapping (7 classes):
  standing      -> 0
  stepping      -> 1
  walking       -> 2
  lying         -> 3
  transitiondown -> 4
  transitionup  -> 5
  sniffingbed   -> 6

Split:
  train:val:test = 6:2:2, assigned per (session_id, tracklet_id) group so
  that every clip/frame of the same cow-track lands in exactly one split.
  Frames sampled from the same or a neighbouring clip are near-duplicates;
  splitting at frame or clip level would leak them into val/test and
  inflate acc/f1. The split is encoded in the output path, so downstream
  loading must use split_type "date" in src.dataset.split_action_dataset_entries.

Pose:
  If HRNET_CKPT exists, an AP-10K HRNet (17 keypoints) runs on every sampled
  frame and the keypoints [x, y, conf] are saved in crop-pixel coordinates
  (CattleActionDataset normalizes them by the crop image size). This enables
  the skeleton-masking augmentation during training. Without the checkpoint,
  pose_path stays empty and training falls back to image-only augmentation.

Output:
  data/action/{split}/crops/{label}/{session}_{clip}_f{n:04d}.jpg
  data/action/{split}/poses/{label}/{session}_{clip}_f{n:04d}.npy  (if HRNet)
  data/annotated/annotated_action.csv
"""

import csv
import os
import random
import sys

import cv2
import numpy as np
import yaml
from PIL import Image
from collections import Counter

# Reuse the HRNet inference helpers from interaction_prep (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Config (see global_config.yaml at the repository root) ────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_REPO_ROOT, "global_config.yaml")) as _f:
    _CFG = yaml.safe_load(_f)


def _resolve(p):
    """Resolve config paths relative to the repo root, not the CWD."""
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


CLIPS_BASE_DIR = _resolve(_CFG["paths"]["clips_dir"])
OUTPUT_DIR     = _resolve(_CFG["action_prep"]["output_dir"])
SAMPLE_EVERY   = _CFG["action_prep"]["sample_every"]
VAL_RATIO      = _CFG["split"]["val_ratio"]
TEST_RATIO     = _CFG["split"]["test_ratio"]
RANDOM_SEED    = _CFG["random_seed"]
HRNET_CKPT     = _resolve(_CFG["paths"]["hrnet_ckpt"])
POSE_BATCH     = _CFG["action_prep"]["pose_batch"]

SPLITS = ("train", "val", "test")

LABEL_MAP = {label: label for label in _CFG["action_prep"]["labels"]}


def load_pose_model():
    """Load the AP-10K HRNet if use_pose is on and HRNET_CKPT exists;
    otherwise continue without pose."""
    if not _CFG["action_prep"]["use_pose"]:
        print("use_pose is off — skipping HRNet; pose_path will be empty.")
        return None, None
    if not os.path.exists(HRNET_CKPT):
        print(f"[WARN] HRNet ckpt not found at '{HRNET_CKPT}'. "
              "pose_path will be empty and skeleton augmentation stays inactive.")
        return None, None
    import torch
    from interaction_prep import load_hrnet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_hrnet(HRNET_CKPT, device)
    print(f"HRNet loaded from '{HRNET_CKPT}' on {device}.")
    return model, device


def estimate_poses(frames_bgr, pose_model, device):
    """Return 17 keypoints [x, y, conf] per frame (whole cow crop) in crop-pixel
    coordinates. CattleActionDataset normalizes them by the image size."""
    from interaction_prep import run_hrnet, decode_heatmaps, HMAP_SIZE
    results = []
    for i in range(0, len(frames_bgr), POSE_BATCH):
        chunk = frames_bgr[i:i + POSE_BATCH]
        pils = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in chunk]
        kps_batch = decode_heatmaps(run_hrnet(pils, pose_model, device))  # (B, J, 3)
        for frame, kps in zip(chunk, kps_batch):
            h, w = frame.shape[:2]
            kps = kps.copy()
            kps[:, 0] *= w / HMAP_SIZE
            kps[:, 1] *= h / HMAP_SIZE
            results.append(kps.astype(np.float32))
    return results


def extract_frames(clip_path, label, session_id, clip_stem, split,
                   pose_model=None, device=None):
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"    SKIP (unreadable): {clip_path}")
        return []
    sampled = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % SAMPLE_EVERY == 0:
            sampled.append((frame_idx, frame))
        frame_idx += 1
    cap.release()

    # CSV paths are stored relative to the repo root so the CSV in
    # data/annotated/ is unambiguous about where the files live.
    rows = []
    for frame_idx, frame in sampled:
        stem = f"{session_id}_{clip_stem}_f{frame_idx:04d}"
        img_abs = os.path.join(OUTPUT_DIR, split, "crops", label, stem + ".jpg")
        cv2.imwrite(img_abs, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append({"image_path": os.path.relpath(img_abs, _REPO_ROOT),
                     "Label": label, "pose_path": ""})

    if pose_model is not None and sampled:
        poses = estimate_poses([f for _, f in sampled], pose_model, device)
        for row, (frame_idx, _), kps in zip(rows, sampled, poses):
            stem = f"{session_id}_{clip_stem}_f{frame_idx:04d}"
            pose_abs = os.path.join(OUTPUT_DIR, split, "poses", label, stem + ".npy")
            np.save(pose_abs, kps)
            row["pose_path"] = os.path.relpath(pose_abs, _REPO_ROOT)

    return rows


def discover_clips():
    """Return list of clip dicts with their (session, tracklet) group key."""
    sessions = sorted(
        e for e in os.listdir(CLIPS_BASE_DIR)
        if os.path.isdir(os.path.join(CLIPS_BASE_DIR, e))
    )
    print(f"Found {len(sessions)} sessions: {sessions}")

    clips = []
    for session_id in sessions:
        session_dir = os.path.join(CLIPS_BASE_DIR, session_id)
        for beh_folder, label in LABEL_MAP.items():
            folder = os.path.join(session_dir, beh_folder)
            if not os.path.isdir(folder):
                continue
            for clip_name in sorted(f for f in os.listdir(folder) if f.endswith(".mp4")):
                stem = clip_name[:-4]
                tracklet_id = stem.split("_")[0]
                clips.append({
                    "session_id": session_id,
                    "label": label,
                    "path": os.path.join(folder, clip_name),
                    "stem": stem,
                    "group": (session_id, tracklet_id),
                })
    return clips


def assign_groups_622(groups, seed):
    """Split (session, tracklet) groups 6:2:2 (train:val:test). Clips from the
    same group (same cow track) never cross splits, preventing near-duplicate
    neighbouring frames from leaking into val/test."""
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    n = len(order)
    if n < 3:
        print(f"[WARN] only {n} group(s); cannot form 6:2:2. All -> train.")
        return {g: "train" for g in order}
    n_test = max(1, round(n * TEST_RATIO))
    n_val = max(1, round(n * VAL_RATIO))
    n_test = min(n_test, n - 2)
    n_val = min(n_val, n - 1 - n_test)
    assignment = {}
    for i, g in enumerate(order):
        if i < n_test:
            assignment[g] = "test"
        elif i < n_test + n_val:
            assignment[g] = "val"
        else:
            assignment[g] = "train"
    return assignment


def main():
    for split in SPLITS:
        for label in LABEL_MAP.values():
            os.makedirs(os.path.join(OUTPUT_DIR, split, "crops", label), exist_ok=True)
            os.makedirs(os.path.join(OUTPUT_DIR, split, "poses", label), exist_ok=True)

    clips = discover_clips()
    print(f"Total clips: {len(clips)}")

    # Assign per (session, tracklet) group 6:2:2 (test contains only unseen cow tracks)
    groups = {c["group"] for c in clips}
    assignment = assign_groups_622(groups, RANDOM_SEED)
    group_counts = Counter(assignment.values())
    print(f"Groups (session, tracklet): {len(groups)}  ->  "
          f"train {group_counts['train']}, val {group_counts['val']}, test {group_counts['test']}")

    # Check class coverage per split before extraction. Class balance is not
    # enforced (per Spec), but a class absent from train cannot be learned and
    # a class absent from test makes f1 undefined, so warn about gaps only.
    coverage = {split: set() for split in SPLITS}
    for clip in clips:
        coverage[assignment[clip["group"]]].add(clip["label"])
    for split in SPLITS:
        missing = [l for l in LABEL_MAP.values() if l not in coverage[split]]
        if missing:
            print(f"[WARN] split '{split}' has NO clips for class(es): {missing}. "
                  f"Consider changing RANDOM_SEED or reviewing the split.")

    pose_model, device = load_pose_model()

    all_rows = []
    split_label_counts = Counter()
    for clip in clips:
        split = assignment[clip["group"]]
        rows = extract_frames(
            clip["path"], clip["label"], clip["session_id"], clip["stem"], split,
            pose_model=pose_model, device=device,
        )
        all_rows.extend(rows)
        split_label_counts[(split, clip["label"])] += len(rows)

    annotated_dir = _resolve(_CFG["paths"]["annotated_dir"])
    os.makedirs(annotated_dir, exist_ok=True)
    csv_path = os.path.join(annotated_dir, "annotated_action.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "Label", "pose_path"])
        writer.writeheader()
        writer.writerows(all_rows)

    n_with_pose = sum(1 for r in all_rows if r["pose_path"])
    print(f"\nDone. Total: {len(all_rows)} frames ({n_with_pose} with pose)")
    for label in LABEL_MAP.values():
        per_split = "  ".join(
            f"{split}: {split_label_counts.get((split, label), 0)}" for split in SPLITS
        )
        print(f"  {label:15s} {per_split}")
    for split in SPLITS:
        total = sum(v for (s, _), v in split_label_counts.items() if s == split)
        print(f"  [{split}] total: {total}")
    print(f"annotated_action.csv -> {csv_path}")
    print('\nNOTE: load with split_type="date" '
          "(src.dataset.split_action_dataset_entries) so the split in the "
          "image path is respected.")


if __name__ == "__main__":
    main()
