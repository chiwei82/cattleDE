"""
Spec:
- model training: YOLOv11
- trackingidentifiers and movement trajectories: ByteTrack
- training, validating and testing using a 6:2:2 split
- without considering class balance

Convert multi-camera tracklets.json files to YOLO OBB dataset format.

Expected directory structure (set DATA_ROOT below):
  camera128/
    128_20250529T062356_20250529T064453/
      tracklets.json
    128_20250529T062356_20250529T064453.mp4
  camera133/
    ...
  camera26/
    ...

Output:
  data/yolo_cattle/
    images/train/  images/val/  images/test/
    labels/train/  labels/val/  labels/test/
    cattle.yaml

YOLO OBB label format per line:
  0 x1 y1 x2 y2 x3 y3 x4 y4   (class_id + 4 corners, normalized 0-1)
"""

import json
import os
import random
import cv2
import yaml

# ── Config (see global_config.yaml at the repository root) ────────────────────
_CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "global_config.yaml")
with open(_CFG_PATH) as _f:
    _CFG = yaml.safe_load(_f)

DATA_ROOT   = _CFG["paths"]["data_root"]
OUTPUT_DIR  = _CFG["yolo_prep"]["output_dir"]
FRAME_STEP  = _CFG["yolo_prep"]["frame_step"]
VAL_RATIO   = _CFG["split"]["val_ratio"]
TEST_RATIO  = _CFG["split"]["test_ratio"]
RANDOM_SEED = _CFG["random_seed"]
CAMERA_DIRS = _CFG["yolo_prep"]["camera_dirs"]
# ──────────────────────────────────────────────────────────────────────────────


def obb_to_normalized(obb, width, height):
    coords = []
    for i, v in enumerate(obb):
        if i % 2 == 0:
            coords.append(max(0.0, min(1.0, v / width)))
        else:
            coords.append(max(0.0, min(1.0, v / height)))
    return coords


def discover_sessions(data_root):
    """Return list of (session_id, tracklets_path, video_path, camera)."""
    sessions = []
    for cam_dir in CAMERA_DIRS:
        cam_path = os.path.join(data_root, cam_dir)
        if not os.path.isdir(cam_path):
            print(f"  WARNING: {cam_path} not found, skipping.")
            continue
        for entry in sorted(os.listdir(cam_path)):
            entry_path = os.path.join(cam_path, entry)
            if not os.path.isdir(entry_path):
                continue
            tracklets_path = os.path.join(entry_path, "tracklets.json")
            video_path     = os.path.join(cam_path, entry + ".mp4")
            if not os.path.exists(tracklets_path):
                print(f"  WARNING: no tracklets.json in {entry_path}, skipping.")
                continue
            if not os.path.exists(video_path):
                print(f"  WARNING: no video at {video_path}, skipping.")
                continue
            sessions.append((entry, tracklets_path, video_path, cam_dir))
    return sessions


def process_session(session_id, tracklets_path, video_path, split, split_counts):
    """Write all sampled frames of a session to its assigned split (train/val/test).
    The split is fixed per camera, so footage from the same camera (background,
    viewpoint, individuals) never crosses splits, and test contains only
    completely unseen cameras."""
    print(f"\nSession: {session_id}  ->  [{split}]")

    with open(tracklets_path) as f:
        tracklets = json.load(f)

    frame_index = {}
    for key, entries in tracklets.items():
        if key == "stats":
            continue
        for e in entries:
            fn = e["frame_number"]
            frame_index.setdefault(fn, []).append(e["obb"])

    sampled = sorted(fn for fn in frame_index if fn % FRAME_STEP == 0)
    print(f"  Annotated frames: {len(frame_index)}, sampled: {len(sampled)}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: cannot open {video_path}, skipping.")
        return
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    saved = 0
    for fn in sampled:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            continue

        stem  = f"{session_id}_frame_{fn:06d}"
        img_path   = os.path.join(OUTPUT_DIR, "images", split, stem + ".jpg")
        label_path = os.path.join(OUTPUT_DIR, "labels", split, stem + ".txt")

        cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(label_path, "w") as lf:
            for obb in frame_index[fn]:
                coords = obb_to_normalized(obb, W, H)
                lf.write("0 " + " ".join(f"{v:.6f}" for v in coords) + "\n")

        saved += 1
        if saved % 100 == 0:
            print(f"  Saved {saved}/{len(sampled)}...")

    cap.release()
    split_counts[split] += saved
    print(f"  Done: {saved} frames saved to {split}.")


def assign_cameras_622(cameras, seed):
    """Split the camera list 6:2:2 (train:val:test). Assigning at camera level
    keeps similar scenes from the same camera (same background, viewpoint,
    individuals) out of both train and test, so test measures generalization
    to unseen cameras."""
    order = sorted(cameras)
    random.Random(seed).shuffle(order)
    n = len(order)
    if n < 3:
        print(f"[WARN] only {n} camera(s); cannot form 6:2:2. All -> train.")
        return {c: "train" for c in order}
    n_test = max(1, round(n * TEST_RATIO))
    n_val = max(1, round(n * VAL_RATIO))
    n_test = min(n_test, n - 2)
    n_val = min(n_val, n - 1 - n_test)
    assignment = {}
    for i, cam in enumerate(order):
        if i < n_test:
            assignment[cam] = "test"
        elif i < n_test + n_val:
            assignment[cam] = "val"
        else:
            assignment[cam] = "train"
    return assignment


def main():
    random.seed(RANDOM_SEED)

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

    sessions = discover_sessions(DATA_ROOT)
    print(f"Found {len(sessions)} sessions:")
    for s, t, v, c in sessions:
        print(f"  [{c}] {s}")

    # Assign cameras 6:2:2 (test contains only unseen cameras)
    cameras = sorted({c for _, _, _, c in sessions})
    assignment = assign_cameras_622(cameras, RANDOM_SEED)
    print("\nCamera split (6:2:2):")
    for cam, sp in sorted(assignment.items(), key=lambda kv: (kv[1], kv[0])):
        print(f"  [{sp}] {cam}")

    split_counts = {"train": 0, "val": 0, "test": 0}
    for session_id, tracklets_path, video_path, camera in sessions:
        process_session(session_id, tracklets_path, video_path,
                        assignment[camera], split_counts)

    yaml_path = os.path.join(OUTPUT_DIR, "cattle.yaml")
    abs_output = os.path.abspath(OUTPUT_DIR)
    with open(yaml_path, "w") as yf:
        yf.write(f"path: {abs_output}\n")
        yf.write("train: images/train\n")
        yf.write("val: images/val\n")
        yf.write("test: images/test\n\n")
        yf.write("nc: 1\n")
        yf.write("names: ['cattle']\n")

    print(f"\n=== Summary ===")
    print(f"Train: {split_counts['train']} frames, "
          f"Val: {split_counts['val']} frames, "
          f"Test: {split_counts['test']} frames")
    print(f"YAML: {yaml_path}")
    print(f"\nTo train:")
    print(f"  yolo obb train model=yolo11n-obb.pt data={yaml_path} epochs=50 imgsz=1280")
    print(f"To evaluate on the unseen test split:")
    print(f"  yolo obb val model=<best.pt> data={yaml_path} split=test imgsz=1280")


if __name__ == "__main__":
    main()
