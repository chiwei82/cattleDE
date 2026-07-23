"""
Auto-fill missing cattle labels in a split using an independent COCO detector.

Motivation: the original labels (from tracklets.json) miss some cattle, which
inflates false positives when evaluating the trained detector. This script runs
an off-the-shelf COCO-pretrained YOLO (class "cow") as an INDEPENDENT second
opinion, and adds any cow it finds that has no overlapping existing label.

No manual annotation is involved. The result is a PSEUDO-LABEL set, not
human-verified ground truth: COCO has its own errors and its cattle are mostly
side-view (vs this dataset's overhead view), so treat the corrected metric as
"evaluated against independent pseudo-labels", not as a clean ground truth.

Output (does NOT touch the original dataset):
  <output_dir><suffix>/
    <split>/images/   -> symlinks to the original images
    <split>/labels/   -> original labels + added cow boxes (axis-aligned OBB)
    object_pseudo.yaml

Evaluate the trained model against it:
  yolo obb val model=checkpoints/yolo.pt \\
      data=data/object_pseudo/object_pseudo.yaml split=<split> imgsz=1280

Usage (from the repo root, on a machine with the venv):
  python prep/pseudo_label_test.py
"""

import os
import shutil
import sys

import cv2
import yaml
from ultralytics import YOLO

# ── Config (see global_config.yaml at the repository root) ────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_REPO_ROOT, "global_config.yaml")) as _f:
    _CFG = yaml.safe_load(_f)


def _resolve(p):
    """Resolve config paths relative to the repo root, not the CWD."""
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


PCFG        = _CFG["pseudo_label"]
DATASET_DIR = _resolve(_CFG["yolo_prep"]["output_dir"])
SPLIT       = PCFG["split"]
PSEUDO_DIR  = _resolve(_CFG["yolo_prep"]["output_dir"] + PCFG["pseudo_suffix"])


def obb_line_to_aabb(line, w, h):
    """Parse one YOLO OBB label line -> pixel AABB (x1, y1, x2, y2)."""
    v = list(map(float, line.split()[1:]))          # drop class id
    xs = [v[i] * w for i in range(0, 8, 2)]
    ys = [v[i] * h for i in range(1, 8, 2)]
    return min(xs), min(ys), max(xs), max(ys)


def iou_aabb(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def aabb_to_obb_norm(x1, y1, x2, y2, w, h):
    """Pixel AABB -> normalized 4-corner OBB line body (class 0)."""
    xs = [x1 / w, x2 / w]; ys = [y1 / h, y2 / h]
    corners = [xs[0], ys[0], xs[1], ys[0], xs[1], ys[1], xs[0], ys[1]]
    corners = [min(1.0, max(0.0, c)) for c in corners]
    return "0 " + " ".join(f"{c:.6f}" for c in corners)


def main():
    images_dir = os.path.join(DATASET_DIR, SPLIT, "images")
    labels_dir = os.path.join(DATASET_DIR, SPLIT, "labels")
    if not os.path.isdir(images_dir):
        print(f"[ERROR] {images_dir} not found. Run prep/yolo_prep.py first.")
        sys.exit(1)

    out_images = os.path.join(PSEUDO_DIR, SPLIT, "images")
    out_labels = os.path.join(PSEUDO_DIR, SPLIT, "labels")
    if os.path.isdir(PSEUDO_DIR):
        shutil.rmtree(PSEUDO_DIR)           # clean rebuild
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    print(f"Loading COCO model {PCFG['model']} ...")
    model = YOLO(PCFG["model"])

    names = sorted(f for f in os.listdir(images_dir) if f.endswith(".jpg"))
    print(f"{len(names)} images in split '{SPLIT}'.")

    n_orig = n_added = n_imgs_touched = 0
    for name in names:
        img_path = os.path.join(images_dir, name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [WARN] cannot read {name}, skipping.")
            continue
        h, w = img.shape[:2]

        # Existing labels (as pixel AABBs for matching) and their raw lines.
        label_path = os.path.join(labels_dir, name[:-4] + ".txt")
        existing_lines, existing_aabbs = [], []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    line = line.strip()
                    if len(line.split()) == 9:
                        existing_lines.append(line)
                        existing_aabbs.append(obb_line_to_aabb(line, w, h))
        n_orig += len(existing_lines)

        # COCO cow detections that don't overlap any existing label.
        res = model.predict(source=img, conf=PCFG["conf"], imgsz=PCFG["imgsz"],
                            classes=[PCFG["coco_cow_class"]], verbose=False)[0]
        added_lines = []
        if res.boxes is not None:
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                cand = (x1, y1, x2, y2)
                max_iou = max((iou_aabb(cand, e) for e in existing_aabbs),
                              default=0.0)
                if max_iou < PCFG["iou_match_thresh"]:
                    added_lines.append(aabb_to_obb_norm(x1, y1, x2, y2, w, h))
        n_added += len(added_lines)
        if added_lines:
            n_imgs_touched += 1

        # Write merged labels + symlink the image into the pseudo dataset.
        with open(os.path.join(out_labels, name[:-4] + ".txt"), "w") as f:
            f.write("\n".join(existing_lines + added_lines))
            if existing_lines or added_lines:
                f.write("\n")
        dst_img = os.path.join(out_images, name)
        os.symlink(os.path.abspath(img_path), dst_img)

    # Dataset yaml for evaluation. Ultralytics requires BOTH `train:` and `val:`
    # keys in every data YAML, so point all splits at the pseudo images; the
    # split= arg selects which one val mode actually loads.
    yaml_path = os.path.join(PSEUDO_DIR, "object_pseudo.yaml")
    rel = f"{SPLIT}/images"
    with open(yaml_path, "w") as yf:
        yf.write(f"path: {os.path.abspath(PSEUDO_DIR)}\n")
        yf.write(f"train: {rel}\n")
        yf.write(f"val: {rel}\n")
        if SPLIT not in ("train", "val"):
            yf.write(f"{SPLIT}: {rel}\n")
        yf.write("\nnc: 1\n")
        yf.write("names: ['object']\n")

    print(f"\n=== Pseudo-labeling summary ({SPLIT}) ===")
    print(f"Original labels:      {n_orig}")
    if n_orig:
        print(f"Added by COCO model:  {n_added}  (+{n_added / n_orig * 100:.1f}%)")
    else:
        print(f"Added by COCO model:  {n_added}")
    print(f"Images gaining boxes: {n_imgs_touched} / {len(names)}")
    print(f"Pseudo dataset:       {PSEUDO_DIR}")
    print(f"YAML:                 {yaml_path}")

    # ── Re-evaluate the trained model against the pseudo-labels ────────────────
    # Use an ABSOLUTE project path: ultralytics prepends its default runs dir to
    # relative project= paths, producing runs/obb/runs/obb/... doubling.
    yolo_ckpt = _resolve(_CFG["paths"]["yolo_ckpt"])
    if not os.path.exists(yolo_ckpt):
        print(f"\n[WARN] trained model not found at {yolo_ckpt}; skipping eval. "
              f"Run manually once it exists:\n"
              f"  yolo obb val model={yolo_ckpt} data={yaml_path} "
              f"split={SPLIT} imgsz={_CFG['yolo_train']['imgsz']}")
        return

    print(f"\n=== Evaluating {yolo_ckpt} against pseudo-labels ({SPLIT}) ===")
    run_dir = os.path.join(_REPO_ROOT, _CFG["yolo_train"]["run_dir"])
    metrics = YOLO(yolo_ckpt).val(
        data=yaml_path,
        split=SPLIT,
        imgsz=_CFG["yolo_train"]["imgsz"],
        project=run_dir,
        name=f"{_CFG['yolo_train']['run_name']}_pseudo_{SPLIT}",
        exist_ok=True,
    )
    print(f"Results saved under {run_dir}/"
          f"{_CFG['yolo_train']['run_name']}_pseudo_{SPLIT}")


if __name__ == "__main__":
    main()
