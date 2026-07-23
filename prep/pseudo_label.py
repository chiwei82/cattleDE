"""
Auto-fill missing cattle labels across all splits using an independent COCO
detector, producing a corrected dataset the OBB model can be RETRAINED on.

Motivation: the original labels (from tracklets.json) miss ~22% of cattle.
During training those unlabeled cattle act as negatives, so the model is
punished for confidently detecting real cows -> it learns to output low
confidence (high recall only at a low confidence threshold). Filling the
missing labels removes that penalty.

Method: run an off-the-shelf COCO-pretrained YOLO (class "cow") as an
INDEPENDENT second opinion on every split, and add any cow it finds that has no
overlapping existing label.

Caveats (state these when reporting results):
  - NO manual annotation — this is a pseudo-label set, not clean ground truth.
  - COCO outputs AXIS-ALIGNED boxes; added labels therefore have no orientation,
    which slightly degrades OBB tightness supervision on the added ~22%.
  - COCO's cattle are mostly side-view vs this dataset's overhead view, so it
    may also miss some cattle (a recall ceiling this cannot fix).

Output (does NOT touch the original dataset):
  <output_dir><suffix>/
    <split>/images/   -> symlinks to the original images
    <split>/labels/   -> original labels + added cow boxes (axis-aligned OBB)
    object_pseudo.yaml

Usage (from the repo root, on a machine with the venv):
  python prep/pseudo_label.py
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
SPLITS      = PCFG["splits"]
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


def process_split(split, model):
    """Build the pseudo-labeled version of one split. Returns (n_orig, n_added)."""
    images_dir = os.path.join(DATASET_DIR, split, "images")
    labels_dir = os.path.join(DATASET_DIR, split, "labels")
    if not os.path.isdir(images_dir):
        print(f"[WARN] {images_dir} not found, skipping split '{split}'.")
        return 0, 0

    out_images = os.path.join(PSEUDO_DIR, split, "images")
    out_labels = os.path.join(PSEUDO_DIR, split, "labels")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    names = sorted(f for f in os.listdir(images_dir) if f.endswith(".jpg"))
    print(f"\n[{split}] {len(names)} images")

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
        os.symlink(os.path.abspath(img_path), os.path.join(out_images, name))

    pct = f"(+{n_added / n_orig * 100:.1f}%)" if n_orig else ""
    print(f"[{split}] original {n_orig}, added {n_added} {pct}, "
          f"images touched {n_imgs_touched}/{len(names)}")
    return n_orig, n_added


def main():
    if "train" not in SPLITS or "val" not in SPLITS:
        print("[ERROR] pseudo_label.splits must include both 'train' and 'val' "
              "(ultralytics requires both keys to train).")
        sys.exit(1)

    if os.path.isdir(PSEUDO_DIR):
        shutil.rmtree(PSEUDO_DIR)           # clean rebuild
    os.makedirs(PSEUDO_DIR, exist_ok=True)

    print(f"Loading COCO model {PCFG['model']} ...")
    model = YOLO(PCFG["model"])

    built, tot_orig, tot_added = [], 0, 0
    for split in SPLITS:
        n_orig, n_added = process_split(split, model)
        if n_orig or n_added:
            built.append(split)
            tot_orig += n_orig
            tot_added += n_added

    # One dataset yaml covering every built split.
    yaml_path = os.path.join(PSEUDO_DIR, "object_pseudo.yaml")
    with open(yaml_path, "w") as yf:
        yf.write(f"path: {os.path.abspath(PSEUDO_DIR)}\n")
        for split in built:
            yf.write(f"{split}: {split}/images\n")
        yf.write("\nnc: 1\n")
        yf.write("names: ['object']\n")

    pct = f"(+{tot_added / tot_orig * 100:.1f}%)" if tot_orig else ""
    print(f"\n=== Pseudo-labeling summary ===")
    print(f"Splits built:   {built}")
    print(f"Total original: {tot_orig}")
    print(f"Total added:    {tot_added} {pct}")
    print(f"Pseudo dataset: {PSEUDO_DIR}")
    print(f"YAML:           {yaml_path}")
    print(f"\nNext: retrain on this dataset (see main_task.sh).")


if __name__ == "__main__":
    main()
