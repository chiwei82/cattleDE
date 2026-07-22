"""
Visualize the YOLO OBB training data produced by yolo_prep.py.

For every image in data/object/{split}/images/, reads the same-name .txt
label, denormalizes the 4 OBB corner points back to pixel coordinates, and
draws each box on the image — i.e. exactly what the model will be trained
on. Results are written to simu/{split}/ with the split and box count
stamped on the image.

Usage (from the repo root):
  python prep/simu_vis.py [--limit N]   # N images per split, default all
"""

import argparse
import os

import cv2
import numpy as np
import yaml

# ── Config (see global_config.yaml at the repository root) ────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_REPO_ROOT, "global_config.yaml")) as _f:
    _CFG = yaml.safe_load(_f)


def _resolve(p):
    """Resolve config paths relative to the repo root, not the CWD."""
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


DATASET_DIR = _resolve(_CFG["yolo_prep"]["output_dir"])
SIMU_DIR    = os.path.join(_REPO_ROOT, "simu")
SPLITS      = ("train", "val", "test")

BOX_COLOR   = (0, 255, 0)     # BGR
BOX_THICK   = 2


def parse_label_file(label_path, width, height):
    """Return a list of (class_id, corners) where corners is (4, 2) int32
    pixel coordinates, denormalized from the 0-1 values in the .txt."""
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 9:
                print(f"  [WARN] malformed line in {label_path}: {line.strip()}")
                continue
            cls = int(parts[0])
            vals = list(map(float, parts[1:]))
            corners = np.array(
                [[vals[i] * width, vals[i + 1] * height] for i in range(0, 8, 2)],
                dtype=np.int32,
            )
            boxes.append((cls, corners))
    return boxes


def draw_boxes(img, boxes, split):
    for cls, corners in boxes:
        cv2.polylines(img, [corners], isClosed=True,
                      color=BOX_COLOR, thickness=BOX_THICK)
        x, y = corners[0]
        cv2.putText(img, str(cls), (int(x), max(0, int(y) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR, 2)
    cv2.putText(img, f"{split}  boxes: {len(boxes)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return img


def main():
    parser = argparse.ArgumentParser(
        description="Render YOLO OBB labels onto their images for inspection."
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="Max images per split (0 = all).")
    args = parser.parse_args()

    total = 0
    for split in SPLITS:
        images_dir = os.path.join(DATASET_DIR, split, "images")
        labels_dir = os.path.join(DATASET_DIR, split, "labels")
        if not os.path.isdir(images_dir):
            print(f"[WARN] {images_dir} not found, skipping split '{split}'.")
            continue

        out_dir = os.path.join(SIMU_DIR, split)
        os.makedirs(out_dir, exist_ok=True)

        names = sorted(f for f in os.listdir(images_dir) if f.endswith(".jpg"))
        if args.limit > 0:
            names = names[:args.limit]

        done = 0
        for name in names:
            img_path   = os.path.join(images_dir, name)
            label_path = os.path.join(labels_dir, name[:-4] + ".txt")

            img = cv2.imread(img_path)
            if img is None:
                print(f"  [WARN] cannot read {img_path}, skipping.")
                continue
            if not os.path.exists(label_path):
                print(f"  [WARN] no label for {name}, skipping.")
                continue

            h, w = img.shape[:2]
            boxes = parse_label_file(label_path, w, h)
            draw_boxes(img, boxes, split)
            cv2.imwrite(os.path.join(out_dir, name), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            done += 1

        total += done
        print(f"[{split}] {done} images -> {out_dir}")

    print(f"\nDone. {total} visualizations in {SIMU_DIR}")


if __name__ == "__main__":
    main()
