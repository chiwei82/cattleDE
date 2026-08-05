"""Dataset for weakly-supervised contact localisation on cattle pair crops.

Data contract (produced by prep/interaction_prep.py, labelled via Label Studio):

  image_path         repo-root-relative path to the pair crop
  bbox1_xyxy         "[x1 y1 x2 y2]" of cow 1, in SOURCE FRAME coordinates
  bbox2_xyxy         "[x1 y1 x2 y2]" of cow 2, in SOURCE FRAME coordinates
  merged_bbox_xyxy   "[x1 y1 x2 y2]" union box; the crop is this region
  label_v1           interaction | no_interaction | not well-cropped | blank
  label_v2           social grooming | interest | sniffing | mount | ...
  source_video       used as the split group; splits are already video-disjoint
  split              train | val | test

The crop is the merged box cut out of the frame WITHOUT resizing, and
prep.safe_crop_bgr clips it at the frame border. The crop origin is therefore
(max(0, mx1), max(0, my1)) and the crop can be SMALLER than the merged box when
the pair sits against an image edge (~12% of rows), so relative boxes are
clamped to the real crop size rather than trusted blindly.

Each sample returns the letterboxed crop, the binary interaction label, and the
contact-candidate region R used as the MIL bag.
"""

import os
import re

import cv2
import numpy as np

# torch is imported lazily so that the CSV/geometry helpers below — and with them
# diagnostics/geometry_baseline.py, the gate that decides whether training is
# worth running — stay usable in a plain numpy environment with no GPU stack.
try:
    import torch
    from torch.utils.data import Dataset

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on numpy-only machines
    torch = None
    Dataset = object
    TORCH_AVAILABLE = False

# Repository root = two levels up from contactTest/src/.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Letterbox padding value; matches the grey used by the YOLO stage.
PAD_VALUE = 114


def parse_bbox(text):
    """Parse "[x1 y1 x2 y2]" (space-, comma- or quote-separated) into an int array."""
    nums = re.findall(r"-?\d+", str(text))
    if len(nums) != 4:
        raise ValueError(f"expected 4 bbox values, got {text!r}")
    return np.array([int(n) for n in nums], dtype=np.int64)


def letterbox(img, size, pad_value=0):
    """Scale the longest side to `size` and pad, preserving the aspect ratio.

    Returned alongside the canvas are the occupied height/width and the scale
    factor, so that boxes and predictions can be mapped in and out of the canvas.
    Padding must be excluded from the MIL bag; it carries no image evidence.
    """
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)

    shape = (size, size) + img.shape[2:]
    canvas = np.full(shape, pad_value, dtype=img.dtype)
    canvas[:nh, :nw] = resized
    return canvas, nh, nw, scale


def binary_label(row, no_names, exclude_names, exclude_positive_v2):
    """Map a CSV row to 0 / 1, or None when the row must be dropped.

    Follows the same rule as train/interaction_with_image.py so that this
    experiment sees exactly the rows the stage-2 classifier is trained on, with
    one extra hook: label_v2 values listed in `exclude_positive_v2` are removed
    from the positive class (see README.md on 'interest').
    """
    v1 = str(row.get("label_v1") or "").strip().lower()
    if not v1 or v1 in exclude_names:
        return None
    if v1 in no_names:
        return 0
    v2 = str(row.get("label_v2") or "").strip().lower()
    if v2 in exclude_positive_v2:
        return None
    return 1


def load_records(cfg, splits=None):
    """Read the annotation CSV into a list of per-pair records.

    Paths are resolved against the repository root; the CSV itself is never
    modified. Rows whose crop file is missing are skipped with a warning.
    """
    import csv

    csv_path = os.path.join(REPO_ROOT, cfg["data"]["csv"])
    lab = cfg["labels"]
    no_names = {str(x).strip().lower() for x in lab["no_interaction_names"]}
    exclude_names = {str(x).strip().lower() for x in lab.get("exclude_names", [])}
    exclude_v2 = {str(x).strip().lower() for x in lab.get("exclude_positive_v2", [])}

    mask_dir = cfg["data"].get("mask_dir")
    mask_root = os.path.join(REPO_ROOT, mask_dir) if mask_dir else None

    records, missing = [], 0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            label = binary_label(row, no_names, exclude_names, exclude_v2)
            if label is None:
                continue
            split = str(row.get("split") or "").strip()
            if splits is not None and split not in splits:
                continue

            rel_image = row["image_path"]
            abs_image = os.path.join(REPO_ROOT, rel_image)
            if not os.path.exists(abs_image):
                missing += 1
                continue

            mask_path = None
            if mask_root is not None:
                candidate = os.path.join(mask_root, os.path.splitext(rel_image)[0] + ".npz")
                if os.path.exists(candidate):
                    mask_path = candidate

            records.append({
                "image_path": abs_image,
                "rel_image": rel_image,
                "mask_path": mask_path,
                "bbox1": parse_bbox(row["bbox1_xyxy"]),
                "bbox2": parse_bbox(row["bbox2_xyxy"]),
                "merged": parse_bbox(row["merged_bbox_xyxy"]),
                "label": label,
                "label_v2": str(row.get("label_v2") or "").strip().lower(),
                "split": split,
                "source_video": str(row.get("source_video") or "").strip(),
                "frame_number": int(row.get("frame_number") or 0),
            })

    if missing:
        print(f"[data] WARNING: {missing} labelled rows skipped — crop file not found")
    return records


def relative_boxes(record, crop_h, crop_w):
    """Convert the two frame-space boxes into crop-space boxes, clamped in range.

    The crop origin is max(0, merged_x1), max(0, merged_y1) because
    prep.safe_crop_bgr clips the merged box against the frame border.
    """
    merged = record["merged"]
    ox, oy = max(0, int(merged[0])), max(0, int(merged[1]))
    out = []
    for key in ("bbox1", "bbox2"):
        b = record[key].astype(np.int64) - np.array([ox, oy, ox, oy])
        x1 = int(np.clip(b[0], 0, crop_w - 1))
        y1 = int(np.clip(b[1], 0, crop_h - 1))
        x2 = int(np.clip(b[2], x1 + 1, crop_w))
        y2 = int(np.clip(b[3], y1 + 1, crop_h))
        out.append((x1, y1, x2, y2))
    return out


def _box_mask(box, h, w):
    m = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = box
    m[y1:y2, x1:x2] = 1
    return m


class ContactPairDataset(Dataset):
    """Pair crops with a binary interaction label and a contact-candidate region.

    The region R is the intersection of the two dilated cow supports, restricted
    to the non-padded part of the canvas. It is the MIL bag: the model may only
    place contact evidence inside it, which is what stops the classifier from
    solving the task with background or global-layout shortcuts.

    Cow support comes from instance masks when `mask_dir` is configured and a
    mask file exists, otherwise from the detector boxes. Boxes are coarser — for
    two axis-aligned rectangles the dilated intersection is itself a rectangle —
    but the sparsity and TV terms still have to concentrate mass inside it, and
    prep guarantees IoU > 0.1 so the intersection is never empty.
    """

    def __init__(self, records, cfg, train=False):
        if not TORCH_AVAILABLE:
            raise ImportError("ContactPairDataset needs torch; the geometry helpers "
                              "in this module do not")
        self.records = records
        self.train = train
        self.size = int(cfg["model"]["image_size"])
        self.dilate_px = int(cfg["region"]["dilate_px"])
        self.min_area = int(cfg["region"]["min_area_px"])
        self.hflip_prob = float(cfg["train"]["hflip_prob"]) if train else 0.0
        k = 2 * self.dilate_px + 1
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    def __len__(self):
        return len(self.records)

    def _supports(self, record, img):
        """Return the two per-cow support masks at crop resolution."""
        h, w = img.shape[:2]
        if record["mask_path"] is not None:
            data = np.load(record["mask_path"])
            mi, mj = data["mi"], data["mj"]
            if mi.shape[:2] != (h, w):
                mi = cv2.resize(mi, (w, h), interpolation=cv2.INTER_NEAREST)
                mj = cv2.resize(mj, (w, h), interpolation=cv2.INTER_NEAREST)
            return (mi > 0).astype(np.uint8), (mj > 0).astype(np.uint8)
        box1, box2 = relative_boxes(record, h, w)
        return _box_mask(box1, h, w), _box_mask(box2, h, w)

    def __getitem__(self, idx):
        record = self.records[idx]
        bgr = cv2.imread(record["image_path"])
        if bgr is None:
            # Corrupt file: fall through to the next record rather than crash the
            # epoch, matching the behaviour of src/dataset.py.
            return self[(idx + 1) % len(self)]
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        support_i, support_j = self._supports(record, img)

        if self.train and np.random.rand() < self.hflip_prob:
            img = np.ascontiguousarray(img[:, ::-1])
            support_i = np.ascontiguousarray(support_i[:, ::-1])
            support_j = np.ascontiguousarray(support_j[:, ::-1])

        canvas, nh, nw, scale = letterbox(img, self.size, PAD_VALUE)
        support_i, _, _, _ = letterbox(support_i, self.size, 0)
        support_j, _, _, _ = letterbox(support_j, self.size, 0)

        # Padding is not image evidence and must never enter the MIL bag.
        valid = np.zeros((self.size, self.size), dtype=np.uint8)
        valid[:nh, :nw] = 1

        di = cv2.dilate(support_i, self.kernel)
        dj = cv2.dilate(support_j, self.kernel)
        region = ((di > 0) & (dj > 0) & (valid > 0)).astype(np.float32)
        if region.sum() < self.min_area:
            # Degenerate overlap (rare; only when the crop is heavily clipped).
            # Fall back to the union so the pooling always has something to pool.
            region = (((di > 0) | (dj > 0)) & (valid > 0)).astype(np.float32)

        x = (canvas.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

        return {
            "image": torch.from_numpy(x).permute(2, 0, 1).contiguous(),
            "region": torch.from_numpy(region)[None],
            "label": torch.tensor(float(record["label"])),
            "index": torch.tensor(idx),
            # Kept for mapping predictions back to the source frame.
            "scale": torch.tensor(np.float32(scale)),
            "valid_hw": torch.tensor([nh, nw], dtype=torch.long),
        }


def split_records(records):
    """Group records by the CSV split column.

    interaction_prep assigns whole videos to a split (assign_videos_622), so the
    splits are already video-disjoint and no additional grouping is required.
    diagnostics/split_leakage.py verifies this rather than assuming it.
    """
    buckets = {"train": [], "val": [], "test": []}
    for r in records:
        buckets.get(r["split"], buckets["train"]).append(r)
    return buckets


def describe(buckets):
    """Print per-split class counts and the label_v2 make-up of the positives."""
    for split in ("train", "val", "test"):
        rows = buckets[split]
        n_pos = sum(1 for r in rows if r["label"] == 1)
        n_neg = len(rows) - n_pos
        print(f"[data] {split:5s} total {len(rows):5d} | neg {n_neg:5d} | pos {n_pos:4d}")
        if n_pos:
            kinds = {}
            for r in rows:
                if r["label"] == 1:
                    kinds[r["label_v2"] or "(blank)"] = kinds.get(r["label_v2"] or "(blank)", 0) + 1
            detail = "  ".join(f"{k}:{v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]))
            print(f"[data]        positives by label_v2 -> {detail}")
