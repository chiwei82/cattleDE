
import os
import re

import cv2
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    Dataset = object
    TORCH_AVAILABLE = False

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONTACT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

PAD_VALUE = 114


def parse_bbox(text):
    nums = re.findall(r"-?\d+", str(text))
    if len(nums) != 4:
        raise ValueError(f"expected 4 bbox values, got {text!r}")
    return np.array([int(n) for n in nums], dtype=np.int64)


def letterbox(img, size, pad_value=0):
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
    v1 = str(row.get("label_v1") or "").strip().lower()
    if not v1 or v1 in exclude_names:
        return None
    if v1 in no_names:
        return 0
    v2 = str(row.get("label_v2") or "").strip().lower()
    if v2 in exclude_positive_v2:
        return None
    return 1


def load_records(cfg, splits=None, require_label=True):
    import csv

    csv_path = os.path.join(REPO_ROOT, cfg["data"]["csv"])
    lab = cfg["labels"]
    no_names = {str(x).strip().lower() for x in lab["no_interaction_names"]}
    exclude_names = {str(x).strip().lower() for x in lab.get("exclude_names", [])}
    exclude_v2 = {str(x).strip().lower() for x in lab.get("exclude_positive_v2", [])}

    mask_dir = cfg["data"].get("mask_dir")
    mask_root = os.path.join(CONTACT_ROOT, mask_dir) if mask_dir else None
    depth_dir = (cfg.get("depth") or {}).get("cache_dir")
    depth_root = os.path.join(CONTACT_ROOT, depth_dir) if depth_dir else None

    records, missing, unlabelled = [], 0, 0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            label = binary_label(row, no_names, exclude_names, exclude_v2)
            if label is None:
                if require_label:
                    continue
                label = -1
                unlabelled += 1
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

            depth_path = None
            if depth_root is not None:
                candidate = os.path.join(depth_root,
                                         os.path.splitext(rel_image)[0] + ".npz")
                if os.path.exists(candidate):
                    depth_path = candidate

            records.append({
                "image_path": abs_image,
                "rel_image": rel_image,
                "mask_path": mask_path,
                "depth_path": depth_path,
                "bbox1": parse_bbox(row["bbox1_xyxy"]),
                "bbox2": parse_bbox(row["bbox2_xyxy"]),
                "merged": parse_bbox(row["merged_bbox_xyxy"]),
                "label": label,
                "label_v1": str(row.get("label_v1") or "").strip().lower(),
                "label_v2": str(row.get("label_v2") or "").strip().lower(),
                "split": split,
                "source_video": str(row.get("source_video") or "").strip(),
                "frame_number": int(row.get("frame_number") or 0),
            })

    if missing:
        print(f"[data] WARNING: {missing} rows skipped — crop file not found")
    if unlabelled:
        print(f"[data] {unlabelled} rows carry no usable annotation (label = -1); "
              "included because require_label=False")
    return records


def relative_boxes(record, crop_h, crop_w):
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

        pcfg = cfg.get("pose", {})
        self.use_pose = bool(pcfg.get("use_in_model", False))
        self.pose_cache = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            pcfg.get("cache_dir", "log/pose_cache"))
        self.kp_min_conf = float(pcfg.get("min_conf", 0.3))
        self.kp_radius = int(pcfg.get("radius_px", 18))
        self.kp_proximity = float(pcfg.get("proximity_px", 60))
        self.num_joints = int(pcfg.get("num_joints", 17))

        self.mask_channels = bool(cfg["data"].get("mask_channels", False))
        self.gap_scale = float(cfg["data"].get("gap_scale_px", 15.0))

        prior = pcfg.get("prior", {}) or {}
        self.prior_enabled = bool(prior.get("enabled", False)) and self.use_pose
        self.prior_sigma_scale = float(prior.get("sigma_scale", 0.5))
        self.prior_sigma_min = float(prior.get("sigma_min_px", 12))
        self.prior_sigma_max = float(prior.get("sigma_max_px", 45))
        self.prior_conf_weighting = bool(prior.get("conf_weighting", True))

    def _load_keypoints(self, record, crop_h, crop_w):
        path = os.path.join(self.pose_cache,
                            os.path.splitext(record["rel_image"])[0] + ".npz")
        if not os.path.exists(path):
            return None
        data = np.load(path)
        keypoints = data["keypoints"].astype(np.float32)
        cached_h, cached_w = [int(v) for v in data["crop_hw"]]
        if (cached_h, cached_w) != (crop_h, crop_w) and cached_h > 0 and cached_w > 0:
            keypoints[..., 0] *= crop_w / cached_w
            keypoints[..., 1] *= crop_h / cached_h
        return keypoints

    def _gate_keypoints(self, keypoints, boxes):
        valid = keypoints[..., 2] >= self.kp_min_conf
        for i in range(keypoints.shape[0]):
            ox1, oy1, ox2, oy2 = boxes[1 - i]
            px = keypoints[i, :, 0]
            py = keypoints[i, :, 1]
            dx = np.maximum(np.maximum(ox1 - px, px - ox2), 0.0)
            dy = np.maximum(np.maximum(oy1 - py, py - oy2), 0.0)
            valid[i] &= np.hypot(dx, dy) <= self.kp_proximity
        return valid

    def _proximity(self, support_i, support_j):
        di = cv2.distanceTransform((support_i == 0).astype(np.uint8), cv2.DIST_L2, 3)
        dj = cv2.distanceTransform((support_j == 0).astype(np.uint8), cv2.DIST_L2, 3)
        return np.exp(-(di + dj) / max(self.gap_scale, 1e-3)).astype(np.float32)

    def _prior_target(self, keypoints, size):
        from .pose import closest_head_link, HEAD_JOINTS

        link = closest_head_link(keypoints, self.kp_min_conf)
        if link is None:
            return None, 0.0
        pa, pb, dist, _, _ = link

        cx, cy = (pa + pb) / 2.0
        if not (0 <= cx < size and 0 <= cy < size):
            return None, 0.0

        sigma = float(np.clip(dist * self.prior_sigma_scale,
                              self.prior_sigma_min, self.prior_sigma_max))
        yy, xx = np.mgrid[0:size, 0:size]
        blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2)))

        weight = 1.0
        if self.prior_conf_weighting:
            conf = keypoints[..., 2]
            best = float(np.sqrt(max(conf[:, HEAD_JOINTS].max(), 1e-6) *
                                 max(conf.max(), 1e-6)))
            weight = float(np.clip(best / (2.0 * self.kp_min_conf), 0.0, 1.0))
        return blob.astype(np.float32), weight

    def _pose_region(self, keypoints, valid, size):
        if not valid.any():
            return None
        region = np.zeros((size, size), np.uint8)
        for i in range(keypoints.shape[0]):
            for j in range(keypoints.shape[1]):
                if not valid[i, j]:
                    continue
                cx, cy = keypoints[i, j, 0], keypoints[i, j, 1]
                if not (0 <= cx < size and 0 <= cy < size):
                    continue
                cv2.circle(region, (int(round(cx)), int(round(cy))),
                           self.kp_radius, 1, -1)
        return region.astype(np.float32)

    def __len__(self):
        return len(self.records)

    def _supports(self, record, img):
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
            return self[(idx + 1) % len(self)]
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        support_i, support_j = self._supports(record, img)
        crop_h, crop_w = img.shape[:2]
        keypoints = self._load_keypoints(record, crop_h, crop_w) if self.use_pose else None
        boxes = relative_boxes(record, crop_h, crop_w)

        if self.train and np.random.rand() < self.hflip_prob:
            img = np.ascontiguousarray(img[:, ::-1])
            support_i = np.ascontiguousarray(support_i[:, ::-1])
            support_j = np.ascontiguousarray(support_j[:, ::-1])
            boxes = [(crop_w - 1 - x2, y1, crop_w - 1 - x1, y2)
                     for (x1, y1, x2, y2) in boxes]
            if keypoints is not None:
                from .pose import flip_keypoints

                keypoints = flip_keypoints(keypoints, crop_w)

        canvas, nh, nw, scale = letterbox(img, self.size, PAD_VALUE)
        support_i, _, _, _ = letterbox(support_i, self.size, 0)
        support_j, _, _, _ = letterbox(support_j, self.size, 0)

        valid = np.zeros((self.size, self.size), dtype=np.uint8)
        valid[:nh, :nw] = 1

        di = cv2.dilate(support_i, self.kernel)
        dj = cv2.dilate(support_j, self.kernel)
        region = ((di > 0) & (dj > 0) & (valid > 0)).astype(np.float32)
        if region.sum() < self.min_area:
            region = (((di > 0) | (dj > 0)) & (valid > 0)).astype(np.float32)

        kp_xy = np.zeros((2, self.num_joints, 2), np.float32)
        kp_valid = np.zeros((2, self.num_joints), bool)
        prior_blob = np.zeros((self.size, self.size), np.float32)
        prior_weight = 0.0
        if keypoints is not None:
            keypoints = keypoints.copy()
            keypoints[..., :2] *= scale
            boxes_canvas = [tuple(v * scale for v in b) for b in boxes]
            kp_valid = self._gate_keypoints(keypoints, boxes_canvas)
            kp_valid &= ((keypoints[..., 0] >= 0) & (keypoints[..., 0] < self.size) &
                         (keypoints[..., 1] >= 0) & (keypoints[..., 1] < self.size))
            kp_xy = keypoints[..., :2].astype(np.float32)

            if self.prior_enabled and record["label"] == 1:
                blob, w = self._prior_target(keypoints, self.size)
                if blob is not None:
                    prior_blob, prior_weight = blob, w

            pose_region = self._pose_region(keypoints, kp_valid, self.size)
            if pose_region is not None:
                combined = pose_region * region
                if combined.sum() >= self.min_area:
                    region = combined
                else:
                    region = pose_region * (valid > 0)

        x = (canvas.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        image = torch.from_numpy(x).permute(2, 0, 1).contiguous()

        if self.mask_channels:
            extra = np.stack([support_i.astype(np.float32),
                              support_j.astype(np.float32),
                              self._proximity(support_i, support_j)], axis=0)
            image = torch.cat([image, torch.from_numpy(extra)], dim=0)

        return {
            "image": image,
            "region": torch.from_numpy(region)[None],
            "label": torch.tensor(float(record["label"])),
            "index": torch.tensor(idx),
            "kp_xy": torch.from_numpy(kp_xy.reshape(-1, 2)),
            "kp_valid": torch.from_numpy(kp_valid.reshape(-1)),
            "prior": torch.from_numpy(prior_blob)[None],
            "prior_weight": torch.tensor(np.float32(prior_weight)),
            "scale": torch.tensor(np.float32(scale)),
            "valid_hw": torch.tensor([nh, nw], dtype=torch.long),
        }


def split_records(records):
    buckets = {"train": [], "val": [], "test": []}
    for r in records:
        buckets.get(r["split"], buckets["train"]).append(r)
    return buckets


def records_for(records, split):
    b = split_records(records)
    if split in ("all", "known_interact"):
        return b["train"] + b["val"] + b["test"]
    return b[split]


def describe(buckets):
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
