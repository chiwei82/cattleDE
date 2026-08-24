"""Skeleton-aware augmentation for the interaction crops.

Ported from reproduction/CattleAct/src/augmentation.py, trimmed to the two
transforms the interaction pipeline uses and adapted to the BINARY label scheme
(0 = no_interaction, 1 = interaction). Both take the merged two-cow crop plus the
two adjusted skeletons and apply label-invariant Cutout that avoids the joints
that define the interaction.
"""

import math
import random

import numpy as np
from PIL import ImageDraw


# Default AK (23-keypoint) body-part grouping used to protect interaction-defining
# joints from masking. Animal Kingdom has no explicit neck point, so the shoulders
# stand in for "neck".
AK_JOINT_MAP = {
    "head": [0, 1, 2, 3, 4, 5, 6],
    "neck": [7, 8],
    "torso": [13, 14, 15],
    "left_front_leg": [7, 9, 11],
    "right_front_leg": [8, 10, 12],
    "left_hind_leg": [14, 16, 18],
    "right_hind_leg": [15, 17, 19],
    "tail": [20, 21, 22],
}


class ImageMaskingFromSkeletonForInteraction:
    """Random Cutout on the two-cow crop that avoids the interaction-defining
    joints of either animal. Binary scheme: negatives (0) are never masked;
    positives (1) protect head/neck/torso."""

    def __init__(
        self,
        joint_map,
        cutout_prob=0.5,
        n_holes=1,
        scale=(0.02, 0.2),
        ratio=(0.3, 3.3),
        max_trials=10,
        margin=10,
        label_to_protected_parts=None,
    ):
        self.joint_map = joint_map
        self.cutout_prob = cutout_prob
        self.max_n_holes = n_holes
        self.scale = scale
        self.ratio = ratio
        self.max_trials = max_trials
        self.margin = margin
        # Binary default: don't mask no_interaction; protect the body core on
        # interaction so the masking stays label-invariant.
        self.label_to_protected_parts = label_to_protected_parts or {
            0: [],
            1: ["head", "neck", "torso"],
        }

    def __call__(self, image, skeleton1, skeleton2, label):
        n_holes = 0
        if random.random() < self.cutout_prob:
            n_holes = random.randint(1, self.max_n_holes)

        image_copy = image.copy()
        draw = ImageDraw.Draw(image_copy)
        img_w, img_h = image.size
        img_area = img_w * img_h

        label_item = label.item() if hasattr(label, "item") else label
        protected_parts = self.label_to_protected_parts.get(label_item)
        if not protected_parts:
            # nothing to protect (e.g. no_interaction) -> leave the image as-is
            return image

        protected_indices = {
            idx for part in protected_parts for idx in self.joint_map.get(part, [])
        }

        all_protected_kpts = []
        for skeleton in [skeleton1, skeleton2]:
            skeleton_np = (
                skeleton.cpu().numpy() if hasattr(skeleton, "cpu")
                else np.array(skeleton)
            )
            if skeleton_np.size == 0:
                continue
            valid_indices = [i for i in protected_indices if i < len(skeleton_np)]
            if not valid_indices:
                continue
            kpts = skeleton_np[valid_indices, :2]
            valid_kpts = kpts[(kpts[:, 0] > 1) & (kpts[:, 1] > 1)]
            if valid_kpts.shape[0] > 0:
                all_protected_kpts.append(valid_kpts)

        if not all_protected_kpts:
            return image
        valid_protected_kpts = np.vstack(all_protected_kpts)

        for _ in range(n_holes):
            for _ in range(self.max_trials):
                hole_area = img_area * random.uniform(self.scale[0], self.scale[1])
                aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
                h = int(round(math.sqrt(hole_area / aspect_ratio)))
                w = int(round(math.sqrt(hole_area * aspect_ratio)))
                if h >= img_h or w >= img_w:
                    continue
                x1 = random.randint(0, img_w - w)
                y1 = random.randint(0, img_h - h)
                x2, y2 = x1 + w, y1 + h
                is_colliding = np.any(
                    (valid_protected_kpts[:, 0] >= x1 - self.margin) &
                    (valid_protected_kpts[:, 0] < x2 + self.margin) &
                    (valid_protected_kpts[:, 1] >= y1 - self.margin) &
                    (valid_protected_kpts[:, 1] < y2 + self.margin)
                )
                if not is_colliding:
                    draw.rectangle([x1, y1, x2, y2], fill="black")
                    break
        return image_copy


class StandardCutout:
    """Random Cutout with no skeleton protection. Signature matches the
    skeleton-aware transform so the dataset can call either interchangeably."""

    def __init__(self, cutout_prob=0.5, n_holes=1, scale=(0.02, 0.2), ratio=(0.3, 3.3)):
        self.cutout_prob = cutout_prob
        self.n_holes = n_holes
        self.scale = scale
        self.ratio = ratio

    def __call__(self, image, skeleton1=None, skeleton2=None, label=None):
        if random.random() > self.cutout_prob:
            return image
        image_copy = image.copy()
        draw = ImageDraw.Draw(image_copy)
        img_w, img_h = image.size
        img_area = img_w * img_h
        for _ in range(self.n_holes):
            hole_area = img_area * random.uniform(self.scale[0], self.scale[1])
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            h = int(round(math.sqrt(hole_area / aspect_ratio)))
            w = int(round(math.sqrt(hole_area * aspect_ratio)))
            if h >= img_h or w >= img_w:
                continue
            x1 = random.randint(0, img_w - w)
            y1 = random.randint(0, img_h - h)
            draw.rectangle([x1, y1, x1 + w, y1 + h], fill="black")
        return image_copy
