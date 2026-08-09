"""HRNet-W32 / AP-10K pose estimation for cattle pair crops.

Reuses the network definition and decoding conventions already in
prep/interaction_prep.py so that keypoints produced here are identical to the
ones that script would have written had use_pose been enabled. Nothing outside
contactTest is modified; the checkpoint and prep/src/hrnet.py are read only.

Keypoints are computed FROM THE PAIR CROP, not from the source video. The crop
is the merged box cut out of the frame, and relative_boxes() gives each animal's
box inside it, so each cow can be re-cropped and posed without touching the
original footage.

AP-10K uses 17 quadruped joints. Their layout matters for this project: 380 of
the 384 positive pairs are head-initiated (grooming is head-to-head, interest is
head-to-body, sniffing is head-led; only the 4 mount pairs are not), so the
nose / eye / neck joints are where contact evidence is expected to sit.
"""

import importlib.util
import os

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from .data import REPO_ROOT, relative_boxes

# AP-10K joint order, as produced by the mmpose 0.x checkpoint.
KEYPOINT_NAMES = [
    "left_eye", "right_eye", "nose", "neck", "root_of_tail",
    "left_shoulder", "left_elbow", "left_front_paw",
    "right_shoulder", "right_elbow", "right_front_paw",
    "left_hip", "left_knee", "left_back_paw",
    "right_hip", "right_knee", "right_back_paw",
]

# Joints that make up the head. Contact in this dataset is overwhelmingly
# head-initiated, so these are the ones worth trusting a prior on.
HEAD_JOINTS = [0, 1, 2, 3]

SKELETON = [
    (0, 1), (0, 2), (1, 2), (2, 3), (3, 4),
    (3, 5), (5, 6), (6, 7),
    (3, 8), (8, 9), (9, 10),
    (4, 11), (11, 12), (12, 13),
    (4, 14), (14, 15), (15, 16),
]

# Left/right joints must swap identity under a horizontal flip. Mirroring the
# coordinates alone would label the animal's left legs as its right ones and
# scramble the anatomy the keypoint output space depends on.
FLIP_PAIRS = [(0, 1), (5, 8), (6, 9), (7, 10), (11, 14), (12, 15), (13, 16)]


def flip_keypoints(keypoints, width):
    """Mirror keypoints about a vertical axis and swap the left/right joints."""
    out = keypoints.copy()
    out[..., 0] = (width - 1) - out[..., 0]
    for a, b in FLIP_PAIRS:
        out[..., [a, b], :] = out[..., [b, a], :]
    return out


def _load_hrnet_class():
    """Import prep/src/hrnet.py by path.

    The repository has two different `src` packages (repo-root and prep/), so a
    plain import would be ambiguous; loading by file path sidesteps the clash
    without mutating sys.path.
    """
    path = os.path.join(REPO_ROOT, "prep", "src", "hrnet.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"HRNet definition not found at {path}")
    spec = importlib.util.spec_from_file_location("contacttest_hrnet_def", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HRNetW32


class _KeypointHead(torch.nn.Module):
    def __init__(self, in_channels=32, num_joints=17):
        super().__init__()
        self.final_layer = torch.nn.Conv2d(in_channels, num_joints, kernel_size=1)

    def forward(self, x):
        return self.final_layer(x)


class HRNetPose(torch.nn.Module):
    """Backbone + keypoint head, matching prep/interaction_prep.py exactly."""

    def __init__(self, num_joints=17):
        super().__init__()
        self.backbone = _load_hrnet_class()()
        self.keypoint_head = _KeypointHead(num_joints=num_joints)

    def forward(self, x):
        return self.keypoint_head(self.backbone(x))


def load_pose_model(cfg, device):
    """Load the AP-10K checkpoint; the state dict is stripped of DDP prefixes."""
    pcfg = cfg["pose"]
    ckpt_path = os.path.join(REPO_ROOT, pcfg["hrnet_ckpt"])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"HRNet checkpoint not found at {ckpt_path}. It is referenced by "
            "global_config.yaml paths.hrnet_ckpt; copy it there or point "
            "pose.hrnet_ckpt at its actual location.")

    model = HRNetPose(int(pcfg["num_joints"])).to(device)
    raw = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = raw.get("state_dict", raw.get("model", raw))
    state = {
        k.removeprefix("module.").removeprefix("model.").removeprefix("net."): v
        for k, v in state.items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[pose] WARNING: {len(missing)} missing keys (e.g. {missing[0]})")
    if unexpected:
        print(f"[pose] WARNING: {len(unexpected)} unexpected keys (e.g. {unexpected[0]})")
    model.eval()
    return model


def build_transform(cfg):
    size = int(cfg["pose"]["input_size"])
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def decode_heatmaps(heatmaps):
    """(N, J, H, W) -> (N, J, 3) as [x, y, conf], with the +/-0.25 px shift.

    Identical to prep/interaction_prep.decode_heatmaps, vectorised over joints.
    """
    n, j, h, w = heatmaps.shape
    flat = heatmaps.reshape(n, j, -1)
    idx = flat.argmax(axis=2)
    conf = np.take_along_axis(flat, idx[..., None], axis=2)[..., 0]
    y = (idx // w).astype(np.float32)
    x = (idx % w).astype(np.float32)

    for a in range(n):
        for b in range(j):
            py, px = int(y[a, b]), int(x[a, b])
            hm = heatmaps[a, b]
            if 1 <= px < w - 1:
                x[a, b] += 0.25 * np.sign(hm[py, px + 1] - hm[py, px - 1])
            if 1 <= py < h - 1:
                y[a, b] += 0.25 * np.sign(hm[py + 1, px] - hm[py - 1, px])
    return np.stack([x, y, conf], axis=2).astype(np.float32)


@torch.no_grad()
def pose_for_pair(record, crop_bgr, model, transform, cfg, device):
    """Keypoints for both cows, in PAIR-CROP pixel coordinates.

    Returns a (2, J, 3) array of [x, y, conf] plus the two relative boxes.
    """
    import cv2

    h, w = crop_bgr.shape[:2]
    boxes = relative_boxes(record, h, w)
    hmap_size = float(cfg["pose"]["hmap_size"])

    tensors, sizes = [], []
    for (x1, y1, x2, y2) in boxes:
        sub = crop_bgr[y1:y2, x1:x2]
        if sub.size == 0:
            sub = crop_bgr
            x1, y1, x2, y2 = 0, 0, w, h
        pil = Image.fromarray(cv2.cvtColor(sub, cv2.COLOR_BGR2RGB))
        tensors.append(transform(pil))
        sizes.append((x2 - x1, y2 - y1, x1, y1))

    batch = torch.stack(tensors).to(device)
    heatmaps = model(batch).cpu().numpy()
    decoded = decode_heatmaps(heatmaps)                     # (2, J, 3)

    # Heatmap space -> the animal's own crop -> pair-crop coordinates.
    out = decoded.copy()
    for i, (bw, bh, ox, oy) in enumerate(sizes):
        out[i, :, 0] = out[i, :, 0] * (bw / hmap_size) + ox
        out[i, :, 1] = out[i, :, 1] * (bh / hmap_size) + oy
    return out, boxes


def closest_head_link(keypoints, min_conf):
    """Shortest link from either cow's head joints to any joint of the other.

    A training-free contact estimate: if the pose is reliable and contact really
    is head-initiated, this segment should already land on the contact site. It
    is the baseline any learned heatmap has to beat before it is worth using.

    Returns (point_a, point_b, distance, joint_a, joint_b) or None.
    """
    best = None
    for src, dst in ((0, 1), (1, 0)):
        for ja in HEAD_JOINTS:
            if keypoints[src, ja, 2] < min_conf:
                continue
            for jb in range(keypoints.shape[1]):
                if keypoints[dst, jb, 2] < min_conf:
                    continue
                pa = keypoints[src, ja, :2]
                pb = keypoints[dst, jb, :2]
                d = float(np.hypot(*(pa - pb)))
                if best is None or d < best[2]:
                    best = (pa, pb, d, f"cow{src + 1}.{KEYPOINT_NAMES[ja]}",
                            f"cow{dst + 1}.{KEYPOINT_NAMES[jb]}")
    return best
