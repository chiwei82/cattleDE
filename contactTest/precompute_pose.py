
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contactTest.src.data import load_records, split_records
from contactTest.src.pose import build_transform, load_pose_model, pose_for_pair
from contactTest.src.utils import load_config

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))


def cache_path(cache_root, rel_image):
    return os.path.join(cache_root, os.path.splitext(rel_image)[0] + ".npz")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(CONTACT_ROOT, "config.yaml"))
    ap.add_argument("--split", default=None, choices=["train", "val", "test"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_root = os.path.join(CONTACT_ROOT, cfg["pose"]["cache_dir"])

    buckets = split_records(load_records(cfg))
    splits = [args.split] if args.split else ["train", "val", "test"]
    records = [r for s in splits for r in buckets[s]]
    if not records:
        raise SystemExit("no labelled rows to process")
    print(f"[pose] caching {len(records)} pairs on {device} -> {cache_root}")

    model = load_pose_model(cfg, device)
    transform = build_transform(cfg)

    done = skipped = failed = 0
    for i, record in enumerate(records):
        out_path = cache_path(cache_root, record["rel_image"])
        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue

        crop = cv2.imread(record["image_path"])
        if crop is None:
            failed += 1
            continue
        keypoints, _ = pose_for_pair(record, crop, model, transform, cfg, device)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, keypoints=keypoints.astype(np.float32),
                            crop_hw=np.array(crop.shape[:2], np.int32))
        done += 1
        if (i + 1) % 200 == 0:
            print(f"[pose] {i + 1}/{len(records)}  written {done}  skipped {skipped}")

    print(f"[pose] done: {done} written, {skipped} already cached, {failed} unreadable")


if __name__ == "__main__":
    main()
