
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contactTest.sam3 import Sam3
from contactTest.sam_contact_region import load_depth
from contactTest.src.data import load_records, relative_boxes, split_records
from contactTest.src.utils import load_config

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))


def depth_as_image(depth, inverse):
    lo, hi = np.percentile(depth, (2.0, 98.0))
    norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    if not inverse:
        norm = 1.0 - norm
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def cache_path(root, rel_image):
    return os.path.join(root, os.path.splitext(rel_image)[0] + ".npz")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(CONTACT_ROOT, "config.yaml"))
    ap.add_argument("--split", default=None, choices=["train", "val", "test"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--text", default="cow")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prompt-source", default="rgb",
                    choices=["rgb", "depth_image"])
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.conf is None:
        args.conf = float(cfg["data"].get("sam3_conf", 0.6))
    cache_root = os.path.join(
        CONTACT_ROOT,
        args.cache_dir or cfg["data"]["mask_dir"] or "log/mask_cache")

    buckets = split_records(load_records(cfg))
    splits = [args.split] if args.split else ["train", "val", "test"]
    records = [r for s in splits for r in buckets[s]]
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise SystemExit("no labelled rows to process")
    print(f"[mask] segmenting {len(records)} pairs -> {cache_root}")

    seg = Sam3(args.weights, args.text, args.conf)
    print(f"[mask] prompt source: {args.prompt_source}")
    done = skipped = failed = no_depth = 0
    areas = []

    for i, record in enumerate(records):
        out_path = cache_path(cache_root, record["rel_image"])
        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue

        bgr = cv2.imread(record["image_path"])
        if bgr is None:
            failed += 1
            continue
        h, w = bgr.shape[:2]
        boxes = relative_boxes(record, h, w)

        image = bgr
        if args.prompt_source == "depth_image":
            dep = load_depth(record, (h, w))
            if dep is None:
                no_depth += 1
                continue
            image = depth_as_image(dep[0], dep[2])

        try:
            masks = seg.assign_to_boxes(image, boxes)
        except Exception as err:
            print(f"[mask] failed on {record['rel_image']}: {err}")
            masks = None
        if masks is None or len(masks) < 2 or masks[0].sum() == 0 or masks[1].sum() == 0:
            failed += 1
            continue

        for m, b in zip(masks, boxes):
            areas.append(m.sum() / max((b[2] - b[0]) * (b[3] - b[1]), 1))

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, mi=masks[0].astype(np.uint8),
                            mj=masks[1].astype(np.uint8),
                            crop_hw=np.array([h, w], np.int32))
        done += 1
        if (i + 1) % 100 == 0:
            print(f"[mask] {i + 1}/{len(records)}  written {done}  failed {failed}")

    print(f"\n[mask] done: {done} written, {skipped} cached, {failed} failed")
    if no_depth:
        print(f"[mask] {no_depth} pairs skipped for want of a cached depth map; "
              "run precompute_depth.py first")
    if areas:
        a = np.array(areas)
        print(f"[mask] mask area / box area: median {np.median(a):.2f}  "
              f"p10 {np.percentile(a, 10):.2f}  p90 {np.percentile(a, 90):.2f}")


if __name__ == "__main__":
    main()
