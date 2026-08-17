"""Cache SAM 3 instance masks for every labelled pair crop.

Usage (from the repository root):

    python -m contactTest.precompute_masks --split train
    python -m contactTest.precompute_masks --split train --overwrite

Writes one .npz per pair under contactTest/log/mask_cache/, mirroring the crop's
relative path, holding uint8 arrays 'mi' and 'mj' at the crop's own resolution.

Prompting is by CONCEPT: `--text cow`. Both animals are black-and-white
Holsteins whose colour statistics are identical and whose boxes overlap by
construction, so nothing keyed on colour or on box geometry separates them. The
class-agnostic SAM 1 backends this file used to carry answered a box with
whichever region best fitted it, and on this footage a uniform patch of pen
floor fits a slightly loose box better than a high-contrast animal does; every
prompt trick here — a centre point, extra points, depth-derived negative points
— existed to steer around that. Asking for the concept removes the premise, so
those backends were deleted rather than kept as a baseline nothing measures.

SAM 3 is reached through `contactTest.sam3.Sam3` and nowhere else, as everything
in contactTest now is. The ultralytics box-prompt backend that used to sit
beside it is gone: it needed a box before it could be prompted, which is the
thing the current questions are trying to do without.

`--prompt-source depth_image` went with it only in part — segmenting the
colourised depth map instead of the photograph is a matter of which IMAGE is
passed, so it still works with a concept prompt. `depth_points` did not survive:
negative point prompts have nowhere to go in a text-prompted model.
"""

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
    """The depth map rendered as a 3-channel BGR image for SAM to segment.

    Worth trying because the failure being chased is a texture failure: SAM keys
    on learned objectness, and on a Holstein the boundary between a black patch
    and a white patch is a stronger edge than the boundary between the animal
    and the shed floor, which is how a mask ends up as one patch of hide. A
    depth map has no coat pattern in it at all, so an animal is a single smooth
    blob and its outline is the only strong edge present.

    The trade is that it also has no eyes, ears or legs, so SAM has less to
    recognise as an object and can merge two animals standing at the same range.
    Which effect wins is a measurement, not a guess.
    """
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
    ap.add_argument("--weights", default=None,
                    help="Hugging Face id or a local snapshot DIRECTORY")
    ap.add_argument("--text", default="cow",
                    help="noun phrase for the concept prompt. SAM 3 segments "
                         "the concept itself, so the pen floor is not a "
                         "candidate answer the way it is for a bare box")
    ap.add_argument("--conf", type=float, default=None,
                    help="SAM 3 score floor; default is data.sam3_conf "
                         "from config.yaml")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N pairs, for a quick quality check")
    ap.add_argument("--prompt-source", default="rgb",
                    choices=["rgb", "depth_image"],
                    help="rgb: segment the photograph. depth_image: segment the "
                         "colourised depth map instead, so coat pattern cannot "
                         "mislead it; needs precompute_depth.py. Both use the "
                         "same concept prompt — only the IMAGE differs")
    ap.add_argument("--cache-dir", default=None,
                    help="override data.mask_dir. Give each prompt source its own "
                         "directory so the variants can be scored against each "
                         "other instead of overwriting one another")
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
            # assign_to_boxes, not detect: the two animals are already chosen by
            # the detector, so what is wanted from SAM 3 is the segmentation and
            # the correspondence to bbox1/bbox2, not a fresh search.
            masks = seg.assign_to_boxes(image, boxes)
        except Exception as err:                   # noqa: BLE001
            print(f"[mask] failed on {record['rel_image']}: {err}")
            masks = None
        if masks is None or len(masks) < 2 or masks[0].sum() == 0 or masks[1].sum() == 0:
            failed += 1
            continue

        # An instance mask should fill a good part of its own box and should not
        # swallow the other animal; log both so quality is auditable.
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
        print("[mask] a cow fills roughly 0.4-0.8 of its axis-aligned box; a median "
              "far outside that means the prompts or the weights are wrong")
    print("[mask] set data.mask_channels: true to feed these to the model")


if __name__ == "__main__":
    main()
