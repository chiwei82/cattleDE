"""Cache SAM instance masks for every labelled pair crop.

Usage (from the repository root):

    python -m contactTest.precompute_masks                # all splits
    python -m contactTest.precompute_masks --split train
    python -m contactTest.precompute_masks --overwrite

Writes one .npz per pair under contactTest/log/mask_cache/, mirroring the crop's
relative path, holding uint8 arrays 'mi' and 'mj' at the crop's own resolution.

Why SAM and not a classical segmenter: both animals are black-and-white
Holsteins whose colour statistics are identical, and the two boxes overlap by
construction (the pair filter keeps IoU > 0.1). GrabCut initialised from either
box therefore returns one animal and an empty mask for the other — measured on
this dataset, so the contact band came out empty. SAM separates touching
instances of the same class because it keys on learned objectness rather than
colour, which is exactly the property needed here.

Prompts are a box PLUS a positive point at its centre, matching panel 2 of
visualize_sam_confusion exactly — SAM has no concept of "cow" and a box alone is
routinely answered with a coherent patch of floor, while the point forces the
mask to contain that pixel. Use --no-point only to reproduce the box-only
behaviour for comparison.

Ultralytics is already a dependency of the repository, so its SAM wrapper is
tried first; the reference segment-anything package is the fallback.
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contactTest.src.data import load_records, relative_boxes, split_records
from contactTest.src.utils import load_config

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))


class _UltralyticsSAM:
    """Box-prompted SAM via ultralytics (already in requirements.txt)."""

    def __init__(self, weights):
        from ultralytics import SAM

        self.model = SAM(weights)

    def __call__(self, bgr, boxes, use_point=True):
        kw = {"bboxes": [list(map(float, b)) for b in boxes]}
        if use_point:
            kw["points"] = [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes]
            kw["labels"] = [1] * len(boxes)
        results = self.model(bgr, verbose=False, **kw)
        masks = getattr(results[0], "masks", None)
        if masks is None or masks.data is None or len(masks.data) < len(boxes):
            return None
        out = []
        for k in range(len(boxes)):
            m = masks.data[k].cpu().numpy().astype(np.uint8)
            if m.shape != bgr.shape[:2]:
                m = cv2.resize(m, (bgr.shape[1], bgr.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            out.append(m)
        return out


class _ReferenceSAM:
    """Box-prompted SAM via the reference segment-anything package."""

    def __init__(self, weights, model_type="vit_b"):
        import torch
        from segment_anything import SamPredictor, sam_model_registry

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[model_type](checkpoint=weights).to(device)
        self.predictor = SamPredictor(sam)

    def __call__(self, bgr, boxes, use_point=True):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)
        out = []
        for b in boxes:
            x1, y1, x2, y2 = map(float, b)
            kw = {"box": np.array([x1, y1, x2, y2], np.float32)[None],
                  "multimask_output": False}
            if use_point:
                kw["point_coords"] = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]],
                                              np.float32)
                kw["point_labels"] = np.array([1], np.int32)
            masks, _, _ = self.predictor.predict(**kw)
            out.append(masks[0].astype(np.uint8))
        return out


def build_segmenter(cfg, backend="auto", model_type="vit_b"):
    """Pick a SAM backend.

    The two are not interchangeable: they wrap different checkpoints and
    post-processing, so the masks differ. That matters because the green band in
    panel 3 of visualize_sam_confusion comes from the reference package, and
    scoring it against masks cached from ultralytics would score a different
    band. Use backend="reference" to keep the two identical.
    """
    weights = cfg["data"].get("sam_weights", "sam_b.pt")
    if backend == "reference":
        seg = _ReferenceSAM(weights, model_type)
        print(f"[mask] using segment-anything ({weights}, {model_type})")
        return seg
    if backend == "ultralytics":
        seg = _UltralyticsSAM(weights)
        print(f"[mask] using ultralytics SAM ({weights})")
        return seg
    try:
        seg = _UltralyticsSAM(weights)
        print(f"[mask] using ultralytics SAM ({weights})")
        print("[mask] NOTE: visualize_sam_confusion uses segment-anything, so "
              "these masks are not the ones its panels were drawn from. Pass "
              "--backend reference to make them match.")
        return seg
    except Exception as err:                       # noqa: BLE001 - report and fall back
        print(f"[mask] ultralytics SAM unavailable ({err}); trying segment-anything")
    seg = _ReferenceSAM(weights, model_type)
    print(f"[mask] using segment-anything ({weights}, {model_type})")
    return seg


def cache_path(root, rel_image):
    return os.path.join(root, os.path.splitext(rel_image)[0] + ".npz")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(CONTACT_ROOT, "config.yaml"))
    ap.add_argument("--split", default=None, choices=["train", "val", "test"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "reference", "ultralytics"],
                    help="which SAM wrapper to use. 'reference' matches "
                         "visualize_sam_confusion exactly, so the cached masks "
                         "are the ones its panels were drawn from")
    ap.add_argument("--model-type", default="vit_b",
                    help="checkpoint variant for the reference backend")
    ap.add_argument("--no-point", action="store_true",
                    help="box prompt only. The default adds a positive point at "
                         "the box centre, which is what stops SAM answering with "
                         "the floor; visualize_sam_confusion uses the same rule, "
                         "so the cached masks match what its panel 2 showed")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N pairs, for a quick quality check")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cache_root = os.path.join(CONTACT_ROOT, cfg["data"]["mask_dir"] or "log/mask_cache")

    buckets = split_records(load_records(cfg))
    splits = [args.split] if args.split else ["train", "val", "test"]
    records = [r for s in splits for r in buckets[s]]
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise SystemExit("no labelled rows to process")
    print(f"[mask] segmenting {len(records)} pairs -> {cache_root}")

    seg = build_segmenter(cfg, args.backend, args.model_type)
    done = skipped = failed = 0
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

        try:
            masks = seg(bgr, boxes, use_point=not args.no_point)
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
    if areas:
        a = np.array(areas)
        print(f"[mask] mask area / box area: median {np.median(a):.2f}  "
              f"p10 {np.percentile(a, 10):.2f}  p90 {np.percentile(a, 90):.2f}")
        print("[mask] a cow fills roughly 0.4-0.8 of its axis-aligned box; a median "
              "far outside that means the prompts or the weights are wrong")
    print("[mask] set data.mask_channels: true to feed these to the model")


if __name__ == "__main__":
    main()
