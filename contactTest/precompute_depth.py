"""Cache a Depth Anything V2 depth map for every pair crop.

Usage (from the repository root):

    python -m contactTest.precompute_depth --split train
    python -m contactTest.precompute_depth --split train --encoder vitl
    python -m contactTest.precompute_depth --overwrite

Writes one .npz per pair under contactTest/log/depth_cache/, mirroring the
crop's relative path, holding a float16 array 'depth' at the crop's own
resolution plus the scalars used to normalise it.

WHY DEPTH IS WORTH ADDING

The contact band is the intersection of two dilated instance masks, which is a
statement about the two silhouettes in the IMAGE PLANE only. From an overhead
camera two animals whose projections meet may be touching, or one may simply be
passing behind the other a metre away. `overlap` in particular means occlusion
at least as often as it means contact, and no amount of tuning the dilation
radius can tell those apart, because the information needed is not in the
silhouettes. Depth is exactly the missing axis.

WHAT IS STORED, AND IN WHICH DIRECTION

The relative Depth Anything V2 checkpoints predict INVERSE depth: the raw output
is larger for nearer surfaces and is defined only up to an unknown scale and
shift. That is fine here for two reasons. The comparison the gate makes is
between two surfaces WITHIN one crop, so the unknown scale cancels as long as
both are read off the same map; and the pen camera looks down from a fixed
height, so the depth range spanned by one crop is small next to the absolute
camera distance, which is the regime where a difference in inverse depth stays
close to proportional to a difference in metres. Neither would hold if crops
were compared against each other, so they never are.

What is cached is the raw prediction plus robust 2nd/98th percentiles of it.
Normalisation is deliberately left to the consumer rather than baked in, so that
the percentile convention can be changed without re-running the network over the
whole dataset, which is the expensive part.

The metric checkpoints (--metric indoor|outdoor) predict metres directly, in the
ordinary direction, and set 'inverse' to 0 in the cache so the consumer can tell.
They need no scale assumption at all, but were trained on indoor scans and
driving footage respectively, so on a cattle pen the relative model is the
safer default and the metric ones are worth a comparison, not a default.
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contactTest.src.data import load_records, split_records
from contactTest.src.utils import load_config

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Hugging Face ids. The relative models are the default; the metric ones are
# offered for comparison because a barn matches neither of their training
# domains especially well.
HF_RELATIVE = {
    "vits": "depth-anything/Depth-Anything-V2-Small-hf",
    "vitb": "depth-anything/Depth-Anything-V2-Base-hf",
    "vitl": "depth-anything/Depth-Anything-V2-Large-hf",
}
HF_METRIC = {
    "indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    "outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
}


class _TransformersDAV2:
    """Depth Anything V2 through transformers' AutoModelForDepthEstimation."""

    def __init__(self, model_id, device):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model.to(device).eval()

    def __call__(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.model(**inputs).predicted_depth
        # The network runs at its own resolution; put it back on the crop's grid
        # so that a depth value and a mask pixel refer to the same place.
        depth = self.torch.nn.functional.interpolate(
            out[:, None], size=bgr.shape[:2], mode="bicubic", align_corners=False
        )[0, 0]
        return depth.float().cpu().numpy()


class _RepoDAV2:
    """Fallback: the official Depth-Anything-V2 repository on sys.path."""

    def __init__(self, encoder, ckpt, device):
        import torch
        from depth_anything_v2.dpt import DepthAnythingV2

        cfgs = {
            "vits": dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384]),
            "vitb": dict(encoder="vitb", features=128, out_channels=[96, 192, 384, 768]),
            "vitl": dict(encoder="vitl", features=256,
                         out_channels=[256, 512, 1024, 1024]),
        }
        self.torch = torch
        self.model = DepthAnythingV2(**cfgs[encoder])
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model.to(device).eval()

    def __call__(self, bgr):
        # infer_image takes BGR and returns a map at the input resolution.
        return np.asarray(self.model.infer_image(bgr), np.float32)


def build_model(args, device):
    """Return a callable bgr -> float32 depth map, and whether it is inverse."""
    if args.metric:
        model_id = HF_METRIC[args.metric]
        return _TransformersDAV2(model_id, device), False, model_id

    if args.ckpt:
        return _RepoDAV2(args.encoder, args.ckpt, device), True, args.ckpt

    model_id = HF_RELATIVE[args.encoder]
    try:
        return _TransformersDAV2(model_id, device), True, model_id
    except ImportError as exc:
        raise SystemExit(
            f"could not load Depth Anything V2 through transformers ({exc}).\n"
            "Either install transformers, or clone the official repository and "
            "pass --ckpt path/to/depth_anything_v2_vitl.pth"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(CONTACT_ROOT, "config.yaml"))
    ap.add_argument("--split", default=None, choices=["train", "val", "test"],
                    help="default: every split")
    ap.add_argument("--encoder", default=None, choices=["vits", "vitb", "vitl"],
                    help="relative model size; default comes from config.yaml")
    ap.add_argument("--metric", default=None, choices=["indoor", "outdoor"],
                    help="use a metric checkpoint (metres) instead of the "
                         "relative one. Changes the cache contents, so pair it "
                         "with a separate --cache-dir")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint for the official repository, if transformers "
                         "is not available")
    ap.add_argument("--cache-dir", default=None,
                    help="default: the depth.cache_dir entry of config.yaml")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default=None, help="cuda | cpu; default autodetect")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg.get("depth") or {}
    if args.encoder is None:
        args.encoder = dcfg.get("encoder", "vitl")
    cache_dir = args.cache_dir or dcfg.get("cache_dir", "log/depth_cache")
    cache_root = os.path.join(CONTACT_ROOT, cache_dir)

    records = load_records(cfg, require_label=False)
    if args.split:
        records = split_records(records)[args.split]
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise SystemExit("no rows to process")

    todo = [r for r in records
            if args.overwrite or not os.path.exists(
                os.path.join(cache_root, os.path.splitext(r["rel_image"])[0] + ".npz"))]
    print(f"[depth] {len(records)} crops, {len(todo)} to compute")
    if not todo:
        print("[depth] nothing to do; pass --overwrite to recompute")
        return

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, inverse, tag = build_model(args, device)
    print(f"[depth] {tag} on {device}  "
          f"({'inverse relative' if inverse else 'metric'} depth)")

    done = failed = 0
    for i, record in enumerate(todo):
        bgr = cv2.imread(record["image_path"])
        if bgr is None:
            failed += 1
            continue
        depth = model(bgr)
        if depth.shape != bgr.shape[:2]:
            depth = cv2.resize(depth, (bgr.shape[1], bgr.shape[0]),
                               interpolation=cv2.INTER_LINEAR)

        # Robust range, stored alongside the map. Plain min/max would be set by a
        # single speck of glare or one stray pixel of railing, and the tolerance
        # the gate applies is expressed as a fraction of this number.
        lo, hi = np.percentile(depth, (2.0, 98.0))
        out_path = os.path.join(cache_root,
                                os.path.splitext(record["rel_image"])[0] + ".npz")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, depth=depth.astype(np.float16),
                            p2=np.float32(lo), p98=np.float32(hi),
                            inverse=np.uint8(1 if inverse else 0))
        done += 1
        if (i + 1) % 200 == 0:
            print(f"[depth] {i + 1}/{len(todo)}")

    print(f"[depth] wrote {done} maps to {cache_root}")
    if failed:
        print(f"[depth] {failed} crops could not be read")
    print("[depth] next: python -m contactTest.score_contact --split "
          f"{args.split or 'train'} --depth-tol 0.05,0.10")


if __name__ == "__main__":
    main()
