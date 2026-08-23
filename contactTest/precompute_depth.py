
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contactTest.src.data import load_records, split_records
from contactTest.src.utils import load_config

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))

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
        depth = self.torch.nn.functional.interpolate(
            out[:, None], size=bgr.shape[:2], mode="bicubic", align_corners=False
        )[0, 0]
        return depth.float().cpu().numpy()


class _RepoDAV2:

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
        return np.asarray(self.model.infer_image(bgr), np.float32)


def build_model(args, device):
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
            f"could not load Depth Anything V2 through transformers ({exc})"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(CONTACT_ROOT, "config.yaml"))
    ap.add_argument("--split", default=None, choices=["train", "val", "test"])
    ap.add_argument("--encoder", default=None, choices=["vits", "vitb", "vitl"])
    ap.add_argument("--metric", default=None, choices=["indoor", "outdoor"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default=None)
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
