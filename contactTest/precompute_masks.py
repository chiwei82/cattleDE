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

from contactTest.sam_contact_region import load_depth
from contactTest.src.data import load_records, relative_boxes, split_records
from contactTest.src.utils import load_config

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))


class _UltralyticsSAM:
    """Box-prompted SAM via ultralytics (already in requirements.txt)."""

    def __init__(self, weights):
        from ultralytics import SAM

        self.model = SAM(weights)

    def __call__(self, bgr, boxes, use_point=True, prompts=None):
        kw = {"bboxes": [list(map(float, b)) for b in boxes]}
        if prompts is not None:
            # One list of points per box, each with its own +/- labels.
            kw["points"] = [[[float(x), float(y)] for x, y in p] for p, _ in prompts]
            kw["labels"] = [[int(v) for v in lab] for _, lab in prompts]
        elif use_point:
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


class _SAM3Text:
    """SAM 3 concept prompting: segment the noun phrase, then match to the boxes.

    This is a different kind of prompt from everything else in this file, and it
    addresses the failure the others work around rather than the symptom. SAM 1
    and SAM 2 are class-agnostic: given a box they return whichever coherent
    region best fits it, and on this footage a uniform patch of pen floor is a
    more coherent region than a high-contrast Holstein, so a slightly loose box
    is answered with the ground. Positive points, negative points and depth
    prompts are all ways of steering that. SAM 3 accepts the concept itself, so
    the floor is not a candidate answer in the first place.

    Promptable concept segmentation returns EVERY instance of the concept in the
    image with its own identity, not one mask per prompt. A pair crop can hold a
    third animal at its edge, so the returned instances are assigned to the two
    detector boxes by mask/box IoU, greedily and without replacement, and a pair
    is rejected if either box finds nothing. That assignment is the one piece
    here with no counterpart in the box-prompted path, and it is where this
    backend can go wrong: check a sample with compare_masks.py before trusting a
    whole cache of it.
    """

    def __init__(self, weights, text="cow", conf=0.25):
        from ultralytics.models.sam import SAM3SemanticPredictor

        self.text = text
        self.predictor = SAM3SemanticPredictor(overrides={
            "conf": conf, "task": "segment", "mode": "predict",
            "model": weights, "quantize": 16, "save": False, "verbose": False})
        print(f"[mask] SAM 3 concept prompting, text={text!r}, conf={conf}")

    @staticmethod
    def _iou_with_box(mask, box):
        x1, y1, x2, y2 = (int(max(0, box[0])), int(max(0, box[1])),
                          int(min(mask.shape[1], box[2])),
                          int(min(mask.shape[0], box[3])))
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = float(mask[y1:y2, x1:x2].sum())
        union = float(mask.sum()) + (x2 - x1) * (y2 - y1) - inter
        return inter / max(union, 1.0)

    def __call__(self, bgr, boxes, use_point=True, prompts=None, path=None):
        # set_image is documented with a path; an array works in the ultralytics
        # predictors, and the path is preferred when we have one so that any
        # internal preprocessing matches the documented behaviour exactly.
        self.predictor.set_image(path if path else bgr)
        results = self.predictor(text=[self.text])
        masks = getattr(results[0], "masks", None)
        if masks is None or masks.data is None or len(masks.data) == 0:
            return None

        inst = []
        for m in masks.data:
            a = m.cpu().numpy().astype(np.uint8)
            if a.shape != bgr.shape[:2]:
                a = cv2.resize(a, (bgr.shape[1], bgr.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            inst.append(a)

        # Greedy assignment over the full IoU matrix: take the best box/instance
        # pair, remove both, repeat. With two boxes this is optimal.
        scores = [[self._iou_with_box(a, b) for a in inst] for b in boxes]
        out = [None] * len(boxes)
        taken = set()
        for _ in range(len(boxes)):
            best, bk, bi = 0.0, None, None
            for k in range(len(boxes)):
                if out[k] is not None:
                    continue
                for i in range(len(inst)):
                    if i in taken:
                        continue
                    if scores[k][i] > best:
                        best, bk, bi = scores[k][i], k, i
            if bk is None or best <= 0.0:
                break
            out[bk] = inst[bi]
            taken.add(bi)
        return None if any(o is None for o in out) else out


class _ReferenceSAM:
    """Box-prompted SAM via the reference segment-anything package."""

    def __init__(self, weights, model_type="vit_b"):
        import torch
        from segment_anything import SamPredictor, sam_model_registry

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[model_type](checkpoint=weights).to(device)
        self.predictor = SamPredictor(sam)

    def __call__(self, bgr, boxes, use_point=True, prompts=None):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)
        out = []
        for k, b in enumerate(boxes):
            x1, y1, x2, y2 = map(float, b)
            kw = {"box": np.array([x1, y1, x2, y2], np.float32)[None],
                  "multimask_output": False}
            if prompts is not None:
                pts, lab = prompts[k]
                kw["point_coords"] = np.asarray(pts, np.float32)
                kw["point_labels"] = np.asarray(lab, np.int32)
            elif use_point:
                kw["point_coords"] = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]],
                                              np.float32)
                kw["point_labels"] = np.array([1], np.int32)
            masks, _, _ = self.predictor.predict(**kw)
            out.append(masks[0].astype(np.uint8))
        return out


def build_segmenter(cfg, backend="auto", model_type="vit_b", text="cow",
                    conf=0.25):
    """Pick a SAM backend.

    The two are not interchangeable: they wrap different checkpoints and
    post-processing, so the masks differ. That matters because the green band in
    panel 3 of visualize_sam_confusion comes from the reference package, and
    scoring it against masks cached from ultralytics would score a different
    band. Use backend="reference" to keep the two identical.
    """
    weights = cfg["data"].get("sam_weights", "sam_b.pt")
    if backend == "sam3_text":
        # sam3.pt is gated: request access at huggingface.co/facebook/sam3 and
        # place the file where sam_weights points, or pass --weights.
        return _SAM3Text(weights, text, conf)
    if backend == "sam3":
        # SAM 3 driven with the SAM 2 style box/point interface, so depth
        # prompts still apply and the only variable is the checkpoint.
        seg = _UltralyticsSAM(weights)
        print(f"[mask] SAM 3 with box/point prompts ({weights})")
        return seg
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


def _sample(region, n, rng, erode=5):
    """Up to n well-interior points of a binary region, spread out.

    Eroding first keeps a prompt off the class boundary, where a point or two of
    depth noise would put it on the wrong side. Sampling is seeded so a cache
    can be reproduced.
    """
    if not region.any():
        return []
    r = region.astype(np.uint8)
    if erode:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1,) * 2)
        er = cv2.erode(r, k)
        if er.any():
            r = er
    ys, xs = np.nonzero(r)
    if len(xs) <= n:
        return list(zip(xs.tolist(), ys.tolist()))
    idx = rng.choice(len(xs), n, replace=False)
    return [(int(xs[i]), int(ys[i])) for i in idx]


def depth_prompts(depth, spread, inverse, boxes, rng, n_pos=3, n_neg=6,
                  min_sep=0.05):
    """Point prompts for each box, derived from the depth map.

    SAM takes negative points natively, and that is the mechanism this uses: the
    floor is the far mode of an overhead crop, so depth can mark it directly and
    say "not this", which is exactly the failure being targeted. RGB cannot do
    the same job, because a Holstein's black and white patches look like a
    stronger boundary than the animal's own outline.

    For each box:
      positive  points at the box centre's own depth  -> the animal
      negative  points lying FURTHER than that        -> the floor under it
      negative  points inside the OTHER box but outside this one -> the other
                animal, which is what stops one mask swallowing both. This part
                needs no depth and is applied regardless.

    Both depth classes hang off the box centre, because that is the only pixel
    the detector guarantees is an animal. An earlier version split the box's
    depth histogram with Otsu and called the near half cattle; that is wrong for
    this camera, whose angled ceiling mount puts railings, feed barriers and
    pipework NEARER the lens than a cow's back, so the near half is as likely to
    be hardware as animal and the positive points would land on a gate.

    Nothing is asserted about what lies nearer than the centre depth: it is
    neither prompted for nor against, and SAM is left to decide. Only the far
    side is claimed, because that is the direction in which this camera can only
    be seeing ground.

    A box whose far side is empty gets no depth-derived negatives at all — it
    contains no visible floor, and `min_sep` is what keeps a box that is
    entirely animal from having negative points planted on the cow. Returns the
    prompts and how many boxes had visible floor, so the caller can report how
    often depth actually contributed.
    """
    out, used = [], 0
    h, w = depth.shape
    for k, b in enumerate(boxes):
        x1, y1, x2, y2 = (int(max(0, b[0])), int(max(0, b[1])),
                          int(min(w, b[2])), int(min(h, b[3])))
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        pts = [(cx, cy)]
        lab = [1]
        if x2 > x1 + 4 and y2 > y1 + 4:
            sub = depth[y1:y2, x1:x2]
            r = max(3, int(0.10 * min(x2 - x1, y2 - y1)))
            disc = depth[max(0, cy - r):min(h, cy + r + 1),
                         max(0, cx - r):min(w, cx + r + 1)]
            ref = float(np.median(disc)) if disc.size else float(depth[cy, cx])
            # Larger means further, whichever way the checkpoint reports depth.
            f = -sub if inverse else sub
            f_ref = -ref if inverse else ref
            near = np.abs(sub - ref) <= (min_sep * spread)   # at cattle depth
            far = f > (f_ref + min_sep * spread)             # beyond it: ground
            if far.any():
                used += 1
                for (px, py) in _sample(far, n_neg, rng):
                    pts.append((px + x1, py + y1)); lab.append(0)
            for (px, py) in _sample(near, n_pos, rng):
                pts.append((px + x1, py + y1)); lab.append(1)

        # The other animal, as negative points. Pure geometry, always applied.
        other = boxes[1 - k] if len(boxes) == 2 else None
        if other is not None:
            mo = np.zeros((h, w), np.uint8)
            mo[int(max(0, other[1])):int(min(h, other[3])),
               int(max(0, other[0])):int(min(w, other[2]))] = 1
            mo[y1:y2, x1:x2] = 0
            for (px, py) in _sample(mo, 3, rng, erode=9):
                pts.append((px, py)); lab.append(0)

        out.append((pts, lab))
    return out, used


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
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "reference", "ultralytics", "sam3",
                             "sam3_text"],
                    help="which SAM wrapper to use. 'reference' matches "
                         "visualize_sam_confusion exactly, so the cached masks "
                         "are the ones its panels were drawn from")
    ap.add_argument("--model-type", default="vit_b",
                    help="checkpoint variant for the reference backend")
    ap.add_argument("--weights", default=None,
                    help="overrides data.sam_weights; use it to point at sam3.pt")
    ap.add_argument("--text", default="cow",
                    help="noun phrase for --backend sam3_text. SAM 3 segments "
                         "the concept itself, so the pen floor is not a "
                         "candidate answer the way it is for a bare box")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="confidence floor for SAM 3 concept segmentation")
    ap.add_argument("--no-point", action="store_true",
                    help="box prompt only. The default adds a positive point at "
                         "the box centre, which is what stops SAM answering with "
                         "the floor; visualize_sam_confusion uses the same rule, "
                         "so the cached masks match what its panel 2 showed")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N pairs, for a quick quality check")
    ap.add_argument("--prompt-source", default="rgb",
                    choices=["rgb", "depth_points", "depth_image"],
                    help="rgb: box + centre point, the original behaviour. "
                         "depth_points: adds positive points on the near mode and "
                         "NEGATIVE points on the floor and on the other animal, "
                         "which is SAM's own mechanism for 'exclude this'. "
                         "depth_image: segment the colourised depth map instead "
                         "of the photograph, so coat pattern cannot mislead it. "
                         "The last two need precompute_depth.py")
    ap.add_argument("--cache-dir", default=None,
                    help="override data.mask_dir. Give each prompt source its own "
                         "directory so the variants can be scored against each "
                         "other instead of overwriting one another")
    ap.add_argument("--min-sep", type=float, default=0.05,
                    help="least depth separation, as a fraction of the crop's "
                         "spread, before a box is deemed to contain visible floor")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.weights:
        cfg["data"]["sam_weights"] = args.weights
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

    seg = build_segmenter(cfg, args.backend, args.model_type, args.text,
                          args.conf)
    print(f"[mask] prompt source: {args.prompt_source}")
    done = skipped = failed = no_depth = 0
    n_split = n_boxes = 0
    rng = np.random.default_rng(int(cfg["random_seed"]))
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

        prompts, image = None, bgr
        if args.prompt_source != "rgb":
            dep = load_depth(record, (h, w))
            if dep is None:
                no_depth += 1
                continue
            depth, spread, inverse = dep
            if args.prompt_source == "depth_points":
                prompts, used = depth_prompts(depth, spread, inverse, boxes,
                                              rng, min_sep=args.min_sep)
                n_split += used
                n_boxes += len(boxes)
            else:
                image = depth_as_image(depth, inverse)

        try:
            kw = {"use_point": not args.no_point, "prompts": prompts}
            if args.backend == "sam3_text":
                kw = {"path": record["image_path"]}
            masks = seg(image, boxes, **kw)
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
    if n_boxes:
        print(f"[mask] depth found visible floor in {n_split}/{n_boxes} boxes "
              f"({n_split / n_boxes:.0%}) at min_sep={args.min_sep}")
        print("[mask] the rest got no depth-derived negative points, so for them "
              "this run differs from 'rgb' only by the other-animal negatives")
    if areas:
        a = np.array(areas)
        print(f"[mask] mask area / box area: median {np.median(a):.2f}  "
              f"p10 {np.percentile(a, 10):.2f}  p90 {np.percentile(a, 90):.2f}")
        print("[mask] a cow fills roughly 0.4-0.8 of its axis-aligned box; a median "
              "far outside that means the prompts or the weights are wrong")
    print("[mask] set data.mask_channels: true to feed these to the model")


if __name__ == "__main__":
    main()
