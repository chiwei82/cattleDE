"""Show each crop next to its cached Depth Anything V2 map.

Usage (from the repository root):

    python -m contactTest.visualize_depth --split train --limit 40
    python -m contactTest.visualize_depth --split train --annotated-only
    python -m contactTest.visualize_depth --split train --raw

Writes to contactTest/log/depth_vis/<split>/ only.

The map is read from the cache written by precompute_depth.py rather than
recomputed, so what is shown is exactly the array the gates threshold. It is
also normalised with the SAME stored 2nd/98th percentiles the gates use, which
is what makes the picture readable as a tolerance: a colour step of a tenth of
the bar is a depth difference of 0.10, the number passed as --depth-tol.

The banner reports the separation between the two animals' body depth and
everything else in the crop, which is the quantity that decides whether the
'body' gate can remove the floor at all. If that number is small the floor and
the cattle are not distinguishable by depth in that crop, and no tolerance will
separate them.
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contactTest.sam_contact_region import load_depth, load_masks
from contactTest.src.data import load_records, split_records
from contactTest.src.utils import load_config

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))


def colourise(depth, spread, p2, inverse, raw):
    """Depth to an RGB image, normalised the way the gates normalise it.

    The relative checkpoints predict INVERSE depth, so a larger value means
    nearer; the metric ones predict metres, where larger means further. The
    cache records which, and the two are flipped to agree here so that bright
    always means near, whichever model produced the map.
    """
    if raw:
        lo, hi = float(depth.min()), float(depth.max())
        norm = (depth - lo) / max(hi - lo, 1e-6)
    else:
        norm = (depth - p2) / spread
    norm = np.clip(norm, 0.0, 1.0)
    if not inverse:
        norm = 1.0 - norm
    u8 = (norm * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_TURBO),
                        cv2.COLOR_BGR2RGB)


def colour_bar(h, w):
    """A vertical near-to-far key down the side of the depth panel.

    One orientation serves both model types: colourise() has already flipped the
    metric case, so a high value always means near by the time it is drawn.
    """
    ramp = np.linspace(1.0, 0.0, h).astype(np.float32)
    bar = cv2.applyColorMap((ramp * 255).astype(np.uint8)[:, None].repeat(w, 1),
                            cv2.COLORMAP_TURBO)
    bar = cv2.cvtColor(bar, cv2.COLOR_BGR2RGB)
    for text, y in (("near", 14), ("far", h - 6)):
        cv2.putText(bar, text, (3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(bar, text, (3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def banner(width, lines, height=40):
    """Two lines of caption, shrunk to fit rather than run off a narrow panel.

    Portrait crops give a panel roughly half the width of a landscape one, so a
    fixed font size silently truncates the text that says whether the gate can
    work at all.
    """
    bar = np.full((height, width, 3), 245, np.uint8)
    longest = max((len(t) for t, _ in lines), default=1)
    scale = float(np.clip((width - 16) / (longest * 19.0), 0.26, 0.42))
    for i, (text, col) in enumerate(lines):
        cv2.putText(bar, text, (8, 16 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    col, 1, cv2.LINE_AA)
    return bar


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(CONTACT_ROOT, "config.yaml"))
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--annotated-only", action="store_true",
                    help="only crops that appear in log/annotate/<split>/contact_gt.csv")
    ap.add_argument("--raw", action="store_true",
                    help="stretch each map over its own min/max instead of the "
                         "stored percentiles. Prettier, but the colours no longer "
                         "correspond to a --depth-tol")
    ap.add_argument("--max-side", type=int, default=520,
                    help="longest side of each panel in the output")
    args = ap.parse_args()

    cfg = load_config(args.config)
    records = split_records(load_records(cfg, require_label=False))[args.split]

    if args.annotated_only:
        import csv
        gt_path = os.path.join(CONTACT_ROOT, "log", "annotate", args.split,
                               "contact_gt.csv")
        if not os.path.exists(gt_path):
            raise SystemExit(f"no ground truth at {gt_path}")
        want = {r["rel_image"] for r in csv.DictReader(open(gt_path))}
        records = [r for r in records if r["rel_image"] in want]

    rng = np.random.default_rng(int(cfg["random_seed"]))
    if args.limit and len(records) > args.limit:
        records = [records[i] for i in
                   rng.choice(len(records), args.limit, replace=False)]
    if not records:
        raise SystemExit(f"no rows in split '{args.split}'")

    out_dir = os.path.join(CONTACT_ROOT, "log", "depth_vis", args.split)
    os.makedirs(out_dir, exist_ok=True)

    made = no_depth = 0
    seps = []
    for i, record in enumerate(records):
        bgr = cv2.imread(record["image_path"])
        if bgr is None:
            continue
        dep = load_depth(record, bgr.shape[:2])
        if dep is None:
            no_depth += 1
            continue
        depth, spread = dep
        data = np.load(record["depth_path"])
        p2 = float(data["p2"])
        inverse = bool(int(data["inverse"])) if "inverse" in data else True

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dcol = colourise(depth, spread, p2, inverse, args.raw)

        h, w = rgb.shape[:2]
        if max(h, w) > args.max_side:
            s = args.max_side / max(h, w)
            size = (int(round(w * s)), int(round(h * s)))
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
            dcol = cv2.resize(dcol, size, interpolation=cv2.INTER_NEAREST)
        h, w = rgb.shape[:2]

        # How far the animals' body depth sits from the rest of the crop. This
        # is the whole basis of the 'body' gate, so it is worth seeing per crop
        # rather than assuming it holds.
        sep = None
        masks = load_masks(record, bgr.shape[:2])
        if masks is not None and (masks[0].any() or masks[1].any()):
            on = (masks[0] > 0) | (masks[1] > 0)
            if on.any() and (~on).any():
                sep = abs(float(np.median(depth[on]))
                          - float(np.median(depth[~on]))) / spread
                seps.append(sep)

        panel = np.hstack([rgb, np.full((h, 6, 3), 250, np.uint8), dcol,
                           colour_bar(h, 34)])
        lines = [(f"{os.path.basename(record['rel_image'])}   crop | DAv2"
                  + ("  [raw]" if args.raw else "  [p2-p98]"), (60, 60, 60))]
        lines.append(
            (f"body vs rest: {sep:.2f} of spread"
             + ("  <- too close to separate" if sep < 0.05
                else "  -> floor is separable"),
             (170, 40, 30) if sep < 0.05 else (25, 110, 40))
            if sep is not None else
            ("no cached mask: separation unknown", (140, 140, 140)))
        panel = np.vstack([banner(panel.shape[1], lines), panel])

        cv2.imwrite(os.path.join(out_dir, f"{i:03d}_"
                                 f"{os.path.basename(record['rel_image'])}"),
                    cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        made += 1

    print(f"\n[depth-vis] wrote {made} images to {out_dir}")
    if no_depth:
        print(f"[depth-vis] {no_depth} crops have no cached map; run "
              "precompute_depth.py to cover them")
    if seps:
        s = np.array(seps)
        print(f"[depth-vis] body-vs-rest separation over {len(s)} crops: "
              f"median {np.median(s):.2f}, 10th pct {np.percentile(s, 10):.2f} "
              "of the depth spread")
        print("[depth-vis] a 'body' tolerance has to sit below that separation to "
              "remove the floor")
        print(f"[depth-vis] it is under 0.05 on {np.mean(s < 0.05):.0%} of crops, "
              "where depth cannot tell the cattle from the ground at all")
    print("[depth-vis] bright = near, dark = far")


if __name__ == "__main__":
    main()
