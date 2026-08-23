
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contactTest.sam_contact_region import body_reference, farness, load_depth
from contactTest.src.data import load_records, relative_boxes, split_records
from contactTest.src.utils import load_config

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))
C_I, C_J = (214, 120, 42), (52, 104, 235)
C_FLOOR = (240, 90, 70)


def load_from(cache_root, rel_image, shape):
    path = os.path.join(cache_root, os.path.splitext(rel_image)[0] + ".npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    mi, mj = data["mi"].astype(np.uint8), data["mj"].astype(np.uint8)
    if mi.shape != shape:
        mi = cv2.resize(mi, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        mj = cv2.resize(mj, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mi, mj


def raggedness(mask):
    area = float(mask.sum())
    if area < 20:
        return float("nan")
    cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_NONE)
    per = sum(cv2.arcLength(c, True) for c in cnt)
    return per / max(2.0 * np.sqrt(np.pi * area), 1e-6)


def measure(masks, boxes, depth, spread, inverse, margin):
    refs = body_reference(depth, boxes, inverse)
    if refs is None:
        return None
    f = farness(depth, inverse)
    out = []
    for k, (m, b) in enumerate(zip(masks, boxes)):
        area = float(m.sum())
        if area < 1:
            out.append({"floor": float("nan"), "fill": 0.0,
                        "ragged": float("nan")})
            continue
        ref = refs[min(k, len(refs) - 1)]
        floor = f > (farness(np.float32(ref), inverse) + margin * spread)
        box_area = max((b[2] - b[0]) * (b[3] - b[1]), 1)
        out.append({"floor": float((floor & (m > 0)).sum()) / area,
                    "fill": area / box_area,
                    "ragged": raggedness(m > 0)})
    return out, refs, f


def tile(rgb, masks, floor_px, title):
    t = (rgb.astype(np.float32) * 0.55).astype(np.uint8)
    for m, c in zip(masks, (C_I, C_J)):
        sel = m > 0
        t[sel] = (t[sel] * 0.55 + np.asarray(c, np.float32) * 0.45).astype(np.uint8)
    bad = floor_px & ((masks[0] > 0) | (masks[1] > 0))
    t[bad] = C_FLOOR
    for text, col in ((title, (255, 255, 255))):
        cv2.putText(t, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(t, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    col, 1, cv2.LINE_AA)
    return t


def banner(width, lines, height=40):
    bar = np.full((height, width, 3), 245, np.uint8)
    longest = max((len(t) for t, _ in lines), default=1)
    scale = float(np.clip((width - 16) / (longest * 19.0), 0.26, 0.44))
    for i, (text, col) in enumerate(lines):
        cv2.putText(bar, text, (8, 16 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    col, 1, cv2.LINE_AA)
    return bar


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(CONTACT_ROOT, "config.yaml"))
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--a", default="log/mask_cache")
    ap.add_argument("--b", required=True)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root_a = os.path.join(CONTACT_ROOT, args.a)
    root_b = os.path.join(CONTACT_ROOT, args.b)
    for r in (root_a, root_b):
        if not os.path.isdir(r):
            raise SystemExit(f"no such cache directory: {r}")

    records = split_records(load_records(cfg, require_label=False))[args.split]
    rng = np.random.default_rng(int(cfg["random_seed"]))

    out_dir = os.path.join(CONTACT_ROOT, "log", "mask_compare", args.split)
    os.makedirs(out_dir, exist_ok=True)

    rows, made, skipped = [], 0, 0
    for record in records:
        if made >= args.limit and rows:
            break
        bgr = cv2.imread(record["image_path"])
        if bgr is None:
            continue
        shape = bgr.shape[:2]
        ma = load_from(root_a, record["rel_image"], shape)
        mb = load_from(root_b, record["rel_image"], shape)
        dep = load_depth(record, shape)
        if ma is None or mb is None or dep is None:
            skipped += 1
            continue
        depth, spread, inverse = dep
        boxes = relative_boxes(record, *shape)

        res_a = measure(ma, boxes, depth, spread, inverse, args.margin)
        res_b = measure(mb, boxes, depth, spread, inverse, args.margin)
        if res_a is None or res_b is None:
            skipped += 1
            continue
        (stat_a, refs, f) = res_a
        (stat_b, _, _) = res_b

        ua = (ma[0] > 0) | (ma[1] > 0)
        ub = (mb[0] > 0) | (mb[1] > 0)
        iou = float((ua & ub).sum()) / max(float((ua | ub).sum()), 1.0)

        row = {"rel_image": record["rel_image"], "iou": iou}
        for tag, st in (("a", stat_a), ("b", stat_b)):
            for key in ("floor", "fill", "ragged"):
                row[f"{tag}_{key}"] = float(np.nanmean([s[key] for s in st]))
        rows.append(row)

        if not args.no_images and made < args.limit:
            ref_far = max(farness(np.float32(r), inverse) for r in refs)
            floor_px = f > (ref_far + args.margin * spread)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            left = tile(rgb, ma, floor_px, os.path.basename(args.a))
            right = tile(rgb, mb, floor_px, os.path.basename(args.b))
            gapcol = np.full((shape[0], 6, 3), 250, np.uint8)
            img = np.hstack([rgb, gapcol, left, gapcol, right])
            drop = row["a_floor"] - row["b_floor"]
            img = np.vstack([banner(img.shape[1], [
                (f"{os.path.basename(record['rel_image'])}   "
                 f"red = mask pixels at FLOOR depth   IoU(a,b) {iou:.2f}",
                 (60, 60, 60)),
                (f"floor share  a {row['a_floor']:.1%} -> b {row['b_floor']:.1%}"
                 f"   ({'-' if drop > 0 else '+'}{abs(drop):.1%})"
                 f"    fill {row['a_fill']:.2f} -> {row['b_fill']:.2f}"
                 f"    ragged {row['a_ragged']:.1f} -> {row['b_ragged']:.1f}",
                 (25, 110, 40) if drop > 0.01 else
                 ((170, 40, 30) if drop < -0.01 else (110, 110, 110)))]),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR)[:, :, ::-1]])
            name = f"{int(round((1 - drop) * 1000)):04d}_" \
                   f"{os.path.basename(record['rel_image'])}"
            cv2.imwrite(os.path.join(out_dir, name),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            made += 1

    if not rows:
        raise SystemExit("nothing comparable: both caches must cover the same "
                         "crops, and precompute_depth must have run")

    def col(name):
        return np.array([r[name] for r in rows], float)

    print(f"\n[compare] {len(rows)} pairs in both caches, {skipped} skipped")
    print(f"[compare] a = {args.a}")
    print(f"[compare] b = {args.b}\n")
    print(f"{'metric':<14}{'a':>10}{'b':>10}{'change':>12}")
    for key, label, better in (("floor", "floor share", "lower"),
                               ("fill", "box fill", "0.4-0.8"),
                               ("ragged", "raggedness", "lower")):
        a, b = np.nanmean(col(f"a_{key}")), np.nanmean(col(f"b_{key}"))
        fmt = ".1%" if key == "floor" else ".2f"
        print(f"{label:<14}{a:>10{fmt}}{b:>10{fmt}}"
              f"{b - a:>+12{fmt}}   ({better} is better)")
    print(f"\n[compare] mean IoU between the two mask sets: {np.nanmean(col('iou')):.2f}")

    d = col("a_floor") - col("b_floor")
    print(f"[compare] floor share fell on {np.mean(d > 0.01):.0%} of pairs, "
          f"rose on {np.mean(d < -0.01):.0%}, unchanged on {np.mean(np.abs(d) <= 0.01):.0%}")
    if not args.no_images:
        print(f"[compare] wrote {made} images to {out_dir} (worst improvement first)")


if __name__ == "__main__":
    main()
