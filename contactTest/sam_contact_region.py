"""Derive candidate contact pixels from two SAM instance masks, four ways.

Usage (from the repository root):

    python -m contactTest.sam_contact_region --limit 16
    python -m contactTest.sam_contact_region --split val --limit 40 --no-images
    python -m contactTest.sam_contact_region --limit 16 --touch-px 12

Writes to contactTest/log/sam_contact/<split>/ only.

The masks are the useful part of SAM here; its per-pixel confidence turned out
not to be (fragmented uncertainty maps, and a crop-mode "mutual claim" that
collapses once the prompts are correct). What remains is a geometry question:
given two good instance masks, which pixels are candidates for contact? The
answer is not unique, so all four readings are computed and drawn side by side.

    overlap   mask_i AND mask_j
              Where the projections literally coincide. Unambiguous but sparse,
              and in an overhead view it means occlusion as often as contact.

    gap       dist_to_i + dist_to_j <= touch_px
              The strip between the two surfaces, thresholded on the true local
              separation rather than on an arbitrary dilation radius. Includes
              the floor visible in the gap when the animals are close but apart.

    surface   the boundary pixels of each mask that lie within touch_px of the
              other animal, dilated into a strip.
              Contact happens ON a body surface, so this keeps skin rather than
              the air between. Closest to "which part of the animal is touching".

    dilated   dilate(mask_i, r) AND dilate(mask_j, r)
              The current default, kept for comparison. r has no physical
              meaning, which is exactly why the others are worth measuring.

`surface` is the one to look at first if the goal is per-pixel contact on the
animals; `gap` if the goal is the interface region between them.
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contactTest.src.data import load_records, relative_boxes, split_records
from contactTest.src.utils import load_config

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))
C_I, C_J = (214, 120, 42), (52, 104, 235)          # RGB, cow i / cow j
C_HIT = (60, 235, 90)


def load_masks(record, shape):
    """Cached SAM masks for a pair, or None."""
    if not record.get("mask_path"):
        return None
    data = np.load(record["mask_path"])
    mi, mj = data["mi"].astype(np.uint8), data["mj"].astype(np.uint8)
    if mi.shape != shape:
        mi = cv2.resize(mi, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        mj = cv2.resize(mj, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mi, mj


def distance_to(mask):
    """Distance from every pixel to the nearest pixel inside `mask`."""
    return cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 3)


def boundary(mask):
    """One-pixel outline of a binary mask."""
    er = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    return (mask.astype(bool) & ~er.astype(bool)).astype(np.uint8)


def contact_readings(mi, mj, touch_px, dilate_px, strip_px):
    """Four candidate definitions of the contact region."""
    di, dj = distance_to(mi), distance_to(mj)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2)
    ks = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * strip_px + 1,) * 2)

    # Surface pixels of one animal that are within touch_px of the other.
    touch_i = (boundary(mi) > 0) & (dj <= touch_px)
    touch_j = (boundary(mj) > 0) & (di <= touch_px)
    surface = cv2.dilate((touch_i | touch_j).astype(np.uint8), ks) > 0

    return {
        "overlap": (mi.astype(bool) & mj.astype(bool)),
        "gap": ((di + dj) <= touch_px),
        "surface": surface,
        "dilated": (cv2.dilate(mi, k) > 0) & (cv2.dilate(mj, k) > 0),
    }


def stats(name, region, mi, mj):
    area = int(region.sum())
    n_comp = max(cv2.connectedComponents(region.astype(np.uint8))[0] - 1, 0)
    on_animal = float((region & (mi.astype(bool) | mj.astype(bool))).sum()) / area \
        if area else 0.0
    return {f"{name}_px": area, f"{name}_components": n_comp,
            f"{name}_on_animal": round(on_animal, 3),
            f"{name}_nonempty": int(area > 0)}


def panel(rgb, mi, mj, readings, order):
    h, w = rgb.shape[:2]
    base = rgb.astype(np.float32).copy()
    for m, c in ((mi, C_I), (mj, C_J)):
        sel = m > 0
        base[sel] = base[sel] * 0.55 + np.asarray(c, np.float32) * 0.45
    tiles = [base.astype(np.uint8)]

    for name in order:
        t = (rgb.astype(np.float32) * 0.45).astype(np.uint8)
        for m, c in ((mi, C_I), (mj, C_J)):
            sel = m > 0
            t[sel] = (t[sel] * 0.6 + np.asarray(c, np.float32) * 0.4).astype(np.uint8)
        t[readings[name]] = C_HIT
        cv2.putText(t, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(t, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)

    gap = np.full((h, 6, 3), 250, np.uint8)
    return np.hstack([x for pair in zip(tiles, [gap] * len(tiles))
                      for x in pair][:-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(CONTACT_ROOT, "config.yaml"))
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--balance", action="store_true",
                    help="half interaction / half not, for viewing only")
    ap.add_argument("--touch-px", type=int, default=10,
                    help="separation at or below which two surfaces count as touching")
    ap.add_argument("--dilate-px", type=int, default=15,
                    help="radius for the 'dilated' reading, for comparison only")
    ap.add_argument("--strip-px", type=int, default=6,
                    help="half-width of the 'surface' strip")
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rows = split_records(load_records(cfg, require_label=False))[args.split]
    rng = np.random.default_rng(int(cfg["random_seed"]))

    def draw(pool, k):
        k = min(len(pool), k)
        return [pool[i] for i in rng.choice(len(pool), k, replace=False)] if k else []

    if args.balance:
        half = args.limit // 2
        records = (draw([r for r in rows if r["label"] == 1], half) +
                   draw([r for r in rows if r["label"] == 0], args.limit - half))
    else:
        records = draw(rows, args.limit)
    if not records:
        raise SystemExit(f"no rows in split '{args.split}'")

    out_dir = os.path.join(CONTACT_ROOT, "log", "sam_contact", args.split)
    os.makedirs(out_dir, exist_ok=True)
    order = ["overlap", "gap", "surface", "dilated"]
    report, missing = [], 0

    for i, record in enumerate(records):
        bgr = cv2.imread(record["image_path"])
        if bgr is None:
            continue
        masks = load_masks(record, bgr.shape[:2])
        if masks is None:
            missing += 1
            continue
        mi, mj = masks

        readings = contact_readings(mi, mj, args.touch_px,
                                    args.dilate_px, args.strip_px)
        row = {"rel_image": record["rel_image"],
               "annotation": {-1: "unlabelled", 0: "no_interaction",
                              1: "interaction"}[record["label"]],
               "crop_px": int(bgr.shape[0] * bgr.shape[1])}
        for name in order:
            row.update(stats(name, readings[name], mi, mj))
        report.append(row)

        if not args.no_images:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            name = f"{i:03d}_{row['annotation'].replace(' ', '-')}_" \
                   f"{os.path.basename(record['rel_image'])}"
            cv2.imwrite(os.path.join(out_dir, name),
                        cv2.cvtColor(panel(rgb, mi, mj, readings, order),
                                     cv2.COLOR_RGB2BGR))

    if missing:
        raise SystemExit(
            f"{missing} pairs have no cached mask. Run precompute_masks.py first:\n"
            "  python -m contactTest.precompute_masks --split " + args.split)
    if not report:
        raise SystemExit("nothing processed")

    print(f"\n[contact] {len(report)} pairs, touch_px={args.touch_px}\n")
    print("面板：mask | " + " | ".join(order))
    print("綠色 = 該定義判定的接觸候選像素\n")
    print(f"{'reading':<10}{'非空':>7}{'區域大小':>11}{'佔畫面':>9}"
          f"{'連通塊':>8}{'落在牛身上':>12}")
    summary = {}
    for name in order:
        ne = np.mean([r[f"{name}_nonempty"] for r in report])
        px = np.median([r[f"{name}_px"] for r in report])
        frac = np.median([r[f"{name}_px"] / max(r["crop_px"], 1) for r in report])
        comp = np.median([r[f"{name}_components"] for r in report])
        onan = np.median([r[f"{name}_on_animal"] for r in report])
        summary[name] = {"nonempty": float(ne), "px_median": float(px),
                         "frac_median": float(frac), "components_median": float(comp),
                         "on_animal_median": float(onan)}
        print(f"{name:<10}{ne:>6.0%}{px:>10.0f}px{frac:>9.1%}{comp:>8.1f}{onan:>11.0%}")

    print("\n『落在牛身上』= 該區域有多少比例壓在任一隻牛的 mask 上。")
    print("接觸發生在體表，所以這個比例越高，圈到的越可能是皮膚而不是牛之間的空氣。")

    with open(os.path.join(out_dir, "contact_report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(report)
    with open(os.path.join(out_dir, "contact_summary.json"), "w") as f:
        json.dump({"split": args.split, "n": len(report),
                   "touch_px": args.touch_px, "readings": summary}, f, indent=2)
    print(f"\n[contact] wrote {out_dir}")


if __name__ == "__main__":
    main()
