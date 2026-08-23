
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
C_I, C_J = (214, 120, 42), (52, 104, 235)
C_HIT = (60, 235, 90)


def load_masks(record, shape):
    if not record.get("mask_path"):
        return None
    data = np.load(record["mask_path"])
    mi, mj = data["mi"].astype(np.uint8), data["mj"].astype(np.uint8)
    if mi.shape != shape:
        mi = cv2.resize(mi, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        mj = cv2.resize(mj, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mi, mj


def load_depth(record, shape):
    if not record.get("depth_path"):
        return None
    data = np.load(record["depth_path"])
    depth = data["depth"].astype(np.float32)
    if depth.shape != shape:
        depth = cv2.resize(depth, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    spread = float(data["p98"]) - float(data["p2"])
    inverse = bool(int(data["inverse"])) if "inverse" in data else True
    return depth, max(spread, 1e-6), inverse


def farness(depth, inverse):
    return -depth if inverse else depth


def body_reference(depth, boxes, inverse, disc_frac=0.10):
    h, w = depth.shape
    refs = []
    for b in boxes:
        x1, y1, x2, y2 = (int(max(0, b[0])), int(max(0, b[1])),
                          int(min(w, b[2])), int(min(h, b[3])))
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        r = max(3, int(disc_frac * min(x2 - x1, y2 - y1)))
        disc = depth[max(0, cy - r):min(h, cy + r + 1),
                     max(0, cx - r):min(w, cx + r + 1)]
        if disc.size:
            refs.append(float(np.median(disc)))
    return refs or None


def ground_split(depth, inverse, boxes, spread, margin=0.05):
    refs = body_reference(depth, boxes, inverse)
    if refs is None:
        return None
    f = farness(depth, inverse)
    ref_far = max(farness(np.float32(r), inverse) for r in refs)
    floor = f > (ref_far + margin * spread)
    if not floor.any():
        return floor, refs, 0.0, 0.0
    sep = float(np.median(f[floor]) - ref_far)
    return floor, refs, sep, float(floor.mean())


def nearest_value(mask, value):
    if not mask.any():
        return None
    _, lab = cv2.distanceTransformWithLabels(
        (mask == 0).astype(np.uint8), cv2.DIST_L2, 5,
        labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(mask)
    lut = np.zeros(int(lab.max()) + 1, np.float32)
    lut[lab[ys, xs]] = value[ys, xs]
    return lut[lab]


DEPTH_STATS = ["body", "step", "pair"]


def depth_stats(mi, mj, depth, spread, inverse=True, boxes=None):
    out = {}

    refs = body_reference(depth, boxes, inverse) if boxes is not None else None
    out["body"] = (np.minimum.reduce([np.abs(depth - r) for r in refs]) / spread
                   if refs else None)

    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    out["step"] = np.sqrt(gx * gx + gy * gy) / (8.0 * spread)

    out["pair"] = depth_disagreement(mi, mj, depth, spread)
    return out


def depth_disagreement(mi, mj, depth, spread):
    mi_only = (mi > 0) & ~(mj > 0)
    mj_only = (mj > 0) & ~(mi > 0)
    di = nearest_value(mi_only.astype(np.uint8), depth)
    dj = nearest_value(mj_only.astype(np.uint8), depth)
    if di is None or dj is None:
        return None
    return np.abs(di - dj) / spread


def distance_to(mask):
    return cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 3)


def boundary(mask):
    er = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    return (mask.astype(bool) & ~er.astype(bool)).astype(np.uint8)


def dilated_band(mi, mj, dilate_px):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2)
    return (cv2.dilate(mi, k) > 0) & (cv2.dilate(mj, k) > 0)


def contact_readings(mi, mj, touch_px, dilate_px, strip_px,
                     depth=None, spread=None, gates=None, inverse=True,
                     boxes=None):
    di, dj = distance_to(mi), distance_to(mj)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2)
    ks = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * strip_px + 1,) * 2)

    touch_i = (boundary(mi) > 0) & (dj <= touch_px)
    touch_j = (boundary(mj) > 0) & (di <= touch_px)
    surface = cv2.dilate((touch_i | touch_j).astype(np.uint8), ks) > 0

    out = {
        "overlap": (mi.astype(bool) & mj.astype(bool)),
        "gap": ((di + dj) <= touch_px),
        "surface": surface,
        "dilated": dilated_band(mi, mj, dilate_px),
    }

    if depth is not None and gates:
        stats_ = depth_stats(mi, mj, depth, spread, inverse, boxes)
        keep = None
        for name, tol in gates.items():
            s = stats_.get(name)
            if s is None:
                continue
            keep = (s <= tol) if keep is None else (keep & (s <= tol))
        if keep is not None:
            out = {n: (r & keep) for n, r in out.items()}
    return out


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
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--touch-px", type=int, default=10)
    ap.add_argument("--dilate-px", type=int, default=15)
    ap.add_argument("--strip-px", type=int, default=6)
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
    print("panels: masks | " + " | ".join(order))
    print(f"{'reading':<10}{'nonempty':>10}{'area':>11}{'of crop':>9}"
          f"{'blobs':>7}{'on animal':>11}")
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
        print(f"{name:<10}{ne:>9.0%}{px:>9.0f}px{frac:>9.1%}{comp:>7.1f}{onan:>10.0%}")


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
