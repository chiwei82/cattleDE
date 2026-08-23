
import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contactTest.sam_contact_region import load_masks
from contactTest.score_contact import read_gt
from contactTest.src.data import load_records, records_for, relative_boxes
from contactTest.src.utils import load_config

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))

RADII = [4, 8, 12, 16, 22, 30, 40, 55]
SCALES = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]

DENSE_RADII = list(range(1, 51))
DENSE_SCALES = list(np.geomspace(0.001, 1.0, 100))

COMPARE_ORDER = ["all", "train", "known_interact"]
MAPPING = {
    "all": "random",
    "train": "fixed_lighting",
    "known_interact": "interaction",
}

FAM_ORDER = ["merged box", "box intersection", "inscribed ellipse",
             "midpoint disc", "SAM 3 overlap", "SAM 3 dilated"]
SINGLE_POINT = {"merged box", "SAM 3 overlap"}
STYLE = {
    "merged box":        ("0.35",             "s"),
    "box intersection":  ((0.85, 0.42, 0.16), "o"),
    "inscribed ellipse": ((0.90, 0.62, 0.20), "^"),
    "midpoint disc":     ((0.55, 0.55, 0.55), "v"),
    "SAM 3 overlap":     ((0.20, 0.47, 0.75), "D"),
    "SAM 3 dilated":     ((0.15, 0.40, 0.70), "o"),
}


def scaled_rect(b, factor, h, w):
    cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    s = float(np.sqrt(max(factor, 0.0)))
    hw, hh = (b[2] - b[0]) / 2.0 * s, (b[3] - b[1]) / 2.0 * s
    return (max(0.0, cx - hw), max(0.0, cy - hh),
            min(float(w), cx + hw), min(float(h), cy + hh))


def rect_mask(shape, r):
    m = np.zeros(shape, bool)
    x1, y1 = int(round(r[0])), int(round(r[1]))
    x2, y2 = int(round(r[2])), int(round(r[3]))
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = True
    return m


def ellipse_mask(shape, r):
    m = np.zeros(shape, np.uint8)
    cx, cy = (r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0
    ax, ay = (r[2] - r[0]) / 2.0, (r[3] - r[1]) / 2.0
    if ax >= 1 and ay >= 1:
        cv2.ellipse(m, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, 1, -1)
    return m.astype(bool)


def disc_mask(shape, c, radius):
    m = np.zeros(shape, np.uint8)
    if radius >= 1:
        cv2.circle(m, (int(c[0]), int(c[1])), int(radius), 1, -1)
    return m.astype(bool)


def band(mi, mj, r):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
    return (cv2.dilate(mi, k) > 0) & (cv2.dilate(mj, k) > 0)


def covered(region, points):
    h, w = region.shape
    n = 0
    for (x, y) in points:
        xx = int(np.clip(x, 0, w - 1)); yy = int(np.clip(y, 0, h - 1))
        n += bool(region[yy, xx])
    return n

def sweep(split, cfg, seg, radii, scales, gt_path=None, limit=0):
    gt_path = gt_path or os.path.join(CONTACT_ROOT, "log", "annotate", split,
                                      "contact_gt.csv")
    if not os.path.exists(gt_path):
        raise SystemExit(f"no ground truth at {gt_path}")
    gt = read_gt(gt_path)
    by_rel = {r["rel_image"]: r for r in
              records_for(load_records(cfg, require_label=False), split)}

    acc, n_img = {}, 0
    no_contact_area, n_no_contact, no_mask = [], 0, 0

    for rel, ann in gt.items():
        if ann["status"] == "skip":
            continue
        rec = by_rel.get(rel)
        if rec is None:
            continue
        bgr = cv2.imread(rec["image_path"])
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        shape = (h, w)
        crop_px = float(h * w)
        b1, b2 = relative_boxes(rec, h, w)
        pts = ann["points"]

        masks = None
        if seg is not False:
            try:
                masks = (load_masks(rec, shape) if seg is None
                         else seg.assign_to_boxes(bgr, [b1, b2]))
            except Exception as err:
                print(f"[roi] SAM 3 failed on {rel}: {err}")
                masks = None
            if masks is None or len(masks) < 2:
                no_mask += 1
                masks = [np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)]
            masks = [np.asarray(m).astype(np.uint8) for m in masks]

        it = (max(b1[0], b2[0]), max(b1[1], b2[1]),
              min(b1[2], b2[2]), min(b1[3], b2[3]))
        mid = ((b1[0] + b1[2] + b2[0] + b2[2]) / 4.0,
               (b1[1] + b1[3] + b2[1] + b2[3]) / 4.0)

        regions = {("merged box", 1.0): np.ones(shape, bool)}
        for s in scales:
            r = scaled_rect(it, s, h, w)
            regions[("box intersection", float(s))] = rect_mask(shape, r)
            regions[("inscribed ellipse", float(s))] = ellipse_mask(shape, r)
        for r in radii:
            regions[("midpoint disc", float(r))] = disc_mask(shape, mid, r)
        if masks is not None:
            regions[("SAM 3 overlap", 0.0)] = (masks[0] > 0) & (masks[1] > 0)
            for r in radii:
                regions[("SAM 3 dilated", float(r))] = band(masks[0], masks[1], r)

        if not pts:
            n_no_contact += 1
            no_contact_area.append(
                {k: float(v.sum()) / crop_px for k, v in regions.items()})
            continue

        n_img += 1
        for k, reg in regions.items():
            c, t, a = acc.get(k, (0, 0, 0.0))
            acc[k] = (c + covered(reg, pts), t + len(pts),
                      a + float(reg.sum()) / crop_px)
        if limit and n_img >= limit:
            break

    if not acc:
        raise SystemExit(f"nothing scored for split '{split}'")

    rows = []
    for (fam, param), (c, t, a) in acc.items():
        a_bar = a / n_img
        rows.append({"family": fam, "param": param, "point_coverage": c / t,
                     "a_bar": a_bar,
                     "coverage_per_area": (c / t) / a_bar if a_bar > 0 else float("nan"),
                     "covered": c, "points": t, "n_images": n_img})
    rows.sort(key=lambda r: (r["family"], r["param"]))
    n_pts = rows[0]["points"]
    print(f"[roi] {split}: {n_img} images with clicks, {n_pts} clicks, "
          f"{n_no_contact} marked 'no contact' (excluded from the curve)"
          + (f", {no_mask} with no usable mask pair" if no_mask else ""))
    return rows, n_img, n_pts


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    return path


def draw_panel(ax, rows, title, n_img, n_pts, mark_r=22.0, legend=False):
    by = {}
    for r in rows:
        by.setdefault(r["family"], []).append(r)
    for v in by.values():
        v.sort(key=lambda r: r["a_bar"])

    for fam in FAM_ORDER:
        r = by.get(fam)
        if not r:
            continue
        col, mk = STYLE[fam]
        if fam in SINGLE_POINT:
            ax.plot([r[0]["a_bar"]], [r[0]["point_coverage"]], marker=mk, color=col,
                    ms=8, lw=0, label=fam, alpha=0.95)
        else:
            ax.plot([x["a_bar"] for x in r], [x["point_coverage"] for x in r],
                    color=col, lw=1.9, label=fam, alpha=0.9)

    for fam in ("SAM 3 dilated", "midpoint disc"):
        hit = [x for x in by.get(fam, []) if x["param"] == mark_r]
        if not hit:
            continue
        ax.scatter([hit[0]["a_bar"]], [hit[0]["point_coverage"]], s=120,
                   facecolors="none", edgecolors="black", linewidths=1.5, zorder=6)
        ax.annotate(f"r={int(mark_r)}", (hit[0]["a_bar"], hit[0]["point_coverage"]),
                    textcoords="offset points", xytext=(7, -12), fontsize=8)
    zero = by.get("SAM 3 overlap")
    if zero:
        ax.annotate("r=0", (zero[0]["a_bar"], zero[0]["point_coverage"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8,
                    color=STYLE["SAM 3 overlap"][0])

    ax.set_xscale("log")
    ax.set_xlabel("a-bar — mean proposed area as a share of the crop (log)",
                  fontsize=9)
    ax.set_title(f"{title}  ({n_img} images, {n_pts} clicks)", fontsize=11)
    ax.grid(alpha=0.2)
    if legend:
        ax.legend(fontsize=8.5, loc="lower right")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(CONTACT_ROOT, "config.yaml"))
    ap.add_argument("--split", default="all",
                    choices=["train", "val", "test", "all", "known_interact"])
    ap.add_argument("--all-compare", action="store_true")
    ap.add_argument("--source", default="sam3_text", choices=["sam3_text", "cache"])
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--text", default="cow")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--radii", default=",".join(str(r) for r in RADII))
    ap.add_argument("--scales", default=",".join(str(s) for s in SCALES))
    ap.add_argument("--mark-r", type=float, default=22.0)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.conf is None:
        args.conf = float(cfg["data"].get("sam3_conf", 0.6))

    seg = False if args.baselines_only else None
    if seg is not False and args.source == "sam3_text":
        from contactTest.sam3 import Sam3
        seg = Sam3(args.weights, args.text, args.conf)

    if args.all_compare:
        radii = DENSE_RADII
        scales = DENSE_SCALES
        print(f"[roi] all-compare: r = {radii[0]}..{radii[-1]} step 1 "
              f"({len(radii)} values), scale = {scales[0]:g}..{scales[-1]:g} "
              f"log-spaced ({len(scales)} values)")
        runs = {}
        for split in COMPARE_ORDER:
            rows, n_img, n_pts = sweep(split, cfg, seg, radii, scales,
                                       limit=args.limit)
            write_csv(rows, os.path.join(CONTACT_ROOT, "log", "roi_sweep",
                                         split, "roi_sweep_dense.csv"))
            runs[split] = (rows, n_img, n_pts)
        out_dir = os.path.join(CONTACT_ROOT, "log", "roi_sweep")
        for split in COMPARE_ORDER:
            print(f"[roi] wrote {os.path.join(out_dir, split, 'roi_sweep_dense.csv')}")
        if args.no_plot:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(COMPARE_ORDER),
                                 figsize=(6.0 * len(COMPARE_ORDER), 5.4),
                                 sharey=True)
        axes = np.atleast_1d(axes)
        for ax, split in zip(axes, COMPARE_ORDER):
            rows, n_img, n_pts = runs[split]
            draw_panel(ax, rows, MAPPING[split], n_img, n_pts,
                       mark_r=args.mark_r, legend=(ax is axes[-1]))
        axes[0].set_ylabel(
            "point coverage — share of clicked contact points inside the region",
            fontsize=9)
        fig.suptitle("ROI proposal: point coverage against area", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        png = os.path.join(out_dir, "roi_sweep_compare.png")
        fig.savefig(png, dpi=150, bbox_inches="tight")
        print(f"[roi] wrote {png}")
        return

    radii = [int(v) for v in args.radii.split(",") if v.strip()]
    scales = [float(v) for v in args.scales.split(",") if v.strip()]
    rows, n_img, n_pts = sweep(args.split, cfg, seg, radii, scales,
                               gt_path=args.gt, limit=args.limit)

    print(f"\n{'family':<20}{'param':>8}{'a-bar':>9}{'coverage':>9}{'cov/area':>13}")
    for fam in FAM_ORDER:
        for r in [x for x in rows if x["family"] == fam]:
            star = "  <-" if fam == "SAM 3 dilated" and r["param"] == args.mark_r else ""
            print(f"{fam:<20}{r['param']:>8.3f}{r['a_bar']:>9.4f}"
                  f"{r['point_coverage']:>9.3f}{r['coverage_per_area']:>13.1f}{star}")

    out_dir = os.path.join(CONTACT_ROOT, "log", "roi_sweep", args.split)
    print(f"\n[roi] wrote {write_csv(rows, os.path.join(out_dir, 'roi_sweep.csv'))}")
    if args.no_plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    draw_panel(ax, rows, MAPPING.get(args.split, args.split), n_img, n_pts,
               mark_r=args.mark_r, legend=True)
    ax.set_ylabel("point coverage — share of clicked contact points inside the region",
                  fontsize=9)
    fig.tight_layout()
    png = os.path.join(out_dir, "roi_sweep.png")
    fig.savefig(png, dpi=150)
    print(f"[roi] wrote {png}")


if __name__ == "__main__":
    main()
