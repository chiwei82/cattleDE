
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contactTest.sam3 import Sam3
from contactTest.sam_contact_region import depth_stats, load_depth
from contactTest.src.data import load_records, relative_boxes, split_records
from contactTest.src.utils import load_config
from contactTest.sam_contact_region import dilated_band

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))
C_I, C_J = (214, 120, 42), (52, 104, 235)



def _heat(img_rgb, m):
    heat = cv2.cvtColor(cv2.applyColorMap((np.clip(m, 0, 1) * 255).astype(np.uint8),
                                          cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    w = np.clip(m, 0, 1)[..., None] * 0.85
    return (img_rgb * (1 - w) + heat * w).astype(np.uint8)

def _mask_tile(img_rgb, mi, mj, points=None):
    out = img_rgb.astype(np.float32).copy()
    for m, c in ((mi, C_I), (mj, C_J)):
        sel = m > 0
        out[sel] = out[sel] * 0.45 + np.asarray(c, np.float32) * 0.55
    out = out.astype(np.uint8)
    for p, c in zip(points or [], (C_I, C_J)):
        px, py = int(round(p[0])), int(round(p[1]))
        cv2.drawMarker(out, (px, py), (0, 0, 0), cv2.MARKER_CROSS, 15, 4,
                       line_type=cv2.LINE_AA)
        cv2.drawMarker(out, (px, py), c, cv2.MARKER_CROSS, 13, 2,
                       line_type=cv2.LINE_AA)
    return out

def _map_tile(img_rgb, m, band=None, cut=None):
    tile = _heat(img_rgb, m)
    if cut is not None and cut.any():
        cnt, _ = cv2.findContours(cut.astype(np.uint8), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(tile, cnt, -1, (190, 130, 240), 1)
    if band is not None and band.any():
        cnt, _ = cv2.findContours(band.astype(np.uint8), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(tile, cnt, -1, (120, 255, 120), 1)
    peak_src = np.where(band, m, -1) if band is not None and band.any() else m
    py, px = np.unravel_index(int(peak_src.argmax()), m.shape)
    cv2.circle(tile, (px, py), 9, (0, 0, 0), 3, lineType=cv2.LINE_AA)
    cv2.circle(tile, (px, py), 9, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    return tile


def panel(rgb, boxes, mi, mj, strict, band, cut=None, points=None):
    tiles = [rgb.copy()]
    for b, c in zip(boxes, (C_I, C_J)):
        cv2.rectangle(tiles[0], (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                      c, 2)
    pts = points if points is not None else [
        ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    tiles.append(_mask_tile(rgb, mi, mj, pts))
    tiles.append(_map_tile(rgb, strict, band, cut))
    gap = np.full((rgb.shape[0], 6, 3), 250, np.uint8)
    return np.hstack([t for pr in zip(tiles, [gap] * len(tiles))
                      for t in pr][:-1])


def uncertainty(pi, pj):
    return (4.0 * pi * (1.0 - pi)) * (4.0 * pj * (1.0 - pj))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(CONTACT_ROOT, "config.yaml"))
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--text", default="cow")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--dilate-px", type=int, default=22)
    ap.add_argument("--depth-tol", type=float, default=None)
    ap.add_argument("--depth-gate", default="pair")
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.conf is None:
        args.conf = float(cfg["data"].get("sam3_conf", 0.6))
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

    try:
        seg = Sam3(args.weights, args.text, args.conf)
    except Exception as err:
        raise SystemExit(f"{err})")

    out_dir = os.path.join(CONTACT_ROOT, "log", "sam3_confusion", args.split)
    os.makedirs(out_dir, exist_ok=True)

    report, failed, no_depth = [], 0, 0

    for i, record in enumerate(records):
        bgr = cv2.imread(record["image_path"])
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        boxes = relative_boxes(record, h, w)

        try:
            got = seg.assign_to_boxes(bgr, boxes, binary=False)
            whole_masks = list(got) if got is not None else None
        except Exception as err:
            print(f"[sam3] failed on {record['rel_image']}: {err}")
            failed += 1
            continue
        if whole_masks is None or len(whole_masks) < 2:
            failed += 1
            continue

        mi = (whole_masks[0] > 0.5).astype(np.uint8)
        mj = (whole_masks[1] > 0.5).astype(np.uint8)
        if mi.sum() == 0 or mj.sum() == 0:
            failed += 1
            continue

        band = dilated_band(mi, mj, args.dilate_px)
        cut = None
        if args.depth_tol is not None:
            dep = load_depth(record, (h, w))
            if dep is None:
                no_depth += 1
            else:
                st = depth_stats(mi, mj, dep[0], dep[1], dep[2], boxes)
                keep = None
                for gname in args.depth_gate.split(","):
                    s_map = st.get(gname.strip())
                    if s_map is None:
                        continue
                    k_ = s_map <= args.depth_tol
                    keep = k_ if keep is None else (keep & k_)
                if keep is not None:
                    cut, band = band & ~keep, band & keep

        strict = uncertainty(whole_masks[0], whole_masks[1])

        report.append({
            "rel_image": record["rel_image"],
            "annotation": {-1: "unlabelled", 0: "no_interaction",
                           1: "interaction"}[record["label"]],
            "mi_px": int(mi.sum()), "mj_px": int(mj.sum()),
            "overlap_px": int((mi & mj).sum()),
            "band_px": int(band.sum()),
            "band_frac": round(float(band.sum()) / (h * w), 4),
            "cut_px": int(cut.sum()) if cut is not None else 0,
        })

        if not args.no_images:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            row = panel(rgb, boxes, mi, mj, strict, band, cut)
            name = f"{i:03d}_{report[-1]['annotation'].replace(' ', '-')}_" \
                   f"{os.path.basename(record['rel_image'])}"
            cv2.imwrite(os.path.join(out_dir, name),
                        cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

    if not report:
        raise SystemExit("nothing processed; every pair failed")

    print(f"\n[sam3] {len(report)} pairs, {failed} failed"
          + (f", {no_depth} without a cached depth map" if no_depth else ""))
    print(f"[sam3] concept prompt, text={args.text!r}, conf={args.conf}")

    b = np.array([r["band_frac"] for r in report], float)
    print(f"\n[sam3] green band: median {np.median(b):.1%} of the crop, "
          f"empty on {np.mean(b == 0):.0%} of pairs")
    if args.depth_tol is not None:
        c = np.array([r["cut_px"] for r in report], float)
        tot = c + np.array([r["band_px"] for r in report], float)
        print(f"[sam3] the depth gate removed {np.sum(c) / max(np.sum(tot), 1):.0%} "
              "of the band overall")

    with open(os.path.join(out_dir, "sam3_report.csv"), "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        wtr.writeheader()
        wtr.writerows(report)
    with open(os.path.join(out_dir, "sam3_summary.json"), "w") as f:
        json.dump({"split": args.split, "n": len(report),
                   "text": args.text, "conf": args.conf,
                   "dilate_px": args.dilate_px,
                   }, f, indent=2)

    print(f"\n[sam3] wrote {out_dir}")
    print("[sam3] green outline = the contact band; "
          + ("purple = what the depth gate removed"
             if args.depth_tol is not None else "no depth gate applied"))


if __name__ == "__main__":
    main()
