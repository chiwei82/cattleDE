
import argparse
import collections
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contactTest.evaluate_contact import evaluate_one, render, banner
from contactTest.sam_contact_region import contact_readings, depth_stats, load_depth
from contactTest.score_contact import read_gt
from contactTest.src.data import load_records, records_for
from contactTest.src.utils import load_config

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))

def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def mask_gap(a, b):
    if not a.any() or not b.any():
        return None
    if (a.astype(bool) & b.astype(bool)).any():
        return 0.0
    d = cv2.distanceTransform((b == 0).astype(np.uint8), cv2.DIST_L2,
                              cv2.DIST_MASK_PRECISE)
    return float(d[a > 0].min())


def mask_box(m):
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max()) + 1,
            float(ys.max()) + 1)


class FrameSource:

    def __init__(self, root):
        self.root = root
        self.cap = None
        self.open_stem = None
        self.index = {}
        for dirpath, _, files in os.walk(root):
            for f in files:
                if os.path.splitext(f)[1].lower() in (".mp4", ".avi", ".mov", ".mkv"):
                    self.index.setdefault(os.path.splitext(f)[0], os.path.join(dirpath, f))

    def frames_for(self, source_video, wanted):
        stem = os.path.splitext(source_video)[0]
        path = self.index.get(stem)
        if path is None:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return
        want = set(int(w) for w in wanted)
        remaining = len(want)
        idx = 0
        while remaining > 0:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in want:
                yield idx, frame
                remaining -= 1
            idx += 1
        cap.release()

    def frames_every(self, source_video, step):
        stem = os.path.splitext(source_video)[0]
        path = self.index.get(stem)
        if path is None:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                yield idx, frame
            idx += 1
        cap.release()

    def fps(self, source_video):
        stem = os.path.splitext(source_video)[0]
        path = self.index.get(stem)
        if path is None:
            return None
        cap = cv2.VideoCapture(path)
        v = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else None
        cap.release()
        return v or None

    def get(self, source_video, frame_number):
        stem = os.path.splitext(source_video)[0]
        path = self.index.get(stem)
        if path is None:
            return None, f"video not found for {stem}"
        if self.open_stem != stem:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(path)
            self.open_stem = stem
            if not self.cap.isOpened():
                return None, f"cannot open {path}"
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ok, frame = self.cap.read()
        return (frame, None) if ok else (None, f"cannot read frame {frame_number}")


def crop_to_frame(points, merged, shape):
    ox, oy = max(0, int(merged[0])), max(0, int(merged[1]))
    h, w = shape
    return [(int(np.clip(x + ox, 0, w - 1)), int(np.clip(y + oy, 0, h - 1)))
            for (x, y) in points]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(CONTACT_ROOT, "config.yaml"))
    ap.add_argument("--split", default="train",
                    choices=["train", "val", "test", "all", "known_interact"])
    ap.add_argument("--video-root", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--text", default="cow")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--iou-low", type=float, default=None)
    ap.add_argument("--iou-high", type=float, default=None)
    ap.add_argument("--scope", default="annotated", choices=["annotated", "frame"])
    ap.add_argument("--reading", default="dilated",
                    choices=["overlap", "gap", "surface", "dilated"])
    ap.add_argument("--dilate-px", type=int, default=22)
    ap.add_argument("--touch-px", type=int, default=10)
    ap.add_argument("--strip-px", type=int, default=6)
    ap.add_argument("--gt-dilate-scale", type=float, default=0.5)
    ap.add_argument("--gt-dilate-px", type=int, default=None)
    ap.add_argument("--depth-tol", type=float, default=None)
    ap.add_argument("--depth-gate", default="pair")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--pair-by", default="box_iou",
                    choices=["box_iou", "mask_gap"])
    ap.add_argument("--pair-gap-px", type=float, default=44.0)
    ap.add_argument("--pairs-from", default="sam3",
                    choices=["sam3", "detector", "detector_all"])
    ap.add_argument("--match-iou", type=float, default=0.3)
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.conf is None:
        args.conf = float(cfg["data"].get("sam3_conf", 0.6))
    if args.video_root is None:
        args.video_root = cfg["data"].get("video_dir")
        if not args.video_root:
            raise SystemExit("no data.video_dir in config.yaml; pass --video-root")
    if args.iou_low is None:
        args.iou_low = float(cfg["data"].get("pair_iou_low", 0.1))
    if args.iou_high is None:
        args.iou_high = float(cfg["data"].get("pair_iou_high", 0.8))
    gt_path = args.gt or os.path.join(CONTACT_ROOT, "log", "annotate", args.split,
                                      "contact_gt.csv")
    if not os.path.exists(gt_path):
        raise SystemExit(f"no ground truth at {gt_path}")
    gt = read_gt(gt_path)
    by_rel = {r["rel_image"]: r
              for r in records_for(load_records(cfg, require_label=False), args.split)}

    frames = collections.defaultdict(list)
    for rel, ann in gt.items():
        if ann["status"] == "skip":
            continue
        rec = by_rel.get(rel)
        if rec is None:
            continue
        frames[(rec["source_video"], int(rec["frame_number"]))].append((rel, ann, rec))
    if not frames:
        raise SystemExit("no annotated crops resolve to a frame")

    all_pairs = collections.defaultdict(list)
    for rec in by_rel.values():
        key = (rec["source_video"], int(rec["frame_number"]))
        if key in frames:
            all_pairs[key].append(rec)

    gt_radius = (args.gt_dilate_px if args.gt_dilate_px is not None
                 else max(1, int(round(args.gt_dilate_scale * args.dilate_px))))

    src = FrameSource(args.video_root)
    needed = {os.path.splitext(v)[0] for (v, _) in frames}
    missing = sorted(needed - set(src.index))
    if missing:
        found = sorted(src.index)[:6]
        raise SystemExit(
            f"{len(missing)} of {len(needed)} videos not found under "
            f"{args.video_root}\n"
            "  missing: " + ", ".join(missing[:4])
            + (" ..." if len(missing) > 4 else "") + "\n"
            + (f"  {len(src.index)} video(s) WERE found there, e.g. "
               + ", ".join(found) + "\n" if src.index
               else "  no video files at all were found there\n"))
    print(f"[wf] {len(needed)} videos resolved under {args.video_root}")

    try:
        from contactTest.sam3 import Sam3
        seg = Sam3(args.weights, args.text, args.conf)
    except Exception as err:
        raise SystemExit(f"could not load SAM 3 ({err}). See visualize_sam3_"
                         "confusion.py for the install and access steps.")

    out_dir = os.path.join(CONTACT_ROOT, "log", "wholeframe", args.split)
    os.makedirs(out_dir, exist_ok=True)

    rows, diag, failed = [], [], 0
    det_yolo, det_sam3, pair_yolo, pair_sam3 = [], [], [], []

    by_video = collections.defaultdict(list)
    for (video, fno) in frames:
        by_video[video].append(fno)

    stop = False
    for video in sorted(by_video):
        if stop:
            break
        wanted = sorted(by_video[video])
        seen = 0
        for fno, frame in src.frames_for(video, wanted):
            seen += 1

            items = frames[(video, fno)]
            if args.limit and len(rows) >= args.limit:
                stop = True
                break
            H, W = frame.shape[:2]

            try:
                inst, boxes3, qscores = seg.detect(frame)
            except Exception as err:
                print(f"[wf] SAM 3 failed on {video} frame {fno}: {err}")
                failed += 1
                continue
            if not inst or len(inst) < 2:
                failed += 1
                continue

            keep = [k for k, b in enumerate(boxes3) if b is not None]
            inst = [inst[k] for k in keep]
            boxes3 = [boxes3[k] for k in keep]
            if not boxes3:
                print(f"[wf] {video} frame {fno}: no pred_boxes")
                failed += 1
                continue

            if args.pairs_from == "sam3":
                if args.pair_by == "box_iou":
                    pairs = [(i, j) for i in range(len(inst))
                             for j in range(i + 1, len(inst))
                             if args.iou_low < box_iou(boxes3[i], boxes3[j])
                             < args.iou_high]
                else:
                    pairs = []
                    for i in range(len(inst)):
                        for j in range(i + 1, len(inst)):
                            g = mask_gap(inst[i], inst[j])
                            if g is not None and g <= args.pair_gap_px:
                                pairs.append((i, j))
            else:
                src_recs = ([rec for _, _, rec in items]
                            if args.pairs_from == "detector"
                            else all_pairs[(video, fno)])
                pairs = []
                for rec in src_recs:
                    b1, b2 = rec["bbox1"], rec["bbox2"]
                    i1 = max(range(len(boxes3)), key=lambda k: box_iou(boxes3[k], b1))
                    i2 = max(range(len(boxes3)), key=lambda k: box_iou(boxes3[k], b2))
                    if i1 != i2 and min(box_iou(boxes3[i1], b1),
                                        box_iou(boxes3[i2], b2)) >= args.match_iou:
                        pairs.append((i1, i2))

            region = np.zeros((H, W), bool)
            for (i, j) in pairs:
                r = contact_readings(inst[i], inst[j], args.touch_px, args.dilate_px,
                                     args.strip_px)[args.reading]
                region |= r

            n_yolo_pairs = len(items)
            det_sam3.append(len(inst))
            pair_sam3.append(len(pairs))
            pair_yolo.append(n_yolo_pairs)

            for rel, ann, rec in items:
                b1, b2 = rec["bbox1"], rec["bbox2"]
                i1 = max(range(len(boxes3)), key=lambda k: box_iou(boxes3[k], b1))
                i2 = max(range(len(boxes3)), key=lambda k: box_iou(boxes3[k], b2))
                m1, m2 = box_iou(boxes3[i1], b1), box_iou(boxes3[i2], b2)
                matched = (m1 >= args.match_iou and m2 >= args.match_iou and i1 != i2)
                pair_iou = box_iou(boxes3[i1], boxes3[i2]) if i1 != i2 else 1.0
                formed = matched and (args.iou_low < pair_iou < args.iou_high)
                mb1, mb2 = mask_box(inst[i1]), mask_box(inst[i2])
                mask_iou = (box_iou(mb1, mb2)
                            if (i1 != i2 and mb1 and mb2) else float("nan"))
                diag.append({"rel_image": rel, "match_iou_1": m1, "match_iou_2": m2,
                             "same_instance": int(i1 == i2), "matched": int(matched),
                             "pair_iou": pair_iou, "pair_formed": int(formed),
                             "mask_extent_iou": mask_iou,
                             "yolo_pair_iou": box_iou(b1, b2)})

            points, scope = [], np.zeros((H, W), bool)
            n_ctrl = 0
            for rel, ann, rec in items:
                merged = rec["merged"]
                x1, y1 = max(0, int(merged[0])), max(0, int(merged[1]))
                x2, y2 = min(W, int(merged[2])), min(H, int(merged[3]))
                if x2 > x1 and y2 > y1:
                    scope[y1:y2, x1:x2] = True
                if ann["status"] == "none":
                    n_ctrl += 1
                    continue
                points.extend(crop_to_frame(ann["points"], merged, (H, W)))

            if args.scope == "annotated":
                region = region & scope

            if args.depth_tol is not None:
                gated = region.copy()
                for rel, ann, rec in items:
                    merged = rec["merged"]
                    x1, y1 = max(0, int(merged[0])), max(0, int(merged[1]))
                    x2, y2 = min(W, int(merged[2])), min(H, int(merged[3]))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    sub_shape = (y2 - y1, x2 - x1)
                    dep = load_depth(rec, sub_shape)
                    if dep is None:
                        continue
                    sub = region[y1:y2, x1:x2]
                    if not sub.any():
                        continue
                    loc = [m[y1:y2, x1:x2] for m in inst]
                    areas = sorted(range(len(loc)), key=lambda k: -loc[k].sum())[:2]
                    if len(areas) < 2 or loc[areas[1]].sum() == 0:
                        continue
                    lb = [(0.0, 0.0, float(sub_shape[1]), float(sub_shape[0]))] * 2
                    st = depth_stats(loc[areas[0]], loc[areas[1]], dep[0], dep[1],
                                     dep[2], lb)
                    keepm = None
                    for g in args.depth_gate.split(","):
                        s_map = st.get(g.strip())
                        if s_map is None:
                            continue
                        k_ = s_map <= args.depth_tol
                        keepm = k_ if keepm is None else (keepm & k_)
                    if keepm is not None:
                        gated[y1:y2, x1:x2] = sub & keepm
                region = gated

            denom = float(scope.sum()) if args.scope == "annotated" else None
            rec_m, lab, hitting = evaluate_one(region, points, (H, W), gt_radius,
                                               denom_px=denom)
            rec_m["rel_image"] = f"{video}#{fno}"
            rec_m["no_contact"] = int(len(points) == 0)
            rec_m["n_sam3_instances"] = len(inst)
            rec_m["n_sam3_pairs"] = len(pairs)
            rec_m["n_yolo_pairs"] = n_yolo_pairs
            rows.append(rec_m)

            if not args.no_images:
                img = render(frame, region, lab, hitting, points, gt_radius, rec_m)
                if args.scope == "annotated":
                    cnt, _ = cv2.findContours(scope.astype(np.uint8), cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img, cnt, -1, (255, 255, 255), 2, cv2.LINE_AA)
                s = rec_m["sensitivity"]
                img = banner(img, [
                    (f"{video} frame {fno}   "
                     + ("no contact" if rec_m["no_contact"] else
                        f"sensitivity {s:.0%} ({rec_m['n_covered']}/{rec_m['n_points']})"),
                     (150, 30, 30) if rec_m["no_contact"] else
                     ((25, 110, 40) if s >= .8 else (170, 60, 25))),
                    (f"SAM 3 found {len(inst)} cattle -> {len(pairs)} pairs   "
                     f"YOLO gave {n_yolo_pairs} annotated pair(s)", (80, 80, 80)),
                    (f"clusters {rec_m['n_clusters']}  blind {rec_m['blind_frac']:.0%}"
                     f"   a_i {rec_m['a_i']:.2%}   scope={args.scope}"
                     + ("  (white outline = scored area)"
                        if args.scope == "annotated" else ""), (80, 80, 80))])
                tag = "none" if rec_m["no_contact"] else f"{int(round(s * 100)):03d}"
                cv2.imwrite(os.path.join(out_dir, f"{tag}_{video}_{fno:08d}.jpg"),
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        if seen < len(wanted):
            short = len(wanted) - seen
            failed += short

    if not rows:
        raise SystemExit("nothing evaluated")

    n_img = len(rows)
    tot_pts = sum(r["n_points"] for r in rows)
    tot_cov = sum(r["n_covered"] for r in rows)
    sensitivity = tot_cov / tot_pts if tot_pts else float("nan")
    tot_fp = sum(r["n_fp_clusters"] for r in rows)
    fppi = tot_fp / n_img
    a_bar = float(np.mean([r["a_i"] for r in rows]))
    lift = sensitivity / a_bar if a_bar > 0 else float("nan")
    wq = [(r["hit_quality"], r["n_covered"]) for r in rows
          if np.isfinite(r["hit_quality"])]
    hit_q = (sum(v * n for v, n in wq) / sum(n for _, n in wq)) if wq else float("nan")
    tot_area = sum(r["area_px"] for r in rows)
    tot_blind = sum(r["blind_area_px"] for r in rows)
    blind = tot_blind / tot_area if tot_area else float("nan")

    print(f"\n{'=' * 66}")
    print(f"  WHOLE FRAME, no YOLO: SAM 3 text={args.text!r}, pairs by "
          f"{args.iou_low} < IoU < {args.iou_high}")
    print(f"  {args.reading} at dilate_px={args.dilate_px}"
          + (f", depth gate {args.depth_gate} <= {args.depth_tol}"
             if args.depth_tol is not None else ", no depth gate"))
    print(f"  {n_img} frames, {tot_pts} GT points, scope={args.scope}, "
          f"pairs from {args.pairs_from}"
          + (f" by {args.pair_by}" if args.pairs_from == "sam3" else ""))
    print(f"{'=' * 66}\n")
    print(f"  1  Sensitivity   {sensitivity:>9.3f}   {tot_cov}/{tot_pts} points covered")
    print(f"  2  FPPI          {fppi:>9.3f}   {tot_fp} empty clusters / {n_img} frames")
    print(f"  3  a-bar         {a_bar:>9.4f}   mean predicted area fraction"
          + ("  (of the SCOPE, matching evaluate_contact's crop denominator)"
             if args.scope == "annotated" else "  (of the whole frame)"))
    print(f"  4  Lift          {lift:>9.1f}")
    print(f"  5  Hit quality   {hit_q:>9.2f}   (GT discs r={gt_radius}px)")
    print(f"  6  Blind area    {blind:>9.3f}   {tot_blind}/{tot_area} px")

    if diag:
        n = len(diag)
        def _cls(d):
            if min(d["match_iou_1"], d["match_iou_2"]) < args.match_iou:
                return "unsegmented"
            if d["same_instance"]:
                return "merged"
            return "formed" if d["pair_formed"] else "outrange"
        kinds = [_cls(d) for d in diag]
        formed = kinds.count("formed")
        outrange = kinds.count("outrange")
        same = kinds.count("merged")
        nomatch = kinds.count("unsegmented")
        print(f"   ({n} annotated pairs)")
        print(f"    pair formed by the same rule   {formed:>4}  {formed / n:>5.0%}")
        print(f"    both animals found, IoU outside {args.iou_low}-{args.iou_high}"
              f"   {outrange:>3}  {outrange / n:>5.0%}")
        print(f"    the two boxes matched ONE instance {same:>3}  {same / n:>5.0%}"
              "   (SAM 3 merged them)")
        print(f"    an animal not segmented at all  {nomatch:>4}  {nomatch / n:>5.0%}"
              f"   (best IoU < {args.match_iou})")
        pi = np.array([d["pair_iou"] for d in diag], float)
        yi = np.array([d["yolo_pair_iou"] for d in diag], float)
        mi_ = np.array([d["mask_extent_iou"] for d in diag], float)
        mi_ = mi_[np.isfinite(mi_)]
        print(f"    median pair IoU, SAM 3 pred_boxes    {np.median(pi):.3f}")
        print(f"    median pair IoU, detector boxes      {np.median(yi):.3f}")
        if mi_.size:
            print(f"    median pair IoU, MASK extents        {np.median(mi_):.3f}"
                  "   (not used for pairing; shown to size the difference)")

    print(f"    SAM 3 found  {np.mean(det_sam3):.1f} cattle per frame "
          f"-> {np.mean(pair_sam3):.1f} pairs at {args.iou_low}-{args.iou_high} IoU")
    print(f"    YOLO left    {np.mean(pair_yolo):.1f} annotated pair(s) per frame")

    if failed:
        print(f"\n  {failed} frames skipped (video or SAM 3 failure)")

    keys = ["rel_image", "no_contact", "n_points", "n_covered", "sensitivity",
            "n_clusters", "n_fp_clusters", "area_px", "a_i", "lift",
            "hit_quality", "gt_disc_px", "blind_area_px", "blind_frac",
            "n_sam3_instances", "n_sam3_pairs", "n_yolo_pairs"]
    with open(os.path.join(out_dir, "wholeframe.csv"), "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=keys)
        wtr.writeheader()
        for r in rows:
            wtr.writerow({k: r.get(k, "") for k in keys})
    if diag:
        with open(os.path.join(out_dir, "pair_recall.csv"), "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(diag[0].keys()))
            wtr.writeheader()
            wtr.writerows(diag)
    with open(os.path.join(out_dir, "wholeframe.json"), "w") as f:
        json.dump({"split": args.split, "scope": args.scope, "text": args.text,
                   "iou_low": args.iou_low, "iou_high": args.iou_high,
                   "reading": args.reading, "dilate_px": args.dilate_px,
                   "n_frames": n_img, "n_points": tot_pts,
                   "sensitivity": sensitivity, "fppi": fppi, "a_bar": a_bar,
                   "lift": lift, "hit_quality": hit_q, "blind_area_frac": blind,
                   "sam3_instances_mean": float(np.mean(det_sam3)),
                   "sam3_pairs_mean": float(np.mean(pair_sam3)),
                   "annotated_pairs_mean": float(np.mean(pair_yolo))}, f, indent=2)
    print(f"\n[wf] wrote {out_dir}")


if __name__ == "__main__":
    main()
