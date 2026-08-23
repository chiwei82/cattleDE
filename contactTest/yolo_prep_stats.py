
import argparse
import csv
import json
import os
import sys

import cv2

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))

DEFAULT_ROOT = "/user/work/sf24225/data/Full_behav/Marco"
DEFAULT_CAMERAS = ["camera128", "camera133", "camera26", "camera27", "camera48"]
DEFAULT_FRAME_STEP = 16


def hms(seconds):
    s = int(round(seconds))
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def count_frames(path):
    cap = cv2.VideoCapture(path)
    n = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        n += 1
    cap.release()
    return n


def video_info(path, exact):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    info = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS)) or 0.0,
        "meta_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    info["frames"] = count_frames(path) if exact else info["meta_frames"]
    info["seconds"] = info["frames"] / info["fps"] if info["fps"] > 0 else 0.0
    return info


def tracklet_stats(path, frame_step):
    with open(path) as f:
        data = json.load(f)
    tracks = boxes = sampled = 0
    frames = set()
    for key, entries in data.items():
        if key == "stats":
            continue
        tracks += 1
        for e in entries:
            fn = e.get("frame_number")
            if fn is None:
                continue
            boxes += 1
            frames.add(fn)
            if fn % frame_step == 0:
                sampled += 1
    return {"tracks": tracks, "boxes": boxes, "boxes_sampled": sampled,
            "annotated_frames": len(frames),
            "sampled_frames": sum(1 for f in frames if f % frame_step == 0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=DEFAULT_ROOT)
    ap.add_argument("--cameras", default=",".join(DEFAULT_CAMERAS))
    ap.add_argument("--frame-step", type=int, default=DEFAULT_FRAME_STEP)
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--csv", default="log/yolo_prep_stats.csv")
    args = ap.parse_args()

    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    if not os.path.isdir(args.data_root):
        raise SystemExit(f"no such data_root: {args.data_root}")
    print(f"[stats] {args.data_root}")
    print(f"[stats] cameras: {', '.join(cameras)}   frame_step={args.frame_step}")

    rows, skipped = [], []
    for cam in cameras:
        cam_path = os.path.join(args.data_root, cam)
        if not os.path.isdir(cam_path):
            print(f"[stats] WARNING: {cam_path} not found")
            continue
        for entry in sorted(os.listdir(cam_path)):
            entry_path = os.path.join(cam_path, entry)
            if not os.path.isdir(entry_path):
                continue
            tpath = os.path.join(entry_path, "tracklets.json")
            vpath = os.path.join(cam_path, entry + ".mp4")
            if not os.path.exists(tpath):
                skipped.append((cam, entry, "no tracklets.json"))
                continue
            if not os.path.exists(vpath):
                skipped.append((cam, entry, "no video"))
                continue
            t = tracklet_stats(tpath, args.frame_step)
            v = video_info(vpath, args.exact) or {}
            rows.append({"camera": cam, "session": entry, **t,
                         "fps": round(v.get("fps", 0.0), 3),
                         "frames": v.get("frames", 0),
                         "meta_frames": v.get("meta_frames", 0),
                         "seconds": round(v.get("seconds", 0.0), 1),
                         "width": v.get("width", 0), "height": v.get("height", 0)})
            print(f"  {cam}/{entry}", flush=True)

    if not rows:
        raise SystemExit("no sessions found")

    print(f"\n{'camera':<11}{'sessions':>9}{'duration':>11}{'frames':>10}"
          f"{'tracks':>8}{'boxes':>10}{'sampled':>9}{'box/frame':>11}")
    print("-" * 79)
    for cam in cameras:
        r = [x for x in rows if x["camera"] == cam]
        if not r:
            continue
        secs = sum(x["seconds"] for x in r)
        fr = sum(x["frames"] for x in r)
        bx = sum(x["boxes"] for x in r)
        af = sum(x["annotated_frames"] for x in r)
        print(f"{cam:<11}{len(r):>9}{hms(secs):>11}{fr:>10}"
              f"{sum(x['tracks'] for x in r):>8}{bx:>10}"
              f"{sum(x['boxes_sampled'] for x in r):>9}"
              f"{(bx / af if af else 0):>11.1f}")
    print("-" * 79)
    secs = sum(x["seconds"] for x in rows)
    bx = sum(x["boxes"] for x in rows)
    af = sum(x["annotated_frames"] for x in rows)
    print(f"{'TOTAL':<11}{len(rows):>9}{hms(secs):>11}"
          f"{sum(x['frames'] for x in rows):>10}"
          f"{sum(x['tracks'] for x in rows):>8}{bx:>10}"
          f"{sum(x['boxes_sampled'] for x in rows):>9}"
          f"{(bx / af if af else 0):>11.1f}")

    print(f"\n[stats] {len(rows)} sessions over "
          f"{len({r['camera'] for r in rows})} cameras")
    print(f"[stats] {bx} source boxes in total; "
          f"{sum(x['boxes_sampled'] for x in rows)} of them "
          f"({sum(x['boxes_sampled'] for x in rows) / max(bx, 1):.1%}) sit on a "
          f"frame divisible by {args.frame_step} and reached data/object")
    print(f"[stats] {af} frames carry at least one box, out of "
          f"{sum(x['frames'] for x in rows)} frames of footage "
          f"({af / max(sum(x['frames'] for x in rows), 1):.1%})")

    if args.exact:
        bad = [r for r in rows if r["frames"] != r["meta_frames"]]
        print(f"[stats] container metadata disagreed with the decoded count on "
              f"{len(bad)}/{len(rows)} videos")

    if skipped:
        print(f"\n[stats] {len(skipped)} directories skipped, as yolo_prep skips them:")
        for cam, entry, why in skipped:
            print(f"    {cam}/{entry}: {why}")

    out = (args.csv if os.path.isabs(args.csv)
           else os.path.join(CONTACT_ROOT, args.csv))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[stats] wrote {out}")


if __name__ == "__main__":
    main()
