"""Save one snapshot per interaction session (the first-second frame).

Sessions are the video files interaction_prep reads from
interaction_prep.video_dir (same discovery: *.avi/*.mp4/*.mov/*.mkv, sorted).
For each video it grabs the frame at ~1 second and writes it to
log/session_snapshot/<video_stem>.jpg, so you can eyeball every session.

    python scripts/session_snapshots.py                 # first-second frame
    python scripts/session_snapshots.py --second 0      # very first frame
"""

import argparse
import os
from pathlib import Path

import cv2
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_REPO_ROOT, "global_config.yaml")) as _f:
    _CFG = yaml.safe_load(_f)

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}


def main():
    ap = argparse.ArgumentParser(description="Per-session first-second snapshots.")
    ap.add_argument("--video_dir", default=_CFG["interaction_prep"]["video_dir"],
                    help="Source folder interaction_prep reads (default: config).")
    ap.add_argument("--out", default=os.path.join(_REPO_ROOT, "log", "session_snapshot"),
                    help="Output directory for the snapshots.")
    ap.add_argument("--second", type=float, default=1.0,
                    help="Which second to grab (0 = the very first frame).")
    args = ap.parse_args()

    if not os.path.isdir(args.video_dir):
        raise SystemExit(f"video_dir not found: {args.video_dir}")
    os.makedirs(args.out, exist_ok=True)

    videos = sorted(p for p in Path(args.video_dir).iterdir()
                    if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        raise SystemExit(f"No videos ({sorted(VIDEO_EXTS)}) in {args.video_dir}")
    print(f"{len(videos)} sessions in {args.video_dir}")

    saved = 0
    for vp in videos:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print(f"  [WARN] cannot open {vp.name}, skipping.")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        target = int(round(args.second * fps))
        if total > 0:
            target = min(target, total - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if not ret:                       # fall back to the first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"  [WARN] cannot read a frame from {vp.name}, skipping.")
            continue

        out_path = os.path.join(args.out, vp.stem + ".jpg")
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1
        print(f"  {vp.name}  (frame {target}) -> {out_path}")

    print(f"\nDone. {saved}/{len(videos)} snapshots in {args.out}")


if __name__ == "__main__":
    main()
