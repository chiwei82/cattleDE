
import argparse
import os
import sys

import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contactTest.src.utils import load_config

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CONTACT_ROOT, ".."))
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def _extract_boxes(results):
    boxes = []
    if results.boxes is not None and len(results.boxes):
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2, conf))
    elif results.obb is not None and len(results.obb):
        corners = results.obb.xyxyxyxy.cpu().numpy()
        confs = results.obb.conf.cpu().numpy()
        for pts, conf in zip(corners, confs):
            x1 = int(pts[:, 0].min()); y1 = int(pts[:, 1].min())
            x2 = int(pts[:, 0].max()); y2 = int(pts[:, 1].max())
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2, float(conf)))
    return boxes


def find_video(root, stem):
    for dirpath, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[0] == stem and f.lower().endswith(VIDEO_EXTS):
                return os.path.join(dirpath, f)
    return None


def read_frame(path, wanted):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    idx, frame = 0, None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if idx == wanted:
            frame = f
            break
        if idx % 2000 == 0 and idx:
            print(f"  ... decoded {idx}/{wanted}", flush=True)
        idx += 1
    cap.release()
    if frame is None:
        raise SystemExit(f"video ended at frame {idx}; {wanted} does not exist")
    return frame


def draw(frame, boxes):
    out = frame.copy()
    h = out.shape[0]
    thick = max(2, h // 540)
    scale = max(0.5, h / 1400.0)
    for k, (x1, y1, x2, y2, conf) in enumerate(boxes):
        cv2.rectangle(out, (x1, y1), (x2, y2), (60, 220, 90), thick)
        text = f"{conf:.2f}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                         scale, thick)
        ty = max(y1, th + base + 2)
        cv2.rectangle(out, (x1, ty - th - base - 2), (x1 + tw + 6, ty),
                      (60, 220, 90), -1)
        cv2.putText(out, text, (x1 + 3, ty - base), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (20, 20, 20), thick, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(CONTACT_ROOT, "config.yaml"))
    ap.add_argument("--video", default="20250802T082154_20250802T084303")
    ap.add_argument("--frame", type=int, default=8125)
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--weights", default="checkpoints/yolo_pseudo.pt")
    ap.add_argument("--conf", type=float, default=0.6)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--out", default="simu/yoloexample.jpg")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = args.video_dir or cfg["data"].get("video_dir")
    path = find_video(root, args.video)
    if path is None:
        raise SystemExit(f"no video with stem {args.video!r} under {root}")
    print(f"[yolo] {path}")

    weights = (args.weights if os.path.isabs(args.weights)
               else os.path.join(REPO_ROOT, args.weights))
    if not os.path.exists(weights):
        raise SystemExit(f"no checkpoint at {weights}")

    print(f"[yolo] decoding to frame {args.frame} (sequential, not seeking)")
    frame = read_frame(path, args.frame)
    print(f"[yolo] frame {args.frame}: {frame.shape[1]}x{frame.shape[0]}")

    from ultralytics import YOLO
    model = YOLO(weights)
    results = model.predict(source=frame, conf=args.conf, imgsz=args.imgsz,
                            verbose=False)[0]
    kind = ("obb" if (results.boxes is None or not len(results.boxes))
            else "axis-aligned")
    boxes = _extract_boxes(results)
    print(f"[yolo] {weights} at conf={args.conf} imgsz={args.imgsz}: "
          f"{len(boxes)} detections ({kind} head)")
    for k, (x1, y1, x2, y2, c) in enumerate(sorted(boxes, key=lambda b: -b[4])):
        print(f"       {k:>2}  conf {c:.3f}   ({x1}, {y1}) -> ({x2}, {y2})")

    out = (args.out if os.path.isabs(args.out)
           else os.path.join(REPO_ROOT, args.out))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, draw(frame, boxes))
    print(f"[yolo] wrote {out}")


if __name__ == "__main__":
    main()
