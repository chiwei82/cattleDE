import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from ultralytics import YOLO

import sys
import random
import yaml
from itertools import combinations
sys.path.append('.')  # add custom module path
from train.action_with_image import LitVisionTransformer
from train.interaction_with_image import LitHybridStreamFusion

# ── Load the repo's global config so paths/params match the current pipeline ──
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_REPO_ROOT, "global_config.yaml")) as _f:
    _CFG = yaml.safe_load(_f)


def _resolve(p):
    """Resolve config paths relative to the repo root, not the CWD."""
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


def extract_boxes_xyxy(results):
    """Axis-aligned (x1, y1, x2, y2) boxes from a YOLO result, handling both
    standard detect models (results.boxes) and OBB models (results.obb) — the
    current pipeline's detector (checkpoints/yolo_pseudo.pt) is OBB."""
    boxes = []
    if results.boxes is not None and len(results.boxes):
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    elif getattr(results, "obb", None) is not None and len(results.obb):
        for pts in results.obb.xyxyxyxy.cpu().numpy():  # (N, 4, 2) corners
            x1 = int(pts[:, 0].min()); y1 = int(pts[:, 1].min())
            x2 = int(pts[:, 0].max()); y2 = int(pts[:, 1].max())
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes

# ==============================================================================
# 1. Helpers (model loading, inference, image processing)
# ==============================================================================

def load_action_classification_model(model_path, device):
    """Load the action classification model from a checkpoint."""
    model = LitVisionTransformer.load_from_checkpoint(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model

def load_interaction_model(model_path, device):
    """Load the interaction classification model from a checkpoint."""
    model = LitHybridStreamFusion.load_from_checkpoint(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model

def run_classification_inference(model, image, class_names, device):
    """Run class-classification inference on a single image."""
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        probabilities = F.softmax(logits, dim=1)
        confidence, predicted_id = torch.max(probabilities, 1)
        predicted_label = class_names.get(predicted_id.item(), "Unknown")
    return predicted_label, confidence.item()

def run_interaction_inference(model, image1, image2, image_context, class_names, device):
    """Run interaction-classification inference on the three images."""
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    t1 = preprocess(image1).unsqueeze(0).to(device)
    t2 = preprocess(image2).unsqueeze(0).to(device)
    t_context = preprocess(image_context).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(t1, t2, t_context)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        probabilities = F.softmax(logits, dim=1)
        confidence, predicted_id = torch.max(probabilities, 1)
        predicted_label = class_names.get(predicted_id.item(), "Unknown")
    return predicted_label, confidence.item()

def crop_image_from_frame(frame, bbox):
    """Crop the given bbox from the frame and return a PIL image."""
    x1, y1, x2, y2 = map(int, bbox)
    h, w, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x1 >= x2 or y1 >= y2:
        return Image.new('RGB', (0, 0))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb).crop((x1, y1, x2, y2))

def calculate_iou(box1, box2):
    """Compute the IoU of two bboxes."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

# ==============================================================================
# 2. Single-frame processing
# ==============================================================================

def process_single_frame(frame, yolo_model, action_model, interaction_model, device, action_class_names, interaction_class_names, yolo_conf, yolo_imgsz):
    """Detect, classify per-cow action and pairwise interaction, and draw on one frame."""
    detection_results = yolo_model.predict(
        source=frame, verbose=False, iou=0.5, conf=yolo_conf, imgsz=yolo_imgsz
    )[0]
    annotated_frame = frame.copy()

    action_color = (255, 0, 0)      # Blue for action
    interaction_color = (255, 0, 255) # Magenta for interaction

    # Axis-aligned (x1,y1,x2,y2) boxes for both OBB and standard models
    all_coords = [list(b) for b in extract_boxes_xyxy(detection_results)]
    if not all_coords:
        return annotated_frame

    num_boxes = len(all_coords)

    # Per-cow action inference and drawing
    for coords in all_coords:
        bbox_image = crop_image_from_frame(frame, coords)
        if bbox_image.width > 0 and bbox_image.height > 0:
            action_label, _ = run_classification_inference(action_model, bbox_image, action_class_names, device)
            x1, y1, x2, y2 = map(int, coords)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), action_color, 2)
            (w, h), _ = cv2.getTextSize(action_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - h - 10), (x1 + w, y1), action_color, -1)
            cv2.putText(annotated_frame, action_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # ── Interaction pair proposals (same config logic as prep/interaction_prep):
    #    keep a pair when iou_low < IoU < iou_high and it is not nested, then
    #    classify it as a BINARY interaction with confidence. ────────────────────
    icfg = _CFG["interaction_prep"]
    iou_low, iou_high, nested_thresh = icfg["iou_low"], icfg["iou_high"], icfg["nested_thresh"]
    for i, j in combinations(range(num_boxes), 2):
        b1, b2 = all_coords[i], all_coords[j]
        iou = calculate_iou(b1, b2)
        if not (iou_low < iou < iou_high):
            continue

        # nested check: smaller box mostly inside the larger -> same animal
        ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        smaller = min(a1, a2)
        if smaller > 0 and inter / smaller > nested_thresh:
            continue

        merged_coords = [min(b1[0], b2[0]), min(b1[1], b2[1]),
                         max(b1[2], b2[2]), max(b1[3], b2[3])]
        box_a, box_b = (b1, b2) if a1 >= a2 else (b2, b1)  # larger box first
        image1 = crop_image_from_frame(frame, box_a)
        image2 = crop_image_from_frame(frame, box_b)
        image_context = crop_image_from_frame(frame, merged_coords)
        if not all(img.width > 0 and img.height > 0 for img in (image1, image2, image_context)):
            continue

        interaction_label, confidence = run_interaction_inference(
            interaction_model, image1, image2, image_context, interaction_class_names, device)
        x1, y1, x2, y2 = merged_coords
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), interaction_color, 3)
        label = f"{interaction_label} {confidence:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        text_x, text_y = x2 + 10, y1
        cv2.rectangle(annotated_frame, (text_x - 5, text_y - 5), (text_x + w + 5, text_y + h + 5), interaction_color, -1)
        cv2.putText(annotated_frame, label, (text_x, text_y + h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(annotated_frame, (text_x, text_y + h // 2), (x2, y1 + (y2 - y1) // 2), interaction_color, 2)

    return annotated_frame

# ==============================================================================
# 3. Video generation main function
# ==============================================================================

def create_processed_video(input_video_path, output_video_path, center_frame_num, 
                           yolo_model_path, action_model_path, interaction_model_path,
                           duration_seconds=4, output_fps=5):
    """Read a video, process the selected range, and save it as a new video."""
    
    # --- 1. Device and model setup ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    yolo_model = YOLO(yolo_model_path)
    # Action classes come from the current pipeline's config (action_prep.labels).
    action_class_names = {i: name for i, name in enumerate(_CFG["action_prep"]["labels"])}
    # Interaction is a BINARY classifier (train/interaction_with_image.py).
    interaction_class_names = {0: 'no_interaction', 1: 'interaction'}

    # YOLO inference params from config (detector is the OBB checkpoint).
    yolo_conf = _CFG["interaction_prep"]["yolo_conf"]
    yolo_imgsz = _CFG["interaction_prep"]["yolo_imgsz"]

    action_model = load_action_classification_model(action_model_path, DEVICE)
    interaction_model = load_interaction_model(interaction_model_path, DEVICE)
    print("All models loaded successfully.")

    # --- 2. Video I/O setup ---
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{input_video_path}'")
        return
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, output_fps, (frame_width, frame_height))
    
    # --- 3. Frame range to process ---
    frame_step = max(1, int(original_fps / output_fps))
    if center_frame_num is None:
        start_frame, end_frame = 0, total_frames          # whole video
    else:
        half = int(duration_seconds / 2 * output_fps) * frame_step
        start_frame = max(0, center_frame_num - half)
        end_frame = min(total_frames, center_frame_num + half)

    print(f"Processing video from frame {start_frame} to {end_frame} (step: {frame_step}).")
    
    # --- 4. Main loop ---
    for frame_idx in range(start_frame, end_frame, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_idx}.")
            continue
        
        print(f"  - Processing frame: {frame_idx}")
        processed_frame = process_single_frame(
            frame, yolo_model, action_model, interaction_model, DEVICE,
            action_class_names, interaction_class_names, yolo_conf, yolo_imgsz
        )
        writer.write(processed_frame)

    # --- 5. Release resources ---
    cap.release()
    writer.release()
    print(f"\nSuccessfully created video: {output_video_path}")

# ==============================================================================
# 4. Entry point
# ==============================================================================

if __name__ == '__main__':
    # Video source = the same folder interaction_prep uses (global_config.yaml).
    VIDEO_DIR = _resolve(_CFG['interaction_prep']['video_dir'])
    OUTPUT_DIR = os.path.join(_REPO_ROOT, 'outputs')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Checkpoint paths (all from global_config.yaml): YOLO = OBB detector,
    # action/interaction = the checkpoints produced by train/*.py.
    YOLO_MODEL_PATH = _resolve(_CFG['paths']['yolo_ckpt'])
    ACTION_MODEL_PATH = _resolve(_CFG['paths']['action_ckpt'])
    INTERACTION_MODEL_PATH = _resolve(_CFG['paths']['interaction_ckpt'])

    # --- Reference demo targets (these NFS paths do not exist here; kept for
    #     reference only, superseded by the interaction_prep video source) ---
    # target_action = 'mount'
    # if target_action == 'conflict':
    #     INPUT_VIDEO_PATH = '/mnt/nfs/CameraData/hiiku2/2025-03-05/17/2025-03-05 17-10-00~17-20-00.avi'
    #     CENTER_FRAME_NUM = 11400
    # elif target_action == 'mount':
    #     INPUT_VIDEO_PATH = '/mnt/nfs/CameraData/hiiku2/2025-03-05/16/2025-03-05 16-40-00~16-50-00.avi'
    #     CENTER_FRAME_NUM = 9510

    for ckpt in (ACTION_MODEL_PATH, INTERACTION_MODEL_PATH):
        if not os.path.exists(ckpt):
            print(f"Error: required checkpoint not found: {ckpt}")
            sys.exit(1)

    video_exts = {'.avi', '.mp4', '.mov', '.mkv'}
    videos = sorted(p for p in os.listdir(VIDEO_DIR)
                    if os.path.splitext(p)[1].lower() in video_exts)
    if not videos:
        print(f"No videos found in {VIDEO_DIR}")
        sys.exit(1)

    # Pick one random video and a random 10-second segment within it.
    SEGMENT_SECONDS = 10
    name = random.choice(videos)
    video_path = os.path.join(VIDEO_DIR, name)
    stem = os.path.splitext(name)[0].replace(' ', '_')

    _cap = cv2.VideoCapture(video_path)
    fps = _cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _cap.release()

    half_frames = int((SEGMENT_SECONDS / 2) * fps)
    if total_frames > 2 * half_frames:
        center_frame = random.randint(half_frames, total_frames - half_frames)
    else:
        center_frame = total_frames // 2  # video shorter than the segment
    print(f"Selected '{name}' — random {SEGMENT_SECONDS}s segment around frame {center_frame}.")

    create_processed_video(
        input_video_path=video_path,
        output_video_path=os.path.join(OUTPUT_DIR, f"{stem}_annotated.mp4"),
        center_frame_num=center_frame,
        yolo_model_path=YOLO_MODEL_PATH,
        action_model_path=ACTION_MODEL_PATH,
        interaction_model_path=INTERACTION_MODEL_PATH,
        duration_seconds=SEGMENT_SECONDS,
        output_fps=5,
    )