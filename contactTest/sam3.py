
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DEFAULT_MODEL_ID = "facebook/sam3"

DEFAULT_CONF = 0.6


class Sam3:

    def __init__(self, weights=None, text="cow", conf=DEFAULT_CONF,
                 device=None):
        import torch
        from transformers import Sam3Model, Sam3Processor

        model_id = DEFAULT_MODEL_ID
        if weights:
            if os.path.isdir(weights):
                model_id = weights
            elif str(weights).endswith((".pt", ".pth", ".safetensors")):
                print(f"[sam3] NOTE: {weights} is not a Hugging Face layout "
                      "(config.json + safetensors). Loading "
                      f"{DEFAULT_MODEL_ID} instead — pass a local snapshot "
                      "directory to use one on disk.")
            else:
                model_id = weights

        self.torch = torch
        self.text = text
        self.conf = conf
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._box_note = False
        self.model = Sam3Model.from_pretrained(model_id).to(self.device).eval()
        self.processor = Sam3Processor.from_pretrained(model_id)
        print(f"[sam3] {model_id} on {self.device}, text={text!r}, conf={conf}")


    def _raw(self, bgr):
        from PIL import Image

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=Image.fromarray(rgb), text=self.text,
                                return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)

        scores = out.pred_logits.sigmoid()[0]
        if getattr(out, "presence_logits", None) is not None:
            scores = scores * out.presence_logits.sigmoid()[0]

        probs = self.torch.sigmoid(out.pred_masks)
        probs = self.torch.nn.functional.interpolate(
            probs.float(), size=bgr.shape[:2], mode="bilinear",
            align_corners=False)[0]

        boxes = None
        pb = getattr(out, "pred_boxes", None)
        if pb is not None:
            boxes = pb[0].float()
            h, w = bgr.shape[:2]
            mx = float(boxes.abs().max()) if boxes.numel() else 0.0
            if mx <= 1.5:
                boxes = boxes * self.torch.tensor(
                    [w, h, w, h], dtype=boxes.dtype, device=boxes.device)
                if not self._box_note:
                    print(f"[sam3] pred_boxes look normalised (max {mx:.3f}); "
                          f"scaled by image size {w}x{h}")
                    self._box_note = True
            elif not self._box_note:
                print(f"[sam3] pred_boxes look absolute (max {mx:.1f}) on a "
                      f"{w}x{h} image; used unscaled")
                self._box_note = True
        return probs, scores.reshape(-1), boxes


    def detect(self, bgr, binary=True):
        probs, scores, boxes = self._raw(bgr)
        keep = (scores >= self.conf).nonzero().reshape(-1).tolist()
        masks, bxs, scs = [], [], []
        for q in keep:
            a = probs[q].cpu().numpy().astype(np.float32)
            masks.append((a > 0.5).astype(np.uint8) if binary else a)
            bxs.append(tuple(float(v) for v in boxes[q].cpu().numpy())
                       if boxes is not None else None)
            scs.append(float(scores[q]))
        return masks, bxs, scs

    def assign_to_boxes(self, bgr, boxes, binary=True):
        masks, _, _ = self.detect(bgr, binary=False)
        if not masks:
            return None
        hard = [(m > 0.5).astype(np.uint8) for m in masks]
        want = hard if binary else masks

        scores = [[iou_mask_box(a, b) for a in hard] for b in boxes]
        out = [None] * len(boxes)
        taken = set()
        for _ in range(len(boxes)):
            best, bk, bi = 0.0, None, None
            for k in range(len(boxes)):
                if out[k] is not None:
                    continue
                for i in range(len(hard)):
                    if i in taken or scores[k][i] <= best:
                        continue
                    best, bk, bi = scores[k][i], k, i
            if bk is None or best <= 0.0:
                break
            out[bk] = want[bi]
            taken.add(bi)
        return None if any(o is None for o in out) else out


def iou_mask_box(mask, box):
    x1, y1 = int(max(0, box[0])), int(max(0, box[1]))
    x2 = int(min(mask.shape[1], box[2]))
    y2 = int(min(mask.shape[0], box[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = float(mask[y1:y2, x1:x2].sum())
    union = float(mask.sum()) + (x2 - x1) * (y2 - y1) - inter
    return inter / max(union, 1.0)


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0
