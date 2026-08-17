"""The one place SAM 3 is called, exposing masks AND boxes together.

Every consumer imports `Sam3` from here. Before this module the backend lived
inside precompute_masks.py, a mask-caching script, and each caller reached in
for whatever it happened to need — which is how `pred_boxes` came to be dropped
in one place and a box measured off the mask substituted in three others without
anyone noticing they were different quantities.

WHAT COMES BACK, AND WHY BOTH

    masks    per-pixel, from sigmoid(pred_masks). Continuous unless `binary`.
    boxes    pred_boxes, SAM 3's OWN box output, in absolute xyxy.
    scores   pred_logits.sigmoid() * presence_logits.sigmoid(), the documented
             form. The presence head is what separates "this query found
             something" from "this query is one of the 200 that matched nothing".

Masks and boxes are not interchangeable and neither is derivable from the other
for the purposes here:

  * The contact region is per-pixel — dilate(mask_i, r) AND dilate(mask_j, r) —
    so it needs masks. Two rectangles cannot produce a band that follows an
    animal's outline.
  * PAIRING consumes a detector's box, because that is what the stage being
    replaced consumed: interaction_prep read YOLO's box.xyxy. Substituting the
    extent of a mask puts a quantity in that neither pipeline produced, and the
    comparison stops being between two detectors.

Whether pred_boxes runs looser or tighter than the mask it belongs to is not
documented and is not assumed anywhere; SAM 3 was trained on its own data
engine's output, so nothing can be read across from familiar box conventions
either. Callers that want to know measure it.

WHY TRANSFORMERS AND NOT ULTRALYTICS

Ultralytics' postprocess ends in `masks = masks > self.model.mask_threshold`,
so the per-pixel score is gone before anything can read it, and
`post_process_instance_segmentation` binarises too. `pred_masks` is emitted as
floats; nothing here thresholds them except where a caller asks for `binary`.
"""

import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DEFAULT_MODEL_ID = "facebook/sam3"

# Matches data.sam3_conf in config.yaml, which every entry point reads. Defined
# here too so a direct construction does not silently get a different, looser
# threshold than the scripts do — which is exactly what happened while this sat
# at a library default nobody had chosen.
DEFAULT_CONF = 0.6


class Sam3:
    """SAM 3 concept segmentation on a whole image."""

    def __init__(self, weights=None, text="cow", conf=DEFAULT_CONF,
                 device=None):
        import torch
        from transformers import Sam3Model, Sam3Processor

        # `weights` may be a Hugging Face id or a local snapshot DIRECTORY. It
        # may not be an ultralytics .pt: that is a different serialisation and
        # Sam3Model.from_pretrained cannot read it. Saying so beats loading the
        # default while the run looks as though the requested file was used.
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

        # No device_map: that routes through accelerate, an extra dependency
        # meant for sharding across several GPUs. One explicit .to() is enough.
        self.torch = torch
        self.text = text
        self.conf = conf
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._box_note = False
        self.model = Sam3Model.from_pretrained(model_id).to(self.device).eval()
        self.processor = Sam3Processor.from_pretrained(model_id)
        print(f"[sam3] {model_id} on {self.device}, text={text!r}, conf={conf}")

    # ── raw model output ─────────────────────────────────────────────────────

    def _raw(self, bgr):
        """Per-query probabilities, scores and boxes, on the image's own grid."""
        from PIL import Image

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=Image.fromarray(rgb), text=self.text,
                                return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)

        scores = out.pred_logits.sigmoid()[0]
        if getattr(out, "presence_logits", None) is not None:
            scores = scores * out.presence_logits.sigmoid()[0]

        probs = self.torch.sigmoid(out.pred_masks)          # (1, Q, H, W)
        probs = self.torch.nn.functional.interpolate(
            probs.float(), size=bgr.shape[:2], mode="bilinear",
            align_corners=False)[0]

        boxes = None
        pb = getattr(out, "pred_boxes", None)
        if pb is not None:
            boxes = pb[0].float()
            # post_process_instance_segmentation takes target_sizes, implying
            # the raw boxes are normalised. Rather than assume, the range is
            # measured and reported once, so a wrong guess is visible instead of
            # silently rescaling every box by a factor of a thousand.
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

    # ── the interface everything else uses ───────────────────────────────────

    def detect(self, bgr, binary=True):
        """Every instance above `conf`, as (masks, boxes, scores).

        The three are index-aligned: masks[k], boxes[k] and scores[k] describe
        the same instance. Anything pairing on boxes and then computing a region
        from masks depends on that, so they are returned together rather than by
        separate calls that could disagree about which queries were kept.
        """
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

    def assign_to_boxes(self, bgr, boxes, binary=True, path=None):
        """The two instances best matching two given boxes, or None.

        Used where a detector has already decided WHICH two animals are in
        question and only the segmentation is wanted from SAM 3. Concept
        segmentation returns instances in no particular order and says nothing
        about which belongs to which box, so the correspondence has to be made
        here; that is identity, not filtering.

        Greedy without replacement: the pair filter upstream keeps only boxes
        overlapping by more than iou_low, so the two always overlap and choosing
        the best instance for each independently could hand the same animal to
        both.
        """
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
    """IoU between a mask and a rectangle. For ASSIGNMENT only, never pairing."""
    x1, y1 = int(max(0, box[0])), int(max(0, box[1]))
    x2 = int(min(mask.shape[1], box[2]))
    y2 = int(min(mask.shape[0], box[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = float(mask[y1:y2, x1:x2].sum())
    union = float(mask.sum()) + (x2 - x1) * (y2 - y1) - inter
    return inter / max(union, 1.0)


def box_iou(a, b):
    """IoU of two rectangles. This is what PAIRING uses."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0
