"""The five-panel figure of visualize_sam_confusion.py, with SAM 3 in place of SAM 1.

Usage (from the repository root):

    python -m contactTest.visualize_sam3_confusion --balance --limit 24 \\
        --weights sam3.pt --dilate-px 22
    python -m contactTest.visualize_sam3_confusion --balance --limit 24 \\
        --weights sam3.pt --text cow --depth-tol 0.10 --depth-gate pair
    python -m contactTest.visualize_sam3_confusion --limit 24 --weights sam3.pt \\
        --prompt-mode boxes --prompt-source depth_points

Writes to contactTest/log/sam3_confusion/<split>/ only.

Same layout, same sampling and the same green band as the SAM 1 version, so the
two are read side by side. `--limit` and the seed pick the same crops in both.

WHAT CHANGES WITH SAM 3, AND WHAT CANNOT BE CARRIED OVER

SAM 1 and SAM 2 are class-agnostic. Given a box they return whichever coherent
region best fits it, and on this footage a uniform patch of pen floor is a more
coherent region than a high-contrast Holstein, so a slightly loose box is often
answered with the ground. Every prompt trick in the SAM 1 script — a centre
point, five points, depth-derived negative points — works around that.

SAM 3 can take different prompts. For example,
in our dataset, we can simply use text promt "cow" to segment a cow. 
since the model has provided general abilities
example:
```
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="<YOUR_TEXT_PROMPT>")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
```
after we get the boxes, we can mapping that back to bbox1 and bbox2

Those instances come back in no particular order, and nothing in them says which
belongs to bbox1 and which to bbox2, so they are assigned to the two detector
boxes by mask/box IoU (reusing the assignment in precompute_masks). The
assignment is for identity, not for filtering out extra animals. It is greedy
without replacement because the pair filter keeps only boxes overlapping by
IoU > 0.1, so choosing independently could give the same animal to both.

The panel built from UNCERTAINTY needs a per-pixel score, not a mask:

    u_x(p)    = 4 * sigmoid(logit_x) * (1 - sigmoid(logit_x))
    strict(p) = u_i(p) * u_j(p)

SAM 3 provides it. `pred_masks` is a float tensor of shape
(batch, num_queries, H, W) and sigmoid turns it into per-pixel probabilities, so
this panel carries over unchanged. That is the reason the text backend goes
through transformers rather than ultralytics: ultralytics' postprocess ends in
`masks = masks > mask_threshold`, discarding the only quantity these two panels
are made of, and `post_process_instance_segmentation` binarises as well. Neither
is used. Queries are kept by the documented score
`pred_logits.sigmoid() * presence_logits.sigmoid()`.

`--prompt-mode boxes` reaches for the same scores below the high-level
ultralytics API. If it cannot get them the run stops rather than drawing
something else: an earlier version substituted the binary mask overlap into
these panels when the scores were missing, which changed what panel 3 MEANS between runs. For a figure built to be read beside the SAM 1 one, a panel whose
meaning varies is worse than a panel that is absent.

Promptable Concept Segmentation (PCS) takes such prompts and returns 
segmentation masks and unique identities for all matching object instances
which means SAM3's output is a pixel-level mask.
example:
```
inputs = processor(images=image, text="ear", return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

# Instance segmentation masks
instance_masks = torch.sigmoid(outputs.pred_masks)  # [batch, num_queries, H, W]

# Semantic segmentation (single channel)
semantic_seg = outputs.semantic_seg  # [batch, 1, H, W]

print(f"Instance masks: {instance_masks.shape}")
print(f"Semantic segmentation: {semantic_seg.shape}")
```

PANELS

    1  crop with the two detector boxes
    2  the two instance masks
    3  the uncertainty product, with the contact band outlined in green
       (and in purple what the depth gate removed, if one is applied)
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contactTest.precompute_masks import _SAM3Text, depth_prompts
from contactTest.sam_contact_region import depth_stats, load_depth
from contactTest.src.data import load_records, relative_boxes, split_records
from contactTest.src.utils import load_config
from contactTest.visualize_sam_confusion import (_mask_tile, _map_tile,
                                                 contact_band)

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))
C_I, C_J = (214, 120, 42), (52, 104, 235)


class Sam3Boxes:
    """SAM 3 driven with the SAM 2 style box/point interface.

    Kept alongside the text backend so the checkpoint and the prompt type can be
    varied separately: a change measured with `--prompt-mode boxes` isolates
    what the newer weights are worth, and `--prompt-mode text` adds what the
    concept prompt is worth on top of that.
    """

    def __init__(self, weights):
        from ultralytics import SAM

        self.model = SAM(weights)
        print(f"[sam3] box/point prompts ({weights})")

    def _prompt_args(self, boxes, prompts):
        if prompts is not None:
            return ([[[float(x), float(y)] for x, y in p] for p, _ in prompts],
                    [[int(v) for v in lab] for _, lab in prompts])
        return ([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes],
                [1] * len(boxes))

    def _logits(self, bgr, boxes, prompts):
        """
        Raw mask logits
        """
        from ultralytics.utils import ops

        p = self.model.predictor
        p.set_image(bgr)
        im = getattr(p, "im", None)
        if im is None:
            return None
        pts, labs = self._prompt_args(boxes, prompts)
        pred, _ = p.inference(
            im,
            bboxes=np.asarray([list(map(float, b)) for b in boxes], np.float32),
            points=np.asarray(pts, np.float32),
            labels=np.asarray(labs, np.int32),
            multimask_output=False)
        # Decoder resolution -> the crop's own grid, the same rescaling the
        # thresholded path applies before it binarises.
        out = ops.scale_masks(pred[None].float(), bgr.shape[:2])[0]
        return [out[k].cpu().numpy().astype(np.float32) for k in range(len(boxes))]

    def __call__(self, bgr, boxes, prompts=None, binary=False):
        """Per-pixel probabilities for each box.

        No fallback to thresholded masks. An earlier version substituted the
        binary overlap into panels 3 and 5 when the logits could not be reached,
        which quietly changed what those panels MEAN from one run to the next —
        fatal for a figure whose whole purpose is to be read beside the SAM 1
        one. If the scores are unavailable the run stops and says so.
        """
        try:
            got = self._logits(bgr, boxes, prompts)
        except Exception as err:                       # noqa: BLE001
            raise RuntimeError(
                f"SAM 3 box mode could not produce per-pixel scores ({err}). "
                "Panel 3 is built from them, so there is nothing "
                "meaningful to draw. Use --prompt-mode text, whose backend "
                "returns pred_masks as floats by construction.") from err
        if got is None or len(got) < len(boxes):
            raise RuntimeError(
                "SAM 3 box mode returned no per-pixel scores; see --prompt-mode "
                "text, which does not depend on reaching past the high-level API.")
        # Logits, not probabilities: sigmoid before they are read as confidences,
        # matching what the SAM 1 script does.
        soft = [1.0 / (1.0 + np.exp(-g)) for g in got]
        return [(a > 0.5).astype(np.uint8) for a in soft] if binary else soft


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(CONTACT_ROOT, "config.yaml"))
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--balance", action="store_true",
                    help="half interaction / half not. Affects which images are "
                         "shown, never how they are measured")
    ap.add_argument("--weights", default="sam3.pt",
                    help="sam3.pt is gated: request access at "
                         "huggingface.co/facebook/sam3 and run 'hf auth login'")
    ap.add_argument("--prompt-mode", default="text", choices=["text", "boxes"],
                    help="text: concept segmentation, then instances assigned to "
                         "the boxes by IoU. boxes: SAM 2 style box+point prompts, "
                         "which isolates the checkpoint from the prompt type")
    ap.add_argument("--text", default="cow",
                    help="noun phrase for concept segmentation. Worth sweeping - "
                         "concept prompting is sensitive to wording")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--dilate-px", type=int, default=22,
                    help="radius for panel 3's green band. 22 is the operating "
                         "point score_contact reports at")
    ap.add_argument("--prompt-source", default="rgb", choices=["rgb", "depth_points"],
                    help="depth_points adds negative point prompts on the ground "
                         "and on the other animal. Applies to --prompt-mode boxes "
                         "only: a concept prompt has nowhere to put them")
    ap.add_argument("--depth-tol", type=float, default=None,
                    help="filter panel 3's band by depth at this tolerance")
    ap.add_argument("--depth-gate", default="pair",
                    help="comma-separated: pair, body, step")
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()

    if args.prompt_source == "depth_points" and args.prompt_mode != "boxes":
        raise SystemExit("--prompt-source depth_points needs --prompt-mode boxes; "
                         "a concept prompt takes no point labels")

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

    try:
        if args.prompt_mode == "text":
            seg = _SAM3Text(args.weights, args.text, args.conf)
        else:
            seg = Sam3Boxes(args.weights)
    except Exception as err:                       # noqa: BLE001
        raise SystemExit(
            f"could not load SAM 3 ({err}).\n"
            "  pip install -U ultralytics          # >= 8.3.237\n"
            "  request access at huggingface.co/facebook/sam3, then: hf auth login\n"
            "If a tokenizer error appears:\n"
            "  pip uninstall clip -y && "
            "pip install git+https://github.com/ultralytics/CLIP.git")

    out_dir = os.path.join(CONTACT_ROOT, "log", "sam3_confusion", args.split)
    os.makedirs(out_dir, exist_ok=True)
    prompt_rng = np.random.default_rng(int(cfg["random_seed"]))

    report, failed, no_depth = [], 0, 0
    n_floor = n_boxes = 0

    for i, record in enumerate(records):
        bgr = cv2.imread(record["image_path"])
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        boxes = relative_boxes(record, h, w)

        prompts = None
        if args.prompt_source == "depth_points":
            dep_p = load_depth(record, (h, w))
            if dep_p is None:
                no_depth += 1
            else:
                prompts, used = depth_prompts(dep_p[0], dep_p[1], dep_p[2],
                                              boxes, prompt_rng)
                n_floor += used
                n_boxes += len(boxes)

        try:
            # Both backends return per-pixel probabilities, so panels 3 and 5
            # mean the same thing whichever is used and whichever figure they
            # are compared against.
            got = (seg(bgr, boxes, path=record["image_path"], binary=False)
                   if args.prompt_mode == "text"
                   else seg(bgr, boxes, prompts=prompts, binary=False))
            whole_masks = list(got) if got is not None else None
        except Exception as err:                   # noqa: BLE001
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

        band = contact_band(mi, mj, args.dilate_px)
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

        # Panel 3: the same uncertainty product the SAM 1 script draws. One
        # definition, always.
        ui = 4.0 * whole_masks[0] * (1.0 - whole_masks[0])
        uj = 4.0 * whole_masks[1] * (1.0 - whole_masks[1])
        strict = ui * uj

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
            tiles = [rgb.copy()]
            for b, c in zip(boxes, (C_I, C_J)):
                cv2.rectangle(tiles[0], (b[0], b[1]), (b[2], b[3]), c, 2)
            pts = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
            tiles.append(_mask_tile(rgb, mi, mj, pts))
            tiles.append(_map_tile(rgb, strict, band, cut))

            gap = np.full((h, 6, 3), 250, np.uint8)
            row = np.hstack([t for pair in zip(tiles, [gap] * len(tiles))
                             for t in pair][:-1])
            name = f"{i:03d}_{report[-1]['annotation'].replace(' ', '-')}_" \
                   f"{os.path.basename(record['rel_image'])}"
            cv2.imwrite(os.path.join(out_dir, name),
                        cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

    if not report:
        raise SystemExit("nothing processed; every pair failed")

    print(f"\n[sam3] {len(report)} pairs, {failed} failed"
          + (f", {no_depth} without a cached depth map" if no_depth else ""))
    print(f"[sam3] prompt mode: {args.prompt_mode}"
          + (f", text={args.text!r}" if args.prompt_mode == "text" else ""))
    if n_boxes:
        print(f"[sam3] depth found visible ground in {n_floor}/{n_boxes} boxes "
              f"({n_floor / n_boxes:.0%})")


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
                   "prompt_mode": args.prompt_mode, "text": args.text,
                   "dilate_px": args.dilate_px,
                   }, f, indent=2)

    print(f"\n[sam3] wrote {out_dir}")
    print("[sam3] panel order:")
    print("      crop + boxes | instance masks | uncertainty + band")
    print("[sam3] green outline = the contact band; "
          + ("purple = what the depth gate removed"
             if args.depth_tol is not None else "no depth gate applied"))
    print("[sam3] to compare against SAM 1, run visualize_sam_confusion with the "
          "same --limit and --balance: the seed makes it draw the same crops")


if __name__ == "__main__":
    main()
