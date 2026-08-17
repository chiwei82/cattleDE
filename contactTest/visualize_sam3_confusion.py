"""Three-panel figure: SAM 3 instance masks and their uncertainty.

Usage (from the repository root):

    python -m contactTest.visualize_sam3_confusion --balance --limit 24
    python -m contactTest.visualize_sam3_confusion --balance --limit 24 \\
        --text cow --depth-tol 0.10 --depth-gate pair

Writes to contactTest/log/sam3_confusion/<split>/ only.

Sampling is seeded, so repeated runs draw the same crops and two settings can be
compared on identical images.

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
this panel carries over unchanged. That is why the model is reached through
transformers rather than ultralytics: ultralytics' postprocess ends in
`masks = masks > mask_threshold`, discarding the only quantity panel 3 is made
of, and `post_process_instance_segmentation` binarises as well. Neither is used.
Queries are kept by the documented score
`pred_logits.sigmoid() * presence_logits.sigmoid()`.

ONE SEGMENTER, ONE IMPORT

Everything here goes through `contactTest.sam3.Sam3`. A second backend used to
live in this file, driving the same checkpoint with SAM 2 style box+point
prompts through ultralytics, so that the weights and the prompt type could be
varied separately. It is gone, for two reasons. It reached past the high-level
API for the per-pixel scores, which broke on every ultralytics change; and a
box-prompted backend cannot answer the question this project is now asking —
whether SAM 3 can replace the detector — because it needs the detector's boxes
before it can be prompted at all.

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

from contactTest.sam3 import Sam3
from contactTest.sam_contact_region import depth_stats, load_depth
from contactTest.src.data import load_records, relative_boxes, split_records
from contactTest.src.utils import load_config
from contactTest.sam_contact_region import dilated_band

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))
C_I, C_J = (214, 120, 42), (52, 104, 235)


# Drawing helpers, moved here when the SAM 1 figure was removed. They are
# presentation only — no model, no measurement — so they live with the
# figure that uses them rather than in a module about geometry.

def _heat(img_rgb, m):
    heat = cv2.cvtColor(cv2.applyColorMap((np.clip(m, 0, 1) * 255).astype(np.uint8),
                                          cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    w = np.clip(m, 0, 1)[..., None] * 0.85
    return (img_rgb * (1 - w) + heat * w).astype(np.uint8)

def _mask_tile(img_rgb, mi, mj, points=None):
    """Masks overlaid, with the prompt points drawn on top.

    The point is what forces the mask to contain a given pixel, so when a mask
    comes back looking wrong the first thing to check is where its point landed.
    Showing it removes the guesswork.
    """
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
        # What the depth gate took out, outlined before the kept band so the
        # green line stays on top where the two touch.
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
    """The three-tile row: crop with boxes, the masks, the uncertainty + band.

    Exposed so anything producing (mi, mj, strict, band) can draw the same
    figure. wholeframe_pairs.py builds those from a whole frame rather than a
    crop, and the picture has to mean the same thing in both or they cannot be
    read against each other.
    """
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
    """strict(p) = u_i(p) * u_j(p), with u_x = 4 p (1 - p).

    Peaks where BOTH masks are undecided, which is where the evidence for
    telling the two animals apart runs out.
    """
    return (4.0 * pi * (1.0 - pi)) * (4.0 * pj * (1.0 - pj))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(CONTACT_ROOT, "config.yaml"))
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--balance", action="store_true",
                    help="half interaction / half not. Affects which images are "
                         "shown, never how they are measured")
    ap.add_argument("--weights", default=None,
                    help="Hugging Face id or a local snapshot DIRECTORY. "
                         "facebook/sam3 is gated: request access, then "
                         "'hf auth login'")
    ap.add_argument("--text", default="cow",
                    help="noun phrase for concept segmentation. Worth sweeping - "
                         "concept prompting is sensitive to wording")
    ap.add_argument("--conf", type=float, default=None,
                    help="SAM 3 score floor; default is data.sam3_conf "
                         "from config.yaml, which mirrors the value the "
                         "detector stage used")
    ap.add_argument("--dilate-px", type=int, default=22,
                    help="radius for panel 3's green band. 22 is the operating "
                         "point score_contact reports at")
    ap.add_argument("--depth-tol", type=float, default=None,
                    help="filter panel 3's band by depth at this tolerance")
    ap.add_argument("--depth-gate", default="pair",
                    help="comma-separated: pair, body, step")
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
    except Exception as err:                       # noqa: BLE001
        raise SystemExit(
            f"could not load SAM 3 ({err}).\n"
            "  request access at huggingface.co/facebook/sam3, then: hf auth login\n"
            "  pip install -U transformers")

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
            # binary=False: panel 3 is an uncertainty product and needs the
            # per-pixel probability, not a threshold of it.
            got = seg.assign_to_boxes(bgr, boxes, binary=False)
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

        # Panel 3: the same uncertainty product the SAM 1 script draws. One
        # definition, always.
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
    print("[sam3] panel order:")
    print("      crop + boxes | instance masks | uncertainty + band")
    print("[sam3] green outline = the contact band; "
          + ("purple = what the depth gate removed"
             if args.depth_tol is not None else "no depth gate applied"))
    print("[sam3] --limit and --balance are seeded, so repeated runs draw the "
          "same crops and settings can be compared on identical images")


if __name__ == "__main__":
    main()
