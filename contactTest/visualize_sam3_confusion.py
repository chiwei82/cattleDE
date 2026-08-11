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

SAM 3 takes the concept itself. `--text cow` asks for cattle, so the floor is
not a candidate answer at all, and promptable concept segmentation returns every
instance with its own identity rather than one mask per prompt. The returned
instances are assigned to the two detector boxes by mask/box IoU (reusing the
tested assignment in precompute_masks), because a pair crop routinely holds a
third animal at its edge.

The two panels built from UNCERTAINTY cannot be carried over unchanged. The SAM
1 script asks the predictor for raw logits and reads

    u_x(p)    = 4 * sigmoid(logit_x) * (1 - sigmoid(logit_x))
    strict(p) = u_i(p) * u_j(p)

which needs a per-pixel score, not a mask. The ultralytics SAM 3 wrapper returns
binary masks. This script probes for a soft output at runtime and says which it
got; when only binary masks are available, panels 3 and 5 show the mask OVERLAP
rather than an uncertainty map, and the tile is labelled so the two are never
confused. Panel 3's green band is unaffected either way — it is built from the
masks alone, and it is the thing score_contact actually measures.

PANELS

    1  crop with the two detector boxes
    2  the two instance masks
    3  overlap or uncertainty, with the contact band outlined in green
       (and in purple what the depth gate removed, if one is applied)
    4  each box segmented in isolation
    5  what both isolated runs claim
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

    def __call__(self, bgr, boxes, prompts=None):
        kw = {"bboxes": [list(map(float, b)) for b in boxes]}
        if prompts is not None:
            kw["points"] = [[[float(x), float(y)] for x, y in p] for p, _ in prompts]
            kw["labels"] = [[int(v) for v in lab] for _, lab in prompts]
        else:
            kw["points"] = [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes]
            kw["labels"] = [1] * len(boxes)
        res = self.model(bgr, verbose=False, **kw)
        return _extract(res, bgr.shape[:2], len(boxes))


def _extract(results, shape, n_expected):
    """Masks out of an ultralytics Results, kept float when the wrapper gives one.

    `masks.data` is normally a binary tensor. If a build ever returns something
    continuous it is passed through untouched, because the uncertainty panels
    can use it; `soft_scores` records which happened so the tiles can be
    labelled honestly rather than a thresholded mask being drawn as a score.
    """
    masks = getattr(results[0], "masks", None)
    if masks is None or masks.data is None or len(masks.data) < n_expected:
        return None, False
    out, soft = [], False
    for k in range(len(masks.data)):
        a = masks.data[k].cpu().numpy().astype(np.float32)
        u = np.unique(a[:: max(1, a.shape[0] // 64)])
        if u.size > 3:                      # more than {0, 1} (and maybe a stray)
            soft = True
        if a.shape != shape:
            a = cv2.resize(a, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        out.append(a)
    return out, soft


def crop_reading(seg_fn, bgr, boxes, pad):
    """Segment each box in isolation and paste the result back onto the crop.

    The SAM 1 version reads this as "what both isolated runs confidently claim":
    blind to the second animal, SAM absorbs whatever overlaps into the object,
    so the pixels both runs claim are the pixels the two bodies share. With
    binary masks the product of the two claims degenerates to their
    intersection, which is still that reading, only without a confidence to
    weight it by.
    """
    h, w = bgr.shape[:2]
    claims = []
    for b in boxes:
        x1 = int(max(0, b[0] - pad)); y1 = int(max(0, b[1] - pad))
        x2 = int(min(w, b[2] + pad)); y2 = int(min(h, b[3] + pad))
        full = np.zeros((h, w), np.float32)
        if x2 <= x1 + 4 or y2 <= y1 + 4:
            claims.append(full)
            continue
        sub = bgr[y1:y2, x1:x2]
        got = seg_fn(sub)
        if got is not None:
            full[y1:y2, x1:x2] = got
        claims.append(full)
    return claims


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
    ap.add_argument("--crop-pad", type=int, default=0,
                    help="context around each box in the isolated reading")
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

    report, failed, no_depth, soft_seen = [], 0, 0, []
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
            if args.prompt_mode == "text":
                got = seg(bgr, boxes, path=record["image_path"])
                whole_masks = ([g.astype(np.float32) for g in got]
                               if got is not None else None)
                soft = False
            else:
                whole_masks, soft = seg(bgr, boxes, prompts=prompts)
        except Exception as err:                   # noqa: BLE001
            print(f"[sam3] failed on {record['rel_image']}: {err}")
            failed += 1
            continue
        if whole_masks is None or len(whole_masks) < 2:
            failed += 1
            continue
        soft_seen.append(bool(soft))

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

        # Panel 3. With a soft output this is the same uncertainty product the
        # SAM 1 script draws; with binary masks it is their overlap, which is a
        # real quantity but a different one, so the tile says which.
        if soft:
            ui = 4.0 * whole_masks[0] * (1.0 - whole_masks[0])
            uj = 4.0 * whole_masks[1] * (1.0 - whole_masks[1])
            strict = ui * uj
        else:
            strict = (mi & mj).astype(np.float32)

        def seg_crop(sub, _seg=seg):
            if args.prompt_mode == "text":
                g = _seg(sub, [(0, 0, sub.shape[1], sub.shape[0])],
                         path=None)
                return g[0].astype(np.float32) if g is not None else None
            g, _ = _seg(sub, [(0, 0, sub.shape[1], sub.shape[0])])
            return g[0] if g is not None else None

        try:
            ci, cj = crop_reading(seg_crop, bgr, boxes, args.crop_pad)
        except Exception as err:                   # noqa: BLE001
            print(f"[sam3] crop reading failed on {record['rel_image']}: {err}")
            ci = cj = np.zeros((h, w), np.float32)
        mutual = ci * cj

        report.append({
            "rel_image": record["rel_image"],
            "annotation": {-1: "unlabelled", 0: "no_interaction",
                           1: "interaction"}[record["label"]],
            "soft_scores": int(bool(soft)),
            "mi_px": int(mi.sum()), "mj_px": int(mj.sum()),
            "overlap_px": int((mi & mj).sum()),
            "band_px": int(band.sum()),
            "band_frac": round(float(band.sum()) / (h * w), 4),
            "cut_px": int(cut.sum()) if cut is not None else 0,
            "mutual_px": int((mutual > 0.5).sum()),
        })

        if not args.no_images:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tiles = [rgb.copy()]
            for b, c in zip(boxes, (C_I, C_J)):
                cv2.rectangle(tiles[0], (b[0], b[1]), (b[2], b[3]), c, 2)
            pts = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
            tiles.append(_mask_tile(rgb, mi, mj, pts))
            t3 = _map_tile(rgb, strict, band, cut)
            label = "uncertainty" if soft else "mask overlap (binary masks)"
            for col, th in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                cv2.putText(t3, label, (6, t3.shape[0] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, th, cv2.LINE_AA)
            tiles.append(t3)
            tiles.append(_mask_tile(rgb, (ci > 0.5).astype(np.uint8),
                                    (cj > 0.5).astype(np.uint8), pts))
            tiles.append(_map_tile(rgb, mutual, None))

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

    if soft_seen and not any(soft_seen):
        print("\n[sam3] the wrapper returned BINARY masks on every pair, so no")
        print("[sam3] per-pixel score exists. Panels 3 and 5 therefore show mask")
        print("[sam3] overlap, not the uncertainty product the SAM 1 figure")
        print("[sam3] draws; they are labelled accordingly and must not be")
        print("[sam3] compared with that figure's panels 3 and 5 as like for")
        print("[sam3] like. Panels 1, 2 and the green band ARE comparable.")
    elif any(soft_seen):
        print(f"\n[sam3] soft scores available on {np.mean(soft_seen):.0%} of "
              "pairs; panel 3 is the same uncertainty product as the SAM 1 figure")

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
                   "soft_scores_any": bool(any(soft_seen))}, f, indent=2)

    print(f"\n[sam3] wrote {out_dir}")
    print("[sam3] panel order:")
    print("      crop + boxes | instance masks | overlap/uncertainty + band"
          " | isolated crops | both claims")
    print("[sam3] green outline = the contact band; "
          + ("purple = what the depth gate removed"
             if args.depth_tol is not None else "no depth gate applied"))
    print("[sam3] to compare against SAM 1, run visualize_sam_confusion with the "
          "same --limit and --balance: the seed makes it draw the same crops")


if __name__ == "__main__":
    main()
