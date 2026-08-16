"""Locate, per image, the region where the two animals share pixels.

Usage (from the repository root):

    python -m contactTest.visualize_sam_confusion --limit 24
    python -m contactTest.visualize_sam_confusion --split val --limit 60 --no-images

Writes to contactTest/log/sam_confusion/<split>/ only.

This is an UNSUPERVISED image measurement, not a predictor. SAM knows nothing
about interaction; prompted with a box it answers "which pixels belong to this
object". Everything below is a property of the picture alone, so the annotation
takes no part in selecting the images, computing the maps, or judging them — it
is written to the CSV for later joins and nothing else.

For every pair, SAM's uncertainty is read from whole-image prompts.

WHOLE-IMAGE PROMPTS — read SAM's uncertainty
    SAM encodes the entire pair crop, then each box is used as a prompt. It can
    see both animals and tries to tell them apart, so the boundary between them
    is where its evidence runs out and the logit sits near 0:

        u_x(p)    = 4 * sigmoid(logit_x(p)) * (1 - sigmoid(logit_x(p)))
        strict(p) = u_i(p) * u_j(p)      both prompts undecided here
        loose(p)  = max(u_i(p), u_j(p))  either one undecided here

    `strict` suppresses animal/floor edges, where one prompt is unsure but the
    other confidently says "not mine". `loose` keeps them.

PROMPTING
    Every prompt is a box PLUS a positive point at its centre. SAM was trained
    on class-agnostic masks and has no concept of "cow": given only a box it
    looks for whatever coherent region best fits that rectangle, and a uniform
    patch of floor is a more coherent region than a high-contrast Holstein. A
    slightly loose box is therefore often answered with the ground. The positive
    point removes that failure by construction — the returned mask has to
    contain that pixel, and the floor does not. `--no-point` reverts to boxes
    alone.

    Reducing SAM's output to one mask offers only what the reference
    implementation recommends. `--mask-select single` (the default) uses
    multimask_output=False, suggested for unambiguous prompts, which a box plus
    a point is; `--mask-select sam_iou` takes the three candidates and keeps the
    one with the highest quality score the model predicts for itself. Selection
    by mask area was tried and dropped: nothing in SAM ranks candidates by size,
    so such a rule is an outside assumption about how much of a view an animal
    fills.

The report describes the maps as regions — non-empty, coherent or speckled, how
large, and where they sit. Whether a region coincides with contact is a separate
question this script deliberately does not ask.
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contactTest.precompute_masks import depth_prompts
from contactTest.sam_contact_region import depth_stats, load_depth
from contactTest.src.data import load_records, relative_boxes, split_records
from contactTest.src.utils import load_config

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))
C_I, C_J = (214, 120, 42), (52, 104, 235)          # RGB, cow i / cow j


class SamLogits:
    """Box-prompted SAM returning per-pixel logits, not just binary masks."""

    def __init__(self, weights, model_type="vit_b"):
        import torch
        from segment_anything import SamPredictor, sam_model_registry

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[model_type](checkpoint=weights).to(device)
        self.predictor = SamPredictor(sam)
        self.last_pick = []
        print(f"[sam] segment-anything {model_type} on {device}")

    def __call__(self, bgr, boxes, use_point=True, use_box=True, select="single",
                 n_points=1, point_spread=0.25, prompts=None):
        """Returns a list of full-resolution logit maps, one per box.

        A box alone only says "the object spans roughly this rectangle", and SAM
        has no notion of "cow" — it was trained on class-agnostic masks and
        simply looks for a coherent region that fits the box. A uniform patch of
        floor is a more coherent region than a high-contrast Holstein, so a
        slightly loose box can easily be answered with the ground. Adding a
        positive point pins the answer down: the mask MUST contain that pixel,
        and the floor does not.

        `select` offers only what the reference implementation recommends:

          single   multimask_output=False. Suggested for unambiguous prompts —
                   a box, or several prompts together — which is what a box plus
                   a point is.
          sam_iou  multimask_output=True, ranked by the quality score the model
                   predicts for each candidate. Suggested for ambiguous prompts
                   such as a lone click, and this is the documented way to
                   reduce the three candidates to one.

        Area-based selection rules were tried and removed: nothing in SAM ranks
        its candidates by size, so any such rule is an outside assumption about
        how much of a view an animal occupies.
        """
        self.predictor.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        self.last_pick = []
        out = []
        for b in boxes:
            x1, y1, x2, y2 = map(float, b)
            kwargs = {"return_logits": True}
            if use_box:
                kwargs["box"] = np.array([x1, y1, x2, y2], np.float32)[None]
            if prompts is not None:
                # Depth-derived prompts replace the fixed pattern entirely: some
                # of these carry label 0, which the pattern below cannot express.
                pp, ll = prompts[len(out)]
                kwargs["point_coords"] = np.asarray(pp, np.float32)
                kwargs["point_labels"] = np.asarray(ll, np.int32)
            elif use_point or not use_box:
                # A point is mandatory without a box: some prompt must be given.
                pts = _point_pattern(x1, y1, x2, y2, n_points, point_spread)
                kwargs["point_coords"] = pts
                # 1 = foreground. Every extra point is another pixel the mask is
                # required to contain, which is the documented way to grow a
                # mask that came back too tight — on these animals SAM segments
                # an individual coat patch precisely, so points spread across
                # neighbouring patches chain them into the whole body.
                kwargs["point_labels"] = np.ones(len(pts), np.int32)

            # One rule for both readings. They are meant to differ in exactly
            # one respect — what SAM can see — so prompt type and mask selection
            # are held identical; otherwise a difference between them cannot be
            # attributed to anything.
            if select == "sam_iou":
                logits, scores, _ = self.predictor.predict(multimask_output=True,
                                                          **kwargs)
                # Ranked by the model's own predicted quality — the mechanism
                # SAM trains an IoU token for.
                pick = int(np.argmax(scores))
                total = float(logits[0].size)
                areas = np.array([(l > 0).sum() for l in logits], np.float64)
                out.append(logits[pick].astype(np.float32))
                self.last_pick.append({"fill": float(areas[pick] / total),
                                       "sam_iou": float(scores[pick]),
                                       "n_cands": int(len(areas))})
            else:
                logits, scores, _ = self.predictor.predict(multimask_output=False,
                                                           **kwargs)
                out.append(logits[0].astype(np.float32))
                self.last_pick.append({"fill": float((logits[0] > 0).mean()),
                                       "sam_iou": float(scores[0]), "n_cands": 1})
        return out


def _point_pattern(x1, y1, x2, y2, n, spread):
    """`n` positive points centred on the box: 1, 5 (cross) or 9 (cross + X).

    Placement is a design choice, not something derivable — there is no reading
    of the geometry that says where on an animal a click belongs. It is kept
    near the centre, at a fraction of the SHORTER side, so the points stay on
    the body for a diagonal pose rather than sliding off the ends.
    """
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if n <= 1:
        return np.array([[cx, cy]], np.float32)
    r = spread * min(x2 - x1, y2 - y1) / 2.0
    cross = [(cx, cy), (cx - r, cy), (cx + r, cy), (cx, cy - r), (cx, cy + r)]
    if n <= 5:
        return np.array(cross[:n], np.float32)
    d = r * 0.7071
    diag = [(cx - d, cy - d), (cx + d, cy - d), (cx - d, cy + d), (cx + d, cy + d)]
    return np.array((cross + diag)[:n], np.float32)


def uncertainty(logit):
    """Bernoulli variance, rescaled to peak at 1 where the logit crosses 0."""
    p = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
    return (4.0 * p * (1.0 - p)).astype(np.float32)


def confusion_maps(logit_i, logit_j):
    """Two readings of the same uncertainty, at different strictness.

    strict = u_i * u_j
        Fires only where BOTH segmentations are undecided about the same pixel,
        so an animal/floor edge — where one is unsure and the other confidently
        says "not mine" — is suppressed. Precise, but it can also suppress a
        genuine contact whenever one of the two masks happens to be confident.

    loose  = max(u_i, u_j)
        Fires wherever EITHER is undecided, so it also lights up floor edges and
        occlusions. Use when a false alarm on the floor is acceptable and
        coverage of the real interface matters more.
    """
    ui, uj = uncertainty(logit_i), uncertainty(logit_j)
    return ui * uj, np.maximum(ui, uj)



def colourise(gray01, rgb):
    """Tint a [0,1] map with a single hue, for overlaying."""
    return (gray01[..., None] * np.asarray(rgb, np.float32)).astype(np.uint8)


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


def panel(img_rgb, boxes, whole, band, cut=None):
    """One row: the crop, the two instance masks, and the confusion map.

    SAM is prompted on the entire pair crop, so it tries to tell the animals
    apart and the signal lives in its uncertainty where the evidence runs out.
    """
    h, w = img_rgb.shape[:2]
    tiles = [img_rgb.copy()]
    for b, c in zip(boxes, (C_I, C_J)):
        cv2.rectangle(tiles[0], (b[0], b[1]), (b[2], b[3]), c, 2)

    pts = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    tiles.append(_mask_tile(img_rgb, whole["mi"], whole["mj"], pts))
    tiles.append(_map_tile(img_rgb, whole["strict"], band, cut))

    gap = np.full((h, 6, 3), 250, np.uint8)
    return np.hstack([t for pair in zip(tiles, [gap] * len(tiles))
                      for t in pair][:-1])


def contact_band(mi, mj, dilate=15):
    """Where the two dilated masks meet — the only place contact can be."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1,) * 2)
    return (cv2.dilate(mi, k) > 0) & (cv2.dilate(mj, k) > 0)


def region_stats(maps, band, thresh=0.5):
    """Describe each confusion map as a REGION, without reference to any label.

    Recorded per reading: whether it produced anything at all, how big it is,
    how fragmented (a coherent interface is one or two connected components,
    speckle is many), its peak, and how much of it falls in the band where the
    two dilated masks meet — the only place an interface between the animals
    can physically be.
    """
    n_band = int(band.sum())
    out = {"band_px": n_band}
    for name, m in maps.items():
        if m is None:
            out.update({f"{name}_nonempty": 0, f"{name}_area_px": 0,
                        f"{name}_components": 0, f"{name}_max": 0.0,
                        f"{name}_frac_in_band": 0.0})
            continue
        binary = (m > thresh).astype(np.uint8)
        area = int(binary.sum())
        n_comp, _ = cv2.connectedComponents(binary)
        out[f"{name}_nonempty"] = int(area > 0)
        out[f"{name}_area_px"] = area
        out[f"{name}_components"] = max(n_comp - 1, 0)      # label 0 is background
        out[f"{name}_max"] = float(m.max())
        out[f"{name}_frac_in_band"] = float((binary & band).sum() / area) if area else 0.0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(CONTACT_ROOT, "config.yaml"))
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=24, help="pairs to process")
    ap.add_argument("--balance", action="store_true",
                    help="draw half interaction / half no_interaction. Affects "
                         "which images are shown, never how they are measured")
    ap.add_argument("--model-type", default="vit_b")
    ap.add_argument("--weights", default=None, help="overrides data.sam_weights")
    ap.add_argument("--no-point", action="store_true",
                    help="box prompt only. A box says where the object roughly "
                         "is; a positive point says the mask must contain that "
                         "pixel, which is what stops SAM answering with floor")
    ap.add_argument("--points", type=int, default=1, choices=[1, 5, 9],
                    help="positive points per prompt: 1 = centre, 5 = centre + a "
                         "cross, 9 = + the diagonals. Each extra point is a pixel "
                         "the mask must contain, which is how a mask that came "
                         "back as a single coat patch is grown into the animal")
    ap.add_argument("--point-spread", type=float, default=0.25,
                    help="how far the extra points sit from the centre, as a "
                         "fraction of the box's SHORTER side. A placement choice, "
                         "not something the geometry dictates")
    ap.add_argument("--mask-select", default="single",
                    choices=["single", "sam_iou"],
                    help="how one mask is chosen, applied to BOTH readings. Only "
                         "the reference implementation's own options are offered. "
                         "'single' (default) = multimask_output=False, suggested "
                         "for unambiguous prompts, which a box plus a point is. "
                         "'sam_iou' = the three candidates ranked by the quality "
                         "score the model predicts for each")
    ap.add_argument("--dilate-px", type=int, default=15,
                    help="radius for panel 3's green band")
    ap.add_argument("--prompt-source", default="rgb",
                    choices=["rgb", "depth_points"],
                    help="depth_points adds NEGATIVE point prompts on the "
                         "ground and on the other animal, so depth changes the "
                         "SEGMENTATION in panels 2 and 3 rather than only "
                         "filtering the band afterwards. Needs "
                         "precompute_depth.py"),
    ap.add_argument("--depth-tol", type=float, default=None,
                    help="apply depth gates to panel 3's band at this "
                         "tolerance. Needs precompute_depth.py")
    ap.add_argument("--depth-gate", default="pair",
                    help="comma-separated: pair (animals at different range), "
                         "body (floor and feet), step (occlusion edges). "
                         "pair is the only one that scored above 1.0 for "
                         "selectivity across a range of tolerances")
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    weights = args.weights or cfg["data"].get("sam_weights", "sam_b.pt")

    # The annotation selects WHICH images to look at, and nothing else. Balanced
    # sampling makes the contact sheet easy to read by eye; it does not enter the
    # maps, the statistics or the verdict, all of which stay label-blind.
    # require_label=False keeps the unannotated and "not well-cropped" rows
    # available, so --balance off really does sample the whole split.
    rows = split_records(load_records(cfg, require_label=False))[args.split]
    if not rows:
        raise SystemExit(f"no rows in split '{args.split}'")
    rng = np.random.default_rng(int(cfg["random_seed"]))

    def draw(pool, k):
        k = min(len(pool), k)
        return [pool[i] for i in rng.choice(len(pool), k, replace=False)] if k else []

    if args.balance:
        pos = [r for r in rows if r["label"] == 1]
        neg = [r for r in rows if r["label"] == 0]
        half = args.limit // 2
        records = draw(pos, half) + draw(neg, args.limit - half)
        print(f"[sam] {len(records)} pairs from '{args.split}': "
              f"{sum(1 for r in records if r['label'] == 1)} interaction / "
              f"{sum(1 for r in records if r['label'] == 0)} no_interaction "
              f"(balanced for viewing only)")
    else:
        records = draw(rows, args.limit)
        print(f"[sam] {len(records)} pairs sampled at random from '{args.split}' "
              f"({len(rows)} available)")

    try:
        sam = SamLogits(weights, args.model_type)
        have_logits = True
    except Exception as err:                       # noqa: BLE001
        raise SystemExit(
            f"could not load segment-anything ({err}).\n"
            "Per-pixel logits need the reference package:\n"
            "  pip install git+https://github.com/facebookresearch/segment-anything.git\n"
            "  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth\n"
            "then pass --weights sam_vit_b_01ec64.pth")

    out_dir = os.path.join(CONTACT_ROOT, "log", "sam_confusion", args.split)
    os.makedirs(out_dir, exist_ok=True)
    report = []
    n_no_depth = n_floor_found = n_boxes_seen = 0
    # Separate from the sampling rng so that adding depth prompts does not
    # change WHICH pairs are drawn; the same --limit and seed give the same
    # crops with and without them, which is what makes the two comparable.
    prompt_rng = np.random.default_rng(int(cfg["random_seed"]))

    for i, record in enumerate(records):
        bgr = cv2.imread(record["image_path"])
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        boxes = relative_boxes(record, h, w)

        # Depth-derived prompts, when asked for. These change what SAM is told
        # before it segments, so they move panels 2 and 3 themselves; the
        # --depth-tol gate further down only filters the band afterwards. The
        # two are independent and can be used together.
        whole_prompts = None
        if args.prompt_source == "depth_points":
            dep_p = load_depth(record, bgr.shape[:2])
            if dep_p is None:
                n_no_depth += 1
            else:
                whole_prompts, used = depth_prompts(
                    dep_p[0], dep_p[1], dep_p[2], boxes, prompt_rng)
                n_floor_found += used
                n_boxes_seen += len(boxes)

        try:
            li, lj = sam(bgr, boxes, use_point=not args.no_point,
                         select=args.mask_select, n_points=args.points,
                         point_spread=args.point_spread,
                         prompts=whole_prompts)
        except Exception as err:                   # noqa: BLE001
            print(f"[sam] failed on {record['rel_image']}: {err}")
            continue

        mi, mj = (li > 0).astype(np.uint8), (lj > 0).astype(np.uint8)
        # How much of its own box each whole-image mask covers. An instance mask
        # should leave some of the box unclaimed — an axis-aligned box around an
        # animal always contains floor. A mask at ~1.0 has taken the box itself.
        whole_fill = []
        for m, (bx1, by1, bx2, by2) in zip((mi, mj), boxes):
            box_area = max((bx2 - bx1) * (by2 - by1), 1)
            whole_fill.append(float(m[by1:by2, bx1:bx2].sum()) / box_area)
        strict, loose = confusion_maps(li, lj)
        band = contact_band(mi, mj, args.dilate_px)

        # Panel 3's band, narrowed by depth. `cut` is what the gate removed and
        # is outlined separately, so the picture shows the band that would be
        # scored AND what paying for it cost — a gate that is working takes away
        # ground and occluded torso, one that is not takes away the place the
        # animals meet.
        cut = None
        if args.depth_tol is not None:
            dep = load_depth(record, bgr.shape[:2])
            if dep is None:
                n_no_depth += 1
            else:
                st = depth_stats(mi.astype(np.uint8), mj.astype(np.uint8),
                                 dep[0], dep[1], dep[2], boxes)
                keep = None
                for gname in args.depth_gate.split(","):
                    s_map = st.get(gname.strip())
                    if s_map is None:
                        continue
                    k_ = s_map <= args.depth_tol
                    keep = k_ if keep is None else (keep & k_)
                if keep is not None:
                    cut, band = band & ~keep, band & keep
        whole = {"mi": mi, "mj": mj, "strict": strict, "loose": loose,
                 "overlap": (mi & mj).astype(np.float32)}

        stats = region_stats({"strict": strict, "loose": loose,
                              "overlap": whole["overlap"]}, band)
        # The annotation is recorded in the CSV so a later analysis can join on
        # it, but it is not used here — not to select images, not to weight them,
        # not to judge the maps.
        annotation = {-1: "unlabelled", 0: "no_interaction", 1: "interaction"}[
            record["label"]]
        if record["label"] == 1 and record["label_v2"]:
            annotation = record["label_v2"]
        stats.update(rel_image=record["rel_image"], annotation=annotation,
                     source_video=record["source_video"],
                     whole_fill_i=round(whole_fill[0], 3),
                     whole_fill_j=round(whole_fill[1], 3))
        report.append(stats)

        if not args.no_images:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            row = panel(rgb, boxes, whole, band, cut)
            name = (f"{i:03d}_{annotation.replace(' ', '-')}_"
                    f"{os.path.basename(record['rel_image'])}")
            cv2.imwrite(os.path.join(out_dir, name), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
        if (i + 1) % 10 == 0:
            print(f"[sam] processed {i + 1}/{len(records)}")

    if not report:
        raise SystemExit("nothing processed")

    wf = np.array([r["whole_fill_i"] for r in report] +
                  [r["whole_fill_j"] for r in report], float)
    print(f"\n[sam] WHOLE: mask as a share of its own box: median {np.median(wf):.0%}  "
          f"p90 {np.percentile(wf, 90):.0%}  above 90%: {np.mean(wf > 0.9):.0%}")
    print("      An axis-aligned box around an animal always contains floor, so a")
    print("      healthy instance mask cannot fill it. Near 100% means the mask")
    print("      has taken the box itself rather than the animal.")
    print("\n[sam] panel order:")
    print("      crop + boxes | masks | strict confusion")
    print("      SAM sees both animals; the signal is in its uncertainty")
    print("      green outline = where the two dilated masks meet; "
          "white ring = the peak")
    if args.prompt_source == "depth_points":
        print(f"      SAM was prompted with depth: negative points on the "
              "ground and on the other animal")
        if n_boxes_seen:
            print(f"      visible ground found in {n_floor_found}/{n_boxes_seen} "
                  f"boxes ({n_floor_found / n_boxes_seen:.0%})")
    if args.depth_tol is not None:
        print(f"      purple outline = band pixels the depth gate "
              f"({args.depth_gate} at {args.depth_tol}) removed; the green "
              "line is what survives")
        if n_no_depth:
            print(f"      {n_no_depth} pairs drawn ungated - no cached depth "
                  "map; run precompute_depth.py")
    print()

    # Everything below describes the maps themselves. The interaction label is
    # NOT used: SAM's uncertainty is a property of the image, so asking whether
    # it separates interacting from non-interacting pairs would impose on SAM a
    # semantics it does not have.
    summary = {}
    print(f"{'reading':<10}{'nonempty':>10}{'area':>12}{'blobs':>8}{'peak':>9}")
    for name in ("strict", "loose", "overlap"):
        nonempty = np.array([r[f"{name}_nonempty"] for r in report], float)
        area = np.array([r[f"{name}_area_px"] for r in report], float)
        comps = np.array([r[f"{name}_components"] for r in report], float)
        peak = np.array([r[f"{name}_max"] for r in report], float)
        inband = np.array([r[f"{name}_frac_in_band"] for r in report], float)
        summary[name] = {"nonempty_frac": float(nonempty.mean()),
                         "area_px_median": float(np.median(area)),
                         "components_median": float(np.median(comps)),
                         "peak_median": float(np.median(peak)),
                         "frac_in_band_median": float(np.median(inband))}
        print(f"{name:<10}{nonempty.mean():>10.0%}{np.median(area):>9.0f} px"
              f"{np.median(comps):>8.1f}{np.median(peak):>9.3f}")
    print("\n(medians; 'area' counts the pixels where that reading exceeds 0.5)")

    print(f"\n{'reading':<10}{'share of region inside the band':>34}")
    for name in ("strict", "loose", "overlap"):
        print(f"{name:<10}{summary[name]['frac_in_band_median']:>33.0%}")

    # Degeneracy is checked first. Every region statistic below rewards a bigger
    # region — area, few components, coverage of the band — so a mask that has
    # swallowed its whole box scores well on all of them while being useless.
    # The size of the masks has to be trusted before their shape means anything.
    if np.median(wf) > 0.9:
        verdict = (f"DEGENERATE - whole-image masks fill {np.median(wf):.0%} of "
                   "their own boxes, i.e. they have taken the box rather than the "
                   "animal. Every region statistic below is inflated by that and "
                   "should not be compared across runs until it is fixed.")
        print(f"\n[sam] verdict: {verdict}")
        with open(os.path.join(out_dir, "confusion_report.csv"), "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(report[0].keys()))
            wtr.writeheader()
            wtr.writerows(report)
        print(f"[sam] wrote {out_dir}")
        return

    s_ok, s_clean = summary["strict"]["nonempty_frac"], summary["strict"]["components_median"]
    if s_ok >= 0.7 and s_clean <= 3:
        verdict = ("USABLE - SAM yields a non-empty, coherent ambiguity "
                   "region on most images; cache it as an input plane.")
    elif s_ok >= 0.7:
        verdict = ("NOISY - regions exist but are fragmented; smooth them or keep "
                   "the largest component before use.")
    elif summary["loose"]["nonempty_frac"] >= 0.7:
        verdict = ("STRICT TOO TIGHT - joint ambiguity is rare, so SAM is "
                   "confident about the boundary between the two animals. Use the "
                   "loose reading, or prompt with points to force a split.")
    else:
        verdict = ("NO SIGNAL - SAM is confident almost everywhere, most likely "
                   "because it returned one animal, or both as a single object, "
                   "rather than an uncertain boundary. Check the mask panel.")
    print(f"\n[sam] verdict: {verdict}")

    with open(os.path.join(out_dir, "confusion_report.csv"), "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        wtr.writeheader()
        wtr.writerows(report)
    with open(os.path.join(out_dir, "confusion_summary.json"), "w") as f:
        json.dump({"split": args.split, "n": len(report),
                   "readings": summary, "verdict": verdict}, f, indent=2)
    print(f"[sam] wrote {out_dir}")


if __name__ == "__main__":
    main()
