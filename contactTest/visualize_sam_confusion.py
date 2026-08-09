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

Two opposite mechanisms are computed for every pair and shown side by side.

WHOLE-IMAGE PROMPTS — read SAM's uncertainty
    SAM encodes the entire pair crop, then each box is used as a prompt. It can
    see both animals and tries to tell them apart, so the boundary between them
    is where its evidence runs out and the logit sits near 0:

        u_x(p)    = 4 * sigmoid(logit_x(p)) * (1 - sigmoid(logit_x(p)))
        strict(p) = u_i(p) * u_j(p)      both prompts undecided here
        loose(p)  = max(u_i(p), u_j(p))  either one undecided here

    `strict` suppresses animal/floor edges, where one prompt is unsure but the
    other confidently says "not mine". `loose` keeps them.

CROPPED PROMPTS — read SAM's confidence
    Each box is cut out and segmented on its own. Blind to the second animal,
    SAM absorbs whatever overlaps into "the object" — it looks like more of the
    same body. Doing that for both boxes, the pixels BOTH crops confidently
    claim are the pixels the two bodies share:

        claim_x(p) = sigmoid(logit from the crop around box x)
        mutual(p)  = claim_i(p) * claim_j(p)

    Outside a box that claim is 0 by construction, so `mutual` is automatically
    confined to the intersection of the two boxes. Where the whole-image reading
    fails because SAM was confident but wrong, this one still fires.

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
        print(f"[sam] segment-anything {model_type} on {device}")

    def __call__(self, bgr, boxes):
        """Returns a list of full-resolution logit maps, one per box."""
        self.predictor.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        out = []
        for b in boxes:
            logits, _, _ = self.predictor.predict(
                box=np.asarray(b, dtype=np.float32)[None],
                multimask_output=False, return_logits=True)
            out.append(logits[0].astype(np.float32))
        return out


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


def claim_logits(sam, bgr, boxes, pad=0):
    """Segment each box's crop in ISOLATION, then paste the result back.

    The opposite mechanism to the whole-image prompts above, and it exploits a
    weakness rather than a strength. Cropped to one animal's box, SAM can no
    longer see that a second animal is present, so wherever the two bodies
    overlap it has every reason to absorb the intruding region into "the object"
    — it looks like more of the same animal. Do that for both boxes and the
    pixels BOTH crops confidently claim are the pixels the two bodies share.

    So this reads SAM's confidence, not its uncertainty:

        claim_x(p) = sigmoid(logit from the crop around box x)
        mutual(p)  = claim_i(p) * claim_j(p)

    Outside a box the corresponding claim is 0 by construction, so `mutual` is
    automatically confined to the intersection of the two boxes.
    """
    h, w = bgr.shape[:2]
    out = []
    for (x1, y1, x2, y2) in boxes:
        x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
        x2p, y2p = min(w, x2 + pad), min(h, y2 + pad)
        sub = bgr[y1p:y2p, x1p:x2p]
        if sub.size == 0:
            out.append(np.full((h, w), -30.0, np.float32))
            continue
        # Prompt with the sub-image's own extent: "the object fills this view".
        sh, sw = sub.shape[:2]
        logits = sam(sub, [[0, 0, sw, sh]])[0]
        # -30 outside the crop: sigmoid(-30) ~ 0, i.e. "this crop makes no claim".
        full = np.full((h, w), -30.0, np.float32)
        full[y1p:y2p, x1p:x2p] = logits
        out.append(full)
    return out


def mutual_claim(logit_i, logit_j):
    """Probability that both isolated segmentations claim the same pixel."""
    sig = lambda l: 1.0 / (1.0 + np.exp(-np.clip(l, -30, 30)))
    return (sig(logit_i) * sig(logit_j)).astype(np.float32)


def colourise(gray01, rgb):
    """Tint a [0,1] map with a single hue, for overlaying."""
    return (gray01[..., None] * np.asarray(rgb, np.float32)).astype(np.uint8)


def _heat(img_rgb, m):
    heat = cv2.cvtColor(cv2.applyColorMap((np.clip(m, 0, 1) * 255).astype(np.uint8),
                                          cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    w = np.clip(m, 0, 1)[..., None] * 0.85
    return (img_rgb * (1 - w) + heat * w).astype(np.uint8)


def _mask_tile(img_rgb, mi, mj):
    out = img_rgb.astype(np.float32).copy()
    for m, c in ((mi, C_I), (mj, C_J)):
        sel = m > 0
        out[sel] = out[sel] * 0.45 + np.asarray(c, np.float32) * 0.55
    return out.astype(np.uint8)


def _map_tile(img_rgb, m, band=None):
    tile = _heat(img_rgb, m)
    if band is not None and band.any():
        cnt, _ = cv2.findContours(band.astype(np.uint8), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(tile, cnt, -1, (120, 255, 120), 1)
    peak_src = np.where(band, m, -1) if band is not None and band.any() else m
    py, px = np.unravel_index(int(peak_src.argmax()), m.shape)
    cv2.circle(tile, (px, py), 9, (0, 0, 0), 3, lineType=cv2.LINE_AA)
    cv2.circle(tile, (px, py), 9, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    return tile


def panel(img_rgb, boxes, whole, cropped, band):
    """One row comparing the two mechanisms.

    boxes | WHOLE: masks, strict | CROP: claims, mutual

    `whole` prompts SAM on the entire pair crop, so it tries to tell the animals
    apart and the signal lives in its uncertainty. `cropped` prompts SAM on each
    box in isolation, so it cannot tell them apart and the signal lives in what
    both crops confidently claim. They are opposite readings of the same scene
    and belong side by side.
    """
    h, w = img_rgb.shape[:2]
    tiles = [img_rgb.copy()]
    for b, c in zip(boxes, (C_I, C_J)):
        cv2.rectangle(tiles[0], (b[0], b[1]), (b[2], b[3]), c, 2)

    tiles.append(_mask_tile(img_rgb, whole["mi"], whole["mj"]))
    tiles.append(_map_tile(img_rgb, whole["strict"], band))
    tiles.append(_mask_tile(img_rgb, cropped["ci"] > 0.5, cropped["cj"] > 0.5))
    tiles.append(_map_tile(img_rgb, cropped["mutual"], None))

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
    ap.add_argument("--model-type", default="vit_b")
    ap.add_argument("--weights", default=None, help="overrides data.sam_weights")
    ap.add_argument("--crop-pad", type=int, default=0,
                    help="pixels of context added around each box in crop mode; "
                         "0 keeps SAM blind to the other animal, which is the "
                         "point of that mechanism")
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    weights = args.weights or cfg["data"].get("sam_weights", "sam_b.pt")

    # A plain random sample. The interaction label does not select the images
    # and does not weight them: SAM's uncertainty is a property of the picture,
    # so the sample should look like the data, not like a balanced test set.
    # require_label=False: every crop in the CSV, including the ones nobody has
    # annotated and the ones a human called badly cropped. SAM's behaviour on
    # those is exactly as relevant as on any other frame.
    rows = split_records(load_records(cfg, require_label=False))[args.split]
    if not rows:
        raise SystemExit(f"no rows in split '{args.split}'")
    rng = np.random.default_rng(int(cfg["random_seed"]))
    n = min(len(rows), args.limit)
    records = [rows[i] for i in sorted(rng.choice(len(rows), n, replace=False))]
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

    for i, record in enumerate(records):
        bgr = cv2.imread(record["image_path"])
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        boxes = relative_boxes(record, h, w)
        try:
            li, lj = sam(bgr, boxes)
        except Exception as err:                   # noqa: BLE001
            print(f"[sam] failed on {record['rel_image']}: {err}")
            continue

        mi, mj = (li > 0).astype(np.uint8), (lj > 0).astype(np.uint8)
        strict, loose = confusion_maps(li, lj)
        band = contact_band(mi, mj)
        whole = {"mi": mi, "mj": mj, "strict": strict, "loose": loose,
                 "overlap": (mi & mj).astype(np.float32)}

        # Second mechanism: each box segmented in isolation, then pasted back.
        try:
            ci_l, cj_l = claim_logits(sam, bgr, boxes, pad=args.crop_pad)
        except Exception as err:                   # noqa: BLE001
            print(f"[sam] crop mode failed on {record['rel_image']}: {err}")
            continue
        sig = lambda l: 1.0 / (1.0 + np.exp(-np.clip(l, -30, 30)))
        cropped = {"ci": sig(ci_l), "cj": sig(cj_l),
                   "mutual": mutual_claim(ci_l, cj_l)}

        stats = region_stats({"strict": strict, "loose": loose,
                              "overlap": whole["overlap"],
                              "mutual": cropped["mutual"]}, band)
        # The annotation is recorded in the CSV so a later analysis can join on
        # it, but it is not used here — not to select images, not to weight them,
        # not to judge the maps.
        annotation = {-1: "unlabelled", 0: "no_interaction", 1: "interaction"}[
            record["label"]]
        if record["label"] == 1 and record["label_v2"]:
            annotation = record["label_v2"]
        stats.update(rel_image=record["rel_image"], annotation=annotation,
                     source_video=record["source_video"])
        report.append(stats)

        if not args.no_images:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            row = panel(rgb, boxes, whole, cropped, band)
            name = f"{i:03d}_{os.path.basename(record['rel_image'])}"
            cv2.imwrite(os.path.join(out_dir, name), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
        if (i + 1) % 10 == 0:
            print(f"[sam] processed {i + 1}/{len(records)}")

    if not report:
        raise SystemExit("nothing processed")

    print("\n[sam] 面板順序：")
    print("      原圖+框 | 【整張】兩個 mask | 【整張】strict confusion"
          " | 【裁切】兩個 claim | 【裁切】mutual claim")
    print("      整張 = SAM 看得到兩頭牛，訊號在它的『不確定』")
    print("      裁切 = SAM 各自只看一個框，訊號在兩邊都『有把握地』claim 的地方")
    print("      綠框 = 兩個膨脹 mask 的交會帶，白圈 = 該圖最高點\n")

    # Everything below describes the maps themselves. The interaction label is
    # NOT used: SAM's uncertainty is a property of the image, so asking whether
    # it separates interacting from non-interacting pairs would impose on SAM a
    # semantics it does not have.
    summary = {}
    print(f"{'reading':<10}{'非空比例':>12}{'區域大小':>13}{'連通塊數':>12}{'峰值':>10}")
    for name in ("strict", "loose", "overlap", "mutual"):
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
        print(f"{name:<10}{nonempty.mean():>11.0%}{np.median(area):>10.0f} px"
              f"{np.median(comps):>12.1f}{np.median(peak):>10.3f}")
    print("\n（中位數；區域 = 該讀法 > 0.5 的像素）")

    print(f"\n{'reading':<10}{'區域落在交會帶內的比例':>24}")
    for name in ("strict", "loose", "overlap", "mutual"):
        print(f"{name:<10}{summary[name]['frac_in_band_median']:>22.0%}")

    s_ok, s_clean = summary["strict"]["nonempty_frac"], summary["strict"]["components_median"]
    if s_ok >= 0.7 and s_clean <= 3:
        verdict = ("USABLE - SAM yields a non-empty, coherent mutual-ambiguity "
                   "region on most images; cache it as an input plane.")
    elif s_ok >= 0.7:
        verdict = ("NOISY - regions exist but are fragmented; smooth them or keep "
                   "the largest component before use.")
    elif summary["loose"]["nonempty_frac"] >= 0.7:
        verdict = ("STRICT TOO TIGHT - mutual ambiguity is rare, so SAM is "
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
