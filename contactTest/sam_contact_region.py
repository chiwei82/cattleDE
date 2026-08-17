"""Derive candidate contact pixels from two SAM instance masks, four ways.

Usage (from the repository root):

    python -m contactTest.sam_contact_region --limit 16
    python -m contactTest.sam_contact_region --split val --limit 40 --no-images
    python -m contactTest.sam_contact_region --limit 16 --touch-px 12

Writes to contactTest/log/sam_contact/<split>/ only.

The masks are the useful part of SAM here; its per-pixel confidence turned out
not to be (fragmented uncertainty maps, and a crop-mode "mutual claim" that
collapses once the prompts are correct). What remains is a geometry question:
given two good instance masks, which pixels are candidates for contact? The
answer is not unique, so all four readings are computed and drawn side by side.

    overlap   mask_i AND mask_j
              Where the projections literally coincide. Unambiguous but sparse,
              and in an overhead view it means occlusion as often as contact.

    gap       dist_to_i + dist_to_j <= touch_px
              The strip between the two surfaces, thresholded on the true local
              separation rather than on an arbitrary dilation radius. Includes
              the floor visible in the gap when the animals are close but apart.

    surface   the boundary pixels of each mask that lie within touch_px of the
              other animal, dilated into a strip.
              Contact happens ON a body surface, so this keeps skin rather than
              the air between. Closest to "which part of the animal is touching".

    dilated   dilate(mask_i, r) AND dilate(mask_j, r)
              The current default, kept for comparison. r has no physical
              meaning, which is exactly why the others are worth measuring.

`surface` is the one to look at first if the goal is per-pixel contact on the
animals; `gap` if the goal is the interface region between them.
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
C_HIT = (60, 235, 90)


def load_masks(record, shape):
    """Cached SAM masks for a pair, or None."""
    if not record.get("mask_path"):
        return None
    data = np.load(record["mask_path"])
    mi, mj = data["mi"].astype(np.uint8), data["mj"].astype(np.uint8)
    if mi.shape != shape:
        mi = cv2.resize(mi, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        mj = cv2.resize(mj, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mi, mj


def load_depth(record, shape):
    """Cached Depth Anything V2 map for a pair as (depth, range), or None.

    `range` is the robust 2nd-to-98th percentile spread of the map, which is what
    a depth tolerance is quoted as a fraction of. The relative checkpoints
    predict inverse depth up to an unknown scale and shift, so no absolute
    meaning survives; a fraction of the spread WITHIN one crop does, and that is
    the only comparison made.
    """
    if not record.get("depth_path"):
        return None
    data = np.load(record["depth_path"])
    depth = data["depth"].astype(np.float32)
    if depth.shape != shape:
        depth = cv2.resize(depth, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    spread = float(data["p98"]) - float(data["p2"])
    inverse = bool(int(data["inverse"])) if "inverse" in data else True
    return depth, max(spread, 1e-6), inverse


def farness(depth, inverse):
    """Depth rewritten so that a larger number always means further away.

    The relative checkpoints predict inverse depth (larger = nearer) and the
    metric ones predict metres (larger = further). Every comparison below is
    about which of two things is further from the lens, so the direction is
    normalised once here instead of being re-derived, and got wrong, at each use.
    """
    return -depth if inverse else depth


def body_reference(depth, boxes, inverse, disc_frac=0.10):
    """The animals' own depth, read where the image is known to be animal.

    The detector put a box round each cow, so the centre of that box is on the
    animal. That single fact is the anchor, and it is deliberately the ONLY
    thing assumed about which pixels are cattle.

    It replaces an earlier rule that took the nearest mode of the depth
    histogram to be the cattle. That is not true of this camera: the ceiling
    mount is angled, so railings, feed barriers and pipework routinely sit
    NEARER the lens than a cow's back, and the near mode is then hardware, not
    an animal. The far end carries no such ambiguity, which is why the floor is
    identified from that end and the cattle from the box centre.

    A small disc is used rather than the single centre pixel so that one noisy
    prediction cannot set the reference. Returns one depth per box, plus the
    disc radius used.
    """
    h, w = depth.shape
    refs = []
    for b in boxes:
        x1, y1, x2, y2 = (int(max(0, b[0])), int(max(0, b[1])),
                          int(min(w, b[2])), int(min(h, b[3])))
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        r = max(3, int(disc_frac * min(x2 - x1, y2 - y1)))
        disc = depth[max(0, cy - r):min(h, cy + r + 1),
                     max(0, cx - r):min(w, cx + r + 1)]
        if disc.size:
            refs.append(float(np.median(disc)))
    return refs or None


def ground_split(depth, inverse, boxes, spread, margin=0.05):
    """Separate the cattle from the ground, anchored on what is actually known.

    The floor is everything lying further from the lens than the animals by more
    than `margin` of the crop's depth spread. Both ends are pinned by something
    reliable: the animals' depth comes from the box centres, and "further than
    that" is the direction in which an angled ceiling camera can only be looking
    at ground.

    Nothing is claimed about what sits NEARER than the cattle. Railings and feed
    barriers live there, and they are not floor, so they are not put in the far
    class — but they are not contact either, which the two-sided `body` reading
    handles without needing to identify them.

    Returns (floor, refs, separation, floor_share), or None when the box centres
    give no usable reading. `floor_share` near zero means no ground is visible
    in this crop, so the separation is not a body-to-floor distance and should
    not be read as one.
    """
    refs = body_reference(depth, boxes, inverse)
    if refs is None:
        return None
    f = farness(depth, inverse)
    # Furthest of the two animals: a pixel only counts as ground when it is
    # beyond BOTH, otherwise the further cow's own back is called floor.
    ref_far = max(farness(np.float32(r), inverse) for r in refs)
    floor = f > (ref_far + margin * spread)
    if not floor.any():
        return floor, refs, 0.0, 0.0
    sep = float(np.median(f[floor]) - ref_far)
    return floor, refs, sep, float(floor.mean())


def nearest_value(mask, value):
    """For every pixel, `value` sampled at the nearest pixel inside `mask`.

    The distance transform's DIST_LABEL_PIXEL labels are turned into a lookup by
    reading each label at the mask pixel that owns it, which is exact and does
    not rely on the order in which OpenCV enumerates them.
    """
    if not mask.any():
        return None
    _, lab = cv2.distanceTransformWithLabels(
        (mask == 0).astype(np.uint8), cv2.DIST_L2, 5,
        labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(mask)
    lut = np.zeros(int(lab.max()) + 1, np.float32)
    lut[lab[ys, xs]] = value[ys, xs]
    return lut[lab]


DEPTH_STATS = ["body", "step", "pair"]


def depth_stats(mi, mj, depth, spread, inverse=True, boxes=None):
    """Three per-pixel depth readings, each in units of the crop's depth spread.

    Depth Anything V2 performs no amodal completion: it emits the range of the
    VISIBLE surface at each pixel and has no notion that anything is hidden. All
    three readings below are therefore plain lookups into that one map — nothing
    here tries to recover an occluded surface, because nothing can.

    body  distance from the NEARER of the two animals' own depths
          How far this pixel sits from cattle depth, where that depth is read at
          the box centres — the one place the detector guarantees is animal.
          This removes the pen surface: the band's habit of running along the
          ground between two animals, and of catching feet where they meet it.

          Two-sided on purpose. Rejecting only what lies FURTHER than the cattle
          would keep the railings and feed barriers that an angled ceiling
          camera sees nearer than a cow's back, and those are not contact
          either. Taking the smaller distance to either animal, rather than to
          one pooled depth, keeps it right when one cow stands nearer than the
          other.

          Anchored neither on the masks (which would inherit their errors) nor
          on the near mode of the histogram (which on this camera is as likely
          to be hardware as an animal). Needs `boxes`; without them there is no
          trustworthy anchor and the reading is not formed at all.

          It cannot distinguish a genuinely low contact from the floor, because
          there is nothing in a depth map that would: a head lowered to another
          animal's leg is at floor depth by definition. Watch selectivity.

    step  |grad depth|, per pixel
          Contact is where two surfaces MEET, so depth runs continuously across
          the junction. Occlusion is where one surface passes in front of
          another, which puts a step in the depth map at the silhouette edge.
          The size of that step is the signal, and it is available precisely
          because the model does not try to smooth occlusion away.

    pair  |depth of the nearest exclusive pixel of i - the same for j|
          Whether the two animals are at the same range at all. Unlike `step`
          this fires over the whole interior of an occluding overlap rather than
          only at its edge, which is the one thing the other two readings miss.
          Each side is read from the part of that animal the other mask does not
          cover: inside the intersection both masks contain the pixel, so
          sampling either one there returns the same single value and the
          difference would be identically zero — an occluding pair would score
          as perfectly agreeing.

    Returns None for a reading that cannot be formed, rather than a zero map that
    would silently pass everything.
    """
    out = {}

    refs = body_reference(depth, boxes, inverse) if boxes is not None else None
    out["body"] = (np.minimum.reduce([np.abs(depth - r) for r in refs]) / spread
                   if refs else None)

    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    # Sobel's 3x3 kernel carries a factor of 8 relative to a per-pixel slope.
    out["step"] = np.sqrt(gx * gx + gy * gy) / (8.0 * spread)

    out["pair"] = depth_disagreement(mi, mj, depth, spread)
    return out


def depth_disagreement(mi, mj, depth, spread):
    """Per pixel: how far apart the two animals' surfaces are in depth, in units
    of the crop's depth spread.

    At a given pixel there is only one depth value, so comparing "the depth
    there" against itself says nothing. The quantity that distinguishes contact
    from occlusion is whether the two SURFACES are at the same range where their
    silhouettes meet, so each animal's own depth is carried in from its nearest
    mask pixel and the two are differenced. Touching animals agree; one passing
    behind the other does not, however much their projections overlap.

    Each depth is read from the part of that animal the OTHER mask does not
    cover. This matters, and getting it wrong makes the gate useless precisely
    where it is needed: inside the intersection both masks contain the pixel, so
    sampling either mask there returns the same single value and the difference
    is identically zero — an occluding pair would be scored as perfectly
    agreeing. A depth map only ever holds the range of the FRONT surface, so the
    occluded animal's depth is not observable inside the intersection at all and
    has to be carried in from where that animal is actually visible.

    Returns None when either animal has no exclusive region, i.e. one silhouette
    lies entirely inside the other. There is then no independent reading of the
    hidden animal and the honest answer is that depth cannot judge this pair.
    """
    mi_only = (mi > 0) & ~(mj > 0)
    mj_only = (mj > 0) & ~(mi > 0)
    di = nearest_value(mi_only.astype(np.uint8), depth)
    dj = nearest_value(mj_only.astype(np.uint8), depth)
    if di is None or dj is None:
        return None
    return np.abs(di - dj) / spread


def distance_to(mask):
    """Distance from every pixel to the nearest pixel inside `mask`."""
    return cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 3)


def boundary(mask):
    """One-pixel outline of a binary mask."""
    er = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    return (mask.astype(bool) & ~er.astype(bool)).astype(np.uint8)


def dilated_band(mi, mj, dilate_px):
    """dilate(mi, r) AND dilate(mj, r) — the ROI, on its own.

    Callers that only want this reading should use it instead of
    contact_readings(...)["dilated"]: the four readings are computed in one pass,
    so asking for all of them costs two distance transforms and a boundary pass
    that are then thrown away, and it forces the caller to supply touch_px and
    strip_px, which have no bearing whatsoever on this band. That is how a file
    drawing only the green outline came to carry two parameters it never used.

    contact_readings calls this, so there is one definition of the band and the
    figures cannot drift from the numbers.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2)
    return (cv2.dilate(mi, k) > 0) & (cv2.dilate(mj, k) > 0)


def contact_readings(mi, mj, touch_px, dilate_px, strip_px,
                     depth=None, spread=None, gates=None, inverse=True,
                     boxes=None):
    """Four candidate definitions of the contact region.

    `gates` is {stat name: tolerance}; a pixel survives only where every named
    reading of depth_stats is at or below its tolerance. Gating can only ever
    REMOVE pixels, so it is worth keeping only if it removes area faster than it
    removes true contact points — which is what score_contact measures, and is
    not something to assume.
    """
    di, dj = distance_to(mi), distance_to(mj)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2)
    ks = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * strip_px + 1,) * 2)

    # Surface pixels of one animal that are within touch_px of the other.
    touch_i = (boundary(mi) > 0) & (dj <= touch_px)
    touch_j = (boundary(mj) > 0) & (di <= touch_px)
    surface = cv2.dilate((touch_i | touch_j).astype(np.uint8), ks) > 0

    out = {
        "overlap": (mi.astype(bool) & mj.astype(bool)),
        "gap": ((di + dj) <= touch_px),
        "surface": surface,
        "dilated": dilated_band(mi, mj, dilate_px),
    }

    if depth is not None and gates:
        stats_ = depth_stats(mi, mj, depth, spread, inverse, boxes)
        keep = None
        for name, tol in gates.items():
            s = stats_.get(name)
            if s is None:
                continue
            keep = (s <= tol) if keep is None else (keep & (s <= tol))
        if keep is not None:
            out = {n: (r & keep) for n, r in out.items()}
    return out


def stats(name, region, mi, mj):
    area = int(region.sum())
    n_comp = max(cv2.connectedComponents(region.astype(np.uint8))[0] - 1, 0)
    on_animal = float((region & (mi.astype(bool) | mj.astype(bool))).sum()) / area \
        if area else 0.0
    return {f"{name}_px": area, f"{name}_components": n_comp,
            f"{name}_on_animal": round(on_animal, 3),
            f"{name}_nonempty": int(area > 0)}


def panel(rgb, mi, mj, readings, order):
    h, w = rgb.shape[:2]
    base = rgb.astype(np.float32).copy()
    for m, c in ((mi, C_I), (mj, C_J)):
        sel = m > 0
        base[sel] = base[sel] * 0.55 + np.asarray(c, np.float32) * 0.45
    tiles = [base.astype(np.uint8)]

    for name in order:
        t = (rgb.astype(np.float32) * 0.45).astype(np.uint8)
        for m, c in ((mi, C_I), (mj, C_J)):
            sel = m > 0
            t[sel] = (t[sel] * 0.6 + np.asarray(c, np.float32) * 0.4).astype(np.uint8)
        t[readings[name]] = C_HIT
        cv2.putText(t, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(t, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)

    gap = np.full((h, 6, 3), 250, np.uint8)
    return np.hstack([x for pair in zip(tiles, [gap] * len(tiles))
                      for x in pair][:-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(CONTACT_ROOT, "config.yaml"))
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--balance", action="store_true",
                    help="half interaction / half not, for viewing only")
    ap.add_argument("--touch-px", type=int, default=10,
                    help="separation at or below which two surfaces count as touching")
    ap.add_argument("--dilate-px", type=int, default=15,
                    help="radius for the 'dilated' reading, for comparison only")
    ap.add_argument("--strip-px", type=int, default=6,
                    help="half-width of the 'surface' strip")
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()

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

    out_dir = os.path.join(CONTACT_ROOT, "log", "sam_contact", args.split)
    os.makedirs(out_dir, exist_ok=True)
    order = ["overlap", "gap", "surface", "dilated"]
    report, missing = [], 0

    for i, record in enumerate(records):
        bgr = cv2.imread(record["image_path"])
        if bgr is None:
            continue
        masks = load_masks(record, bgr.shape[:2])
        if masks is None:
            missing += 1
            continue
        mi, mj = masks

        readings = contact_readings(mi, mj, args.touch_px,
                                    args.dilate_px, args.strip_px)
        row = {"rel_image": record["rel_image"],
               "annotation": {-1: "unlabelled", 0: "no_interaction",
                              1: "interaction"}[record["label"]],
               "crop_px": int(bgr.shape[0] * bgr.shape[1])}
        for name in order:
            row.update(stats(name, readings[name], mi, mj))
        report.append(row)

        if not args.no_images:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            name = f"{i:03d}_{row['annotation'].replace(' ', '-')}_" \
                   f"{os.path.basename(record['rel_image'])}"
            cv2.imwrite(os.path.join(out_dir, name),
                        cv2.cvtColor(panel(rgb, mi, mj, readings, order),
                                     cv2.COLOR_RGB2BGR))

    if missing:
        raise SystemExit(
            f"{missing} pairs have no cached mask. Run precompute_masks.py first:\n"
            "  python -m contactTest.precompute_masks --split " + args.split)
    if not report:
        raise SystemExit("nothing processed")

    print(f"\n[contact] {len(report)} pairs, touch_px={args.touch_px}\n")
    print("panels: masks | " + " | ".join(order))
    print("green = the contact candidates that reading selects\n")
    print(f"{'reading':<10}{'nonempty':>10}{'area':>11}{'of crop':>9}"
          f"{'blobs':>7}{'on animal':>11}")
    summary = {}
    for name in order:
        ne = np.mean([r[f"{name}_nonempty"] for r in report])
        px = np.median([r[f"{name}_px"] for r in report])
        frac = np.median([r[f"{name}_px"] / max(r["crop_px"], 1) for r in report])
        comp = np.median([r[f"{name}_components"] for r in report])
        onan = np.median([r[f"{name}_on_animal"] for r in report])
        summary[name] = {"nonempty": float(ne), "px_median": float(px),
                         "frac_median": float(frac), "components_median": float(comp),
                         "on_animal_median": float(onan)}
        print(f"{name:<10}{ne:>9.0%}{px:>9.0f}px{frac:>9.1%}{comp:>7.1f}{onan:>10.0%}")

    print("\n'on animal' = the share of the region that lies on either mask.")
    print("Contact happens on a body surface, so the higher this is, the more of")
    print("what was selected is skin rather than the air between the animals.")

    with open(os.path.join(out_dir, "contact_report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(report)
    with open(os.path.join(out_dir, "contact_summary.json"), "w") as f:
        json.dump({"split": args.split, "n": len(report),
                   "touch_px": args.touch_px, "readings": summary}, f, indent=2)
    print(f"\n[contact] wrote {out_dir}")


if __name__ == "__main__":
    main()
