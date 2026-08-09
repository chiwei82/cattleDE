"""Predict contact heatmaps and run the annotation-free deletion test.

Usage (from the repository root):

    # overlay heatmaps for the highest-scoring test pairs
    python -m contactTest.infer_contact --split test --limit 40

    # self-consistency check that needs no contact annotation
    python -m contactTest.infer_contact --split test --deletion-test

Outputs go to contactTest/log/contact/vis/. Predictions are also written in
SOURCE-FRAME coordinates so they can be composited back onto the original video
frames without re-deriving the letterbox transform.

The deletion test asks whether the highlighted region is the evidence the model
actually uses. For each positive pair the top-k% of the heatmap is blanked and
the pair is re-scored; the same area is then blanked at a random location and at
the region's lowest-response location. If the model's own region causes a much
larger score drop than the controls, the heatmap is pointing at the evidence
rather than decorating a decision made elsewhere. This is not a substitute for
ground-truth contact boxes, but it is a real result obtainable with zero extra
annotation.
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contactTest.src.data import (ContactPairDataset, letterbox, load_records,
                                  split_records)
from contactTest.src.model import ContactMIL

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))


def load_checkpoint(path, device):
    state = torch.load(path, map_location=device)
    cfg = state["config"]
    model = ContactMIL(cfg).to(device)
    model.load_state_dict(state["model"])
    model.set_tau(cfg["model"]["tau_end"])
    model.eval()
    return model, cfg


def overlay(canvas_rgb, heat, alpha=0.55):
    """Blend a colour map of the heatmap over the letterboxed crop.

    Alpha is proportional to the heat rather than gated by a threshold. A fixed
    threshold hides everything when the model produces a very peaked heatmap —
    the response is real but occupies too few pixels to clear the cut-off — and
    a proportional blend lets weak structure fade in instead of vanishing.
    """
    heat_u8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    colour = cv2.applyColorMap(heat_u8, cv2.COLORMAP_INFERNO)
    colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)
    weight = np.clip(heat[..., None], 0.0, 1.0) * alpha
    return np.clip(canvas_rgb * (1 - weight) + colour * weight, 0, 255).astype(np.uint8)


def to_frame_coords(record, y, x, scale):
    """Map a canvas pixel back to source-frame coordinates.

    Letterboxing scales by `scale` and pastes at the canvas origin, and the crop
    itself starts at max(0, merged_x1), max(0, merged_y1) in the frame.
    """
    merged = record["merged"]
    ox, oy = max(0, int(merged[0])), max(0, int(merged[1]))
    return int(round(x / scale)) + ox, int(round(y / scale)) + oy


@torch.no_grad()
def run_inference(model, dataset, records, device, batch_size, num_workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    results = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        region = batch["region"].to(device, non_blocking=True)
        z, pooled, _ = model(image, region) if model.pooling != "keypoint" else \
            model(image, region, batch["kp_xy"].to(device), batch["kp_valid"].to(device))
        heat = (torch.sigmoid(z.float()) * region).cpu().numpy()[:, 0]
        scores = torch.sigmoid(pooled.float()).cpu().numpy()
        for i in range(len(scores)):
            idx = int(batch["index"][i])
            record = records[idx]
            h = heat[i]
            py, px = np.unravel_index(int(h.argmax()), h.shape)
            region_np = batch["region"][i, 0].numpy()
            lit = float((h > 0.05).sum())
            scale = float(batch["scale"][i])
            fx, fy = to_frame_coords(record, py, px, scale)
            results.append({
                "index": idx,
                "rel_image": record["rel_image"],
                "label": record["label"],
                "label_v2": record["label_v2"],
                "source_video": record["source_video"],
                "frame_number": record["frame_number"],
                "score": float(scores[i]),
                "peak_canvas_xy": (int(px), int(py)),
                "peak_frame_xy": (fx, fy),
                # Magnitude diagnostics: a peak near the -4.0 init floor
                # (sigmoid(-4) = 0.018) with almost no lit pixels means the
                # heatmap has collapsed to a point estimate rather than a region.
                "peak_heat": float(h.max()),
                "mean_heat_in_region": float(h.sum() / max(region_np.sum(), 1.0)),
                "lit_px": lit,
                "region_px": float(region_np.sum()),
                "heat": h,
                "region": region_np,
            })
    return results


@torch.no_grad()
def deletion_test(model, dataset, records, device, top_frac=0.2, max_pairs=200, seed=0):
    """Blank the predicted region and measure the score drop against controls."""
    rng = np.random.default_rng(seed)
    positives = [i for i, r in enumerate(records) if r["label"] == 1][:max_pairs]
    rows = []

    for idx in positives:
        sample = dataset[idx]
        image = sample["image"][None].to(device)
        region = sample["region"][None].to(device)

        kp = ((sample["kp_xy"][None].to(device), sample["kp_valid"][None].to(device))
              if model.pooling == "keypoint" else ())
        z, pooled, _ = model(image, region, *kp)
        base = float(torch.sigmoid(pooled.float())[0])
        heat = (torch.sigmoid(z.float()) * region).cpu().numpy()[0, 0]
        region_np = sample["region"].numpy()[0]

        inside = np.flatnonzero(region_np.ravel() > 0.5)
        if inside.size < 16:
            continue
        k = max(1, int(round(top_frac * inside.size)))
        values = heat.ravel()[inside]

        # Three blanking policies over the same number of pixels: the model's own
        # top-k, its bottom-k, and a random k inside the region.
        picks = {
            "model": inside[np.argsort(-values)[:k]],
            "worst": inside[np.argsort(values)[:k]],
            "random": rng.choice(inside, size=k, replace=False),
        }

        scores = {}
        for name, flat_idx in picks.items():
            occluded = image.clone()
            mask = torch.zeros(region_np.size, dtype=torch.bool)
            mask[torch.from_numpy(np.asarray(flat_idx))] = True
            mask = mask.reshape(region_np.shape).to(device)
            # Zero in normalised space == the ImageNet mean colour.
            occluded[0, :, mask] = 0.0
            _, pooled_occ, _ = model(occluded, region, *kp)
            scores[name] = float(torch.sigmoid(pooled_occ.float())[0])

        rows.append({
            "rel_image": records[idx]["rel_image"],
            "label_v2": records[idx]["label_v2"],
            "base": base,
            "drop_model": base - scores["model"],
            "drop_random": base - scores["random"],
            "drop_worst": base - scores["worst"],
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None,
                    help="defaults to contactTest/log/contact/contact_mil.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=40, help="number of overlays to write")
    ap.add_argument("--positives-only", action="store_true",
                    help="visualise only pairs labelled as interaction")
    ap.add_argument("--deletion-test", action="store_true")
    ap.add_argument("--raw", action="store_true",
                    help="draw absolute heat values. By default each heatmap is "
                         "divided by its own maximum, so the spatial pattern is "
                         "visible even when the absolute response is tiny; the "
                         "true magnitudes are always reported in predictions.csv")
    ap.add_argument("--top-frac", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.checkpoint or os.path.join(CONTACT_ROOT, "log", "contact", "contact_mil.pt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt} — run train_contact.py first")
    model, cfg = load_checkpoint(ckpt, device)

    records = split_records(load_records(cfg))[args.split]
    if not records:
        raise SystemExit(f"no labelled rows in split '{args.split}'")
    dataset = ContactPairDataset(records, cfg, train=False)

    vis_dir = os.path.join(CONTACT_ROOT, cfg["output"]["vis_dir"], args.split)
    os.makedirs(vis_dir, exist_ok=True)

    if args.deletion_test:
        rows = deletion_test(model, dataset, records, device, top_frac=args.top_frac)
        if not rows:
            raise SystemExit("no positive pairs available for the deletion test")
        out_csv = os.path.join(vis_dir, "deletion_test.csv")
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "n_pairs": len(rows),
            "top_frac": args.top_frac,
            "mean_drop_model": float(np.mean([r["drop_model"] for r in rows])),
            "mean_drop_random": float(np.mean([r["drop_random"] for r in rows])),
            "mean_drop_worst": float(np.mean([r["drop_worst"] for r in rows])),
        }
        summary["model_over_random"] = summary["mean_drop_model"] - summary["mean_drop_random"]
        with open(os.path.join(vis_dir, "deletion_test.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
        print("[infer] a mean_drop_model clearly above mean_drop_random means the "
              "heatmap marks the evidence the classifier relies on")
        return

    results = run_inference(model, dataset, records, device, args.batch_size, args.num_workers)
    if args.positives_only:
        results = [r for r in results if r["label"] == 1]
    results.sort(key=lambda r: -r["score"])
    results = results[:args.limit]

    # Magnitude report. If the peak sits near the sigmoid(-4) = 0.018 floor the
    # model never committed to any location; if the peak is high but only a
    # handful of pixels are lit, it collapsed to a point instead of a region.
    peaks = np.array([r["peak_heat"] for r in results])
    lit = np.array([r["lit_px"] for r in results])
    frac = np.array([r["lit_px"] / max(r["region_px"], 1.0) for r in results])
    print(f"[infer] peak heat   median {np.median(peaks):.3f}  "
          f"min {peaks.min():.3f}  max {peaks.max():.3f}")
    print(f"[infer] lit pixels  median {np.median(lit):.0f} px  "
          f"({np.median(frac):.2%} of the candidate region)")
    if np.median(peaks) < 0.05:
        print("[infer] WARNING: peaks are at the initialisation floor — the model "
              "did not commit to any location. Lower model.tau_end and retrain.")
    elif np.median(lit) < 100:
        print("[infer] WARNING: the response is a point, not a region. Lower "
              "model.tau_end and loss.lambda_sparsity, then retrain.")

    index_rows = []
    for rank, res in enumerate(results):
        record = records[res["index"]]
        bgr = cv2.imread(record["image_path"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        canvas, nh, nw, _ = letterbox(rgb, cfg["model"]["image_size"], 114)
        heat = res["heat"]
        if not args.raw:
            # Per-image contrast stretch, so the SHAPE of the response is
            # visible regardless of its absolute magnitude.
            heat = heat / max(heat.max(), 1e-6)
        blended = overlay(canvas, heat)
        px, py = res["peak_canvas_xy"]
        # Hollow ring rather than a filled cross: the peak is the pixel the
        # reader most needs to see, so the marker must circle it, not cover it.
        # Black under white keeps it legible over both hot and cold colours.
        cv2.circle(blended, (px, py), 9, (0, 0, 0), 3, lineType=cv2.LINE_AA)
        cv2.circle(blended, (px, py), 9, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        # Drop the letterbox padding bars; they carry no image evidence.
        blended = blended[:nh, :nw]

        name = f"{rank:03d}_{record['source_video'].replace('.mp4', '')}_" \
               f"{os.path.basename(record['rel_image'])}"
        cv2.imwrite(os.path.join(vis_dir, name), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        index_rows.append({
            "file": name,
            "rel_image": res["rel_image"],
            "label": res["label"],
            "label_v2": res["label_v2"],
            "score": f"{res['score']:.4f}",
            "peak_heat": f"{res['peak_heat']:.4f}",
            "mean_heat_in_region": f"{res['mean_heat_in_region']:.4f}",
            "lit_px": int(res["lit_px"]),
            "region_px": int(res["region_px"]),
            "peak_frame_x": res["peak_frame_xy"][0],
            "peak_frame_y": res["peak_frame_xy"][1],
            "source_video": res["source_video"],
            "frame_number": res["frame_number"],
        })

    with open(os.path.join(vis_dir, "predictions.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"[infer] wrote {len(index_rows)} overlays and predictions.csv to {vis_dir}")


if __name__ == "__main__":
    main()
