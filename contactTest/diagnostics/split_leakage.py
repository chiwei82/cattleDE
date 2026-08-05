"""Diagnostic C — how much near-duplicate overlap is there across the splits?

Usage (from the repository root):

    python -m contactTest.diagnostics.split_leakage

Crops originate from video sampled at 1 fps, so consecutive saved frames of the
same pair can be visually near-identical. If such twins straddle the train/test
boundary the test score is inflated. interaction_prep.assign_videos_622 already
assigns whole videos to a split, so this should be clean — this script verifies
that rather than assuming it, and quantifies the residual similarity.

Two things are reported:

  1. whether any source_video appears in more than one split (a hard error if so)
  2. for every evaluation crop, the highest cosine similarity to any training
     crop, under the real video-disjoint split and under a deliberately shuffled
     split for contrast

A large gap between the two distributions is what a random split would have cost.
Similarity uses a 32x32 mean-removed intensity signature, which is sensitive to
near-duplicates and cheap enough to run over the whole set.
"""

import json
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from contactTest.src.data import load_records, split_records
from contactTest.src.utils import load_config

CONTACT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SIGNATURE_SIDE = 32


def signature(path):
    """Aspect-agnostic intensity signature, mean-removed and L2-normalised."""
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    small = cv2.resize(gray, (SIGNATURE_SIDE, SIGNATURE_SIDE),
                       interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    small -= small.mean()
    norm = np.linalg.norm(small)
    return small / norm if norm > 1e-6 else None


def signatures(records):
    out, kept = [], []
    for record in records:
        sig = signature(record["image_path"])
        if sig is not None:
            out.append(sig)
            kept.append(record)
    return np.stack(out) if out else np.zeros((0, SIGNATURE_SIDE ** 2)), kept


def nearest_neighbour(train_sig, eval_sig, chunk=512):
    """Max cosine similarity from each eval row to any train row."""
    if len(train_sig) == 0 or len(eval_sig) == 0:
        return np.zeros(len(eval_sig))
    best = np.empty(len(eval_sig), dtype=np.float32)
    for start in range(0, len(eval_sig), chunk):
        block = eval_sig[start:start + chunk]
        best[start:start + chunk] = (block @ train_sig.T).max(axis=1)
    return best


def summarise(name, sims):
    stats = {
        "n": int(len(sims)),
        "median": float(np.median(sims)) if len(sims) else float("nan"),
        "p90": float(np.percentile(sims, 90)) if len(sims) else float("nan"),
        "frac_above_0.95": float(np.mean(sims > 0.95)) if len(sims) else float("nan"),
        "frac_above_0.99": float(np.mean(sims > 0.99)) if len(sims) else float("nan"),
    }
    print(f"[diag-C] {name:<16s} median {stats['median']:.3f}  p90 {stats['p90']:.3f}  "
          f">0.95 {stats['frac_above_0.95']:.1%}  >0.99 {stats['frac_above_0.99']:.1%}")
    return stats


def main():
    cfg = load_config(os.path.join(CONTACT_ROOT, "config.yaml"))
    records = load_records(cfg)
    buckets = split_records(records)

    # 1. Structural check: is each video confined to a single split?
    video_splits = {}
    for record in records:
        video_splits.setdefault(record["source_video"], set()).add(record["split"])
    straddling = {v: sorted(s) for v, s in video_splits.items() if len(s) > 1}
    if straddling:
        print(f"[diag-C] ERROR: {len(straddling)} videos appear in multiple splits:")
        for video, splits in list(straddling.items())[:10]:
            print(f"           {video} -> {splits}")
    else:
        print(f"[diag-C] OK: all {len(video_splits)} source videos are confined "
              "to a single split")

    eval_split = "test" if buckets["test"] else "val"
    print(f"[diag-C] computing signatures (train + {eval_split})")
    train_sig, _ = signatures(buckets["train"])
    eval_sig, _ = signatures(buckets[eval_split])

    real = nearest_neighbour(train_sig, eval_sig)
    stats = {"video_disjoint": summarise("video-disjoint", real)}

    # 2. Contrast: what a random split would have produced on the same crops.
    rng = np.random.default_rng(int(cfg["random_seed"]))
    pooled = np.concatenate([train_sig, eval_sig], axis=0)
    order = rng.permutation(len(pooled))
    cut = len(train_sig)
    shuffled = nearest_neighbour(pooled[order[:cut]], pooled[order[cut:]])
    stats["random_split"] = summarise("random-split", shuffled)

    gap = stats["random_split"]["frac_above_0.95"] - stats["video_disjoint"]["frac_above_0.95"]
    stats["near_duplicate_gap"] = float(gap)
    if gap > 0.05:
        verdict = ("Random splitting would have leaked near-duplicates; the existing "
                   "video-disjoint split is doing real work — keep it.")
    else:
        verdict = ("Near-duplicate rates are similar either way; 1 fps sampling is "
                   "sparse enough that this dataset was never at much risk.")
    print(f"[diag-C] {verdict}")

    stats["videos_straddling_splits"] = straddling
    stats["verdict"] = verdict
    out_dir = os.path.join(CONTACT_ROOT, cfg["output"]["diag_dir"])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "split_leakage.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[diag-C] wrote {os.path.join(out_dir, 'split_leakage.json')}")


if __name__ == "__main__":
    main()
