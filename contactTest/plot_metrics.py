"""Per-image distributions of the six metrics, as one figure.

Usage (from the repository root):

    python -m contactTest.plot_metrics --split train
    python -m contactTest.plot_metrics \\
        --csv log/evaluate/train/evaluation.csv \\
        --csv log/wholeframe/train/wholeframe.csv \\
        --label crop --label "whole frame"

Writes to contactTest/log/metric_plots/ only. Reads the per-image CSV that
evaluate_contact.py (or evaluate_wholeframe.py) already wrote — it runs no
model and needs no GPU.

WHY PER IMAGE

The printed summary gives one pooled number per metric. A pooled 0.88
sensitivity is consistent with every crop scoring about 0.88, and equally
consistent with four fifths scoring 1.0 and the rest scoring 0. Those are
different results and they call for different next steps, and only the
distribution separates them.

PANELS

    1  Sensitivity, histogram        piles at 0 and 1 are the thing to look for
    2  a_i, violin + strip           proposed area as a share of the crop
    3  Lift, violin on a log axis    a ratio, so it is read multiplicatively;
                                     1.0 is chance and is drawn
    4  Blind area, violin + strip    share of prediction covering no GT point
    5  Hit quality, violin + strip   cluster area over GT disc area; 1.0 drawn

The strip is every image drawn individually next to its violin, because a violin
is a smoothed estimate and with 81 crops the smoothing can invent shapes the
data does not have. The points are the data.

CROPS MARKED "NO CONTACT"

They are not a control group and are not held apart. A "no contact" mark is the
annotator's finding about that one image, so those images are counted with the
rest everywhere they have a value.

Panels 2 and 4 include them. Panels 1, 3 and 5 do not, and that is arithmetic
rather than a choice: sensitivity is covered/0, lift divides by it and hit
quality averages over covered points, so all three are UNDEFINED without GT
points — not zero. A NaN cannot be plotted.

In panel 4 they sit at 1.0, because with no GT points every cluster is blind by
construction. An annotated crop sitting at 1.0 there is something else entirely —
a genuine total miss — and it is the same image that appears at 0 in panel 1.
The caption gives the count so the two can be told apart.

Crops marked "skip" during annotation are not-well-cropped and never reach this
file: evaluate_contact drops them before anything is computed.
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

CONTACT_ROOT = os.path.abspath(os.path.dirname(__file__))

C_SERIES = [(0.20, 0.47, 0.75), (0.85, 0.42, 0.16), (0.35, 0.62, 0.35)]


def read_csv(path):
    """Per-image rows as float arrays, keeping NaN rather than dropping it."""
    cols = {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} has no rows")
    for k in ("sensitivity", "a_i", "lift", "blind_frac", "hit_quality",
              "no_contact"):
        vals = []
        for r in rows:
            v = r.get(k, "")
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                vals.append(np.nan)
        cols[k] = np.array(vals, float)
    cols["n"] = len(rows)
    return cols


def finite(v):
    return v[np.isfinite(v)]


def violin_strip(ax, series, labels, log=False, ref=None, ref_label=None):
    """A violin per series with every image drawn beside it."""
    rng = np.random.default_rng(0)
    data, keep = [], []
    for k, v in enumerate(series):
        v = finite(v)
        if log:
            v = v[v > 0]
        if len(v) >= 2:
            data.append(v)
            keep.append(k)
    if data:
        parts = ax.violinplot([np.log10(d) if log else d for d in data],
                              positions=range(len(data)), showextrema=False,
                              widths=0.8)
        for k, b in zip(keep, parts["bodies"]):
            b.set_facecolor(C_SERIES[k % len(C_SERIES)])
            b.set_alpha(0.35)
            b.set_edgecolor("none")
    for pos, (k, d) in enumerate(zip(keep, data)):
        y = np.log10(d) if log else d
        x = pos + rng.normal(0, 0.055, len(y))
        ax.scatter(x, y, s=11, color=C_SERIES[k % len(C_SERIES)], alpha=0.55,
                   linewidths=0)
        med = float(np.median(y))
        ax.hlines(med, pos - 0.34, pos + 0.34, color="0.15", lw=1.6, zorder=5)
        ax.annotate(f"{10 ** med:.2f}" if log else f"{med:.3f}",
                    (pos + 0.36, med), fontsize=7.5, va="center", color="0.15")
    if ref is not None:
        r = np.log10(ref) if log else ref
        ax.axhline(r, color="0.45", ls="--", lw=1.0, zorder=1)
        if ref_label:
            ax.annotate(ref_label, (len(data) - 0.5, r), fontsize=7,
                        color="0.45", va="bottom", ha="right")
    ax.set_xlim(-0.62, max(len(data) - 1, 0) + 0.62)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([labels[k] for k in keep], fontsize=8)
    if log:
        lo, hi = ax.get_ylim()
        ticks = np.arange(np.floor(lo), np.ceil(hi) + 1)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{10 ** t:g}" for t in ticks], fontsize=8)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="all",
                    choices=["train", "val", "test", "all"],
                    help="picks the default --csv, log/evaluate/<split>/"
                         "evaluation.csv. 'all' matches what annotate_contact "
                         "samples and evaluate_contact writes")
    ap.add_argument("--csv", action="append", default=None,
                    help="per-image CSV, relative to contactTest/. Repeat to "
                         "overlay runs; default is the evaluate_contact output "
                         "for --split")
    ap.add_argument("--label", action="append", default=None,
                    help="name for each --csv, in the same order")
    ap.add_argument("--out", default=None, help="output png")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    paths = args.csv or [os.path.join("log", "evaluate", args.split,
                                      "evaluation.csv")]
    labels = args.label or ([os.path.basename(os.path.dirname(p)) for p in paths]
                            if len(paths) > 1 else [args.split])
    if len(labels) != len(paths):
        raise SystemExit(f"{len(paths)} --csv but {len(labels)} --label")

    runs = []
    for p in paths:
        full = p if os.path.isabs(p) else os.path.join(CONTACT_ROOT, p)
        if not os.path.exists(full):
            raise SystemExit(f"no such CSV: {full}\n"
                             "run evaluate_contact.py first — this script only "
                             "plots what that wrote")
        runs.append(read_csv(full))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Five panels on a 2x3 grid: the sixth cell is removed rather than left
    # blank, so tight_layout does not reserve space for an axis that is not there.
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6))
    fig.delaxes(axes[1][2])
    used = [axes[0][0], axes[0][1], axes[0][2], axes[1][0], axes[1][1]]
    fig.suptitle(args.title or f"per-image metric distributions — "
                 + ", ".join(f"{l} (n={r['n']})" for l, r in zip(labels, runs)),
                 fontsize=12)

    # 1 — sensitivity histogram
    ax = axes[0][0]
    bins = np.linspace(0, 1, 21)
    for k, r in enumerate(runs):
        v = finite(r["sensitivity"])
        ax.hist(v, bins=bins, alpha=0.55, color=C_SERIES[k % len(C_SERIES)],
                label=f"{labels[k]}  n={len(v)}", edgecolor="white", linewidth=0.4)
    ax.set_title("1  Sensitivity (per image)", fontsize=10)
    ax.set_xlabel("covered GT points / GT points", fontsize=8)
    ax.set_ylabel("images", fontsize=8)
    ax.legend(fontsize=7.5)

    # 2 — a_i
    ax = axes[0][1]
    violin_strip(ax, [r["a_i"] for r in runs], labels)
    ax.set_title("2  a$_i$  — proposed area / image area", fontsize=10)
    ax.set_ylabel("share of the crop", fontsize=8)

    # 3 — lift, log axis
    ax = axes[0][2]
    violin_strip(ax, [r["lift"] for r in runs], labels, log=True,
                 ref=1.0, ref_label="chance")
    ax.set_title("3  Lift = Sensitivity / a$_i$   (log axis)", fontsize=10)
    ax.set_ylabel("x chance", fontsize=8)

    # 4 — blind area, every image in one distribution. Crops marked "no contact"
    # are not a separate arm and are not drawn apart; they do sit at 1.0 by
    # construction, which is said in the annotation rather than by splitting the
    # data.
    ax = axes[1][0]
    violin_strip(ax, [r["blind_frac"] for r in runs], labels)
    ax.set_title("4  Blind area — prediction covering no GT point", fontsize=10)
    ax.set_ylabel("share of proposed pixels", fontsize=8)
    n_at_one = sum(int(np.nansum(r["no_contact"] == 1)) for r in runs)
    if n_at_one:
        ax.annotate(f"{n_at_one} image(s) marked 'no contact' sit at 1.0:\n"
                    "with no GT points every cluster is blind.\n"
                    "Included here, as in every other number",
                    (0.03, 0.97), xycoords="axes fraction", fontsize=7,
                    color="0.4", va="top")

    # 5 — hit quality
    ax = axes[1][1]
    violin_strip(ax, [r["hit_quality"] for r in runs], labels, ref=1.0,
                 ref_label="cluster = GT discs")
    ax.set_title("5  Hit quality — cluster area / GT disc area", fontsize=10)
    ax.set_ylabel("x", fontsize=8)

    n_ctrl = [int(np.nansum(r["no_contact"] == 1)) for r in runs]
    for a in used:
        a.grid(alpha=0.18, linewidth=0.6)
        a.tick_params(labelsize=8)
    note = ("Crops marked 'no contact' have no GT points, so sensitivity, lift "
            "and hit quality are UNDEFINED for them and cannot be plotted in "
            "panels 1, 3 and 5; panels 2 and 4 include them, at 1.0 in panel 4 "
            "by construction. Crops marked 'skip' are not-well-cropped and were "
            "dropped before any number was computed. "
            + ", ".join(f"{l}: {c} of {r['n']} marked 'no contact'"
                        for l, c, r in zip(labels, n_ctrl, runs)))
    fig.text(0.5, 0.012, note, ha="center", fontsize=7.5, color="0.3", wrap=True)
    fig.tight_layout(rect=[0, 0.035, 1, 0.96])

    out_dir = os.path.join(CONTACT_ROOT, "log", "metric_plots")
    os.makedirs(out_dir, exist_ok=True)
    out = args.out or os.path.join(out_dir, f"metrics_{'_vs_'.join(
        l.replace(' ', '-') for l in labels)}.png")
    fig.savefig(out, dpi=150)
    print(f"\n[plot] wrote {out}")

    for l, r in zip(labels, runs):
        s, a = finite(r["sensitivity"]), finite(r["a_i"])
        print(f"\n[plot] {l}: {r['n']} images"
              f"  ({int(np.nansum(r['no_contact'] == 1))} marked 'no contact')")
        if len(s):
            print(f"        sensitivity  median {np.median(s):.3f}   "
                  f"at 0: {np.mean(s == 0):.0%}   at 1: {np.mean(s == 1):.0%}")
            if np.mean(s == 0) + np.mean(s == 1) > 0.5:
                print("        -> more than half the images are at one extreme, so")
                print("           the pooled mean describes almost none of them")
        if len(a):
            print(f"        a_i          median {np.median(a):.4f}   "
                  f"p90 {np.percentile(a, 90):.4f}")


if __name__ == "__main__":
    main()
