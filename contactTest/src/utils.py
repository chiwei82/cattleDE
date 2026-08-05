"""Small helpers shared by training and diagnostics.

Deliberately free of torch imports: diagnostics/geometry_baseline.py is the gate
that decides whether training is worth running at all, so it must work in a plain
numpy environment before any GPU stack is installed.
"""

import numpy as np
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def roc_auc(labels, scores):
    """Rank-based ROC AUC with tie correction.

    Implemented directly so the training loop does not pull in sklearn; the
    diagnostics use sklearn for models but share this metric for comparability.
    Returns NaN when one class is absent.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average the ranks inside each tie group, so constant scores give 0.5.
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]

    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
