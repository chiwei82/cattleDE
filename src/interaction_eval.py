"""Per-sample prediction logging for the binary interaction model.

Collects one row per evaluated pair — image_path, predicted (T/F), truth (T/F) —
so you can (a) compute a confusion matrix and (b) open the actual crops behind
every false case. Label convention: T = interaction (label 1),
F = no_interaction (label 0).

Used by train/interaction_with_image.py: a PredictionCollector is attached to
the LightningModule, fed in test_step, and dumped in on_test_epoch_end.
"""

import csv
import os


class PredictionCollector:
    def __init__(self, out_dir, split="test"):
        self.out_dir = out_dir
        self.split = split
        self.rows = []          # list of {"image_path", "predicted", "truth"}

    @staticmethod
    def _tf(v):
        """0/1 (or bool) -> 'F'/'T'. 1 = interaction = T."""
        return "T" if int(v) == 1 else "F"

    def add(self, image_paths, preds, truths):
        """image_paths: iterable[str]; preds/truths: iterable of 0/1 (or bool).
        Lengths must match (one entry per sample in the batch)."""
        for path, pred, truth in zip(image_paths, preds, truths):
            self.rows.append({
                "image_path": path,
                "predicted": self._tf(pred),
                "truth": self._tf(truth),
            })

    def reset(self):
        self.rows = []

    # ── analysis ────────────────────────────────────────────────────────────
    def confusion(self):
        """Return {'TP','FP','TN','FN'} counts (positive = interaction = T)."""
        tp = fp = tn = fn = 0
        for r in self.rows:
            pred = r["predicted"] == "T"
            truth = r["truth"] == "T"
            if pred and truth:
                tp += 1
            elif pred and not truth:
                fp += 1
            elif (not pred) and (not truth):
                tn += 1
            else:
                fn += 1
        return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}

    def false_cases(self):
        """Rows where predicted != truth, each tagged with case = 'FP' | 'FN'.
        FP = predicted T but truth F; FN = predicted F but truth T."""
        out = []
        for r in self.rows:
            if r["predicted"] == r["truth"]:
                continue
            case = "FP" if r["predicted"] == "T" else "FN"
            out.append({**r, "case": case})
        return out

    def metrics(self):
        """Derived scores from the confusion matrix (interaction = positive).
        Includes the counts (TP/TN/FP/FN), the four rates (TPR/TNR/FPR/FNR) and
        balanced accuracy = (TPR + TNR) / 2."""
        c = self.confusion()
        tp, fp, tn, fn = c["TP"], c["FP"], c["TN"], c["FN"]
        total = tp + fp + tn + fn
        tpr = tp / (tp + fn) if (tp + fn) else 0.0   # recall / sensitivity
        tnr = tn / (tn + fp) if (tn + fp) else 0.0   # specificity
        fpr = fp / (fp + tn) if (fp + tn) else 0.0   # 1 - TNR
        fnr = fn / (fn + tp) if (fn + tp) else 0.0   # 1 - TPR
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (2 * precision * tpr / (precision + tpr)
              if (precision + tpr) else 0.0)
        accuracy = (tp + tn) / total if total else 0.0
        balanced_accuracy = (tpr + tnr) / 2
        return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "tpr": tpr, "tnr": tnr, "fpr": fpr, "fnr": fnr,
                "precision": precision, "recall": tpr, "f1": f1,
                "accuracy": accuracy, "balanced_accuracy": balanced_accuracy,
                "n": total}

    # ── output ──────────────────────────────────────────────────────────────
    def _dump(self, path, rows, fieldnames):
        os.makedirs(self.out_dir, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def write(self):
        """Write all predictions + a false-cases-only CSV. Returns both paths."""
        pred_csv = os.path.join(self.out_dir, f"{self.split}_predictions.csv")
        self._dump(pred_csv, self.rows, ["image_path", "predicted", "truth"])
        false_csv = os.path.join(self.out_dir, f"{self.split}_false_cases.csv")
        self._dump(false_csv, self.false_cases(),
                   ["image_path", "predicted", "truth", "case"])
        return pred_csv, false_csv

    def report(self):
        """Human-readable confusion matrix + derived metrics as a string."""
        c = self.confusion()
        m = self.metrics()
        return (
            f"[interaction] confusion matrix ({self.split}, positive = interaction/T)\n"
            f"                 truth T   truth F\n"
            f"    pred T   {c['TP']:8d}  {c['FP']:8d}   (TP, FP)\n"
            f"    pred F   {c['FN']:8d}  {c['TN']:8d}   (FN, TN)\n"
            f"    counts   TP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}  n={m['n']}\n"
            f"    rates    TPR={m['tpr']:.4f}  TNR={m['tnr']:.4f}  "
            f"FPR={m['fpr']:.4f}  FNR={m['fnr']:.4f}\n"
            f"    balanced_accuracy={m['balanced_accuracy']:.4f}  "
            f"accuracy={m['accuracy']:.4f}\n"
            f"    precision={m['precision']:.4f}  recall(TPR)={m['tpr']:.4f}  "
            f"f1(interaction)={m['f1']:.4f}\n"
            f"    false cases: {c['FP']} FP + {c['FN']} FN = "
            f"{c['FP'] + c['FN']}"
        )
