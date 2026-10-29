"""Classification metrics computed from held-out labels and probabilities.

Keeping this separate from the UI lets the Model Report page recompute everything
live when the user moves the decision-threshold slider.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)


def metrics_at_threshold(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],  # [[TN, FP], [FN, TP]]
    }


def best_f1_threshold(y_true, y_prob):
    """Threshold that maximises F1 on the given data. Returns (threshold, f1)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    i = int(np.argmax(f1))
    return float(thresholds[i]), float(f1[i])
