"""Anomaly-detection metrics for the autoencoder (reconstruction quality + detection quality)."""

from __future__ import annotations

from typing import Dict

import numpy as np
from skimage.metrics import structural_similarity
from sklearn.metrics import average_precision_score, roc_auc_score


def mean_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM over (N,1,P,P) arrays in [0,1]."""
    if len(a) == 0:
        return float("nan")
    return float(np.mean([structural_similarity(x[0], y[0], data_range=1.0) for x, y in zip(a, b)]))


def detection_metrics(err_normal: np.ndarray, err_anom: np.ndarray, threshold: float) -> Dict[str, float]:
    """Errors are reconstruction errors; label 1 = anomaly. Positive prediction = error > threshold."""
    y = np.r_[np.zeros(len(err_normal)), np.ones(len(err_anom))]
    s = np.r_[err_normal, err_anom]
    pred = s > threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "n_normal": int(len(err_normal)), "n_anomaly": int(len(err_anom)), "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan"),
        "pr_auc": float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan"),
        "precision": prec, "recall": rec, "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def select_threshold(err_normal: np.ndarray, err_anom: np.ndarray, max_fpr: float = 0.10) -> Dict[str, float]:
    """Pick the threshold on VALIDATION errors: maximise F1 subject to FPR <= max_fpr."""
    cands = np.unique(np.r_[np.percentile(err_normal, np.linspace(50, 99.9, 200)),
                            np.percentile(err_anom, np.linspace(1, 99, 200))])
    best, best_f1 = None, -1.0
    for t in cands:
        m = detection_metrics(err_normal, err_anom, float(t))
        if m["false_positive_rate"] <= max_fpr and m["f1"] > best_f1:
            best, best_f1 = m, m["f1"]
    if best is None:                                            # nothing satisfies the FPR cap: use its quantile
        t = float(np.quantile(err_normal, 1.0 - max_fpr))
        best = detection_metrics(err_normal, err_anom, t)
    return best
