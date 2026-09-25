"""
Fit the decision-fusion weights from REAL labelled sonar:   python scripts/train_fusion.py

Data: Anoma valid half A (55 frames). Test half B is never touched here.
Each SA-CFAR candidate becomes one sample:
    features = [cfar, anomaly, recon_ratio, snr, shadow, log_area, contrast]      (backend/pipeline/fusion.py)
    label    = 1 if its centre lies inside an annotated anomaly box, else 0
A class-balanced logistic regression is fitted; the decision threshold is chosen from OUT-OF-FOLD predictions
(frame-grouped 5-fold CV, so no frame is in both fit and threshold selection), maximising object-level F1.
Output: weights/fusion_model.json  (coefficients + scaling + validated threshold + CV metrics)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from backend.config import load_config  # noqa: E402
from backend.pipeline.fusion import FEATURE_NAMES  # noqa: E402
from backend.pipeline.runner import SonarPipeline  # noqa: E402
from models.autoencoder import patches as P  # noqa: E402
from scripts.evaluate_pipeline import _center_in, score_frames  # noqa: E402


def collect(pipe, frames):
    X, y, g, recs_by_frame = [], [], [], []
    for gi, (path, boxes) in enumerate(frames):
        res = pipe.run(path, save_artifacts=False)
        recs = [d for d in res.detections if d["detection_type"] != "known"]
        for d in recs:
            X.append([d["features"][n] for n in FEATURE_NAMES])
            y.append(int(any(_center_in(b, d["bbox_global"]) for b in boxes)))
            g.append(gi)
        recs_by_frame.append((boxes, recs))
    return np.array(X), np.array(y), np.array(g), recs_by_frame


def fit(X, y):
    mean, std = X.mean(0), X.std(0)
    clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000)
    clf.fit((X - mean) / np.where(std > 0, std, 1.0), y)
    return clf, mean, std


def main():
    cfg = load_config()
    # collect with hand-set fusion and a permissive threshold so EVERY candidate is recorded as a sample
    pipe = SonarPipeline(cfg.override(**{"fusion.noise_max_fused": 0.0, "fusion.learned_model": ""}))
    val_a, _ = P.split_by_group(P.list_frames("valid", ROOT / "samples" / "anoma"), [0.5, 0.5], seed=cfg.pipeline.seed)
    X, y, g, by_frame = collect(pipe, val_a)
    print(f"samples {len(X)} | positives {int(y.sum())} ({y.mean():.1%}) | frames {len(val_a)}")

    # out-of-fold probabilities (frames are grouped -> no leakage between fit and threshold selection)
    oof = np.zeros(len(X))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        clf, mean, std = fit(X[tr], y[tr])
        oof[te] = clf.predict_proba((X[te] - mean) / np.where(std > 0, std, 1.0))[:, 1]

    # object-level F1 over thresholds, using the same scorer as the final evaluation
    pos, sweep = 0, []
    scored = []
    for gi, (boxes, recs) in enumerate(by_frame):
        r2 = []
        for d in recs:
            d = dict(d)
            d["confidence"] = float(oof[pos])
            d["anomaly_score"] = 0.0                                   # force the learned score to be the only gate
            r2.append(d)
            pos += 1
        scored.append((boxes, r2))
    for t in np.round(np.arange(0.30, 0.96, 0.02), 2):
        sweep.append(score_frames(scored, float(t), 0.3, "region"))
    best = max(sweep, key=lambda r: (r["f1"], r["noise_threshold"]))

    clf, mean, std = fit(X, y)
    out = {"features": FEATURE_NAMES, "mean": mean.tolist(), "std": std.tolist(), "coef": clf.coef_[0].tolist(),
           "intercept": float(clf.intercept_[0]), "threshold": float(best["noise_threshold"]),
           "trained_on": "Anoma valid half A, real SA-CFAR candidates; label = centre inside annotated anomaly box",
           "n_candidates": int(len(X)), "n_positive": int(y.sum()), "frames": len(val_a),
           "cv": {"folds": 5, "grouped_by": "frame", "objective": "object-level region F1", **{k: best[k] for k in (
               "precision", "recall", "f1", "false_alarms_per_frame")}},
           "config_hash": cfg.hash}
    dst = cfg.path("fusion.learned_model")
    dst.write_text(json.dumps(out, indent=2))
    print("coefficients:", {n: round(c, 3) for n, c in zip(FEATURE_NAMES, out["coef"])})
    print("threshold", out["threshold"], "| out-of-fold:", {k: round(out["cv"][k], 3) for k in ("precision", "recall", "f1", "false_alarms_per_frame")})
    print("saved", dst)


if __name__ == "__main__":
    main()
