"""
Fit and evaluate the acoustic-signature verifier.   python scripts/train_signature.py

Model    : multinomial logistic regression on standardised signature features (train split).
Question : given the class the detector CLAIMS, is the object's acoustic signature consistent with it?
Metrics  (all on the held-out, survey-disjoint TEST split; thresholds chosen on VAL):
  * 27-class accuracy from the signature alone (top-1 / top-3), family accuracy, and a majority-class baseline
  * verification AUROC: score = P(claimed class | signature); positives = true class claimed, negatives = a random WRONG class
  * error catching against the REAL YOLO11-Seg detector: of YOLO's wrong class calls, how many does the signature flag,
    and how many correct calls does it wrongly flag?
Outputs  : weights/signature_model.json, outputs/evaluation/signature_metrics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import load_config  # noqa: E402
from backend.pipeline.acoustic_signature import CLASS_TO_FAMILY, FEATURE_NAMES, extract_signature, to_vector  # noqa: E402


def load():
    d = np.load(ROOT / "outputs" / "evaluation" / "signature_features.npz", allow_pickle=True)
    return list(d["names"]), {s: (d[f"{s}_X"], d[f"{s}_y"], d[f"{s}_src"]) for s in ("train", "val", "test")}


def probs(clf, mean, std, X, n_classes):
    p = np.zeros((len(X), n_classes))
    p[:, clf.classes_] = clf.predict_proba((X - mean) / std)
    return p


def verification_auc(P, y, rng):
    wrong = np.array([rng.choice([c for c in range(P.shape[1]) if c != t]) for t in y])
    pos, neg = P[np.arange(len(y)), y], P[np.arange(len(y)), wrong]
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))


def main():
    cfg = load_config()
    names, sp = load()
    (Xtr, ytr, _), (Xva, yva, _), (Xte, yte, ste) = sp["train"], sp["val"], sp["test"]
    mean, std = Xtr.mean(0), np.where(Xtr.std(0) > 0, Xtr.std(0), 1.0)
    clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit((Xtr - mean) / std, ytr)
    K = len(names)
    Pte, Pva = probs(clf, mean, std, Xte, K), probs(clf, mean, std, Xva, K)

    fam = lambda i: CLASS_TO_FAMILY.get(names[i], "other")
    top1 = float((Pte.argmax(1) == yte).mean())
    top3 = float(np.mean([yte[i] in np.argsort(-Pte[i])[:3] for i in range(len(yte))]))
    fam_acc = float(np.mean([fam(Pte[i].argmax()) == fam(yte[i]) for i in range(len(yte))]))
    prior = np.bincount(ytr, minlength=K) / len(ytr)
    baseline = float((prior.argmax() == yte).mean())
    rng = np.random.default_rng(cfg.pipeline.seed)
    auc_te, auc_va = verification_auc(Pte, yte, rng), verification_auc(Pva, yva, np.random.default_rng(1))

    # per-class verification AUC (which classes does the signature actually help for?)
    per_class = {}
    for c in range(K):
        idx = np.where(yte == c)[0]
        if len(idx) >= 10:
            others = np.where(yte != c)[0]
            pos, neg = Pte[idx, c], Pte[others, c]
            per_class[names[c]] = {"n": int(len(idx)), "auc": float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))}

    # ---------- ablation: how much is genuinely ACOUSTIC (vs object geometry)? ----------
    geo = {"log_area", "elongation", "fill_ratio"}
    ablation = {}
    for label, idx in (("all_features", list(range(len(FEATURE_NAMES)))),
                       ("acoustic_only", [i for i, n in enumerate(FEATURE_NAMES) if n not in geo]),
                       ("geometry_only", [i for i, n in enumerate(FEATURE_NAMES) if n in geo])):
        m_, s_ = Xtr[:, idx].mean(0), np.where(Xtr[:, idx].std(0) > 0, Xtr[:, idx].std(0), 1.0)
        c_ = LogisticRegression(max_iter=3000).fit((Xtr[:, idx] - m_) / s_, ytr)
        P_ = np.zeros((len(Xte), K)); P_[:, c_.classes_] = c_.predict_proba((Xte[:, idx] - m_) / s_)
        ablation[label] = {"n_features": len(idx), "top1": float((P_.argmax(1) == yte).mean()),
                           "top3": float(np.mean([yte[i] in np.argsort(-P_[i])[:3] for i in range(len(yte))])),
                           "verification_auroc": verification_auc(P_, yte, np.random.default_rng(0))}

    # ---------- real-detector error catching ----------
    from models.yolo11_seg.inference import YOLOSegDetector
    det = YOLOSegDetector(cfg.path("yolo_seg.weights"), imgsz=cfg.yolo_seg.imgsz, conf=cfg.yolo_seg.conf, iou=cfg.yolo_seg.iou)
    root = ROOT / "SIH_Dataset_27class_seg"

    def yolo_rows(split):
        rows = []
        for lab in sorted((root / split / "labels").glob("*.txt")):
            im = cv2.imread(str(root / split / "images" / (lab.stem + ".png")))
            h, w = im.shape[:2]
            gt = []
            for line in lab.read_text().strip().splitlines():
                v = line.split()
                pts = np.array(v[1:], np.float32).reshape(-1, 2)
                gt.append((int(v[0]), [pts[:, 0].min() * w, pts[:, 1].min() * h, pts[:, 0].max() * w, pts[:, 1].max() * h]))
            for d in det.detect(im):
                best, gc = 0.0, None
                for c, b in gt:
                    ix1, iy1, ix2, iy2 = max(d["bbox"][0], b[0]), max(d["bbox"][1], b[1]), min(d["bbox"][2], b[2]), min(d["bbox"][3], b[3])
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    u = (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
                    if inter / max(u, 1e-9) > best:
                        best, gc = inter / max(u, 1e-9), c
                if best < 0.5:
                    continue                                      # not localised on an annotated object: not a class call
                pc = [k for k, n in enumerate(names) if n == d["class_name"]][0]
                f = extract_signature(im, d["bbox"], d["mask"], waterfall=False)
                if f is not None:
                    rows.append((pc, gc, to_vector(f)))
        return rows

    def consistency(rows):
        X = np.array([r[2] for r in rows])
        P = probs(clf, mean, std, X, K)
        claimed = np.array([r[0] for r in rows])
        rank = np.array([int((P[i] > P[i, claimed[i]]).sum()) for i in range(len(rows))])   # 0 = signature's top choice
        ratio = P[np.arange(len(rows)), claimed] / P.max(1)          # claimed class vs the signature's best guess
        return P[np.arange(len(rows)), claimed], rank, np.array([r[0] == r[1] for r in rows]), ratio

    pv, rv, okv, qv = consistency(yolo_rows("val"))
    pt, rt, okt, qt = consistency(yolo_rows("test"))

    # CONFIRMED needs the claimed class to be COMPETITIVE with the signature's best guess (ratio >= r), not merely in its
    # top-3: a call the signature rates at 1% while it is 98% sure of another class is not "confirmed". r is chosen on
    # VAL by Youden's J = (share of correct calls confirmed) - (share of wrong calls confirmed).
    best_r, best_j = 1.0, -1.0
    for r_ in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
        conf_v = qv >= r_
        j = (conf_v[okv].mean() if okv.any() else 0) - (conf_v[~okv].mean() if (~okv).any() else 0)
        if j > best_j:
            best_r, best_j = r_, j
    confirm_ratio = best_r

    # policy chosen on VAL: MISMATCH if the claimed class is outside the signature's top-`k` AND p < tau; pick the
    # setting that flags the most YOLO errors while wrongly flagging at most 10% of correct calls.
    best = None
    for k in (3, 5, 8, 12):
        for tau in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08):
            flag = (rv >= k) & (pv < tau)
            fp = flag[okv].mean() if okv.any() else 0
            tp = flag[~okv].mean() if (~okv).any() else 0
            if fp <= 0.10 and (best is None or tp > best[0]):
                best = (tp, k, tau, fp)
    tp_v, k, tau, fp_v = best if best else (0, 12, 0.005, 0)
    flag_t = (rt >= k) & (pt < tau)
    confirmed = qt >= confirm_ratio
    confirmed_precision = float(okt[confirmed].mean()) if confirmed.any() else None
    metrics = {
        "signature_only_classification": {"top1": top1, "top3": top3, "family_accuracy": fam_acc, "majority_class_baseline": baseline,
                                          "chance_top1": 1.0 / K, "n_test_objects": int(len(yte))},
        "verification_auroc": {"test": auc_te, "val": auc_va,
                               "meaning": "P(claimed class | signature): true class vs a random wrong class (0.5 = no information)"},
        "ablation": ablation,
        "per_class_auc": dict(sorted(per_class.items(), key=lambda kv: -kv[1]["auc"])),
        "detector_error_catching": {
            "detector": "YOLO11-Seg (real predictions on the test split)", "n_calls": int(len(rt)),
            "detector_accuracy": float(okt.mean()), "n_detector_errors": int((~okt).sum()),
            "policy": {"mismatch_if_rank_ge": int(k), "and_p_lt": tau, "confirm_min_ratio": float(confirm_ratio),
                       "chosen_on": "validation: mismatch flags most errors at <=10% of correct calls; confirm ratio maximises Youden J",
                       "val": {"errors_flagged": float(tp_v), "correct_flagged": float(fp_v), "confirm_youden_j": float(best_j)}},
            "test": {"errors_flagged": float(flag_t[~okt].mean()) if (~okt).any() else None,
                     "correct_calls_flagged": float(flag_t[okt].mean()),
                     "confirmed_share_of_correct": float(confirmed[okt].mean()),
                     "confirmed_share_of_wrong": float(confirmed[~okt].mean()) if (~okt).any() else None,
                     "precision_of_confirmed": confirmed_precision, "base_detector_accuracy": float(okt.mean())}},
        "material_family_check": {"note": "families assigned by me from class names (no material labels exist)",
                                  "median_impedance_contrast_idx": {f: float(np.median(Xtr[[CLASS_TO_FAMILY.get(names[c]) == f for c in ytr], 0]))
                                                                    for f in ("metal", "glass", "plastic", "rubber", "structure")}},
        "feature_importance": dict(zip(FEATURE_NAMES, [float(v) for v in np.abs(clf.coef_).mean(0)])),
        "features": FEATURE_NAMES,
    }
    model = {"features": FEATURE_NAMES, "classes": names, "mean": mean.tolist(), "std": std.tolist(),
             "coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist(), "class_index": clf.classes_.tolist(),
             "policy": metrics["detector_error_catching"]["policy"], "config_hash": cfg.hash}
    (ROOT / "weights" / "signature_model.json").write_text(json.dumps(model))
    (ROOT / "outputs" / "evaluation" / "signature_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("signature_only_classification", "verification_auroc", "detector_error_catching")}, indent=2))
    print("ablation:", {k: {a: round(b, 3) for a, b in v.items() if a != "n_features"} for k, v in ablation.items()})


if __name__ == "__main__":
    main()
