"""
End-to-end evaluation of the FULL pipeline on real data:   python scripts/evaluate_pipeline.py

A) Unknown/real anomalies - Anoma sonar frames (held-out half B; half A is used ONLY to pick the fusion noise
   threshold, so the reported numbers are not tuned on the test frames). Object-level recall / precision.
B) Known debris - SIH 27-class test split through the whole pipeline: is the right class found?

Everything runs through SonarPipeline (QC -> ... -> fusion -> tracking), i.e. this measures the system, not a
single model. Frames taller than --max-side px are skipped for (B) to keep the run short (count reported).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import load_config  # noqa: E402
from backend.pipeline.runner import SonarPipeline  # noqa: E402
from models.autoencoder import patches as P  # noqa: E402


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def keep(det, t):
    """Fusion rule re-applied at a different noise threshold (records were produced with threshold 0)."""
    return det["detection_type"] == "known" or (det.get("anomaly_score") or 0) >= 0.5 or det["confidence"] >= t


def _center_in(box, det_box):
    cx, cy = (det_box[0] + det_box[2]) / 2, (det_box[1] + det_box[3]) / 2
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def score_frames(runs, thr, iou_thr, mode="region"):
    """mode='region': an annotated object is FOUND when >=1 kept detection's centre lies inside its box; detections
    inside an object are not false alarms (fragments of the same object). Right yardstick for a candidate generator
    that proposes small specks inside large annotated regions.  mode='iou': strict box IoU >= iou_thr."""
    tp = fn = fp = n_det = 0
    n_gt = 0
    for gts, dets in runs:
        dets = [d for d in dets if keep(d, thr)]
        n_det += len(dets)
        for g in gts:
            n_gt += 1
            hit = any((_center_in(g, d["bbox_global"]) if mode == "region" else iou(g, d["bbox_global"]) >= iou_thr) for d in dets)
            tp += int(hit)
            fn += int(not hit)
        for d in dets:
            ok = any((_center_in(g, d["bbox_global"]) if mode == "region" else iou(g, d["bbox_global"]) >= iou_thr) for g in gts)
            fp += int(not ok)
    p_ = (n_det - fp) / max(1, n_det)
    r_ = tp / max(1, n_gt)
    return {"mode": mode, "noise_threshold": thr, "n_gt": n_gt, "objects_found": tp, "objects_missed": fn,
            "detections": n_det, "false_alarms": fp, "precision": p_, "recall": r_,
            "f1": 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0, "false_alarms_per_frame": fp / max(1, len(runs))}


def run_frames(pipe, frames):
    out, times = [], []
    for path, boxes in frames:
        t0 = time.perf_counter()
        res = pipe.run(path, save_artifacts=False)
        times.append(time.perf_counter() - t0)
        out.append((boxes, res.detections))
    return out, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--known-samples", type=int, default=250)
    ap.add_argument("--max-side", type=int, default=1000)
    ap.add_argument("--fusion", default="learned", choices=["learned", "hand"], help="learned = logistic fusion model; hand = fixed weights")
    ap.add_argument("--out", default="pipeline_e2e_metrics.json")
    ap.add_argument("--merge-gap", type=int, default=None, help="override sa_cfar.merge_gap_px")
    ap.add_argument("--context", type=int, default=None, help="override autoencoder.context_px (0 = tight ROI)")
    ap.add_argument("--yolo-input", default=None, choices=["enhanced", "calibrated"], help="override yolo_seg.input")
    ap.add_argument("--ae-input", default=None, choices=["enhanced", "calibrated"], help="override autoencoder.input")
    ap.add_argument("--only", default="both", choices=["both", "known", "anomaly"])
    a = ap.parse_args()

    cfg = load_config()
    ov = {"fusion.noise_max_fused": 0.0}                                          # keep everything, threshold post-hoc
    if a.fusion == "hand":
        ov["fusion.learned_model"] = ""
    if a.merge_gap is not None:
        ov["sa_cfar.merge_gap_px"] = a.merge_gap
    if a.context is not None:
        ov["autoencoder.context_px"] = a.context
    if a.yolo_input is not None:
        ov["yolo_seg.input"] = a.yolo_input
    if a.ae_input is not None:
        ov["autoencoder.input"] = a.ae_input
    pipe = SonarPipeline(cfg.override(**ov))
    report = {"fusion_mode": a.fusion, "config_hash": cfg.hash, "model_version": pipe.model_version, "iou_threshold": a.iou}

    # ---------------- A) real anomalies (Anoma) ----------------
    val_a, test_b = P.split_by_group(P.list_frames("valid", ROOT / "samples" / "anoma"), [0.5, 0.5], seed=cfg.pipeline.seed)
    if a.only == "known":
        val_a, test_b = val_a[:1], test_b[:1]                      # skip part A (kept minimal so the report structure is unchanged)
    runs_a, _ = run_frames(pipe, val_a)
    sweep = [score_frames(runs_a, t, a.iou, "region") for t in np.round(np.arange(0.20, 0.71, 0.05), 2)]
    best = max(sweep, key=lambda r: (r["f1"], -r["noise_threshold"]))
    runs_b, times_b = run_frames(pipe, test_b)
    report["unknown_anomalies_anoma"] = {
        "note": "noise threshold selected on Anoma valid half A; metrics below are on the disjoint half B",
        "selected_noise_threshold": best["noise_threshold"], "validation_at_selected": best,
        "validation_sweep": sweep, "test_region_hit": score_frames(runs_b, best["noise_threshold"], a.iou, "region"),
        "test_strict_iou": score_frames(runs_b, best["noise_threshold"], a.iou, "iou"),
        "test_region_hit_at_default_config_threshold": score_frames(runs_b, float(cfg.fusion.noise_max_fused), a.iou, "region"),
        "why_two_metrics": "Anoma annotations are large regions (median 303 px of 640); SA-CFAR proposes small specks (median 10 px). "
                           "Region-hit is the fair yardstick; strict IoU is reported for transparency and is capped by that size gap.",
        "frames": len(test_b), "mean_seconds_per_frame": float(np.mean(times_b)),
        "yolo_active": pipe.yolo.available, "ae_active": pipe.scorer.available,
        "limitation": "SA-CFAR is a small-target detector: objects larger than its guard window are missed by the CFAR path",
    }

    # ---------------- B) known debris through the whole system ----------------
    if pipe.yolo.available and a.only != "anomaly":
        import cv2
        import yaml
        root = ROOT / "SIH_Dataset_27class_seg"
        names = yaml.safe_load((root / "data.yaml").read_text())["names"]
        names = list(names.values()) if isinstance(names, dict) else list(names)
        imgs = sorted((root / "test" / "images").glob("*.png"))
        rng = np.random.default_rng(cfg.pipeline.seed)
        rng.shuffle(imgs)
        sample, skipped = [], 0
        for p in imgs:
            im = cv2.imread(str(p))
            if max(im.shape[:2]) > a.max_side:
                skipped += 1
                continue
            sample.append(p)
            if len(sample) >= a.known_samples:
                break
        ok = wrong = unknown_only = nothing = extra = total_known = 0
        times = []
        for p in sample:
            im = cv2.imread(str(p))
            h, w = im.shape[:2]
            gts = []
            for line in (root / "test" / "labels" / (p.stem + ".txt")).read_text().strip().splitlines():
                v = line.split()
                pts = np.array(v[1:], np.float32).reshape(-1, 2)
                gts.append((names[int(v[0])], [pts[:, 0].min() * w, pts[:, 1].min() * h, pts[:, 0].max() * w, pts[:, 1].max() * h]))
            t0 = time.perf_counter()
            res = pipe.run(p, save_artifacts=False)
            times.append(time.perf_counter() - t0)
            known = [d for d in res.detections if d["detection_type"] == "known"]
            total_known += len(known)
            good = [d for d in known if any(d["class_name"] == g[0] and iou(g[1], d["bbox_global"]) >= a.iou for g in gts)]
            bad = [d for d in known if d not in good]
            ok += int(bool(good))
            wrong += int(not good and bool(known))
            unknown_only += int(not known and any(d["detection_type"] == "unknown" for d in res.detections))
            nothing += int(not res.detections)
            extra += len(bad)
        n = max(1, len(sample))
        report["known_debris_27class"] = {
            "images": len(sample), "skipped_large_frames": skipped, "correct_class_and_box_rate": ok / n,
            "wrong_known_only_rate": wrong / n, "unknown_only_rate": unknown_only / n, "nothing_found_rate": nothing / n,
            "known_precision": (total_known - extra) / max(1, total_known), "false_known_per_image": extra / n,
            "mean_seconds_per_image": float(np.mean(times)), "iou_threshold": a.iou}
    else:
        report["known_debris_27class"] = {"skipped": "YOLO11-Seg weights not available"}

    out = ROOT / "outputs" / "evaluation" / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    u = report["unknown_anomalies_anoma"]
    print(f"   validation half A @ selected thr: F1 {u['validation_at_selected']['f1']:.3f} (P {u['validation_at_selected']['precision']:.3f} R {u['validation_at_selected']['recall']:.3f}, FA/frame {u['validation_at_selected']['false_alarms_per_frame']:.1f})")
    for k in ("test_region_hit", "test_strict_iou"):
        print(f"A) Anoma test half B [{k}] @ noise thr {float(u['selected_noise_threshold']):.2f}:", {kk: round(float(v), 3) for kk, v in u[k].items() if isinstance(v, (int, float)) and kk not in ('noise_threshold',)})
    if "correct_class_and_box_rate" in report["known_debris_27class"]:
        print("B) known:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in report["known_debris_27class"].items()})


if __name__ == "__main__":
    main()
