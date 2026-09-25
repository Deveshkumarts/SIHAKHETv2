"""MODEL EVALUATION view: YOLO11-Seg, Autoencoder and LSTM metrics read from outputs/evaluation/*.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

EVAL_DIR = Path(__file__).resolve().parents[2] / "outputs" / "evaluation"


def _load(name: str) -> Optional[Dict[str, Any]]:
    p = EVAL_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def _fmt(v, nd=3):
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _cards(items, cols=4):
    row = st.columns(cols)
    for i, (label, val) in enumerate(items):
        row[i % cols].metric(label, val)


def yolo_panel() -> None:
    st.subheader("YOLO11-Seg  (known debris: box + class + mask)")
    r = _load("yolo11_seg_metrics.json")
    if not r:
        st.warning("Not evaluated yet - run `python -m models.yolo11_seg.evaluate`.")
        return
    b, m, iou, lat, mod = r["box"], r["mask"], r["mask_iou"], r["latency"], r["model"]
    _cards([("Precision", _fmt(b["precision"])), ("Recall", _fmt(b["recall"])), ("F1", _fmt(b["f1"])),
            ("mAP@50", _fmt(b["map50"])), ("mAP@50:95", _fmt(b["map50_95"])), ("Mask mAP@50", _fmt(m["map50"])),
            ("Mask mAP@50:95", _fmt(m["map50_95"])), ("Mask IoU (pseudo)", _fmt(iou["mean_mask_iou"]))])
    _cards([("Latency / image", f"{lat['mean_ms']:.1f} ms"), ("FPS", f"{lat['fps']:.1f}"),
            ("Model size", f"{mod['size_mb']} MB"), ("Parameters", f"{mod['parameters'] / 1e6:.2f} M")])
    st.caption(f"Device: {r['device']} | imgsz {r['imgsz']} | RSS delta {r['memory']['process_rss_delta_mb']} MB | "
               f"{r['memory']['gpu_note'] or 'GPU peak ' + str(r['memory']['gpu_peak_mb']) + ' MB'}")
    st.warning(r["caveat"])
    with st.expander("Per-class AP"):
        st.dataframe(pd.DataFrame(r["per_class"]).T.round(3), use_container_width=True)


def ae_panel() -> None:
    st.subheader("Convolutional Autoencoder  (unknown anomalies)")
    r = _load("autoencoder_metrics.json")
    if not r:
        st.warning("Not evaluated yet - run `python -m models.autoencoder.train`.")
        return
    real, syn = r["test_real"], r["test_synthetic"]
    _cards([("ROC-AUC (real)", _fmt(real["roc_auc"])), ("PR-AUC (real)", _fmt(real["pr_auc"])),
            ("Precision", _fmt(real["precision"])), ("Recall", _fmt(real["recall"])), ("F1", _fmt(real["f1"])),
            ("False-positive rate", _fmt(real["false_positive_rate"])), ("False-negative rate", _fmt(real["false_negative_rate"])),
            ("Threshold (validation)", _fmt(r["threshold"], 5))])
    st.dataframe(pd.DataFrame([{"set": k, "n anomalies": v["n_anomaly"], "ROC-AUC": v["roc_auc"], "PR-AUC": v["pr_auc"],
                                "precision": v["precision"], "recall": v["recall"], "F1": v["f1"], "FPR": v["false_positive_rate"],
                                "MSE normal": v["mse_normal"], "MSE anomaly": v["mse_anomaly"], "MAE normal": v["mae_normal"],
                                "MAE anomaly": v["mae_anomaly"], "SSIM normal": v["ssim_normal"], "SSIM anomaly": v["ssim_anomaly"]}
                               for k, v in (("real (primary)", real), ("synthetic (secondary)", syn))]).round(4),
                 use_container_width=True, hide_index=True)
    st.caption(f"Threshold selected on {r['threshold_selected_on']}; the test frames were never used to tune it.")


def lstm_panel() -> None:
    st.subheader("LSTM tracker")
    r = _load("lstm_metrics.json")
    if not r:
        st.warning("Not evaluated yet - run `python -m models.lstm.evaluate`.")
        return
    l, b = r["lstm"], r["constant_velocity_baseline"]
    _cards([("Position MAE (1-step)", f"{l['forecast']['mae_px_step1']:.2f} px"), ("Position RMSE (1-step)", f"{l['forecast']['rmse_px_step1']:.2f} px"),
            ("Trajectory error (ADE)", f"{l['forecast']['trajectory_ade_px']:.2f} px"), ("Final-point error (FDE)", f"{l['forecast']['trajectory_fde_px']:.2f} px"),
            ("Track continuity", _fmt(l["tracking"]["track_continuity"])), ("ID switches", l["tracking"]["id_switches"]),
            ("Tracking success", _fmt(l["tracking"]["tracking_success_rate"])), ("FPS / latency", f"{l['tracking']['fps']:.0f} / {l['tracking']['latency_ms_per_frame']:.1f} ms")])
    st.dataframe(pd.DataFrame({"LSTM": {**l["forecast"], **l["tracking"]}, "Constant-velocity baseline": {**b["forecast"], **b["tracking"]}}).round(4),
                 use_container_width=True)
    st.warning(r["data"])


def pipeline_panel() -> None:
    st.subheader("End-to-end pipeline  (real Anoma frames + 27-class test set)")
    r = _load("pipeline_e2e_metrics.json") or _load("e2e_gap20.json")
    if not r:
        st.info("Run `python scripts/evaluate_pipeline.py` for end-to-end numbers.")
        return
    u = r["unknown_anomalies_anoma"]
    t = u["test_region_hit"]
    _cards([("Anomaly regions found", f"{t['objects_found']}/{t['n_gt']}"), ("Recall (region-hit)", _fmt(t["recall"])),
            ("Precision", _fmt(t["precision"])), ("False alarms / frame", _fmt(t["false_alarms_per_frame"], 1))])
    st.caption(u["why_two_metrics"])
    k = r.get("known_debris_27class", {})
    if "correct_class_and_box_rate" in k:
        _cards([("Correct class + box", _fmt(k["correct_class_and_box_rate"])), ("Known precision", _fmt(k["known_precision"])),
                ("Nothing found", _fmt(k["nothing_found_rate"])), ("s / image", _fmt(k["mean_seconds_per_image"], 2))])


def render_metrics() -> None:
    yolo_panel()
    st.divider()
    ae_panel()
    st.divider()
    lstm_panel()
    st.divider()
    pipeline_panel()
    onnx = _load("onnx_export.json")
    if onnx:
        st.divider()
        st.subheader("Edge deployment (ONNX export parity)")
        st.dataframe(pd.DataFrame(onnx["models"]).T, use_container_width=True)
        st.caption(f"onnxruntime {onnx.get('onnxruntime')} | providers {onnx.get('providers_available')} | "
                   "TensorRT engines are built on the Jetson (deploy/jetson/build_engines.sh) and were NOT run in this environment.")
