"""LIVE / ANALYSIS views: original + enhanced sonar, SA-CFAR candidates, YOLO11-Seg masks, unknown-anomaly markers,
tracked + predicted trajectories, and reliability information."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st


def _img(path: str | None, caption: str) -> None:
    if path and Path(path).exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"{caption}: not available")


def render_images(result: Dict[str, Any]) -> None:
    a = result.get("artifacts", {})
    c1, c2 = st.columns(2)
    with c1:
        _img(a.get("original"), "Original sonar image")
    with c2:
        _img(a.get("enhanced"), "Enhanced (median > bilateral > CLAHE)")
    c3, c4 = st.columns(2)
    with c3:
        _img(a.get("candidates"), "SA-CFAR candidates (cyan boxes)")
    with c4:
        _img(a.get("detections"), "YOLO11-Seg masks (amber) | unknown anomalies (magenta diamonds) | "
                                  "tracked trajectory (green) | LSTM-predicted path (orange)")


def render_reliability(result: Dict[str, Any]) -> None:
    st.subheader("Reliability")
    scqi, snr, qc = result["scqi"], result["snr"], result["qc"]
    c = st.columns(5)
    c[0].metric("SCQI", f"{scqi.get('overall_score', 0):.1f}", scqi.get("grade"))
    c[1].metric("SNR", f"{snr.get('snr_db', 0):.1f} dB")
    c[2].metric("Data QC", "PASS" if qc["passed"] else "FAIL")
    c[3].metric("Resurvey", "recommended" if scqi.get("resurvey_recommended") else "not needed")
    c[4].metric("Pipeline time", f"{result['total_seconds'] * 1000:.0f} ms")
    if result.get("telemetry_synthetic"):
        st.warning("This input has no navigation data, so positions use SYNTHETIC telemetry and are not real locations.")
    for w in result.get("warnings", []):
        st.caption(f"Note: {w}")
    failed = [ch for ch in qc["checks"] if not ch["passed"]]
    if failed:
        st.error("QC checks failed: " + ", ".join(f"{ch['name']}={ch['value']} (limit {ch['limit']})" for ch in failed))


def render_detections(result: Dict[str, Any]) -> None:
    st.subheader(f"Detections ({len(result['detections'])})")
    if not result["detections"]:
        st.info("No known debris or unknown anomalies were detected.")
        return
    rows: List[Dict[str, Any]] = []
    for d in result["detections"]:
        rows.append({"track": d["track_id"], "type": d["detection_type"], "class": d["class_name"],
                     "confidence": round(d["confidence"], 3), "mask px": d["mask_area_px"],
                     "anomaly score": None if d["anomaly_score"] is None else round(d["anomaly_score"], 3),
                     "recon error": None if d["reconstruction_error"] is None else round(d["reconstruction_error"], 5),
                     "lat": round(d["latitude"], 6), "lon": round(d["longitude"], 6),
                     "+/- m (95%)": round(d["positional_uncertainty_m"], 1), "tile SNR dB": round(d["snr_db"] or 0, 1),
                     "frame": d["frame_id"], "why": "; ".join(d["reasons"][:2])})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_tracks(result: Dict[str, Any]) -> None:
    st.subheader(f"LSTM tracks ({len(result['tracks'])})")
    if not result["tracks"]:
        st.info("No confirmed tracks.")
        return
    st.dataframe(pd.DataFrame([{
        "track": t["track_id"], "class": t["class_name"], "state": t["state"], "hits": t["hits"],
        "x": round(t["x"], 1), "y": round(t["y"], 1), "vx px/frame": round(t["velocity"][0], 2),
        "vy px/frame": round(t["velocity"][1], 2), "predicted +1": [round(v, 1) for v in t["predicted"][0]],
        "predicted +5": [round(v, 1) for v in t["predicted"][-1]], "predictor": t["predictor"]} for t in result["tracks"]]),
        use_container_width=True, hide_index=True)


def render_stage_timings(result: Dict[str, Any]) -> None:
    st.subheader("Stage execution log")
    st.dataframe(pd.DataFrame([{"stage": s["stage"], "ms": round(s["seconds"] * 1000, 1), "ok": s["ok"],
                                "output": str(s["summary"])[:140]} for s in result["stages"]]),
                 use_container_width=True, hide_index=True)
    st.caption(f"run {result['run_id']} | config {result['config_hash']} | models {result['model_version']}")
