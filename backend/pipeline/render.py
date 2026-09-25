"""Overlay rendering for the dashboard: SA-CFAR candidates, YOLO11-Seg masks, unknown anomalies, tracks."""

from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

KNOWN_COLOR = (0, 215, 255)       # amber (BGR)
UNKNOWN_COLOR = (255, 0, 255)     # magenta
CFAR_COLOR = (255, 200, 0)        # cyan-ish
TRACK_COLOR = (0, 255, 0)
PRED_COLOR = (0, 165, 255)


def draw_candidates(image_bgr: np.ndarray, cands_global: List[Dict[str, Any]]) -> np.ndarray:
    out = image_bgr.copy()
    for c in cands_global:
        x1, y1, x2, y2 = [int(v) for v in c["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), CFAR_COLOR, 1)
    return out


def draw_detections(image_bgr: np.ndarray, records: List[Dict[str, Any]], tracks: List[Dict[str, Any]]) -> np.ndarray:
    """records use GLOBAL pixel coordinates (bbox_global / mask_polygon_global)."""
    out = image_bgr.copy()
    overlay = out.copy()
    for r in records:
        color = KNOWN_COLOR if r["detection_type"] == "known" else UNKNOWN_COLOR
        poly = r.get("mask_polygon_global") or []
        if len(poly) >= 3:
            cv2.fillPoly(overlay, [np.round(np.array(poly)).astype(np.int32)], color)
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
    for r in records:
        color = KNOWN_COLOR if r["detection_type"] == "known" else UNKNOWN_COLOR
        x1, y1, x2, y2 = [int(v) for v in r["bbox_global"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{r['class_name']} {r['confidence']:.2f}" + (f" #{r['track_id']}" if r.get("track_id") else "")
        cv2.putText(out, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        if r["detection_type"] == "unknown":                      # unknown-anomaly marker
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.drawMarker(out, (cx, cy), UNKNOWN_COLOR, cv2.MARKER_DIAMOND, 14, 2)
    for t in tracks:
        pts = np.round(np.array(t["trajectory"])).astype(np.int32)
        if len(pts) >= 2:
            cv2.polylines(out, [pts], False, TRACK_COLOR, 2, cv2.LINE_AA)
        pred = np.round(np.array([[t["x"], t["y"]]] + t["predicted"])).astype(np.int32)
        if len(pred) >= 2:
            cv2.polylines(out, [pred], False, PRED_COLOR, 1, cv2.LINE_AA)
            cv2.circle(out, tuple(pred[-1]), 3, PRED_COLOR, -1)
    return out
