"""Stage 10 - LSTM tracking service: persistent track IDs, current + predicted position, trajectory."""

from __future__ import annotations

from typing import Any, Dict, List

from backend.pipeline.fusion import NOISE, FusedObject
from models.lstm.tracker import LSTMTracker


def make_tracker(cfg) -> LSTMTracker:
    t = cfg.tracking
    return LSTMTracker(cfg.path("tracking.weights"), seq_len=t.seq_len, horizon=t.horizon, max_age=t.max_age,
                       match_dist_px=t.match_dist_px, min_hits=int(cfg.get("tracking.min_hits", 3)),
                       device=t.device,
                       onnx=cfg.path("export.onnx_dir") / "lstm_tracker.onnx" if cfg.get("export.backend", "pytorch") == "onnx" else None)


def track_objects(tracker: LSTMTracker, objects: List[FusedObject], frame_offset_y: float, t: float) -> List[Dict[str, Any]]:
    """Feed non-noise objects of one frame to the tracker. Positions are global (frame y + offset) so a static
    object keeps a consistent coordinate across overlapping frames. Returns the tracker snapshot; each fused
    object gets `track_id` assigned in place."""
    dets = []
    live = [o for o in objects if o.category != NOISE]
    for o in live:
        cx, cy = o.centroid
        w, h = o.bbox[2] - o.bbox[0], o.bbox[3] - o.bbox[1]
        dets.append({"x": cx, "y": cy + frame_offset_y, "w": w, "h": h, "conf": o.confidence,
                     "class_name": o.class_name, "payload": id(o)})
    snapshot = tracker.update(dets, t=t)
    for o, d in zip(live, dets):
        o.track_id = d.get("track_id")
    return snapshot
