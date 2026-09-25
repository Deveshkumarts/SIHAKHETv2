"""
Simulated multi-object sonar scenes for training / evaluating the tracker.

NOTE: no labelled real track sequences exist in this repository. Objects move through the sensor frame
because the platform advances (common along-track flow) plus small individual drift and acceleration
noise; detections have measurement noise, missed detections and single-frame clutter false positives.
Results on this simulator show the tracker works as designed; they are not a substitute for real
survey validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Scene:
    frames: List[List[Dict]]                    # per-frame detections: x,y,w,h,conf,class_name,gt_id (-1 = clutter)
    gt_tracks: Dict[int, List[Tuple[int, float, float]]] = field(default_factory=dict)   # id -> [(frame,x,y)]


def simulate_scene(rng: np.random.Generator, n_frames: int = 40, frame_w: int = 640, frame_h: int = 640,
                   n_objects: Tuple[int, int] = (2, 6), miss_prob: float = 0.10, meas_sigma: float = 1.5,
                   clutter_rate: float = 0.8, dt: float = 1.0) -> Scene:
    n = int(rng.integers(n_objects[0], n_objects[1] + 1))
    # Half the scenes are in a global (ground-fixed) frame where debris is ~static; the rest are in a
    # sensor frame where everything flows with the platform advance. The tracker must handle both.
    static = bool(rng.random() < 0.5)
    # Non-static scenes flow in a RANDOM direction (any heading), not a fixed one - otherwise the LSTM learns a
    # direction bias instead of motion (caught by test_prediction_shape_and_accuracy_on_straight_motion).
    speed = 0.0 if static else float(rng.uniform(4, 24))          # px / frame
    theta = float(rng.uniform(0, 2 * np.pi))
    flow_x, flow_y = speed * float(np.cos(theta)), speed * float(np.sin(theta))
    drift_x, drift_y = (0.4, 0.3) if static else (1.0, 0.8)
    objs = []
    for gid in range(n):
        objs.append({"id": gid, "x": float(rng.uniform(40, frame_w - 40)), "y": float(rng.uniform(frame_h * 0.3, frame_h)),
                     "vx": flow_x + float(rng.normal(0, drift_x)), "vy": flow_y + float(rng.normal(0, drift_y)),
                     "w": float(rng.uniform(14, 60)), "h": float(rng.uniform(14, 60)),
                     "cls": f"class_{int(rng.integers(0, 6))}", "base_conf": float(rng.uniform(0.55, 0.9))})
    frames: List[List[Dict]] = []
    gt: Dict[int, List[Tuple[int, float, float]]] = {o["id"]: [] for o in objs}
    for f in range(n_frames):
        dets = []
        for o in objs:
            o["vx"] += float(rng.normal(0, 0.15 if static else 0.35))
            o["vy"] += float(rng.normal(0, 0.15 if static else 0.35))
            o["x"] += o["vx"] * dt
            o["y"] += o["vy"] * dt
            if not (0 <= o["x"] < frame_w and 0 <= o["y"] < frame_h):
                continue
            gt[o["id"]].append((f, o["x"], o["y"]))
            if rng.random() > miss_prob:
                dets.append({"x": o["x"] + float(rng.normal(0, meas_sigma)), "y": o["y"] + float(rng.normal(0, meas_sigma)),
                             "w": o["w"] + float(rng.normal(0, 1.0)), "h": o["h"] + float(rng.normal(0, 1.0)),
                             "conf": float(np.clip(rng.normal(o["base_conf"], 0.08), 0.05, 0.99)),
                             "class_name": o["cls"], "gt_id": o["id"], "t": f * dt})
        for _ in range(int(rng.poisson(clutter_rate))):
            dets.append({"x": float(rng.uniform(0, frame_w)), "y": float(rng.uniform(0, frame_h)),
                         "w": float(rng.uniform(10, 40)), "h": float(rng.uniform(10, 40)),
                         "conf": float(rng.uniform(0.05, 0.5)), "class_name": f"class_{int(rng.integers(0, 6))}",
                         "gt_id": -1, "t": f * dt})
        rng.shuffle(dets)
        frames.append(dets)
    return Scene(frames, {k: v for k, v in gt.items() if len(v) >= 3})


def make_training_windows(scenes: List[Scene], seq_len: int, horizon: int):
    """(history features source, target displacement) pairs built from GT-associated observations."""
    from models.lstm.model import make_features
    X, Y = [], []
    for sc in scenes:
        per_obj: Dict[int, List[Dict]] = {}
        truth: Dict[int, Dict[int, Tuple[float, float]]] = {}
        for gid, pts in sc.gt_tracks.items():
            truth[gid] = {f: (x, y) for f, x, y in pts}
        for f, dets in enumerate(sc.frames):
            for d in dets:
                if d["gt_id"] >= 0:
                    per_obj.setdefault(d["gt_id"], []).append(d)
        for gid, hist in per_obj.items():
            for i in range(2, len(hist)):
                last = hist[i]
                f_last = int(round(last["t"]))
                future = [truth[gid].get(f_last + k) for k in range(1, horizon + 1)]
                if any(p is None for p in future):
                    continue
                X.append(make_features(hist[max(0, i - seq_len + 1): i + 1], seq_len))
                Y.append(np.array([[p[0] - last["x"], p[1] - last["y"]] for p in future], np.float32) / 100.0)
    return np.stack(X), np.stack(Y)
