"""
Evaluate the tracker on held-out simulated scenes:  python -m models.lstm.evaluate

Reports, for the LSTM predictor AND a constant-velocity baseline (so the benefit is measurable):
position MAE / RMSE (1-step and horizon), trajectory error (ADE / FDE), track continuity, ID switches,
tracking success rate, false-track rate, FPS and per-frame latency.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np

from backend.config import ROOT_DIR, load_config
from models.lstm.simulate import Scene, make_training_windows, simulate_scene
from models.lstm.tracker import LSTMTracker


def forecast_errors(tracker: LSTMTracker, scenes: List[Scene]) -> Dict[str, float]:
    from models.lstm.model import make_features
    X, Y = make_training_windows(scenes, tracker.seq_len, tracker.horizon)
    if tracker.model is not None:
        import torch
        with torch.no_grad():
            P = tracker.model(torch.from_numpy(X)).numpy()
    else:
        from models.lstm.model import constant_velocity_forecast
        # rebuild histories from features is lossy; evaluate CV on windows directly via displacement features
        P = np.zeros_like(Y)
        for i, f in enumerate(X):
            hist = [{"x": r[0] * 100.0, "y": r[1] * 100.0, "w": 0, "h": 0, "conf": 0, "t": float(k)} for k, r in enumerate(f)]
            P[i] = constant_velocity_forecast(hist, tracker.horizon) / 100.0
    err = (P - Y) * 100.0                                   # px
    d = np.linalg.norm(err, axis=-1)                        # (N,H)
    return {"n_windows": int(len(X)),
            "mae_px_step1": float(np.abs(err[:, 0]).mean()), "rmse_px_step1": float(np.sqrt((err[:, 0] ** 2).mean())),
            "mae_px_horizon": float(np.abs(err).mean()), "rmse_px_horizon": float(np.sqrt((err ** 2).mean())),
            "trajectory_ade_px": float(d.mean()), "trajectory_fde_px": float(d[:, -1].mean())}


def tracking_metrics(make_tracker, scenes: List[Scene], min_hits: int = 3) -> Dict[str, float]:
    id_switches = frag = n_gt = success = 0
    continuity: List[float] = []
    false_tracks = total_tracks = 0
    lat: List[float] = []
    for sc in scenes:
        tr: LSTMTracker = make_tracker()
        assigned: Dict[int, List[int]] = defaultdict(list)      # gt_id -> track ids over time
        track_votes: Dict[int, Counter] = defaultdict(Counter)  # track id -> Counter(gt ids)
        for f, dets in enumerate(sc.frames):
            inp = [dict(d) for d in dets]
            t0 = time.perf_counter()
            tr.update(inp, t=float(f))
            lat.append(time.perf_counter() - t0)
            for d in inp:
                tid = d.get("track_id")
                if tid is not None:
                    track_votes[tid][d["gt_id"]] += 1
                    if d["gt_id"] >= 0:
                        assigned[d["gt_id"]].append(tid)
        for gid, seq in assigned.items():
            n_gt += 1
            switches = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
            id_switches += switches
            dom = Counter(seq).most_common(1)[0][1] / len(seq)
            continuity.append(dom)
            frag += len(set(seq)) - 1
            success += int(dom >= 0.8 and switches == 0)
        for tid, votes in track_votes.items():
            if sum(votes.values()) >= min_hits:                     # confirmed tracks only
                total_tracks += 1
                false_tracks += int(votes.most_common(1)[0][0] == -1)
    lat_arr = np.array(lat)
    return {"n_gt_tracks": n_gt, "id_switches": int(id_switches), "id_switches_per_track": id_switches / max(1, n_gt),
            "track_continuity": float(np.mean(continuity)) if continuity else 0.0, "fragmentations": int(frag),
            "tracking_success_rate": success / max(1, n_gt), "false_track_rate": false_tracks / max(1, total_tracks),
            "latency_ms_per_frame": float(lat_arr.mean() * 1000), "fps": float(1.0 / lat_arr.mean())}


def main():
    cfg = load_config()
    rng = np.random.default_rng(cfg.pipeline.seed + 20_000)        # held-out stream (train uses seed, val seed+10000)
    scenes = [simulate_scene(rng) for _ in range(200)]
    tc = cfg.tracking
    weights = str(cfg.path("tracking.weights"))
    report = {}
    for name, kw in (("lstm", {"predictor": "auto"}), ("constant_velocity_baseline", {"predictor": "cv"})):
        mk = lambda kw=kw: LSTMTracker(weights, tc.seq_len, tc.horizon, tc.max_age, tc.match_dist_px, min_hits=tc.min_hits, device="cpu", **kw)
        proto = mk()
        report[name] = {"predictor_used": proto.predictor_name, "forecast": forecast_errors(proto, scenes),
                        "tracking": tracking_metrics(mk, scenes, tc.min_hits)}
    report["data"] = "simulated held-out scenes (200); no real labelled tracks exist in this repo"
    out = ROOT_DIR / "outputs" / "evaluation" / "lstm_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    for k in ("lstm", "constant_velocity_baseline"):
        print(k, json.dumps({**report[k]["forecast"], **report[k]["tracking"]}, indent=1, default=lambda o: round(o, 4)))


if __name__ == "__main__":
    main()
