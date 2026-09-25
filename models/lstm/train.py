"""
Train the LSTM motion model on simulated sonar scenes.   python -m models.lstm.train

Held-out scenes (different RNG stream) are used for validation and for the final evaluation in evaluate.py.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from backend.config import ROOT_DIR, load_config
from models.lstm.model import TrackLSTM
from models.lstm.simulate import make_training_windows, simulate_scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=1500)
    ap.add_argument("--val-scenes", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    cfg = load_config()
    seed, seq_len, horizon = cfg.pipeline.seed, cfg.tracking.seq_len, cfg.tracking.horizon
    torch.manual_seed(seed)
    torch.set_num_threads(a.threads)

    tr_rng, va_rng = np.random.default_rng(seed), np.random.default_rng(seed + 10_000)
    Xt, Yt = make_training_windows([simulate_scene(tr_rng) for _ in range(a.scenes)], seq_len, horizon)
    Xv, Yv = make_training_windows([simulate_scene(va_rng) for _ in range(a.val_scenes)], seq_len, horizon)
    print(f"train windows {len(Xt)}  val windows {len(Xv)}")
    Xt, Yt, Xv, Yv = (torch.from_numpy(v) for v in (Xt, Yt, Xv, Yv))

    model = TrackLSTM(horizon, a.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best, state, hist, t0 = 1e9, None, [], time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), a.batch):
            idx = perm[i:i + a.batch]
            loss = torch.mean((model(Xt[idx]) - Yt[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        model.eval()
        with torch.no_grad():
            vpx = torch.sqrt(torch.mean((model(Xv) - Yv) ** 2)).item() * 100.0
        sched.step(vpx)
        hist.append({"epoch": ep + 1, "train_mse_norm": tot / len(Xt), "val_rmse_px": vpx})
        print(f"epoch {ep + 1:02d}  train {tot / len(Xt):.6f}  val RMSE {vpx:.3f}px", flush=True)
        if vpx < best:
            best, state = vpx, {k: v.clone() for k, v in model.state_dict().items()}
    dst = cfg.path("tracking.weights")
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "seq_len": seq_len, "horizon": horizon, "hidden": a.hidden}, dst)
    (ROOT_DIR / "outputs" / "training").mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / "outputs" / "training" / "lstm_train.json").write_text(json.dumps({
        "trained_on": "simulated sonar scenes (no real labelled track data available)", "best_val_rmse_px": best,
        "epochs": len(hist), "wall_seconds": round(time.time() - t0, 1), "history": hist, "config_hash": cfg.hash}, indent=2))
    print("best val RMSE", round(best, 3), "px ->", dst)


if __name__ == "__main__":
    main()
