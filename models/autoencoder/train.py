"""
Train the anomaly autoencoder on NORMAL seabed patches only, calibrate the threshold on validation
data, and report held-out test metrics. All from real Anoma side-scan sonar.

    python -m models.autoencoder.train --epochs 40

Data flow (source frames are grouped by frame id so no frame appears in two roles):
  Anoma train           -> normal patches                       : TRAIN the autoencoder
  Anoma valid (half A)  -> normal + real anomalies + synthetic  : early stopping + THRESHOLD selection
  Anoma valid (half B)  -> normal + real anomalies + synthetic  : held-out TEST (reported, never tuned on)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from backend.config import ROOT_DIR, load_config
from models.autoencoder import metrics as M
from models.autoencoder import patches as P
from models.autoencoder.anomaly import reconstruct
from models.autoencoder.conv_ae import PatchConvAE


def build_sets(patch: int, seed: int, per_frame_train: int, per_frame_eval: int):
    rng = np.random.default_rng(seed)
    root = ROOT_DIR / "samples" / "anoma"
    train_frames = P.list_frames("train", root)
    valid_frames = P.list_frames("valid", root)
    val_a, test_b = P.split_by_group(valid_frames, [0.5, 0.5], seed=seed)
    sizes = P.box_side_sizes(train_frames + valid_frames)

    def make(frames, per_frame):
        normal = P.normal_patches(frames, patch, per_frame, sizes, rng)
        real = P.anomaly_patches(frames, patch)
        synth = P.synthetic_anomalies(normal[: max(len(real), 200)], rng)
        return normal, real, synth

    tr_normal = P.normal_patches(train_frames, patch, per_frame_train, sizes, rng)
    return tr_normal, make(val_a, per_frame_eval), make(test_b, per_frame_eval), {
        "train_frames": len(train_frames), "val_frames": len(val_a), "test_frames": len(test_b),
        "unlabelled_split_unused": "samples/anoma/text has images but no labels",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--per-frame-train", type=int, default=24)
    ap.add_argument("--per-frame-eval", type=int, default=12)
    ap.add_argument("--max-fpr", type=float, default=0.10)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--recalibrate", action="store_true",
                    help="reuse saved weights; only re-select the threshold and re-evaluate")
    args = ap.parse_args()

    cfg = load_config()
    seed, patch, latent = cfg.pipeline.seed, cfg.autoencoder.patch_size, cfg.autoencoder.latent_dim
    torch.manual_seed(seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    tr_normal, (va_n, va_real, va_syn), (te_n, te_real, te_syn), prov = build_sets(
        patch, seed, args.per_frame_train, args.per_frame_eval)
    X = P.to_tensor_array(tr_normal)
    Va_n = P.to_tensor_array(va_n)
    print(f"train normal patches {len(X)} | val normal {len(va_n)} real-anom {len(va_real)} synth {len(va_syn)} "
          f"| test normal {len(te_n)} real-anom {len(te_real)} synth {len(te_syn)}")

    model = PatchConvAE(patch, latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_state, bad, hist = 1e9, None, 0, []
    t0 = time.time()
    if args.recalibrate:
        ckpt = torch.load(cfg.path("autoencoder.weights"), map_location=device, weights_only=True)
        best_state = ckpt["state_dict"]
        prev = json.loads(cfg.path("autoencoder.calibration").read_text())
        hist = prev.get("history", [])
    else:
        for ep in range(args.epochs):
            model.train()
            perm = np.random.default_rng(seed + ep).permutation(len(X))
            tot = 0.0
            for i in range(0, len(X), args.batch):
                xb = torch.from_numpy(X[perm[i:i + args.batch]])
                if np.random.rand() < 0.5:
                    xb = torch.flip(xb, dims=[3])                       # horizontal flip augmentation
                loss = torch.mean((model(xb) - xb) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(xb)
            model.eval()
            _, v_mse, _ = reconstruct(model, Va_n, device)
            vloss = float(v_mse.mean())
            hist.append({"epoch": ep + 1, "train_mse": tot / len(X), "val_normal_mse": vloss})
            print(f"epoch {ep + 1:02d} train {tot / len(X):.5f}  val-normal {vloss:.5f}")
            if vloss < best - 1e-6:
                best, bad = vloss, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    print("early stop")
                    break
    model.load_state_dict(best_state)
    model.eval()

    def errors(arrs):
        a = P.to_tensor_array(arrs)
        recon, mse, mae = reconstruct(model, a, device)
        return a, recon, mse, mae

    # ---- threshold selected on VALIDATION (real + synthetic anomalies vs normals) ----
    # Real anomalies are the primary benchmark, so the threshold is tuned on REAL validation anomalies only;
    # synthetic ones are reported as a secondary check and never tune the threshold.
    _, _, e_vn, _ = errors(va_n)
    _, _, e_va, _ = errors(va_real)
    _, _, e_vs, _ = errors(va_syn)
    chosen = M.select_threshold(e_vn, e_va, max_fpr=args.max_fpr)
    thr = float(chosen["threshold"])

    # ---- held-out TEST: real anomalies are the primary benchmark, synthetic secondary ----
    report = {"threshold_selected_on": "validation (Anoma valid half A), REAL anomalies only", "threshold": thr,
              "max_fpr_constraint": args.max_fpr, "validation": chosen,
              "validation_synthetic": M.detection_metrics(e_vn, e_vs, thr)}
    for tag, anoms in (("real", te_real), ("synthetic", te_syn), ("real+synthetic", te_real + te_syn)):
        an, arec, e_an, mae_an = errors(anoms) if anoms else (np.zeros((0, 1, patch, patch), np.float32),) * 2 + (np.zeros(0),) * 2
        nn_, nrec, e_n, mae_n = errors(te_n)
        m = M.detection_metrics(e_n, e_an, thr)
        m.update({"mse_normal": float(e_n.mean()), "mse_anomaly": float(e_an.mean()) if len(e_an) else None,
                  "mae_normal": float(mae_n.mean()), "mae_anomaly": float(mae_an.mean()) if len(mae_an) else None,
                  "ssim_normal": M.mean_ssim(nn_, nrec), "ssim_anomaly": M.mean_ssim(an, arec) if len(an) else None})
        report[f"test_{tag}"] = m

    # ---- save ----
    wpath, cpath = cfg.path("autoencoder.weights"), cfg.path("autoencoder.calibration")
    wpath.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "patch": patch, "latent": latent}, wpath)
    calib = {"threshold": thr, "score_sharpness": 3.0, "patch_size": patch, "latent_dim": latent,
             "normal_error_mean": float(e_vn.mean()), "normal_error_p95": float(np.percentile(e_vn, 95)),
             "trained_on": "normal seabed patches only (real Anoma sonar)", "provenance": prov,
             "train_patches": int(len(X)), "epochs_run": len(hist), "wall_seconds": round(time.time() - t0, 1), "recalibrated_only": bool(args.recalibrate),
             "config_hash": cfg.hash, "history": hist, "report": report}
    cpath.write_text(json.dumps(calib, indent=2))
    _, _, e_te_n, _ = errors(te_n)
    _, _, e_te_r, _ = errors(te_real)
    _, _, e_te_s, _ = errors(te_syn)
    np.savez_compressed(ROOT_DIR / "outputs" / "evaluation" / "autoencoder_errors.npz", normal=e_te_n, real=e_te_r,
                        synthetic=e_te_s, threshold=thr)
    out = ROOT_DIR / "outputs" / "evaluation" / "autoencoder_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in report["test_real"].items()}, indent=2))
    print("saved:", wpath, cpath)


if __name__ == "__main__":
    main()
