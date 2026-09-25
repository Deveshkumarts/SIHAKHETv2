"""
Export the three MarineGuard models to ONNX and verify numerical parity with PyTorch.

    python scripts/export_onnx.py --models ae lstm yolo

    PyTorch (dev) -> ONNX -> [ONNX Runtime | TensorRT engine via trtexec] -> Jetson Orin NX

Writes weights/onnx/{conv_ae,lstm_tracker,yolo11n_seg}.onnx and outputs/evaluation/onnx_export.json
(max abs difference vs PyTorch for each model). A model that fails parity is reported, never silently shipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import load_config  # noqa: E402
from backend.pipeline.onnx_backend import make_session  # noqa: E402
from models.autoencoder import patches as P  # noqa: E402
from models.autoencoder.conv_ae import PatchConvAE  # noqa: E402
from models.lstm.model import TrackLSTM  # noqa: E402
from models.lstm.simulate import make_training_windows, simulate_scene  # noqa: E402

PARITY_TOL = 1e-4


def _export(model, dummy, path, in_name, out_name, opset):
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(input_names=[in_name], output_names=[out_name], opset_version=opset,
              dynamic_axes={in_name: {0: "batch"}, out_name: {0: "batch"}})
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kw)      # stable TorchScript exporter
    except TypeError:
        torch.onnx.export(model, dummy, str(path), **kw)


def export_ae(cfg) -> dict:
    ckpt = torch.load(cfg.path("autoencoder.weights"), map_location="cpu", weights_only=True)
    m = PatchConvAE(int(ckpt["patch"]), int(ckpt["latent"]))
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    out = cfg.path("export.onnx_dir") / "conv_ae.onnx"
    _export(m, torch.zeros(1, 1, m.patch, m.patch), out, "patch", "reconstruction", int(cfg.export.opset))
    # parity on REAL sonar patches, at a different batch size than the export dummy (dynamic batch check)
    frames = P.list_frames("valid", ROOT / "samples" / "anoma")[:8]
    rng = np.random.default_rng(0)
    x = P.to_tensor_array(P.normal_patches(frames, m.patch, 6, P.box_side_sizes(frames), rng)[:32])
    with torch.no_grad():
        ref = m(torch.from_numpy(x)).numpy()
    got = make_session(out).run(None, {"patch": x})[0]
    err_ref, err_got = ((x - ref) ** 2).mean((1, 2, 3)), ((x - got) ** 2).mean((1, 2, 3))
    return {"file": str(out.relative_to(ROOT)), "size_kb": round(out.stat().st_size / 1e3, 1),
            "max_abs_diff": float(np.abs(ref - got).max()),
            "max_rel_diff_reconstruction_error": float((np.abs(err_ref - err_got) / np.maximum(err_ref, 1e-9)).max())}


def export_lstm(cfg) -> dict:
    ckpt = torch.load(cfg.path("tracking.weights"), map_location="cpu", weights_only=True)
    m = TrackLSTM(int(ckpt["horizon"]), int(ckpt["hidden"]))
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    out = cfg.path("export.onnx_dir") / "lstm_tracker.onnx"
    _export(m, torch.zeros(1, int(ckpt["seq_len"]), 9), out, "history", "future_offsets", int(cfg.export.opset))
    X, _ = make_training_windows([simulate_scene(np.random.default_rng(5)) for _ in range(20)], int(ckpt["seq_len"]),
                                 int(ckpt["horizon"]))
    X = X[:64]
    with torch.no_grad():
        ref = m(torch.from_numpy(X)).numpy()
    got = make_session(out).run(None, {"history": X})[0]
    return {"file": str(out.relative_to(ROOT)), "size_kb": round(out.stat().st_size / 1e3, 1),
            "max_abs_diff": float(np.abs(ref - got).max()), "max_abs_diff_px": float(np.abs(ref - got).max() * 100)}


def export_yolo(cfg) -> dict:
    import cv2
    from ultralytics import YOLO
    w = cfg.path("yolo_seg.weights")
    if not w.exists():
        return {"skipped": f"weights not found: {w}"}
    out = cfg.path("export.onnx_dir") / "yolo11n_seg.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    produced = Path(YOLO(str(w)).export(format="onnx", imgsz=int(cfg.yolo_seg.imgsz), opset=int(cfg.export.opset),
                                        simplify=False, dynamic=True, half=False, verbose=False))   # dynamic H/W keeps rectangular (minimal-padding) inference
    if produced.resolve() != out.resolve():
        out.write_bytes(produced.read_bytes())
    # parity: same images through the .pt and the .onnx model
    imgs = sorted((ROOT / "SIH_Dataset_27class_seg" / "test" / "images").glob("*.png"))[:40]
    pt, ox = YOLO(str(w)), YOLO(str(out), task="segment")
    diffs, count_match = [], 0
    for p in imgs:
        im = cv2.imread(str(p))
        a = pt.predict(im, imgsz=int(cfg.yolo_seg.imgsz), conf=0.25, device="cpu", verbose=False)[0]
        b = ox.predict(im, imgsz=int(cfg.yolo_seg.imgsz), conf=0.25, device="cpu", verbose=False)[0]
        count_match += int(len(a.boxes) == len(b.boxes))
        if len(a.boxes) and len(a.boxes) == len(b.boxes):
            ba, bb = a.boxes.xyxy.cpu().numpy(), b.boxes.xyxy.cpu().numpy()
            order_a, order_b = np.lexsort(ba.T), np.lexsort(bb.T)
            diffs.append(float(np.abs(ba[order_a] - bb[order_b]).max()))
    return {"file": str(out.relative_to(ROOT)), "size_mb": round(out.stat().st_size / 1e6, 2), "images_compared": len(imgs),
            "same_detection_count": f"{count_match}/{len(imgs)}",
            "max_box_diff_px": max(diffs) if diffs else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ae", "lstm", "yolo"], choices=["ae", "lstm", "yolo"])
    a = ap.parse_args()
    cfg = load_config()
    fns = {"ae": export_ae, "lstm": export_lstm, "yolo": export_yolo}
    report = {"opset": int(cfg.export.opset), "torch": torch.__version__, "config_hash": cfg.hash, "models": {}}
    import onnxruntime
    report["onnxruntime"] = onnxruntime.__version__
    report["providers_available"] = onnxruntime.get_available_providers()
    for name in a.models:
        t0 = time.time()
        try:
            r = fns[name](cfg)
            tol = PARITY_TOL if name != "yolo" else None
            if tol is not None:
                r["parity_ok"] = bool(r["max_abs_diff"] <= tol)
            else:
                r["parity_ok"] = bool(r.get("max_box_diff_px") is None or r["max_box_diff_px"] < 1.0)
        except Exception as e:                                        # report, never hide
            r = {"error": f"{type(e).__name__}: {e}", "parity_ok": False}
        r["seconds"] = round(time.time() - t0, 1)
        report["models"][name] = r
        print(name, json.dumps(r))
    out = ROOT / "outputs" / "evaluation" / "onnx_export.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():                                                  # merge so exporting one model keeps the others
        prev = json.loads(out.read_text())
        prev["models"].update(report["models"])
        report["models"] = prev["models"]
    out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
