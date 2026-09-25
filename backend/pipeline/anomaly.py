"""Stage 7b - autoencoder service (unknown anomalies from reconstruction error, validated threshold)."""

from __future__ import annotations

from models.autoencoder.anomaly import AnomalyScorer


def load_anomaly_scorer(cfg) -> AnomalyScorer:
    a = cfg.autoencoder
    onnx = cfg.path("export.onnx_dir") / "conv_ae.onnx" if cfg.get("export.backend", "pytorch") == "onnx" else None
    return AnomalyScorer(cfg.path("autoencoder.weights"), cfg.path("autoencoder.calibration"), device=a.device, onnx=onnx)
