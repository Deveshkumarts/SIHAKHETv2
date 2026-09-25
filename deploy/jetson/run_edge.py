"""
Run the MarineGuard pipeline locally on an AUV/ROV (Jetson Orin NX) with NO network access.

    python deploy/jetson/run_edge.py --input survey.xtf [--backend onnx|pytorch] [--device cuda:0]

Uses the same code as the workstation pipeline. Only the execution engine changes:
  * export.backend=onnx  -> YOLO / autoencoder / LSTM run through ONNX Runtime; on the Jetson the TensorRT
    provider is picked automatically when onnxruntime-gpu (with TensorRT EP) is installed, else CUDA, else CPU.
  * results go to the local SQLite store and outputs/runs/<run_id>/ - nothing is sent to a cloud service.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.config import load_config  # noqa: E402
from backend.database.repository import DetectionRepository  # noqa: E402
from backend.pipeline.onnx_backend import available_providers  # noqa: E402
from backend.pipeline.runner import SonarPipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--backend", default="onnx", choices=["onnx", "pytorch"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fp16", action="store_true", help="half precision for the YOLO model (CUDA only)")
    a = ap.parse_args()

    cfg = load_config().override(**{
        "export.backend": a.backend, "yolo_seg.device": a.device, "autoencoder.device": a.device,
        "yolo_seg.half": a.fp16, "database.backend": "sqlite",          # local store: no cloud dependency
    })
    print("execution providers:", available_providers() if a.backend == "onnx" else "pytorch")
    pipe = SonarPipeline(cfg)
    res = pipe.run(a.input)
    DetectionRepository(cfg).save_run(
        {"survey_id": res.survey_id, "run_id": res.run_id, "source_file": res.source, "scqi": res.scqi.get("overall_score"),
         "snr_db": res.snr.get("snr_db"), "config_hash": res.config_hash, "model_version": res.model_version}, res.detections)
    print(json.dumps(res.summary(), indent=2))
    print("stage timings (ms):", {s["stage"]: round(s["seconds"] * 1000, 1) for s in res.stages})


if __name__ == "__main__":
    main()
