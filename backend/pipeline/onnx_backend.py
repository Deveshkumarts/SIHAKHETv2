"""
ONNX Runtime backend helper. Same pipeline code, different execution engine:

  PyTorch (development)  ->  ONNX  ->  ONNX Runtime with TensorRT / CUDA / CPU providers
                                   ->  (Jetson) trtexec engines built from the same .onnx files (see deploy/jetson)

The provider order prefers TensorRT, then CUDA, then CPU, and only uses what is actually installed, so the
identical code runs on a laptop (CPU) and on a Jetson Orin NX (TensorRT) with no cloud connection.
"""

from __future__ import annotations

from pathlib import Path
from typing import List


def available_providers() -> List[str]:
    import onnxruntime as ort
    order = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    have = set(ort.get_available_providers())
    return [p for p in order if p in have] or ["CPUExecutionProvider"]


def make_session(path: str | Path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=so, providers=available_providers())
