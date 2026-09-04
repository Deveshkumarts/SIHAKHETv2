"""
Device & Hardware Detection Utilities for YOLO11.
Supports automatic CUDA/GPU detection, CPU fallback, and VRAM management.
"""

import sys
import torch


def get_device_info() -> dict:
    """
    Query system dynamically for PyTorch, Python, CUDA, and GPU hardware information.
    Accurately detects whatever GPU is currently running (RTX 3050, RTX 4050, etc.)
    or gracefully reports CPU mode.
    """
    try:
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False

    gpu_name = "None (CPU)"
    short_name = "CPU Mode"
    vram_gb = 0.0
    device_count = 0
    cuda_version = getattr(torch.version, "cuda", None) or "N/A"

    if cuda_available:
        try:
            device_count = torch.cuda.device_count()
            if device_count > 0:
                raw_name = torch.cuda.get_device_name(0)
                gpu_name = raw_name if raw_name else "NVIDIA GPU"
                # Create a concise short name (e.g., "RTX 3050", "RTX 4050", "GTX 1650")
                short_name = gpu_name.replace("NVIDIA GeForce ", "").replace(" Laptop GPU", "").strip()
                vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        except Exception as e:
            gpu_name = f"CUDA GPU (Detection Error: {e})"
            short_name = "CUDA GPU"

    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "device_count": device_count,
        "gpu_name": gpu_name,
        "short_gpu_name": short_name,
        "vram_gb": vram_gb,
    }



def select_device(requested: str = "auto") -> str:
    """
    Select optimal execution device based on hardware availability and user preference.

    Args:
        requested: 'auto', 'cpu', '0', '0,1', etc.

    Returns:
        Device identifier string recognized by Ultralytics YOLO ('0', 'cpu', etc.)
    """
    if requested is None or requested.lower() == "auto":
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[Hardware] CUDA GPU detected: {gpu_name} ({vram:.2f} GB VRAM). Using device='0'.")
            return "0"
        else:
            print("[Hardware] No CUDA GPU detected. Falling back to device='cpu'.")
            return "cpu"

    if requested.lower() == "cpu":
        return "cpu"

    # Specific GPU index or list passed (e.g. '0' or '0,1')
    if torch.cuda.is_available():
        return requested
    else:
        print(f"[Hardware Warning] GPU '{requested}' requested but CUDA is unavailable. Falling back to 'cpu'.")
        return "cpu"


def recommend_batch_size(imgsz: int = 640, device_str: str = "0") -> int:
    """
    Recommend a safe batch size based on available VRAM and image resolution.
    """
    if device_str == "cpu" or not torch.cuda.is_available():
        return 4

    try:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if imgsz <= 640:
            if vram_gb >= 16:
                return 32
            elif vram_gb >= 8:
                return 16
            elif vram_gb >= 4:
                return 16
            else:
                return 8
        elif imgsz <= 832:
            if vram_gb >= 16:
                return 24
            elif vram_gb >= 8:
                return 12
            else:
                return 8
        else:  # >= 1024
            if vram_gb >= 16:
                return 16
            elif vram_gb >= 8:
                return 8
            else:
                return 4
    except Exception:
        return 8
