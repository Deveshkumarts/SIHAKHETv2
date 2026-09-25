"""Shared fixtures: an isolated config (tmp dirs), a deterministic sample XTF, and small test images."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.config import load_config  # noqa: E402
from utils.sonar_raw_ingestion import generate_synthetic_xtf  # noqa: E402


@pytest.fixture(scope="session")
def base_cfg():
    return load_config()


@pytest.fixture()
def cfg(base_cfg, tmp_path):
    """Config whose outputs (runs, logs, db) land in a temp dir so tests never touch project data."""
    return base_cfg.override(**{
        "paths.runs_dir": str(tmp_path / "runs"),
        "logging.dir": str(tmp_path / "logs"),
        "database.sqlite_path": str(tmp_path / "test.db"),
        "database.backend": "sqlite",
    })


@pytest.fixture(scope="session")
def sample_xtf(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("xtf") / "sample.xtf"
    generate_synthetic_xtf(p, num_pings=160, samples_per_channel=400, seed=0)   # same params as scripts/generate_sample_xtf.py
    return p


@pytest.fixture(scope="session")
def sample_image() -> np.ndarray:
    """Deterministic noisy seabed with one small bright highlight + dark shadow (a fake object).
    CFAR is a small-target detector: the object must fit inside the guard window (~9 px) or it
    contaminates its own reference ring, so the highlight is deliberately ~8x10 px."""
    rng = np.random.default_rng(0)
    img = np.clip(rng.normal(90, 14, (256, 256)), 0, 255).astype(np.uint8)
    img[100:108, 120:130] = 240
    img[100:108, 130:150] = 8
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


@pytest.fixture(scope="session")
def anoma_image_path() -> Path:
    files = sorted((ROOT / "samples" / "anoma" / "valid" / "images").glob("*.jpg"))
    if not files:
        pytest.skip("Anoma sample images not available")
    return files[0]
