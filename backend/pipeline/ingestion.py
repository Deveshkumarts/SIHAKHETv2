"""Stage 1 - ingestion: XTF/JSF binary logs or plain sonar images -> (image, telemetry, metadata)."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from utils.sonar_raw_ingestion import ingest_raw_sonar_file
from utils.telemetry_parser import TelemetryRecord, generate_synthetic_telemetry

RAW_EXTS = {".xtf", ".jsf"}


@dataclass
class IngestResult:
    image_bgr: np.ndarray
    telemetry: List[TelemetryRecord]
    meta: Dict[str, Any] = field(default_factory=dict)
    kind: str = "image"                     # 'xtf' | 'jsf' | 'image'
    synthetic_telemetry: bool = False       # True when the file carried no navigation data


def ingest(source: Union[str, Path, bytes], filename: Optional[str] = None, cfg=None) -> IngestResult:
    name = filename or (str(source) if isinstance(source, (str, Path)) else "upload.bin")
    ext = Path(name).suffix.lower()
    supported = set(cfg.ingestion.supported) if cfg is not None else RAW_EXTS | {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if ext not in supported:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {sorted(supported)}")

    if ext in RAW_EXTS:
        img, tele, meta = ingest_raw_sonar_file(source if not isinstance(source, bytes) else io.BytesIO(source), filename=name)
        meta = dict(meta)
        meta["filename"] = name
        return IngestResult(img, list(tele), meta, kind=ext.lstrip("."))

    if isinstance(source, bytes):
        img = cv2.imdecode(np.frombuffer(source, np.uint8), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError(f"Could not decode image '{name}'")

    # Plain images carry no navigation: say so instead of silently inventing a real track.
    tele = generate_synthetic_telemetry(num_pings=1)
    meta = {"filename": name, "height": img.shape[0], "width": img.shape[1], "telemetry_source": "synthetic"}
    return IngestResult(img, tele, meta, kind="image", synthetic_telemetry=True)
