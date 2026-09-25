"""
ROI/patch data for the anomaly autoencoder, built from REAL side-scan sonar (Anoma dataset, 640x640).

* normal patches  : random crops that do not overlap any labelled object (seabed background)
* real anomalies  : the labelled Anoma objects (Debris Target / Fragment / Cluster / Linear structure)
* synthetic anom. : blobs, shadows, lines, blocks injected into real normal patches (secondary benchmark)

Normal crops are sampled with the SAME side-length distribution as the real object boxes, then resized
to the patch size exactly like pipeline ROIs, so scale is not a confound between normal and anomalous.
Images are grouped by source-frame id so the same frame never lands in two splits.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

ANOMA_ROOT = Path("samples/anoma")


def frame_group(path: Path) -> str:
    """'000012_jpg.rf.<hash>.jpg' -> '000012'  (roboflow augmentations share the source frame id)."""
    m = re.match(r"^(.*?)_(?:jpg|png|jpeg)\.rf\.", path.name)
    return m.group(1) if m else path.stem


def read_boxes(label_path: Path, w: int, h: int) -> List[List[int]]:
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            p = line.split()
            if len(p) >= 5:
                cx, cy, bw, bh = (float(v) for v in p[1:5])
                boxes.append([int((cx - bw / 2) * w), int((cy - bh / 2) * h),
                              int((cx + bw / 2) * w), int((cy + bh / 2) * h)])
    return boxes


def list_frames(split: str, root: Path = ANOMA_ROOT) -> List[Tuple[Path, List[List[int]]]]:
    d = root / split / "images"
    frames = []
    for img in sorted(d.glob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        im = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        h, w = im.shape
        frames.append((img, read_boxes(root / split / "labels" / (img.stem + ".txt"), w, h)))
    return frames


def split_by_group(frames, fractions: Sequence[float], seed: int = 0):
    """Split frames into disjoint groups (by source-frame id). fractions must sum to 1."""
    groups: Dict[str, list] = defaultdict(list)
    for f in frames:
        groups[frame_group(f[0])].append(f)
    keys = sorted(groups)
    np.random.default_rng(seed).shuffle(keys)
    cuts = np.cumsum(fractions)[:-1] * len(keys)
    out, start = [], 0
    for cut in list(cuts.astype(int)) + [len(keys)]:
        out.append([f for k in keys[start:cut] for f in groups[k]])
        start = cut
    return out


def _resize(patch: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA if patch.shape[0] >= size else cv2.INTER_CUBIC)


def to_tensor_array(patches: List[np.ndarray]) -> np.ndarray:
    return (np.stack(patches).astype(np.float32) / 255.0)[:, None]


def box_side_sizes(frames, pad: int = 8) -> np.ndarray:
    sizes = [max(b[2] - b[0], b[3] - b[1]) + 2 * pad for _, boxes in frames for b in boxes]
    return np.array(sizes if sizes else [64], dtype=np.int32)


def _overlaps(box, boxes, margin: int) -> bool:
    for b in boxes:
        if not (box[2] < b[0] - margin or box[0] > b[2] + margin or box[3] < b[1] - margin or box[1] > b[3] + margin):
            return True
    return False


def normal_patches(frames, patch: int, per_frame: int, sizes: np.ndarray, rng: np.random.Generator,
                   margin: int = 12) -> List[np.ndarray]:
    out = []
    for path, boxes in frames:
        im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        h, w = im.shape
        got, tries = 0, 0
        while got < per_frame and tries < per_frame * 12:
            tries += 1
            s = int(np.clip(rng.choice(sizes), 24, min(h, w) - 1))
            x, y = int(rng.integers(0, w - s)), int(rng.integers(0, h - s))
            if _overlaps([x, y, x + s, y + s], boxes, margin):
                continue
            out.append(_resize(im[y:y + s, x:x + s], patch))
            got += 1
    return out


def anomaly_patches(frames, patch: int, pad: int = 8) -> List[np.ndarray]:
    out = []
    for path, boxes in frames:
        im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        h, w = im.shape
        for x1, y1, x2, y2 in boxes:
            x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
            if x2 - x1 >= 8 and y2 - y1 >= 8:
                out.append(_resize(im[y1:y2, x1:x2], patch))
    return out


def synthetic_anomalies(normals: List[np.ndarray], rng: np.random.Generator) -> List[np.ndarray]:
    """Inject a plausible unknown object into a REAL normal patch (bright highlight + dark shadow / line / block)."""
    out = []
    for base in normals:
        p = base.astype(np.float32).copy()
        s = p.shape[0]
        kind = int(rng.integers(0, 4))
        cx, cy = int(rng.integers(s // 4, 3 * s // 4)), int(rng.integers(s // 4, 3 * s // 4))
        hi = float(rng.uniform(70, 140))
        layer = np.zeros_like(p)
        if kind == 0:                                      # highlight + acoustic shadow tail
            a, b = int(rng.integers(4, s // 4)), int(rng.integers(3, s // 5))
            cv2.ellipse(layer, (cx, cy), (a, b), float(rng.uniform(0, 180)), 0, 360, hi, -1)
            cv2.ellipse(layer, (min(s - 1, cx + a + 3), cy), (a + 4, b), 0, 0, 360, -hi * 0.8, -1)
        elif kind == 1:                                    # linear structure (cable / pipe-like)
            x2, y2 = int(rng.integers(0, s)), int(rng.integers(0, s))
            cv2.line(layer, (cx, cy), (x2, y2), hi, int(rng.integers(2, 5)))
        elif kind == 2:                                    # rectangular block
            bw, bh = int(rng.integers(6, s // 3)), int(rng.integers(6, s // 3))
            cv2.rectangle(layer, (cx, cy), (min(s - 1, cx + bw), min(s - 1, cy + bh)), hi, -1)
        else:                                              # scattered fragments
            for _ in range(int(rng.integers(3, 7))):
                fx, fy = int(rng.integers(2, s - 2)), int(rng.integers(2, s - 2))
                cv2.circle(layer, (fx, fy), int(rng.integers(1, 4)), hi, -1)
        layer = cv2.GaussianBlur(layer, (0, 0), 0.8)
        out.append(np.clip(p + layer, 0, 255).astype(np.uint8))
    return out
