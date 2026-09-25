"""Extract acoustic-signature features for every object in the survey-disjoint seg dataset (uses the instance masks).

    python scripts/extract_signatures.py  ->  outputs/evaluation/signature_features.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.pipeline.acoustic_signature import FEATURE_NAMES, extract_signature, to_vector  # noqa: E402


def main():
    root = ROOT / "SIH_Dataset_27class_seg"
    names = yaml.safe_load((root / "data.yaml").read_text())["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    out = {s: {"X": [], "y": [], "src": []} for s in ("train", "val", "test")}
    for split in out:
        for lab in sorted((root / split / "labels").glob("*.txt")):
            img_p = root / split / "images" / (lab.stem + ".png")
            im = cv2.imread(str(img_p))
            if im is None:
                continue
            h, w = im.shape[:2]
            for line in lab.read_text().strip().splitlines():
                v = line.split()
                pts = np.array(v[1:], np.float32).reshape(-1, 2)
                bbox = [pts[:, 0].min() * w, pts[:, 1].min() * h, pts[:, 0].max() * w, pts[:, 1].max() * h]
                m = np.zeros((h, w), np.uint8)
                cv2.fillPoly(m, [np.round(pts * [w, h]).astype(np.int32)], 1)
                f = extract_signature(im, bbox, m, waterfall=False)
                if f is None:
                    continue
                out[split]["X"].append(to_vector(f))
                out[split]["y"].append(int(v[0]))
                out[split]["src"].append(lab.stem)
        print(split, len(out[split]["y"]), "objects", flush=True)
    dst = ROOT / "outputs" / "evaluation" / "signature_features.npz"
    np.savez_compressed(dst, names=np.array(names), features=np.array(FEATURE_NAMES),
                        **{f"{s}_{k}": np.array(v) for s in out for k, v in out[s].items()})
    print("saved", dst)


if __name__ == "__main__":
    main()
