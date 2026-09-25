"""
Runtime acoustic-signature verification: stage between fusion and tracking.

For every non-noise object it computes the acoustic signature and, for KNOWN debris, checks whether that signature is
consistent with the class YOLO11-Seg claimed:

  CONFIRMED  claimed class is COMPETITIVE with the signature's best guess: P(claimed) >= confirm_min_ratio * P(best)
             (validated cut-off; merely being in the top-3 is not enough if the signature is sure of something else)
  MISMATCH   claimed class is ranked >= k by the signature AND its probability < tau     (k, tau validated on val data)
  UNCERTAIN  anything in between
  N/A        unknown anomalies (no claimed class) or inputs outside the validated domain

UNKNOWN anomalies get descriptive signature values and the signature's nearest known classes as a hint only.
Verification NEVER changes an object's category: a MISMATCH is a flag for review, not a rejection.

Domain: the verifier was trained and validated on the 27-class object chips. On raw side-scan waterfalls the same
features can be computed but the verdict is reported as N/A (not validated there).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from backend.pipeline.acoustic_signature import FEATURE_NAMES, extract_signature, to_vector

CONFIRMED, UNCERTAIN, MISMATCH, NA = "CONFIRMED", "UNCERTAIN", "MISMATCH", "N/A"


class SignatureVerifier:
    def __init__(self, model_path: str | Path):
        self.path = Path(model_path)
        self.available = self.path.exists()
        if not self.available:
            return
        m = json.loads(self.path.read_text())
        assert m["features"] == FEATURE_NAMES, "signature model was trained with a different feature set"
        self.classes: List[str] = m["classes"]
        self.mean, self.std = np.array(m["mean"]), np.array(m["std"])
        self.coef, self.intercept = np.array(m["coef"]), np.array(m["intercept"])
        self.class_index = np.array(m["class_index"])
        self.k, self.tau = int(m["policy"]["mismatch_if_rank_ge"]), float(m["policy"]["and_p_lt"])
        self.confirm_ratio = float(m["policy"].get("confirm_min_ratio", 0.3))

    def probs(self, f: Dict[str, float]) -> np.ndarray:
        z = ((to_vector(f) - self.mean) / self.std) @ self.coef.T + self.intercept
        e = np.exp(z - z.max())
        p = np.zeros(len(self.classes))
        p[self.class_index] = e / e.sum()
        return p

    def verify(self, f: Dict[str, float], claimed: Optional[str], validated_domain: bool = True) -> Dict[str, Any]:
        p = self.probs(f)
        order = np.argsort(-p)
        top3 = [{"class": self.classes[i], "p": round(float(p[i]), 4)} for i in order[:3]]
        out: Dict[str, Any] = {"features": {k: round(float(v), 4) for k, v in f.items()}, "nearest_classes": top3,
                               "verdict": NA, "claimed_class": claimed, "domain_validated": validated_domain}
        if claimed and claimed in self.classes:
            ci = self.classes.index(claimed)
            rank = int((p > p[ci]).sum())
            ratio = float(p[ci] / p.max())
            out.update(claimed_probability=round(float(p[ci]), 4), claimed_rank=rank + 1, claimed_vs_best=round(ratio, 4))
            if validated_domain:
                out["verdict"] = (CONFIRMED if ratio >= self.confirm_ratio else
                                  (MISMATCH if (rank >= self.k and p[ci] < self.tau) else UNCERTAIN))
        return out


def verify_objects(image_bgr: np.ndarray, objects: List[Any], verifier: SignatureVerifier, waterfall: bool,
                   geometry: Optional[Dict[str, float]] = None) -> Dict[str, int]:
    """Attach `.signature` (dict) to each non-noise FusedObject in place. Returns verdict counts."""
    counts = {CONFIRMED: 0, UNCERTAIN: 0, MISMATCH: 0, NA: 0}
    for o in objects:
        o.signature = None
        if o.category == "NOISE" or not verifier.available:
            continue
        mask = o.yolo["mask"] if o.yolo is not None and o.yolo.get("mask") is not None and o.yolo["mask"].any() else None
        f = extract_signature(image_bgr, o.bbox, mask, waterfall=waterfall, geometry=geometry)
        if f is None:
            continue
        claimed = o.class_name if o.detection_type == "known" else None
        o.signature = verifier.verify(f, claimed, validated_domain=not waterfall)
        counts[o.signature["verdict"]] += 1
    return counts
