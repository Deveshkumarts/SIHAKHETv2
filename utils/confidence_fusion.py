"""
Multi-Evidence Mathematical Confidence & Reliability Fusion Engine.
SIH26057 - Akhet: Marine Guard ("Turning Echoes into Impact")

Implements the multi-evidence fusion from Slide 3:
  AI Confidence + Anomaly Score + Geometry + SNR + Motion Quality
  --> Calibrated Reliability (0-100): Confirmed / Probable / Uncertain
"""

import math
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import numpy as np
from utils.confidence_calibration import TemperatureScaler


@dataclass
class FusedConfidenceReport:
    final_confidence_pct: float       # [0.0, 100.0]
    final_confidence_norm: float      # [0.0, 1.0]
    calibrated_yolo_conf: float       # [0.0, 1.0]
    raw_yolo_conf: float              # [0.0, 1.0]
    cfar_contrast_score: float        # [0.0, 1.0]
    ae_consistency_score: float       # [0.0, 1.0]
    shadow_contrast_score: float      # [0.0, 1.0]
    acoustic_snr_score: float         # [0.0, 1.0]
    motion_quality_score: float       # [0.0, 1.0]
    mc_uncertainty_penalty: float     # Subtracted penalty
    reliability_tier: str             # "CONFIRMED", "PROBABLE", "UNCERTAIN"
    reliability_color: str            # Hex code: green / yellow / red
    evidence_breakdown: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MultiEvidenceConfidenceFusion:
    """
    Mathematical Confidence Score Fusion Engine for Side-Scan Sonar.
    Applies weighted evidential aggregation across AI, physics, acoustics, and vehicle motion.
    """

    def __init__(
        self,
        weight_yolo: float = 0.35,
        weight_cfar: float = 0.15,
        weight_ae: float = 0.15,
        weight_shadow: float = 0.15,
        weight_snr: float = 0.10,
        weight_motion: float = 0.10,
        uncertainty_lambda: float = 1.25,
        temperature: float = 1.35
    ):
        self.w_yolo = weight_yolo
        self.w_cfar = weight_cfar
        self.w_ae = weight_ae
        self.w_shadow = weight_shadow
        self.w_snr = weight_snr
        self.w_motion = weight_motion
        self.unc_lambda = uncertainty_lambda
        self.scaler = TemperatureScaler(temperature=temperature)

    def fuse_detection_confidence(
        self,
        raw_yolo_conf: float,
        cfar_contrast_ratio: float = 1.5,
        ae_anomaly_score: float = 0.20,
        has_shadow: bool = True,
        shadow_contrast: float = 0.45,
        calibrated_snr_db: float = 12.0,
        motion_quality_score: float = 1.0,
        mc_epistemic_variance: float = 0.010
    ) -> FusedConfidenceReport:
        # 1. Temperature-calibrated YOLO score
        p_cal = self.scaler.calibrate_probability(raw_yolo_conf)

        # 2. OS-CFAR contrast score (normalized to [0, 1])
        s_cfar = float(np.clip((cfar_contrast_ratio - 1.0) / 2.0, 0.0, 1.0))

        # 3. Autoencoder consistency score (1 - anomaly loss)
        s_ae = float(np.clip(1.0 - ae_anomaly_score, 0.0, 1.0))

        # 4. Acoustic shadow score (geometry evidence)
        if has_shadow:
            s_shadow = float(np.clip(1.0 - shadow_contrast, 0.2, 1.0))
        else:
            s_shadow = 0.10  # Severe penalty for lack of acoustic shadow

        # 5. Acoustic SNR score
        s_snr = float(np.clip(calibrated_snr_db / 20.0, 0.0, 1.0))

        # 6. Motion quality score (1.0 = stable, 0.0 = severe roll/pitch)
        s_motion = float(np.clip(motion_quality_score, 0.0, 1.0))

        # 7. Epistemic uncertainty penalty
        unc_penalty = float(self.unc_lambda * max(0.0, mc_epistemic_variance))

        # Evidential combination
        weighted_sum = (
            self.w_yolo * p_cal +
            self.w_cfar * s_cfar +
            self.w_ae * s_ae +
            self.w_shadow * s_shadow +
            self.w_snr * s_snr +
            self.w_motion * s_motion
        )

        final_norm = float(np.clip(weighted_sum - unc_penalty, 0.0, 1.0))
        final_pct = round(final_norm * 100.0, 1)

        # 3-Tier Calibrated Reliability Categorization (Slide 3)
        if final_pct >= 75.0:
            tier = "CONFIRMED"
            color = "#00e676"  # Emerald green
        elif final_pct >= 45.0:
            tier = "PROBABLE"
            color = "#ffc107"  # Amber yellow
        else:
            tier = "UNCERTAIN"
            color = "#ff5252"  # Coral red

        breakdown = {
            "Calibrated YOLO (35%)": round(self.w_yolo * p_cal * 100.0, 1),
            "OS-CFAR Highlight (15%)": round(self.w_cfar * s_cfar * 100.0, 1),
            "AE Consistency (15%)": round(self.w_ae * s_ae * 100.0, 1),
            "Acoustic Shadow (15%)": round(self.w_shadow * s_shadow * 100.0, 1),
            "Acoustic SNR (10%)": round(self.w_snr * s_snr * 100.0, 1),
            "Motion Quality (10%)": round(self.w_motion * s_motion * 100.0, 1),
            "Uncertainty Penalty (-)": round(unc_penalty * 100.0, 1),
        }

        return FusedConfidenceReport(
            final_confidence_pct=final_pct,
            final_confidence_norm=round(final_norm, 3),
            calibrated_yolo_conf=round(p_cal, 3),
            raw_yolo_conf=round(raw_yolo_conf, 3),
            cfar_contrast_score=round(s_cfar, 3),
            ae_consistency_score=round(s_ae, 3),
            shadow_contrast_score=round(s_shadow, 3),
            acoustic_snr_score=round(s_snr, 3),
            motion_quality_score=round(s_motion, 3),
            mc_uncertainty_penalty=round(unc_penalty, 4),
            reliability_tier=tier,
            reliability_color=color,
            evidence_breakdown=breakdown
        )
