"""
Multi-Evidence Mathematical Confidence Fusion Engine.
Fuses:
  1. Calibrated YOLO Probability (via Temperature Scaling)
  2. 2D OS-CFAR Highlight Contrast Score
  3. Convolutional Autoencoder Consistency Score (1 - Anomaly Loss)
  4. Physical Acoustic Shadow Verification Score
  5. Calibrated Acoustic SNR Index
  6. Monte Carlo Dropout Epistemic Variance Uncertainty Penalty

Outputs a normalized, robust, calibrated 0–100% final confidence score with full evidence transparency.
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
    mc_uncertainty_penalty: float     # Subtracted penalty
    evidence_breakdown: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MultiEvidenceConfidenceFusion:
    """
    Mathematical Confidence Score Fusion Engine for Side-Scan Sonar.
    Applies weighted logit-scale evidential aggregation with uncertainty regularization.
    """

    def __init__(
        self,
        weight_yolo: float = 0.40,
        weight_cfar: float = 0.15,
        weight_ae: float = 0.15,
        weight_shadow: float = 0.20,
        weight_snr: float = 0.10,
        uncertainty_lambda: float = 1.25,
        temperature: float = 1.35
    ):
        self.w_yolo = weight_yolo
        self.w_cfar = weight_cfar
        self.w_ae = weight_ae
        self.w_shadow = weight_shadow
        self.w_snr = weight_snr
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
        mc_epistemic_variance: float = 0.010
    ) -> FusedConfidenceReport:
        """
        Calculates the multi-evidence fused confidence score.

        Formula:
          S_fused = w_yolo * P_cal + w_cfar * S_cfar + w_ae * (1 - S_ae)
                    + w_shadow * S_shadow + w_snr * S_snr - lambda * sigma^2_mc
        """
        # 1. Temperature-calibrated YOLO score
        p_cal = self.scaler.calibrate_probability(raw_yolo_conf)

        # 2. OS-CFAR contrast score (normalized to [0, 1], saturated at contrast=3.0)
        s_cfar = float(np.clip((cfar_contrast_ratio - 1.0) / 2.0, 0.0, 1.0))

        # 3. Autoencoder consistency score (1 - anomaly loss)
        s_ae = float(np.clip(1.0 - ae_anomaly_score, 0.0, 1.0))

        # 4. Acoustic shadow score (1 - shadow intensity ratio)
        if has_shadow:
            s_shadow = float(np.clip(1.0 - shadow_contrast, 0.2, 1.0))
        else:
            s_shadow = 0.10 # Severe penalty for lack of shadow

        # 5. Acoustic SNR score (normalized from [0 dB, 20 dB] -> [0.0, 1.0])
        s_snr = float(np.clip(calibrated_snr_db / 20.0, 0.0, 1.0))

        # 6. Epistemic uncertainty penalty
        unc_penalty = float(self.unc_lambda * max(0.0, mc_epistemic_variance))

        # Evidential combination
        weighted_sum = (
            self.w_yolo * p_cal +
            self.w_cfar * s_cfar +
            self.w_ae * s_ae +
            self.w_shadow * s_shadow +
            self.w_snr * s_snr
        )

        final_norm = float(np.clip(weighted_sum - unc_penalty, 0.0, 1.0))
        final_pct = round(final_norm * 100.0, 1)

        breakdown = {
            "Calibrated YOLO (40%)": round(self.w_yolo * p_cal * 100.0, 1),
            "OS-CFAR Highlight (15%)": round(self.w_cfar * s_cfar * 100.0, 1),
            "AE Consistency (15%)": round(self.w_ae * s_ae * 100.0, 1),
            "Acoustic Shadow (20%)": round(self.w_shadow * s_shadow * 100.0, 1),
            "Acoustic SNR (10%)": round(self.w_snr * s_snr * 100.0, 1),
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
            mc_uncertainty_penalty=round(unc_penalty, 4),
            evidence_breakdown=breakdown
        )
