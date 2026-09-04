"""
Confidence Calibration & Temperature Scaling Engine.
Implements:
  1. Temperature Scaling for logit calibration: p_cal = softmax(logits / T)
  2. Expected Calibration Error (ECE) & Maximum Calibration Error (MCE)
  3. Interactive Reliability Diagrams & Calibration Curves
"""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional


class TemperatureScaler:
    """
    Temperature Scaling for post-hoc confidence calibration.
    Scales logits by optimal learned temperature parameter T > 0.
    """

    def __init__(self, temperature: float = 1.35):
        self.temperature = max(0.1, temperature)

    def calibrate_probability(self, uncalibrated_conf: float) -> float:
        """
        Calibrate a single probability using logistic inverse logit scaling.
        """
        conf = np.clip(uncalibrated_conf, 1e-6, 1.0 - 1e-6)
        # Logit
        logit = np.log(conf / (1.0 - conf))
        # Scaled logit
        scaled_logit = logit / self.temperature
        # Calibrated probability
        calibrated_conf = 1.0 / (1.0 + np.exp(-scaled_logit))
        return float(np.clip(calibrated_conf, 0.0, 1.0))

    def calibrate_array(self, confidences: np.ndarray) -> np.ndarray:
        confs = np.clip(confidences, 1e-6, 1.0 - 1e-6)
        logits = np.log(confs / (1.0 - confs))
        scaled_logits = logits / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled_logits))


def compute_calibration_metrics(
    confidences: List[float],
    ground_truth_correctness: List[int],
    num_bins: int = 10
) -> Dict[str, any]:
    """
    Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
    and reliability bin statistics.

    Args:
        confidences: List of predicted confidence scores [0.0, 1.0]
        ground_truth_correctness: Binary list [1 if prediction was correct, 0 otherwise]
        num_bins: Number of confidence bins (standard is 10)
    """
    if len(confidences) == 0:
        return {"ece": 0.0, "mce": 0.0, "bin_confs": [], "bin_accs": [], "bin_counts": []}

    confs = np.array(confidences)
    corrects = np.array(ground_truth_correctness)
    n_samples = len(confs)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    bin_confs = []
    bin_accs = []
    bin_counts = []
    ece = 0.0
    mce = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Samples falling into this confidence interval
        in_bin = (confs > bin_lower) & (confs <= bin_upper) if i > 0 else (confs >= bin_lower) & (confs <= bin_upper)
        count = int(np.sum(in_bin))
        bin_counts.append(count)

        if count > 0:
            avg_conf = float(np.mean(confs[in_bin]))
            avg_acc = float(np.mean(corrects[in_bin]))
            bin_confs.append(avg_conf)
            bin_accs.append(avg_acc)

            cal_err = abs(avg_acc - avg_conf)
            ece += (count / n_samples) * cal_err
            mce = max(mce, cal_err)
        else:
            bin_confs.append((bin_lower + bin_upper) / 2.0)
            bin_accs.append(0.0)

    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "bin_confs": bin_confs,
        "bin_accs": bin_accs,
        "bin_counts": bin_counts,
        "bin_boundaries": bin_boundaries.tolist()
    }


def generate_reliability_diagram(
    uncalibrated_confs: List[float],
    calibrated_confs: List[float],
    ground_truth_correctness: List[int],
    num_bins: int = 10
) -> go.Figure:
    """
    Generates an interactive Plotly Reliability Diagram comparing
    Uncalibrated vs Temperature-Scaled confidence curves against perfect calibration.
    """
    uncal_metrics = compute_calibration_metrics(uncalibrated_confs, ground_truth_correctness, num_bins)
    cal_metrics = compute_calibration_metrics(calibrated_confs, ground_truth_correctness, num_bins)

    fig = go.Figure()

    # 1. Perfect calibration reference line (y = x)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Perfect Calibration (y = x)",
        line=dict(color="rgba(255, 255, 255, 0.4)", dash="dash", width=2)
    ))

    # 2. Uncalibrated Curve
    fig.add_trace(go.Scatter(
        x=uncal_metrics["bin_confs"],
        y=uncal_metrics["bin_accs"],
        mode="lines+markers",
        name=f"Raw Model (ECE: {uncal_metrics['ece']:.3f}, MCE: {uncal_metrics['mce']:.3f})",
        line=dict(color="#ff4757", width=2.5),
        marker=dict(size=8, symbol="circle")
    ))

    # 3. Temperature-Scaled Calibrated Curve
    fig.add_trace(go.Scatter(
        x=cal_metrics["bin_confs"],
        y=cal_metrics["bin_accs"],
        mode="lines+markers",
        name=f"Temperature Scaled (ECE: {cal_metrics['ece']:.3f}, MCE: {cal_metrics['mce']:.3f})",
        line=dict(color="#2ecc71", width=3),
        marker=dict(size=9, symbol="diamond")
    ))

    fig.update_layout(
        title=dict(
            text=f"Confidence Calibration Reliability Diagram (ECE Reduction: {uncal_metrics['ece']:.3f} → {cal_metrics['ece']:.3f})",
            font=dict(color="#ffffff", size=14)
        ),
        xaxis=dict(
            title="Mean Predicted Confidence",
            range=[0, 1],
            gridcolor="rgba(0, 140, 200, 0.12)",
            tickfont=dict(color="#8da8ba")
        ),
        yaxis=dict(
            title="Empirical Accuracy / True Positive Fraction",
            range=[0, 1],
            gridcolor="rgba(0, 140, 200, 0.12)",
            tickfont=dict(color="#8da8ba")
        ),
        paper_bgcolor="#060c18",
        plot_bgcolor="#081426",
        legend=dict(
            x=0.03, y=0.95,
            bgcolor="rgba(6, 12, 24, 0.85)",
            bordercolor="#1f4260",
            borderwidth=1,
            font=dict(color="#ddd", size=10)
        ),
        height=400,
        margin=dict(l=40, r=20, t=50, b=40)
    )

    return fig
