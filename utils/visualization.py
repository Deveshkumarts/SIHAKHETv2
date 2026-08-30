"""
Visualization utilities for sea debris detection.
Handles bounding box drawing, class badges, confidence scores, and metrics plotting.
Optimized for underwater background contrast (bright high-visibility color palette).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path


# High-contrast color palette (BGR format for OpenCV) to stand out on blue/green underwater backgrounds
MARINE_PALETTE_BGR = [
    (0, 255, 255),    # Yellow (Plastic bottle)
    (0, 165, 255),    # Orange (Plastic bag)
    (0, 0, 255),      # Red (Fishing net)
    (255, 0, 255),    # Magenta (Can)
    (0, 255, 128),    # Spring Green (Glass)
    (255, 255, 0),    # Cyan (Other debris)
    (255, 128, 0),    # Blue-Orange
    (128, 0, 255),    # Purple
    (0, 255, 0),      # Bright Lime
    (255, 192, 203),  # Pink
]


def get_class_color(class_id: int) -> Tuple[int, int, int]:
    """Get distinct BGR color for class ID."""
    return MARINE_PALETTE_BGR[class_id % len(MARINE_PALETTE_BGR)]


def draw_bounding_box(
    image: np.ndarray,
    box_xyxy: Tuple[int, int, int, int],
    label: str,
    color: Tuple[int, int, int],
    line_thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding box and readable text badge on an image in-place.
    """
    x1, y1, x2, y2 = box_xyxy
    h, w = image.shape[:2]

    # Clamp coordinates
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))

    # Draw box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

    # Calculate text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(0.7, w / 1200))
    font_thickness = max(1, int(font_scale * 2))
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

    # Badge background position
    badge_y1 = max(0, y1 - th - baseline - 4)
    badge_y2 = y1
    badge_x1 = x1
    badge_x2 = min(w, x1 + tw + 6)

    # If badge would clip top of image, put inside box
    if y1 - th - baseline - 4 < 0:
        badge_y1 = y1
        badge_y2 = y1 + th + baseline + 4

    # Draw badge background
    cv2.rectangle(image, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)

    # Text color: dark text on light badge
    badge_luminance = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
    text_color = (0, 0, 0) if badge_luminance > 128 else (255, 255, 255)

    text_pos_y = badge_y2 - baseline - 2 if y1 - th - baseline - 4 >= 0 else badge_y2 - baseline
    cv2.putText(
        image,
        label,
        (badge_x1 + 3, text_pos_y),
        font,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA
    )
    return image


def draw_detections(
    image: np.ndarray,
    boxes_xyxy: List[List[float]],
    scores: List[float],
    class_ids: List[int],
    class_names: Dict[int, str],
    show_conf: bool = True
) -> np.ndarray:
    """
    Draw multiple model predictions onto an image.
    """
    annotated = image.copy()
    for box, score, cid in zip(boxes_xyxy, scores, class_ids):
        cname = class_names.get(cid, f"cls_{cid}")
        label = f"{cname} {score:.2f}" if show_conf else cname
        color = get_class_color(cid)
        draw_bounding_box(annotated, tuple(box), label, color)
    return annotated


def draw_ground_truth(
    image: np.ndarray,
    yolo_boxes: List[Tuple[int, float, float, float, float]],
    class_names: Dict[int, str]
) -> np.ndarray:
    """
    Draw ground truth YOLO normalized boxes (cls, cx, cy, w, h) onto an image.
    """
    annotated = image.copy()
    h, w = image.shape[:2]

    for cid, cx, cy, bw, bh in yolo_boxes:
        xmin = int((cx - bw / 2.0) * w)
        ymin = int((cy - bh / 2.0) * h)
        xmax = int((cx + bw / 2.0) * w)
        ymax = int((cy + bh / 2.0) * h)

        cname = class_names.get(cid, f"cls_{cid}")
        label = f"GT: {cname}"
        color = get_class_color(cid)
        draw_bounding_box(annotated, (xmin, ymin, xmax, ymax), label, color, line_thickness=2)

    return annotated


def draw_fps_and_stats(
    image: np.ndarray,
    fps: float,
    detection_count: int,
    model_name: str = "YOLO11"
) -> np.ndarray:
    """Draw FPS and detection count HUD on video/webcam frame."""
    hud_text = f"{model_name} | FPS: {fps:.1f} | Detections: {detection_count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(hud_text, font, font_scale, thickness)

    # Semi-transparent dark background for HUD banner
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (20 + tw, 20 + th + baseline), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    cv2.putText(image, hud_text, (15, 15 + th), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
    return image


def plot_metrics_summary(
    metrics: Dict[str, float],
    class_metrics: Optional[Dict[str, Dict[str, float]]],
    output_path: Union[str, Path]
):
    """
    Plot precision, recall, mAP50, and mAP50-95 bar chart summary.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    keys = ["Precision", "Recall", "mAP@50", "mAP@50-95"]
    vals = [
        metrics.get("precision", 0.0),
        metrics.get("recall", 0.0),
        metrics.get("map50", 0.0),
        metrics.get("map50_95", 0.0),
    ]

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    bars = ax.bar(keys, vals, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11, fontweight="bold")
    ax.set_title("YOLO11 Marine Debris Evaluation Summary", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
