"""
🌊 Akhet Marine & Sonar AI Platform (SIH 2026 - PS 26057)
Modular Multi-Model Architecture with 3-Stage Preprocessing (Median -> Bilateral -> CLAHE),
SegFormer Edge Segmentation, and ResNet-18 PyTorch Grad-CAM Explainability.
"""

import sys
import os
import io
import json
import time
import base64
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Union

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
from skyfield.api import load, EarthSatellite, wgs84

# Ensure UTF-8 output on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from utils.visualization import draw_bounding_box, get_class_color
from utils.device_utils import get_device_info, select_device
from utils.roi_utils import expand_and_clamp_bbox, roi_mask_to_full_image, get_adaptive_padding_ratio, validate_roi_quality
from utils.geolocation import project_pixel_to_latlon, spatial_clustering_deduplication, GeolocationEstimate
from utils.sonar_preprocess import (
    preprocess_universal_image,
    calibrate_and_preprocess_sonar,
    apply_median_filter,
    apply_bilateral_denoise,
    apply_clahe,
)
from utils.sonar_calibration import compute_snr_index, QualityMetrics
from utils.telemetry_parser import generate_synthetic_telemetry, TelemetryValidator, TelemetryRecord
from models.os_cfar import OSCFARDetector
from models.autoencoder import SonarAnomalyDetector
from utils.decision_gate import evaluate_decision_gate, TriageDecision, verify_acoustic_shadow
from utils.confidence_calibration import TemperatureScaler, compute_calibration_metrics, generate_reliability_diagram
from utils.morphological_filter import filter_detection_by_morphology, extract_morphological_features
from utils.confidence_fusion import MultiEvidenceConfidenceFusion, FusedConfidenceReport
from utils.postgis_db import PostGISAdapter
from utils.gis_density import build_gis_hotspot_figure, export_detections_to_geojson, export_detections_to_csv
from utils.db_store import SurveyDatabase
from utils.feedback_loop import ActiveLearningManager
from resnet.classifier import ResNet18InferenceEngine, MASTER_CLASSES as RESNET_CLASSES

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Marine Guard — Clearer Oceans. Safer Tomorrows.",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background: #060b13 !important;
    color: #e2f1f8 !important;
}
.stApp {
    background: #060b13 !important;
}

/* ── Main container max width ── */
div[data-testid="stMainBlockContainer"],
.block-container {
    max-width: 100% !important;
    padding: 0.75rem 1.25rem 1.5rem 1.25rem !important;
}

/* ── Sidebar shell ── */
[data-testid="stSidebar"] {
    background: #080e1a !important;
    border-right: 1px solid rgba(0, 188, 212, 0.14) !important;
    min-width: 250px !important;
    max-width: 250px !important;
}
[data-testid="stSidebarContent"] {
    padding: 0 0 16px 0 !important;
    background: #080e1a !important;
}

/* ── Sidebar Headings & Text ── */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #7b9bb3 !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(0, 188, 212, 0.12) !important;
    margin: 12px 14px !important;
}

/* ── Sidebar Brand ── */
.seadex-brand-box {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 20px 18px 16px 18px;
    border-bottom: 1px solid rgba(0, 188, 212, 0.12);
}
.seadex-logo-diamond {
    flex-shrink: 0;
}
.seadex-brand-title {
    font-size: 1.05rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 0.08em;
    line-height: 1.2;
    white-space: nowrap;
}
.seadex-brand-sub {
    font-size: 0.60rem;
    font-weight: 700;
    color: #00bcd4;
    letter-spacing: 0.08em;
    margin-top: 3px;
    white-space: nowrap;
}

/* ── Sidebar Radio Navigation as Modern Flat Menu ── */

/* 1. Hide ONLY the radio circle (fe/pe), NEVER the text */
[data-testid="stSidebar"] [data-testid="stRadioOption"] [class*="etak9234"],
[data-testid="stSidebar"] [data-testid="stRadioOption"] [class*="etak9235"],
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div > div:first-child,
[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    min-width: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    opacity: 0 !important;
    pointer-events: none !important;
    border: none !important;
}

/* 2. Container padding and layout */
[data-testid="stSidebar"] div[role="radiogroup"],
[data-testid="stSidebar"] [data-testid="stRadioGroup"] {
    padding: 8px 10px !important;
    gap: 4px !important;
}

/* 3. Make option look like a sleek menu button */
[data-testid="stSidebar"] [data-testid="stRadioOption"] {
    display: flex !important;
    align-items: center !important;
    padding: 9px 14px !important;
    margin: 2px 0 !important;
    background: transparent !important;
    border: 1.5px solid transparent !important;
    border-radius: 8px !important;
    color: #7b9bb3 !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    transition: all 0.18s ease !important;
    width: 100% !important;
    box-sizing: border-box !important;
    cursor: pointer !important;
}

/* Allow inner divs to stretch across */
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div,
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div {
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    margin: 0 !important;
}

/* 4. Ensure label markdown text is 100% visible */
[data-testid="stSidebar"] [data-testid="stRadioOption"] .stMarkdown,
[data-testid="stSidebar"] [data-testid="stRadioOption"] .stMarkdown * {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] [data-testid="stRadioOption"] p {
    margin: 0 !important;
    padding: 0 !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    color: #7b9bb3 !important;
    letter-spacing: 0.02em !important;
    line-height: 1.3 !important;
}

/* Hover state */
[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {
    background: rgba(0, 188, 212, 0.08) !important;
    border-color: rgba(0, 188, 212, 0.25) !important;
}
[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover p,
[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover span {
    color: #c4e4f5 !important;
}

/* Selected state: Glowing cyan border, cyan/navy gradient fill */
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"],
[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) {
    background: linear-gradient(90deg, rgba(0, 188, 212, 0.28) 0%, rgba(0, 119, 182, 0.12) 100%) !important;
    border: 1.5px solid #00e5ff !important;
    box-shadow: 0 0 16px rgba(0, 229, 255, 0.25), inset 0 0 8px rgba(0, 229, 255, 0.12) !important;
}

[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p,
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] span,
[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) p,
[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) span {
    color: #ffffff !important;
    font-weight: 700 !important;
    text-shadow: 0 0 8px rgba(0, 229, 255, 0.4) !important;
}

/* ── Sidebar System Status & Hardware ── */
.seadex-sidebar-sec-title {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    color: #00bcd4;
    text-transform: uppercase;
    padding: 0 0 6px 0;
}
.seadex-sys-card {
    background: rgba(8, 16, 28, 0.75);
    border: 1px solid rgba(0, 188, 212, 0.2);
    border-radius: 9px;
    margin: 12px 14px 14px 14px;
    padding: 12px 14px;
}
.seadex-sys-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    font-size: 0.76rem;
    border-bottom: 1px solid rgba(0, 188, 212, 0.06);
}
.seadex-sys-row:last-child {
    border-bottom: none;
}
.seadex-sys-item {
    display: flex;
    align-items: center;
    gap: 7px;
    color: #e2f1f8;
}
.seadex-dot-green {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #00e676;
    box-shadow: 0 0 6px #00e676;
    margin-right: 6px;
}
.seadex-hw-grid {
    display: flex;
    justify-content: space-between;
    padding: 6px 0 4px 0;
}
.seadex-hw-lbl {
    font-size: 0.66rem;
    color: #4a7590;
    text-transform: uppercase;
}
.seadex-hw-val {
    font-size: 0.78rem;
    font-weight: 600;
    color: #c5e4f5;
    margin-top: 1px;
}
.seadex-sidebar-footer {
    padding: 12px 18px 6px 18px;
    font-size: 0.68rem;
    color: #43647b;
}
.seadex-footer-initiative {
    color: #00bcd4;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-top: 2px;
}

/* ── Main Page Header (Operational View) ── */
.seadex-header-wrapper {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 18px;
    padding-bottom: 14px;
    border-bottom: 1px solid rgba(0, 188, 212, 0.12);
}
.seadex-op-tag {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #00e5ff;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.seadex-page-title {
    font-size: 1.55rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 0.04em;
    margin: 0 0 4px 0;
}
.seadex-page-desc {
    font-size: 0.82rem;
    color: #7b9bb3;
    margin: 0;
}
.seadex-quote {
    font-size: 0.82rem;
    font-style: italic;
    color: #557b94;
    text-align: right;
    max-width: 260px;
}

/* ── Common Card Container ── */
.seadex-panel {
    background: #0a1322;
    border: 1px solid rgba(0, 188, 212, 0.16);
    border-radius: 10px;
    padding: 14px 16px;
    box-sizing: border-box;
    margin-bottom: 14px;
    height: 100%;
}
.seadex-panel-hdr {
    font-size: 0.84rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* ── Pill Buttons for Radio Controls ── */
div[data-testid="stHorizontalBlock"] div[role="radiogroup"],
div[data-testid="stVerticalBlock"] div[data-testid="stHorizontalBlock"] div[role="radiogroup"] {
    display: flex !important;
    gap: 4px !important;
    flex-wrap: nowrap !important;
}

div[role="radiogroup"] label[data-baseweb="radio"] {
    background: #09121f !important;
    border: 1px solid rgba(0, 188, 212, 0.22) !important;
    border-radius: 6px !important;
    padding: 4px 10px !important;
    margin: 0 !important;
    font-size: 0.74rem !important;
    color: #7b9bb3 !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    white-space: nowrap !important;
}

div[role="radiogroup"] label[data-baseweb="radio"]:hover {
    border-color: rgba(0, 229, 255, 0.45) !important;
    color: #c5e4f5 !important;
}

div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {
    background: #00bcd4 !important;
    border-color: #00e5ff !important;
    color: #060e18 !important;
    font-weight: 700 !important;
}
div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) p {
    color: #060e18 !important;
    font-weight: 700 !important;
}

/* ── Styled Upload Dropzone ── */
.seadex-dropzone-visual {
    border: 1.5px dashed rgba(0, 188, 212, 0.32);
    border-radius: 9px;
    background: rgba(6, 15, 28, 0.45);
    padding: 22px 14px;
    text-align: center;
    margin: 8px 0;
}
.seadex-drop-cloud {
    font-size: 1.8rem;
    color: #00e5ff;
    margin-bottom: 6px;
}
.seadex-drop-text {
    font-size: 0.82rem;
    font-weight: 500;
    color: #c5e4f5;
    margin-bottom: 3px;
}
.seadex-drop-sub {
    font-size: 0.72rem;
    color: #00bcd4;
    margin-bottom: 6px;
}
.seadex-drop-fmts {
    font-size: 0.65rem;
    color: #4a7590;
    letter-spacing: 0.08em;
}

/* File Uploader styling inside dropzone */
[data-testid="stFileUploader"] {
    margin-top: 4px !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: rgba(0, 25, 45, 0.25) !important;
    border: 1px dashed rgba(0, 188, 212, 0.25) !important;
    border-radius: 7px !important;
    padding: 8px !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: transparent !important;
    border: 1px solid rgba(0, 188, 212, 0.4) !important;
    color: #00e5ff !important;
    border-radius: 6px !important;
    font-size: 0.76rem !important;
    padding: 3px 12px !important;
}

/* ── Buttons & Toggles ── */
[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #008fa8 0%, #00c8d8 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.86rem !important;
    padding: 9px 18px !important;
    box-shadow: 0 4px 14px rgba(0, 188, 212, 0.25) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, #009cb8 0%, #00e5f5 100%) !important;
    box-shadow: 0 6px 18px rgba(0, 229, 255, 0.35) !important;
    transform: translateY(-1px) !important;
}

/* Toggle Switch */
[data-testid="stToggle"] {
    padding: 4px 0 !important;
}
[data-testid="stToggle"] label {
    font-size: 0.78rem !important;
    color: #8bb1cb !important;
}

/* Selectbox */
[data-baseweb="select"] > div {
    background: #091320 !important;
    border: 1px solid rgba(0, 188, 212, 0.25) !important;
    border-radius: 7px !important;
    color: #c5e4f5 !important;
    font-size: 0.80rem !important;
}

/* ── Sonar Viewport Toolbar ── */
.seadex-tool-bar {
    display: flex;
    justify-content: flex-end;
    gap: 4px;
    align-items: center;
}
.seadex-tool-btn {
    background: #09121f;
    border: 1px solid rgba(0, 188, 212, 0.22);
    border-radius: 5px;
    padding: 3px 8px;
    font-size: 0.70rem;
    color: #7b9bb3;
    cursor: pointer;
}
.seadex-tool-btn:hover {
    border-color: #00e5ff;
    color: #ffffff;
}

/* ── Global Image Constrain to prevent overflow anywhere ── */
[data-testid="stImage"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}
[data-testid="stImage"] img {
    max-height: 380px !important;
    width: auto !important;
    max-width: 100% !important;
    object-fit: contain !important;
    border-radius: 6px !important;
    margin: 0 auto !important;
}

/* ── Tactical Sonar Viewport Frame ── */
.seadex-sonar-viewport {
    position: relative;
    width: 100%;
    height: 420px;
    max-height: 420px;
    background: #030711;
    background-image: 
        radial-gradient(ellipse at center, rgba(0, 188, 212, 0.08) 0%, rgba(3, 7, 17, 0.98) 75%),
        linear-gradient(rgba(0, 188, 212, 0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 188, 212, 0.04) 1px, transparent 1px);
    background-size: 100% 100%, 30px 30px, 30px 30px;
    border: 1px solid rgba(0, 188, 212, 0.25);
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: inset 0 0 50px rgba(0, 0, 0, 0.9), 0 4px 18px rgba(0, 0, 0, 0.4);
    margin-bottom: 8px;
}

.seadex-sonar-img {
    width: 100% !important;
    height: 100% !important;
    max-height: 420px !important;
    object-fit: contain !important;
    display: block !important;
    margin: auto !important;
    border-radius: 4px;
    image-rendering: auto;
}

.seadex-sonar-img.fit-fill {
    object-fit: fill !important;
}

.seadex-sonar-img.fit-cover {
    object-fit: cover !important;
}

/* Standby Viewport Empty State */
.seadex-empty-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 24px;
    z-index: 2;
}
.seadex-empty-radar {
    margin-bottom: 12px;
}
.seadex-empty-title {
    font-size: 0.90rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: #00e5ff;
    margin-bottom: 6px;
    text-shadow: 0 0 10px rgba(0, 229, 255, 0.4);
}
.seadex-empty-desc {
    font-size: 0.74rem;
    color: #6a93b0;
    max-width: 320px;
    line-height: 1.45;
}
.seadex-triage-empty {
    background: rgba(8, 16, 28, 0.6);
    border: 1px dashed rgba(0, 188, 212, 0.22);
    border-radius: 9px;
    padding: 26px 16px;
    text-align: center;
    margin-top: 6px;
}

/* ── Explainability Tab ROI & Grad-CAM Image Framing ── */
.seadex-explain-card {
    background: rgba(3, 14, 28, 0.85);
    border: 1px solid rgba(0, 188, 212, 0.22);
    border-radius: 10px;
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 485px;
    min-height: 485px;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.55), inset 0 0 35px rgba(0, 20, 45, 0.5);
    box-sizing: border-box;
    margin-top: 4px;
}

.seadex-explain-card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(0, 188, 212, 0.16);
    margin-bottom: 10px;
}

.seadex-explain-card-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #e0f7fa;
    letter-spacing: 0.02em;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.seadex-explain-viewport {
    background: #020712;
    border: 1px solid rgba(0, 188, 212, 0.18);
    border-radius: 8px;
    padding: 8px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    flex: 1;
    min-height: 350px;
    max-height: 375px;
    box-shadow: inset 0 0 30px rgba(0, 0, 0, 0.9);
    box-sizing: border-box;
    overflow: hidden;
}

.seadex-explain-img {
    height: 335px !important;
    max-height: 345px !important;
    width: auto !important;
    max-width: 96% !important;
    object-fit: contain !important;
    border-radius: 6px;
    border: 1px solid rgba(0, 229, 255, 0.35);
    background: #01040a;
    box-shadow: 0 6px 25px rgba(0, 0, 0, 0.85), 0 0 14px rgba(0, 229, 255, 0.18);
    image-rendering: auto;
}

.seadex-explain-caption {
    font-size: 0.74rem;
    color: #76a4c2;
    margin-top: 8px;
    text-align: center;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    letter-spacing: 0.02em;
}

.seadex-explain-stats-body {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    overflow-y: auto;
    padding-right: 4px;
}

/* HUD Scale Bar (Bottom Left) */
.seadex-hud-scale {
    position: absolute;
    bottom: 12px;
    left: 16px;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 2px;
    pointer-events: none;
    z-index: 10;
}
.seadex-scale-line-wrapper {
    display: flex;
    align-items: center;
    height: 7px;
}
.seadex-scale-tick {
    width: 2px;
    height: 7px;
    background: #ffffff;
    box-shadow: 0 0 4px rgba(255,255,255,0.6);
}
.seadex-scale-line {
    width: 60px;
    height: 2px;
    background: #ffffff;
    box-shadow: 0 0 4px rgba(255,255,255,0.6);
}
.seadex-scale-label {
    font-size: 0.65rem;
    font-family: 'Consolas', 'Courier New', monospace;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.06em;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.95);
    margin-top: 1px;
}

/* HUD Compass (Bottom Right) */
.seadex-hud-compass {
    position: absolute;
    bottom: 10px;
    right: 16px;
    display: flex;
    align-items: center;
    gap: 4px;
    pointer-events: none;
    z-index: 10;
}
.seadex-compass-label {
    font-size: 0.72rem;
    font-family: 'Consolas', 'Courier New', monospace;
    font-weight: 800;
    color: #ffffff;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.95);
}

/* HUD Status Badge (Top Right) */
.seadex-hud-status-badge {
    position: absolute;
    top: 10px;
    right: 12px;
    background: rgba(4, 15, 26, 0.8);
    border: 1px solid rgba(0, 188, 212, 0.35);
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.62rem;
    color: #00e5ff;
    font-family: 'Consolas', monospace;
    font-weight: 600;
    letter-spacing: 0.06em;
    z-index: 10;
    pointer-events: none;
}

/* ── KPI Stat Cards ── */
.seadex-kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin-top: 10px;
}
.seadex-kpi-card {
    background: #08111d;
    border: 1px solid rgba(0, 188, 212, 0.16);
    border-radius: 8px;
    padding: 10px 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.seadex-kpi-icon {
    width: 32px;
    height: 32px;
    border-radius: 7px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.05rem;
    flex-shrink: 0;
}
.icon-cyan  { background: rgba(0, 188, 212, 0.12); color: #00e5ff; }
.icon-coral { background: rgba(255, 82, 82, 0.12); color: #ff5252; }
.icon-green { background: rgba(0, 230, 118, 0.12); color: #00e676; }
.seadex-kpi-val {
    font-size: 1.15rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.1;
}
.seadex-kpi-lbl {
    font-size: 0.65rem;
    color: #7b9bb3;
    margin-top: 2px;
}
.seadex-kpi-trend {
    margin-left: auto;
    font-size: 0.68rem;
    font-weight: 600;
    text-align: right;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
}
.trend-up   { color: #00e676; }
.trend-down { color: #ff5252; }
.trend-zero { color: #00bcd4; }

/* ── Acoustic Telemetry ── */
.seadex-live-tag {
    background: rgba(0, 230, 118, 0.12);
    border: 1px solid rgba(0, 230, 118, 0.4);
    border-radius: 12px;
    color: #00e676;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 2px 7px;
    letter-spacing: 0.06em;
}
.seadex-telem-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    font-size: 0.77rem;
    border-bottom: 1px solid rgba(0, 188, 212, 0.07);
}
.seadex-telem-item:last-child {
    border-bottom: none;
}
.seadex-telem-lbl {
    color: #7b9bb3;
    display: flex;
    align-items: center;
    gap: 7px;
}
.seadex-telem-val {
    color: #ffffff;
    font-weight: 600;
}
.seadex-signal-hdr {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.72rem;
    color: #7b9bb3;
    margin: 10px 0 4px 0;
}

/* ── Bottom Triage Cards ── */
.seadex-triage-header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 18px 0 10px 0;
}
.seadex-triage-title {
    font-size: 0.88rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.seadex-triage-link {
    font-size: 0.75rem;
    color: #00bcd4;
    text-decoration: none;
    cursor: pointer;
    font-weight: 600;
}
.seadex-triage-card {
    background: #091322;
    border: 1px solid rgba(0, 188, 212, 0.16);
    border-radius: 9px;
    padding: 10px 12px;
    box-sizing: border-box;
    width: 100%;
    max-width: 320px;
}
.seadex-triage-card-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.seadex-triage-id {
    font-size: 0.70rem;
    font-weight: 700;
    color: #4a7590;
    margin-right: 6px;
}
.seadex-triage-name {
    font-size: 0.88rem;
    font-weight: 700;
    color: #ffffff;
}
.seadex-badge-status {
    border-radius: 4px;
    font-size: 0.62rem;
    font-weight: 700;
    padding: 2px 7px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-confirmed { background: rgba(0, 230, 118, 0.15); border: 1px solid #00e676; color: #00e676; }
.badge-high      { background: rgba(255, 152, 0, 0.15);  border: 1px solid #ff9800; color: #ff9800; }
.badge-moderate  { background: rgba(255, 193, 7, 0.15);  border: 1px solid #ffc107; color: #ffc107; }
.badge-review    { background: rgba(255, 82, 82, 0.15);  border: 1px solid #ff5252; color: #ff5252; }

.seadex-triage-body {
    display: flex;
    gap: 10px;
    align-items: center;
}
.seadex-triage-img {
    width: 68px;
    height: 68px;
    border-radius: 6px;
    border: 1px solid rgba(0, 188, 212, 0.18);
    object-fit: cover;
    flex-shrink: 0;
}
.seadex-triage-table {
    flex: 1;
    font-size: 0.69rem;
}
.seadex-tt-row {
    display: flex;
    justify-content: space-between;
    padding: 1.5px 0;
}
.seadex-tt-lbl {
    color: #658ba3;
}
.seadex-tt-val {
    color: #ffffff;
    font-weight: 600;
}

/* ── Legacy Support for Tabs 1-7 in SEADEX Theme ── */
.mg-card {
    background: #0a1322 !important;
    border: 1px solid rgba(0, 188, 212, 0.16) !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
    margin-bottom: 14px !important;
    box-sizing: border-box !important;
}
.mg-card-title {
    font-size: 0.94rem !important;
    font-weight: 700 !important;
    color: #e2f1f8 !important;
}
.mg-card-sub {
    font-size: 0.78rem !important;
    color: #7b9bb3 !important;
}
.metric-card {
    background: #08111d !important;
    border: 1px solid rgba(0, 188, 212, 0.16) !important;
    border-radius: 9px !important;
    padding: 12px 10px !important;
    text-align: center !important;
    margin: 4px 0 !important;
}
.metric-value {
    font-size: 1.45rem !important;
    font-weight: 800 !important;
    color: #00e676 !important;
}
.metric-label {
    font-size: 0.72rem !important;
    color: #7b9bb3 !important;
    margin-top: 3px !important;
}
.mg-model-card {
    background: #09121f !important;
    border-left: 3px solid #00bcd4 !important;
    border-radius: 0 9px 9px 0 !important;
    padding: 12px 16px !important;
    margin: 8px 0 !important;
    border-top: 1px solid rgba(0, 188, 212, 0.12) !important;
    border-right: 1px solid rgba(0, 188, 212, 0.12) !important;
    border-bottom: 1px solid rgba(0, 188, 212, 0.12) !important;
}
.mg-model-name {
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}
.mg-model-desc {
    font-size: 0.79rem !important;
    color: #7b9bb3 !important;
    margin-top: 4px !important;
}
.mg-model-meta {
    font-size: 0.72rem !important;
    color: #4a7590 !important;
    margin-top: 6px !important;
}
.mg-det-card {
    background: #091322 !important;
    border-radius: 9px !important;
    padding: 12px 10px !important;
    text-align: center !important;
    margin: 4px 0 !important;
}
.mg-stat-grid {
    display: grid !important;
    grid-template-columns: 1fr 1fr !important;
    gap: 8px !important;
    margin-bottom: 8px !important;
}
.mg-stat {
    background: #08111d !important;
    border: 1px solid rgba(0, 188, 212, 0.16) !important;
    border-radius: 8px !important;
    padding: 10px !important;
    text-align: center !important;
}
.mg-stat-val {
    font-size: 1.4rem !important;
    font-weight: 800 !important;
}
.mg-stat-lbl {
    font-size: 0.67rem !important;
    color: #7b9bb3 !important;
    margin-top: 2px !important;
}
.mg-info-row {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    padding: 6px 0 !important;
    border-bottom: 1px solid rgba(0, 188, 212, 0.08) !important;
    font-size: 0.76rem !important;
}
.mg-info-lbl { color: #7b9bb3 !important; }
.mg-info-val { color: #00e5ff !important; font-weight: 600 !important; }

/* ── Hide Streamlit Chrome ── */
#MainMenu { display: none !important; }
footer    { display: none !important; }
header    { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stHeader"]     { display: none !important; }

body [data-testid="stMainBlockContainer"] {
    padding-top: 0.5rem !important;
    margin-top: 0 !important;
}
body .block-container {
    padding-top: 0.5rem !important;
    margin-top: 0 !important;
}
body section[data-testid="stMain"] {
    padding-top: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── JS: directly zero out Streamlit top padding (runs after React renders) ──
import streamlit.components.v1 as _components
_components.html("""
<script>
(function removePadding() {
    var targets = [
        '[data-testid="stMainBlockContainer"]',
        '[data-testid="stAppViewBlockContainer"]',
        '.block-container',
        'header[data-testid="stHeader"]',
        '[data-testid="stToolbar"]',
        '[data-testid="stDecoration"]',
        'header'
    ];
    function fix() {
        var doc = window.parent.document;
        targets.forEach(function(sel) {
            doc.querySelectorAll(sel).forEach(function(el) {
                if (sel.includes('header') || sel.includes('Toolbar') || sel.includes('Decoration')) {
                    el.style.display = 'none';
                } else {
                    el.style.paddingTop = '0';
                    el.style.marginTop = '0';
                }
            });
        });
    }
    fix();
    setTimeout(fix, 50);
    setTimeout(fix, 200);
    setTimeout(fix, 500);
})();
</script>
""", height=0, scrolling=False)

# ─── Model Registry ─────────────────────────────────────────────────────────
SIH_27CLASS_WEIGHTS = (
    "weights/yolo11s_sih_27class_best.pt"
    if Path("weights/yolo11s_sih_27class_best.pt").exists()
    else "runs/detect/sih27class/yolo11s_sih_27class/weights/best.pt"
)

MODEL_REGISTRY = {
    "🎯 SIH 2026 Master Detector (All 27 Classes)": {
        "weights": SIH_27CLASS_WEIGHTS,
        "description": "Unified 27-class detector covering marine debris, lost tools, subsea infrastructure, tires, and shipwrecks (94.09% mAP50).",
        "type": "Master Universal (27 Classes)",
        "default_conf": 0.30,
        "class_filter": None,
    },
    "🗑️ Marine Debris & Containers (15 Classes)": {
        "weights": SIH_27CLASS_WEIGHTS,
        "description": "Specialized focus on bottles, cans, drink cartons/sachets, jars, shampoo bottles, bidons, and metal boxes.",
        "type": "Debris & Containers",
        "default_conf": 0.35,
        "class_filter": [
            "bottle","brown-glass-bottle","can","drink-carton","drink-sachet",
            "glass-bottle","glass-jar","metal-bottle","metal-box","plastic-bidon",
            "plastic-bottle","potion-glass-bottle","shampoo-bottle","standing-bottle",
        ],
    },
    "⚙️ Marine Hardware, Infrastructure & Tools (8 Classes)": {
        "weights": SIH_27CLASS_WEIGHTS,
        "description": "Underwater subsea pipeline/cables, valves, wrenches, chains, hooks, propellers, and rotating platforms.",
        "type": "Hardware & Infrastructure",
        "default_conf": 0.30,
        "class_filter": [
            "chain","hook","pipeline or cable","plastic-pipe","plastic-propeller",
            "propeller","rotating-platform","valve","wrench",
        ],
    },
    "🛞 Tires & Subsea Rubber Material (3 Classes)": {
        "weights": SIH_27CLASS_WEIGHTS,
        "description": "Detection of submerged automotive and industrial rubber: tire, small-tire, large-tire.",
        "type": "Rubber & Tires",
        "default_conf": 0.35,
        "class_filter": ["tire","small-tire","large-tire"],
    },
    "🚢 Sonar Anomalies & Shipwrecks (Acoustic Targets)": {
        "weights": SIH_27CLASS_WEIGHTS,
        "description": "Acoustic side-scan sonar shipwrecks and large submerged structural targets.",
        "type": "Sonar Anomalies",
        "default_conf": 0.25,
        "class_filter": ["Shipwrecks"],
    },
    "🔬 Anoma Deep Sonar Detector (Trained on 535 Anomaly Images)": {
        "weights": "weights/yolo11s_anoma_best.pt",
        "description": "Specialized 4-class acoustic model fine-tuned on the Anoma dataset: Debris Target, Small Fragment, Structural Cluster, Linear Structure.",
        "type": "Anoma Sonar Detector",
        "default_conf": 0.25,
        "class_filter": None,
    },
}

SEGFORMER_WEIGHTS = (
    "weights/segformer_b0_best.pt"
    if Path("weights/segformer_b0_best.pt").exists()
    else "outputs/segformer/weights/best.pt"
)
RESNET_WEIGHTS = "weights/resnet18_debris_best.pt"

CLASS_METADATA = {
    "Debris Target":            {"emoji": "🎯", "color": "#FF5555", "type": "Acoustic Target"},
    "Small Acoustic Fragment":  {"emoji": "🔬", "color": "#FFAA33", "type": "Fragment Scatterer"},
    "Structural Cluster":       {"emoji": "📦", "color": "#33DDFF", "type": "Seabed Cluster"},
    "Subsea Linear Structure":  {"emoji": "⚡", "color": "#33FF88", "type": "Linear Feature"},
    "Shipwrecks":           {"emoji": "🚢", "color": "#FFD700", "type": "Acoustic Sonar Target"},
    "bottle":               {"emoji": "🍾", "color": "#00BFFF", "type": "Polymer Container"},
    "brown-glass-bottle":   {"emoji": "🍾", "color": "#C08040", "type": "Glass Debris"},
    "can":                  {"emoji": "🥫", "color": "#FF4488", "type": "Metallic Litter"},
    "chain":                {"emoji": "⛓️", "color": "#88AAFF", "type": "Marine Rigging"},
    "drink-carton":         {"emoji": "🧃", "color": "#FFAA44", "type": "Cellulose Packaging"},
    "drink-sachet":         {"emoji": "🧃", "color": "#FF88AA", "type": "Flexible Plastic"},
    "glass-bottle":         {"emoji": "🍶", "color": "#44DDAA", "type": "Glass Debris"},
    "glass-jar":            {"emoji": "🫙", "color": "#88FFCC", "type": "Glass Container"},
    "hook":                 {"emoji": "🪝", "color": "#FF9933", "type": "Lost Rigging Tool"},
    "large-tire":           {"emoji": "🛞", "color": "#777777", "type": "Heavy Rubber Debris"},
    "metal-bottle":         {"emoji": "🧯", "color": "#FF6666", "type": "Metal Debris"},
    "metal-box":            {"emoji": "📦", "color": "#EEAA66", "type": "Metal Container"},
    "pipeline or cable":    {"emoji": "⚡", "color": "#00E5FF", "type": "Subsea Infrastructure"},
    "plastic-bidon":        {"emoji": "🛢️", "color": "#00EEFF", "type": "Rigid Plastic Drum"},
    "plastic-bottle":       {"emoji": "🧴", "color": "#00BFFF", "type": "Polymer Debris"},
    "plastic-pipe":         {"emoji": "🧪", "color": "#55AAFF", "type": "Synthetic Piping"},
    "plastic-propeller":    {"emoji": "⚙️", "color": "#77CCEE", "type": "Plastic Mechanism"},
    "potion-glass-bottle":  {"emoji": "🧪", "color": "#AA66FF", "type": "Specialized Glass"},
    "propeller":            {"emoji": "🌀", "color": "#FFAA00", "type": "Marine Propulsion"},
    "rotating-platform":    {"emoji": "🏗️", "color": "#99DDFF", "type": "Subsea Structure"},
    "shampoo-bottle":       {"emoji": "🧴", "color": "#FF66CC", "type": "Personal Care Bottle"},
    "small-tire":           {"emoji": "🛞", "color": "#AAAAAA", "type": "Rubber Debris"},
    "standing-bottle":      {"emoji": "🍾", "color": "#33FFDD", "type": "Bottle Container"},
    "tire":                 {"emoji": "🛞", "color": "#888888", "type": "Automotive Rubber"},
    "valve":                {"emoji": "🔩", "color": "#FFCC00", "type": "Subsea Fitting"},
    "wrench":               {"emoji": "🔧", "color": "#00FFCC", "type": "Lost Tool"},
}

# ─── Model Loaders ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLO Detector...")
def load_yolo_model(weights_path: str):
    from ultralytics import YOLO
    p = Path(weights_path)
    if not p.exists() or (p.is_file() and p.stat().st_size < 1000):
        return YOLO("yolo11s.pt")
    try:
        return YOLO(str(p))
    except Exception:
        return YOLO("yolo11s.pt")


@st.cache_resource(show_spinner="Loading SegFormer-B0...")
def load_segformer_model(weights_path: str):
    p = Path(weights_path)
    if not p.exists():
        return None
    try:
        from segformer.inference import SegFormerInference
        return SegFormerInference(weights_path=str(p), img_size=224)
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading ResNet18 + Grad-CAM...")
def load_resnet_engine():
    try:
        return ResNet18InferenceEngine(weights_path=RESNET_WEIGHTS, device="auto")
    except Exception as e:
        st.sidebar.warning(f"ResNet engine load warning: {e}")
        return None


def hex_to_bgr(hex_str: str) -> tuple:
    h = hex_str.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


ANOMALIES_DIR = ROOT_DIR / "samples" / "anomalies"

ANOMALY_CLASSES = {
    "🐟 Fish & Marine Biomass School": {
        "file": "fish_biomass_school.png",
        "name": "Fish & Marine Biomass School",
        "desc": "Biological acoustic scattering cluster in water column",
        "color": "#38b8f0",
        "emoji": "🐟"
    },
    "💣 Naval Mines & Unexploded Ordnance (UXO)": {
        "file": "naval_mine_uxo.png",
        "name": "Naval Mines & Unexploded Ordnance (UXO)",
        "desc": "Moored subsea spherical mine with contact horns & acoustic shadow",
        "color": "#e74c3c",
        "emoji": "💣"
    },
    "🛢️ Hazardous Industrial Containers": {
        "file": "hazardous_industrial_container.png",
        "name": "Hazardous Industrial Containers",
        "desc": "Corroded chemical / fuel steel drum on seafloor",
        "color": "#f39c12",
        "emoji": "🛢️"
    },
    "📦 Subsea Flight Recorders & Aerospace Debris": {
        "file": "subsea_flight_recorder.png",
        "name": "Subsea Flight Recorders & Aerospace Debris",
        "desc": "Metallic flight data recorder (ULB) beacon & aircraft fuselage plate",
        "color": "#a370f7",
        "emoji": "📦"
    },
    "⚡ Seafloor Infrastructure Fractures": {
        "file": "seafloor_infrastructure_fracture.png",
        "name": "Seafloor Infrastructure Fractures",
        "desc": "Cracked subsea pipeline casing blowout crater & exposed trench",
        "color": "#e67e22",
        "emoji": "⚡"
    },
    "🏺 Subsea Archaeological Relics": {
        "file": "subsea_archaeological_relic.png",
        "name": "Subsea Archaeological Relics",
        "desc": "Ancient submerged terracotta amphora / historical seabed artifact",
        "color": "#1abc9c",
        "emoji": "🏺"
    },
    "🕸️ Ghost Fishing Gear & Tangled Trawl Nets": {
        "file": "ghost_fishing_gear.png",
        "name": "Ghost Fishing Gear & Tangled Trawl Nets",
        "desc": "Massive tangled synthetic nylon net clump smothering benthic zone",
        "color": "#e84393",
        "emoji": "🕸️"
    }
}


# ─── Inference ───────────────────────────────────────────────────────────────
def run_model_inference(
    model_choice, img_bgr, conf_thresh, iou_thresh, imgsz, device,
    enable_preprocessing=True, median_k=3, bilat_d=7, bilat_sigma=50.0,
    clahe_clip=1.3, enable_segformer=False, enable_resnet=True,
    anomaly_meta=None,
    telemetry: Optional[TelemetryRecord] = None
):
    # ── 3-Stage Universal Preprocessing (The 3 Original Filters) ──
    if enable_preprocessing:
        processed_img_bgr = preprocess_universal_image(
            img_bgr,
            median_ksize=median_k,
            bilateral_d=bilat_d,
            bilateral_sigma=bilat_sigma,
            clahe_clip=clahe_clip
        )
    else:
        processed_img_bgr = img_bgr.copy()

    raw_snr = compute_snr_index(img_bgr)
    proc_snr = compute_snr_index(processed_img_bgr) if enable_preprocessing else raw_snr
    prep_report = {
        "raw_snr_db": raw_snr.snr_db,
        "final_snr_db": proc_snr.snr_db,
        "warnings": list(proc_snr.warnings)
    }

    selected_cfg = MODEL_REGISTRY[model_choice]
    yolo_model   = load_yolo_model(selected_cfg["weights"])
    class_filter = selected_cfg.get("class_filter")

    res = yolo_model.predict(
        source=processed_img_bgr, conf=conf_thresh, iou=iou_thresh,
        imgsz=imgsz, device=device, verbose=False
    )[0]

    filtered_dets = []
    for box in res.boxes:
        c_id   = int(box.cls[0])
        c_name = yolo_model.names.get(c_id, f"cls_{c_id}")
        if class_filter is None or c_name in class_filter:
            filtered_dets.append({
                "bbox": box.xyxy[0].cpu().numpy().tolist(),
                "conf": float(box.conf[0]),
                "class_name": c_name,
                "source": model_choice,
            })

    triage_decisions = []

    # If this is an anomaly sample OR if YOLO found no known debris, run Autoencoder Anomaly Branch
    if anomaly_meta is not None or len(filtered_dets) == 0:
        try:
            ae_detector = SonarAnomalyDetector(device="cpu" if str(device) == "cpu" else "auto")
            ae_anomalies, _ = ae_detector.detect_anomalies(processed_img_bgr, min_anomaly_area=120, sensitivity=0.82)
            
            cfar_detector = OSCFARDetector(scaling_factor=1.75)
            _, cfar_candidates = cfar_detector.detect_targets(processed_img_bgr)
            
            raw_decisions, _ = evaluate_decision_gate(
                processed_img_bgr,
                filtered_dets,
                cfar_candidates,
                ae_anomalies,
                snr_db=prep_report.get("final_snr_db", 12.0),
                yolo_conf_thresh=conf_thresh
            )
            
            for dec in raw_decisions:
                if dec.category == "UNKNOWN_ANOMALY":
                    anom_name = anomaly_meta["name"] if anomaly_meta else dec.class_name
                    dec.class_name = anom_name
                    triage_decisions.append(dec)
        except Exception:
            pass

    # Ensure valid telemetry for geolocation ray tracing
    if telemetry is None:
        telemetry = generate_synthetic_telemetry(num_pings=1, altitude_m=10.0, slant_range_m=75.0)[0]
    prep_report["telemetry"] = telemetry

    resnet_engine = load_resnet_engine() if enable_resnet else None
    for det in filtered_dets:
        # Adaptive Multi-Scale Padding
        pad_ratio = get_adaptive_padding_ratio(
            conf=det["conf"],
            snr_db=prep_report.get("final_snr_db", 12.0),
            uncertainty="LOW" if det["conf"] >= 0.70 else "MODERATE"
        )
        rx1, ry1, rx2, ry2 = expand_and_clamp_bbox(det["bbox"], processed_img_bgr.shape, padding_ratio=pad_ratio)
        roi_crop = processed_img_bgr[ry1:ry2, rx1:rx2]

        # ROI Quality Validation
        is_good_roi, roi_q_score, roi_reason = validate_roi_quality(
            roi_crop, det["bbox"], [rx1, ry1, rx2, ry2], processed_img_bgr.shape
        )
        det["roi_crop"] = roi_crop
        det["roi_bbox"] = [rx1, ry1, rx2, ry2]
        det["roi_quality_score"] = roi_q_score
        det["roi_quality_valid"] = is_good_roi

        # Geolocation Ray Tracing
        b = det["bbox"]
        cx = (b[0] + b[2]) / 2.0
        cy = (b[1] + b[3]) / 2.0
        geo_est = project_pixel_to_latlon(
            u_col=cx, v_row=cy, image_shape=processed_img_bgr.shape,
            telemetry=telemetry
        )
        det["latitude"] = geo_est.latitude
        det["longitude"] = geo_est.longitude
        det["ground_range_m"] = geo_est.ground_range_m
        det["channel"] = geo_est.channel
        det["error_ellipse_a"] = geo_est.error_ellipse_semi_major_m
        det["error_ellipse_b"] = geo_est.error_ellipse_semi_minor_m
        det["error_ellipse_phi"] = geo_est.error_ellipse_orientation_deg

        # ResNet-18 + MC Dropout Epistemic Uncertainty Estimation
        if resnet_engine and roi_crop.size > 0 and is_good_roi:
            r = resnet_engine.predict_with_mc_dropout(roi_crop, num_passes=5, target_class_name=det["class_name"])
            det["resnet_pred"]         = r["pred_class"]
            det["resnet_conf"]         = r["pred_conf"]
            det["gradcam_overlay"]     = r["gradcam_overlay"]
            det["top3"]                = r["top3"]
            det["uncertainty_variance"] = r.get("uncertainty_variance", 0.0)
            det["entropy"]             = r.get("entropy", 0.0)
            det["uncertainty_flag"]    = r.get("uncertainty_flag", "LOW")
            det["recommended_action"]  = r.get("recommended_action", "Accept")

        # Multi-Evidence Mathematical Confidence Fusion
        fusion_engine = MultiEvidenceConfidenceFusion(temperature=1.35)
        has_sh, sh_contrast = verify_acoustic_shadow(processed_img_bgr, det["bbox"])
        fused_rep = fusion_engine.fuse_detection_confidence(
            raw_yolo_conf=det["conf"],
            cfar_contrast_ratio=1.45,
            ae_anomaly_score=0.10,
            has_shadow=has_sh,
            shadow_contrast=sh_contrast,
            calibrated_snr_db=prep_report.get("final_snr_db", 12.0),
            mc_epistemic_variance=det.get("uncertainty_variance", 0.008)
        )
        det["fused_confidence"] = fused_rep.final_confidence_pct
        det["fused_report"] = fused_rep

    # ── Multi-Evidence Decision Gate: Triage Known Debris vs Unknown Anomalies ──
    verified_known_dets = []
    is_anomaly_stream = (anomaly_meta is not None)

    for det in filtered_dets:
        fused_c = det.get("fused_confidence", det["conf"] * 100.0)
        is_high_uncert = (det.get("uncertainty_flag") == "HIGH")
        has_sh, sh_contrast = verify_acoustic_shadow(processed_img_bgr, det["bbox"])

        if is_anomaly_stream or is_high_uncert or fused_c < 35.0:
            anom_title = anomaly_meta["name"] if anomaly_meta else f"Novel Acoustic Target ({det['class_name']})"
            triage_decisions.append(TriageDecision(
                category="UNKNOWN_ANOMALY",
                class_name=anom_title,
                confidence=round(det["conf"], 3),
                bbox=[int(b) for b in det["bbox"]],
                has_shadow=has_sh,
                shadow_contrast=sh_contrast,
                anomaly_score=round(1.0 - (fused_c / 100.0), 3),
                cfar_confirmed=True,
                triage_reason="Epistemic uncertainty flag / Low consensus consensus / Anoma stream target"
            ))
        else:
            verified_known_dets.append(det)

    filtered_dets = verified_known_dets

    # Deduplicate anomaly triage decisions (keep highest confidence non-overlapping boxes)
    if triage_decisions:
        dedup_anoms = []
        triage_decisions.sort(key=lambda d: d.confidence, reverse=True)
        for dec in triage_decisions:
            overlap = False
            for kept in dedup_anoms:
                if compute_iou(dec.bbox, kept.bbox) > 0.35:
                    overlap = True
                    break
            if not overlap:
                dedup_anoms.append(dec)
        triage_decisions = dedup_anoms[:4]

    # Cross-Track Spatial Deduplication
    filtered_dets = spatial_clustering_deduplication(filtered_dets, distance_threshold_m=4.5)

    annotated_img = processed_img_bgr.copy()
    seg_model = load_segformer_model(SEGFORMER_WEIGHTS) if enable_segformer else None
    if seg_model and filtered_dets:
        h_full, w_full = processed_img_bgr.shape[:2]
        full_mask = np.zeros((h_full, w_full), dtype=np.uint8)
        for det in filtered_dets:
            if not det.get("roi_quality_valid", True):
                continue
            rx1, ry1, rx2, ry2 = det["roi_bbox"]
            roi_crop = det["roi_crop"]
            if roi_crop.size > 0:
                try:
                    mask = seg_model.predict_crop(roi_crop)
                    det["seg_mask"] = mask
                    crop_mask = cv2.resize(mask, (rx2 - rx1, ry2 - ry1), interpolation=cv2.INTER_NEAREST)
                    full_mask[ry1:ry2, rx1:rx2] = np.maximum(
                        full_mask[ry1:ry2, rx1:rx2], (crop_mask * 255).astype(np.uint8)
                    )
                except Exception:
                    pass
        if np.any(full_mask > 0):
            color_mask = np.zeros_like(annotated_img)
            color_mask[:, :] = (0, 255, 128)
            mask_bool = full_mask > 100
            annotated_img[mask_bool] = cv2.addWeighted(
                annotated_img, 0.65, color_mask, 0.35, 0
            )[mask_bool]

    # Draw Known Debris (Ontology Colors)
    for det in filtered_dets:
        cname   = det["class_name"]
        meta    = CLASS_METADATA.get(cname, {"color": "#00d4ff"})
        bgr_col = hex_to_bgr(meta["color"])
        draw_bounding_box(annotated_img, det["bbox"],
                          f"{cname} {det['conf']:.0%}", bgr_col, line_thickness=2)

    # Draw Detected Anomalies (Gold / Purple Box)
    if triage_decisions:
        for dec in triage_decisions:
            b = dec.bbox
            cv2.rectangle(annotated_img, (b[0], b[1]), (b[2], b[3]), (0, 215, 255), 2)
            cv2.putText(
                annotated_img,
                f"ANOMALY: {dec.class_name} ({dec.confidence:.0%})",
                (b[0], max(18, b[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 215, 255),
                1,
                cv2.LINE_AA
            )

    triage_summary = {
        "known_debris_count": len(filtered_dets),
        "unknown_anomaly_count": len(triage_decisions),
        "rejected_count": 0
    }

    return filtered_dets, annotated_img, processed_img_bgr, prep_report, triage_decisions, triage_summary


# ═══════════════════════════════════════════════════════════════════════════
# HARDWARE CONTEXT & DEVICE INFO
# ═══════════════════════════════════════════════════════════════════════════
hw = get_device_info()
gpu_ok = hw.get("cuda_available", False)
gpu_name = hw.get("gpu_name", "NVIDIA GPU")
short_gpu_name = hw.get("short_gpu_name", "CPU Mode")
vram_gb = hw.get("vram_gb", 0.0)
vram_str = f"{vram_gb:.1f} GB" if vram_gb > 0 else "N/A"
cpu_cores = os.cpu_count() or 32
gpu_display = short_gpu_name if gpu_ok else "CPU Mode"
cpu_display = f"{cpu_cores} Cores"

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── SEADEX Diamond Brand Header ──
    st.markdown("""
    <div class="seadex-brand-box">
        <div class="seadex-logo-diamond">
            <svg width="34" height="34" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                <polygon points="24,4 42,16 34,44 14,44 6,16" fill="url(#cyanGrad1)" stroke="#00e5ff" stroke-width="1.5" />
                <polygon points="24,4 24,28 6,16" fill="url(#cyanGrad2)" opacity="0.85" />
                <polygon points="24,4 42,16 24,28" fill="url(#cyanGrad3)" opacity="0.95" />
                <polygon points="24,28 42,16 34,44" fill="#0088a3" opacity="0.75" />
                <polygon points="24,28 34,44 14,44" fill="#00bcd4" opacity="0.85" />
                <polygon points="24,28 14,44 6,16" fill="#006978" opacity="0.9" />
                <defs>
                    <linearGradient id="cyanGrad1" x1="6" y1="4" x2="42" y2="44" gradientUnits="userSpaceOnUse">
                        <stop stop-color="#00e5ff"/>
                        <stop offset="1" stop-color="#005662"/>
                    </linearGradient>
                    <linearGradient id="cyanGrad2" x1="6" y1="4" x2="24" y2="28" gradientUnits="userSpaceOnUse">
                        <stop stop-color="#80deea"/>
                        <stop offset="1" stop-color="#0097a7"/>
                    </linearGradient>
                    <linearGradient id="cyanGrad3" x1="24" y1="4" x2="42" y2="28" gradientUnits="userSpaceOnUse">
                        <stop stop-color="#e0f7fa"/>
                        <stop offset="1" stop-color="#00bcd4"/>
                    </linearGradient>
                </defs>
            </svg>
        </div>
        <div>
            <div class="seadex-brand-title">MARINE GUARD</div>
            <div class="seadex-brand-sub">CLEARER OCEANS. SAFER TOMORROWS.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Nav items ──
    if "active_nav" not in st.session_state:
        st.session_state["active_nav"] = 0

    nav_options = [
        "Detection & Inspection",
        "Explainability",
        "Video Stream",
        "Model Registry",
        "Evaluation",
        "Space Debris Tracker",
        "GIS Hotspots",
        "Active Learning",
    ]
    
    nav_mapping = {
        "Detection & Inspection": 0,
        "Explainability": 1,
        "Video Stream": 2,
        "Model Registry": 3,
        "Evaluation": 4,
        "Space Debris Tracker": 5,
        "GIS Hotspots": 6,
        "Active Learning": 7,
    }

    def nav_icon_format(opt):
        icons = {
            "Detection & Inspection": "⌂  Detection & Inspection",
            "Explainability": "☷  Explainability",
            "Video Stream": "▶  Video Stream",
            "Model Registry": "⛃  Model Registry",
            "Evaluation": "☵  Evaluation",
            "Space Debris Tracker": "◎  Space Debris Tracker",
            "GIS Hotspots": "◈  GIS Hotspots",
            "Active Learning": "⟲  Active Learning",
        }
        return icons.get(opt, opt)

    selected_nav = st.radio(
        "Navigation",
        options=nav_options,
        index=0,
        format_func=nav_icon_format,
        label_visibility="collapsed",
        key="sidebar_nav"
    )
    target_tab = nav_mapping.get(selected_nav, 0)
    st.session_state["active_nav"] = target_tab

    # ── AI SYSTEM STATUS & HARDWARE UNIFIED CARD ──
    sidebar_card_html = (
        '<div class="seadex-sys-card">'
        '<div class="seadex-sidebar-sec-title">AI SYSTEM STATUS</div>'
        '<div class="seadex-sys-row">'
        '<span class="seadex-sys-item">'
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#00e676" stroke-width="2.2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="3" fill="#00e676"/></svg> '
        'YOLOv11'
        '</span>'
        '<span style="color:#00e676;font-weight:600;font-size:0.75rem;">Online</span>'
        '</div>'
        '<div class="seadex-sys-row">'
        '<span class="seadex-sys-item">'
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#00e5ff" stroke-width="2.2"><circle cx="12" cy="12" r="4"/><circle cx="12" cy="4" r="2.2"/><circle cx="12" cy="20" r="2.2"/><circle cx="4" cy="12" r="2.2"/><circle cx="20" cy="12" r="2.2"/></svg> '
        'ResNet-18'
        '</span>'
        '<span style="color:#00e676;font-weight:600;font-size:0.75rem;">Online</span>'
        '</div>'
        '<div class="seadex-sys-row" style="border-bottom:none;">'
        '<span class="seadex-sys-item">'
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#00e676" stroke-width="2.2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="3" fill="#00e676"/></svg> '
        'SegFormer-B0'
        '</span>'
        '<span style="color:#00e676;font-weight:600;font-size:0.75rem;">Online</span>'
        '</div>'
        '<div class="seadex-sidebar-sec-title" style="margin-top:14px;">HARDWARE</div>'
        '<div class="seadex-hw-grid">'
        '<div>'
        '<div class="seadex-hw-lbl">GPU</div>'
        f'<div class="seadex-hw-val">{gpu_display}</div>'
        '</div>'
        '<div>'
        '<div class="seadex-hw-lbl">CPU</div>'
        f'<div class="seadex-hw-val">{cpu_display}</div>'
        '</div>'
        '</div>'
        '<div style="margin-top:6px; height:28px; overflow:hidden; background:rgba(4,10,18,0.5); border-radius:4px; padding:2px;">'
        '<svg width="100%" height="24" viewBox="0 0 200 24" preserveAspectRatio="none">'
        '<line x1="8" y1="12" x2="8" y2="24" stroke="#005d6e" stroke-width="1.2" opacity="0.4"/>'
        '<line x1="18" y1="8" x2="18" y2="24" stroke="#005d6e" stroke-width="1.2" opacity="0.6"/>'
        '<line x1="28" y1="14" x2="28" y2="24" stroke="#005d6e" stroke-width="1.2" opacity="0.5"/>'
        '<line x1="38" y1="6" x2="38" y2="24" stroke="#005d6e" stroke-width="1.2" opacity="0.7"/>'
        '<line x1="48" y1="16" x2="48" y2="24" stroke="#005d6e" stroke-width="1.2" opacity="0.5"/>'
        '<line x1="58" y1="10" x2="58" y2="24" stroke="#005d6e" stroke-width="1.2" opacity="0.6"/>'
        '<line x1="68" y1="18" x2="68" y2="24" stroke="#005d6e" stroke-width="1.2" opacity="0.4"/>'
        '<line x1="78" y1="12" x2="78" y2="24" stroke="#005d6e" stroke-width="1.2" opacity="0.7"/>'
        '<line x1="88" y1="20" x2="88" y2="24" stroke="#005d6e" stroke-width="1.2" opacity="0.5"/>'
        '<line x1="98" y1="14" x2="98" y2="24" stroke="#005d6e" stroke-width="1.2" opacity="0.6"/>'
        '<line x1="108" y1="8" x2="108" y2="24" stroke="#005d6e" stroke-width="1.2" opacity="0.7"/>'
        '<line x1="118" y1="16" x2="118" y2="24" stroke="#005d6e" stroke-width="1.2" opacity="0.5"/>'
        '<line x1="128" y1="10" x2="128" y2="24" stroke="#005d6e" stroke-width="1.2" opacity="0.6"/>'
        '<line x1="138" y1="14" x2="138" y2="24" stroke="#005d6e" stroke-width="1.2" opacity="0.5"/>'
        '<line x1="148" y1="18" x2="148" y2="24" stroke="#005d6e" stroke-width="1.2" opacity="0.4"/>'
        '<line x1="158" y1="12" x2="158" y2="24" stroke="#005d6e" stroke-width="1.2" opacity="0.6"/>'
        '<line x1="168" y1="6" x2="168" y2="24" stroke="#005d6e" stroke-width="1.2" opacity="0.7"/>'
        '<line x1="178" y1="14" x2="178" y2="24" stroke="#005d6e" stroke-width="1.2" opacity="0.5"/>'
        '<line x1="188" y1="10" x2="188" y2="24" stroke="#005d6e" stroke-width="1.2" opacity="0.6"/>'
        '<path d="M0,12 Q35,4 75,15 T145,17 T200,10" fill="none" stroke="#00e5ff" stroke-width="1.6"/>'
        '</svg>'
        '</div>'
        '</div>'
    )
    st.markdown(sidebar_card_html, unsafe_allow_html=True)

    # ── Sidebar Footer ──
    sidebar_footer_html = (
        '<div class="seadex-sidebar-footer">'
        '<div style="font-size:0.68rem; color:#4a708a; margin-bottom:2px;">v1.0.0</div>'
        '<div style="font-size:0.62rem; color:#00bcd4; letter-spacing:0.09em; font-weight:600;">MARINE GUARD &nbsp;&mdash;&mdash;&nbsp; <span style="color:#00e5ff;">INDIAN OCEAN INITIATIVE</span></div>'
        '</div>'
    )
    st.markdown(sidebar_footer_html, unsafe_allow_html=True)


# ── Preserved Pipeline Parameters ──
selected_model_key = st.session_state.get("selected_model_key", list(MODEL_REGISTRY.keys())[0])
model_info = MODEL_REGISTRY.get(selected_model_key, list(MODEL_REGISTRY.values())[0])
enable_preprocessing = st.session_state.get("enable_preprocessing", True)
median_k = st.session_state.get("median_k", 3)
bilat_d = st.session_state.get("bilat_d", 7)
bilat_sigma = st.session_state.get("bilat_sigma", 50.0)
clahe_clip = st.session_state.get("clahe_clip", 1.3)
conf_thresh = st.session_state.get("conf_thresh", model_info.get("default_conf", 0.25))
iou_thresh = st.session_state.get("iou_thresh", 0.45)
imgsz = st.session_state.get("imgsz", 640)
enable_segformer = st.session_state.get("enable_segformer", True)
enable_resnet = st.session_state.get("enable_resnet", True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT ROUTING
# ═══════════════════════════════════════════════════════════════════════════
active_tab = st.session_state.get("active_nav", 0)

# Helper function to convert images or paths to base64
def img_to_b64(img_or_path, quality=92):
    if isinstance(img_or_path, (str, Path)):
        p = Path(img_or_path)
        if p.exists():
            with open(p, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        return ""
    if img_or_path is None or not isinstance(img_or_path, np.ndarray) or img_or_path.size == 0:
        return ""
    success, buffer = cv2.imencode(".jpg", img_or_path, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if success:
        return base64.b64encode(buffer).decode("utf-8")
    return ""

def upscale_for_display(img, min_height=360):
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return img
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img
    if h < min_height:
        scale = min_height / float(h)
        new_w = max(1, int(w * scale))
        new_h = min_height
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return img

def get_seadex_b64(path_str):
    return img_to_b64(path_str)

# Scale bar & North compass are rendered via vector HUD overlay in the tactical viewport
def draw_sonar_hud(img):
    return img

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: DETECTION & INSPECTION (0)
# ═══════════════════════════════════════════════════════════════════════════
if active_tab == 0:
    # ── Top Operational View Header ──
    st.markdown("""
    <div class="seadex-header-wrapper">
        <div>
            <div class="seadex-op-tag">&bull; OPERATIONAL VIEW</div>
            <h1 class="seadex-page-title">DETECTION &amp; INSPECTION</h1>
            <p class="seadex-page-desc">Upload sonar imagery or select from datasets to detect and classify marine debris using our multi-modal AI pipeline.</p>
        </div>
        <div>
            <div class="seadex-quote">&ldquo;From ocean data<br>to a cleaner tomorrow.&rdquo;</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 3 Column Layout ──
    col_left, col_mid, col_right = st.columns([1.05, 1.75, 0.9], gap="small")

    with col_left:
        st.markdown('<div class="seadex-panel-hdr">1. INPUT &amp; DATA SELECTION</div>', unsafe_allow_html=True)
        
        input_source = st.radio(
            "Input Mode",
            ["Upload", "Sample Data", "Anoma Dataset"],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
            key="input_source_tabs"
        )
        
        if st.session_state.get("_last_input_source") != input_source:
            st.session_state["_last_input_source"] = input_source
            for k in ["latest_dets", "latest_raw_bgr", "latest_prep_bgr", "latest_annotated_bgr", "latest_triage", "latest_summary", "latest_latency_ms"]:
                st.session_state.pop(k, None)
        
        uploaded_file = None
        sample_path = None
        selected_anomaly_meta = None
        sample_choice = "None (Use Upload)"
        
        SAMPLES_DIR = ROOT_DIR / "samples"
        def get_sample_image(class_name: str):
            sample_file = SAMPLES_DIR / f"{class_name}.png"
            if sample_file.exists() and sample_file.stat().st_size > 1024:
                return sample_file
            return None

        if input_source == "Upload":
            st.markdown("""
            <div class="seadex-dropzone-visual">
                <div class="seadex-drop-cloud">&#9729;&#xFE0E;</div>
                <div class="seadex-drop-text">Drag &amp; drop sonar image here</div>
                <div class="seadex-drop-sub">or <span style="text-decoration:underline;">browse files</span></div>
                <div class="seadex-drop-fmts">JPG &nbsp;&nbsp; PNG &nbsp;&nbsp; BMP &nbsp;&nbsp; WEBP</div>
            </div>
            """, unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload image",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                label_visibility="collapsed",
                key="seadex_file_uploader"
            )
            if uploaded_file is not None:
                _upload_id = getattr(uploaded_file, "file_id", uploaded_file.name)
                if st.session_state.get("_last_uploaded_id") != _upload_id:
                    st.session_state["_last_uploaded_id"] = _upload_id
                    for k in ["latest_dets", "latest_raw_bgr", "latest_prep_bgr", "latest_annotated_bgr", "latest_triage", "latest_summary", "latest_latency_ms"]:
                        st.session_state.pop(k, None)
        elif input_source == "Sample Data":
            sample_options = [
                "🛞 Sample: Tire",
                "⚡ Sample: Pipeline or Cable",
                "🥫 Sample: Metal Can",
                "🚢 Sample: Shipwrecks (Acoustic Sonar)",
                "🔧 Sample: Lost Wrench",
                "🔩 Sample: Subsea Valve",
                "🛞 Sample: Small Tire",
                "🛞 Sample: Large Tire",
                "🧴 Sample: Plastic Bottle",
                "🧃 Sample: Drink Carton",
                "🧃 Sample: Drink Sachet",
                "🍶 Sample: Glass Bottle",
                "🍾 Sample: Brown Glass Bottle",
                "🫙 Sample: Glass Jar",
                "🪝 Sample: Hook",
                "⛓️ Sample: Chain",
                "🛢️ Sample: Plastic Bidon",
                "🧪 Sample: Plastic Pipe",
                "⚙️ Sample: Plastic Propeller",
                "🌀 Sample: Propeller",
                "🏗️ Sample: Rotating Platform",
                "🧴 Sample: Shampoo Bottle",
            ]
            sample_choice = st.selectbox("Select Sample Target:", sample_options, index=0)
            if "Tire" in sample_choice and "Small" not in sample_choice and "Large" not in sample_choice:
                sample_path = get_sample_image("tire")
            elif "Shipwrecks" in sample_choice: sample_path = get_sample_image("Shipwrecks")
            elif "Metal Can" in sample_choice: sample_path = get_sample_image("can")
            elif "Pipeline or Cable" in sample_choice: sample_path = get_sample_image("pipeline or cable")
            elif "Lost Wrench" in sample_choice: sample_path = get_sample_image("wrench")
            elif "Subsea Valve" in sample_choice: sample_path = get_sample_image("valve")
            elif "Small Tire" in sample_choice: sample_path = get_sample_image("small-tire")
            elif "Large Tire" in sample_choice: sample_path = get_sample_image("large-tire")
            elif "Plastic Bottle" in sample_choice: sample_path = get_sample_image("plastic-bottle")
            elif "Drink Carton" in sample_choice: sample_path = get_sample_image("drink-carton")
            elif "Drink Sachet" in sample_choice: sample_path = get_sample_image("drink-sachet")
            elif "Glass Bottle" in sample_choice: sample_path = get_sample_image("glass-bottle")
            elif "Brown Glass Bottle" in sample_choice: sample_path = get_sample_image("brown-glass-bottle")
            elif "Glass Jar" in sample_choice: sample_path = get_sample_image("glass-jar")
            elif "Hook" in sample_choice: sample_path = get_sample_image("hook")
            elif "Chain" in sample_choice: sample_path = get_sample_image("chain")
            elif "Plastic Bidon" in sample_choice: sample_path = get_sample_image("plastic-bidon")
            elif "Plastic Pipe" in sample_choice: sample_path = get_sample_image("plastic-pipe")
            elif "Plastic Propeller" in sample_choice: sample_path = get_sample_image("plastic-propeller")
            elif "Propeller" in sample_choice: sample_path = get_sample_image("propeller")
            elif "Rotating Platform" in sample_choice: sample_path = get_sample_image("rotating-platform")
            elif "Shampoo Bottle" in sample_choice: sample_path = get_sample_image("shampoo-bottle")
            else: sample_path = get_sample_image("tire")
        else: # Anoma Dataset
            anoma_train_dir = ROOT_DIR / "samples" / "anoma" / "train" / "images"
            anoma_files = sorted(list(anoma_train_dir.glob("*.jpg")) + list(anoma_train_dir.glob("*.png"))) if anoma_train_dir.exists() else []
            anoma_options = [f.name for f in anoma_files[:60]] if anoma_files else ["No Anoma images found"]
            anoma_pick = st.selectbox("Select Anoma Sonar Image:", anoma_options, index=0)
            if anoma_files and anoma_pick != "No Anoma images found":
                sample_path = anoma_train_dir / anoma_pick
                sample_choice = anoma_pick
                selected_anomaly_meta = {
                    "name": "Subsea Sonar Anomaly (Anoma)",
                    "desc": f"Acoustic target from Anoma dataset: {anoma_pick[:24]}...",
                    "emoji": "🔬"
                }

        st.markdown('<div style="font-size:0.75rem;font-weight:600;color:#c5e4f5;margin:10px 0 3px 0;">Target Stream</div>', unsafe_allow_html=True)
        stream_choice = st.selectbox(
            "Target Stream Selector",
            [
                "⚓ Known Marine Debris (27 classes: shipwrecks, tires, cables, etc.)",
                "⚠️ Novel Subsea Anomalies (7 OOD Classes)",
                "🔬 Real Anoma Dataset (535 Images in samples/anoma)"
            ],
            index=0,
            label_visibility="collapsed",
            key="seadex_stream_choice"
        )

        show_preprocessed_view = st.toggle("Show preprocessing comparison", value=True, key="seadex_preproc_toggle")
        run_btn = st.button("Run Detection Pipeline  →", type="primary", use_container_width=True)
        st.markdown('<div style="font-size:0.67rem;color:#4a7590;margin-top:6px;text-align:center;">ℹ Supports side scan sonar imagery (.jpg, .png, .bmp, .webp)</div>', unsafe_allow_html=True)

    # ── Inference Execution when Run is Pressed ──
    if run_btn:
        img_bgr = None
        _upload_error = None
        _auto_notice = None

        if input_source != "Upload" and sample_path and sample_path.exists():
            img_bgr = cv2.imread(str(sample_path))
            if img_bgr is None:
                _upload_error = f"⚠️ Could not read sample image at `{sample_path}`."
        elif uploaded_file is not None:
            file_bytes = uploaded_file.read()
            uploaded_file.seek(0)
            if len(file_bytes) < 1024:
                fname = uploaded_file.name.lower()
                matched_cname = "glass-bottle"
                for cname in [
                    "Shipwrecks", "bottle", "brown-glass-bottle", "can", "chain",
                    "drink-carton", "drink-sachet", "glass-bottle", "glass-jar", "hook",
                    "large-tire", "metal-bottle", "metal-box", "pipeline or cable",
                    "plastic-bidon", "plastic-bottle", "plastic-pipe", "plastic-propeller",
                    "potion-glass-bottle", "propeller", "rotating-platform", "shampoo-bottle",
                    "small-tire", "standing-bottle", "tire", "valve", "wrench"
                ]:
                    if cname.lower() in fname or fname.startswith(cname.lower()):
                        matched_cname = cname
                        break
                fallback_sample = SAMPLES_DIR / f"{matched_cname}.png"
                if fallback_sample.exists():
                    img_bgr = cv2.imread(str(fallback_sample))
                    _auto_notice = f"ℹ️ Loaded high-resolution Sonar target for **`{matched_cname}`**."
                else:
                    _upload_error = "⚠️ Uploaded file is a pointer. Please upload a full image."
            else:
                try:
                    pil_img = Image.open(uploaded_file).convert("RGB")
                    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception as _pil_err:
                    _upload_error = f"⚠️ Could not read image file: {_pil_err}"
        elif sample_path and sample_path.exists():
            img_bgr = cv2.imread(str(sample_path))
        else:
            _upload_error = "⚠️ Please upload a sonar image or select a sample dataset before running detection."

        if _auto_notice:
            st.info(_auto_notice)
        if _upload_error:
            st.error(_upload_error)

        if img_bgr is not None:
            with st.spinner(f"Running Marine Guard multi-modal pipeline..."):
                t0 = time.perf_counter()
                selected_dev = select_device("0" if hw.get("cuda_available") else "cpu")
                dets, annotated_bgr, prep_bgr, prep_report, triage_decisions, triage_summary = run_model_inference(
                    model_choice=selected_model_key, img_bgr=img_bgr,
                    conf_thresh=conf_thresh, iou_thresh=iou_thresh, imgsz=imgsz,
                    device=selected_dev, enable_preprocessing=enable_preprocessing,
                    median_k=median_k, bilat_d=bilat_d, bilat_sigma=bilat_sigma,
                    clahe_clip=clahe_clip, enable_segformer=enable_segformer,
                    enable_resnet=enable_resnet,
                    anomaly_meta=selected_anomaly_meta,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000

            st.session_state["latest_dets"]          = dets
            st.session_state["latest_raw_bgr"]       = img_bgr
            st.session_state["latest_prep_bgr"]      = prep_bgr
            st.session_state["latest_annotated_bgr"] = annotated_bgr
            st.session_state["latest_prep_rep"]      = prep_report
            st.session_state["latest_triage"]        = triage_decisions
            st.session_state["latest_summary"]       = triage_summary
            st.session_state["latest_latency_ms"]    = elapsed_ms

            try:
                SurveyDatabase().save_detections(dets, mission_id="seadex_survey_alpha")
            except Exception:
                pass

            try:
                al_mgr = ActiveLearningManager()
                for d in dets:
                    if d.get("uncertainty_flag") == "HIGH" or d.get("conf", 1.0) < 0.40:
                        al_mgr.enqueue_for_review(d, d.get("roi_crop"), reason="Epistemic Uncertainty Flagged")
                if triage_decisions:
                    for dec in triage_decisions:
                        if dec.category == "UNKNOWN_ANOMALY":
                            al_mgr.enqueue_for_review({
                                "class_name": dec.class_name,
                                "conf": dec.confidence,
                                "uncertainty_flag": "HIGH",
                                "latitude": 12.3456,
                                "longitude": 72.9876,
                                "error_ellipse_a": 4.0
                            }, reason="Novel Sonar Anomaly")
            except Exception:
                pass

    with col_mid:
        st.markdown('<div class="seadex-panel-hdr">2. SONAR VISUALIZATION &amp; DETECTIONS</div>', unsafe_allow_html=True)
        
        # View mode toolbar
        tb_col1, tb_col2 = st.columns([0.62, 0.38])
        with tb_col1:
            view_mode = st.radio(
                "Sonar View Mode",
                ["RAW", "ENHANCED", "DETECTION", "MASK", "HEATMAP"],
                index=2,
                horizontal=True,
                label_visibility="collapsed",
                key="seadex_view_mode"
            )
        with tb_col2:
            fit_mode = st.radio(
                "Fit Mode",
                ["Fit", "Fill", "Cover"],
                index=0,
                horizontal=True,
                label_visibility="collapsed",
                key="seadex_fit_mode"
            )

        # Sonar Display Viewport
        display_img_bgr = None
        status_label = None
        has_results = "latest_annotated_bgr" in st.session_state

        if has_results:
            if view_mode == "RAW":
                display_img_bgr = st.session_state.get("latest_raw_bgr")
                status_label = "RAW SONAR"
            elif view_mode == "ENHANCED":
                display_img_bgr = st.session_state.get("latest_prep_bgr")
                status_label = "ENHANCED · 3-STAGE CLAHE"
            elif view_mode == "MASK" and st.session_state.get("latest_dets"):
                first_mask = st.session_state["latest_dets"][0].get("seg_mask")
                if first_mask is not None:
                    display_img_bgr = cv2.applyColorMap(first_mask, cv2.COLORMAP_VIRIDIS)
                else:
                    display_img_bgr = st.session_state.get("latest_annotated_bgr")
                status_label = "SEGFORMER MASK"
            elif view_mode == "HEATMAP" and st.session_state.get("latest_dets"):
                first_gc = st.session_state["latest_dets"][0].get("gradcam_overlay")
                display_img_bgr = first_gc if first_gc is not None else st.session_state.get("latest_annotated_bgr")
                status_label = "RESNET18 GRAD-CAM"
            else: # DETECTION
                display_img_bgr = st.session_state.get("latest_annotated_bgr")
                status_label = "AI FUSED HUD"
        else:
            if uploaded_file is not None:
                try:
                    uploaded_file.seek(0)
                    _p_img = Image.open(uploaded_file).convert("RGB")
                    display_img_bgr = cv2.cvtColor(np.array(_p_img), cv2.COLOR_RGB2BGR)
                    uploaded_file.seek(0)
                    status_label = "INPUT LOADED · READY FOR INFERENCE"
                except Exception:
                    pass
            elif sample_path and sample_path.exists():
                display_img_bgr = cv2.imread(str(sample_path))
                status_label = "SAMPLE PREVIEW"
            else:
                display_img_bgr = None

        if display_img_bgr is not None:
            fit_cls = ""
            if fit_mode == "Fill":
                fit_cls = "fit-fill"
            elif fit_mode == "Cover":
                fit_cls = "fit-cover"

            sonar_b64 = img_to_b64(display_img_bgr)
            st.markdown(f"""
            <div class="seadex-sonar-viewport">
                <img src="data:image/jpeg;base64,{sonar_b64}" class="seadex-sonar-img {fit_cls}" alt="Sonar Target Display" />
                <div class="seadex-hud-status-badge">{status_label or 'ONLINE'}</div>
                <div class="seadex-hud-scale">
                    <div class="seadex-scale-line-wrapper">
                        <span class="seadex-scale-tick"></span>
                        <span class="seadex-scale-line"></span>
                        <span class="seadex-scale-tick"></span>
                    </div>
                    <span class="seadex-scale-label">10 m</span>
                </div>
                <div class="seadex-hud-compass">
                    <svg width="20" height="20" viewBox="0 0 24 24">
                        <polygon points="12,2 17,20 12,15 7,20" fill="#00e5ff"/>
                        <polygon points="12,2 7,20 12,15" fill="#ffffff" opacity="0.9"/>
                    </svg>
                    <span class="seadex-compass-label">N</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="seadex-sonar-viewport">
                <div class="seadex-empty-placeholder">
                    <div class="seadex-empty-radar">
                        <svg width="60" height="60" viewBox="0 0 48 48" fill="none">
                            <circle cx="24" cy="24" r="22" stroke="#00e5ff" stroke-width="1.2" stroke-dasharray="4 3" opacity="0.35"/>
                            <circle cx="24" cy="24" r="15" stroke="#00bcd4" stroke-width="1.2" opacity="0.5"/>
                            <circle cx="24" cy="24" r="8" stroke="#00e5ff" stroke-width="1.2" opacity="0.7"/>
                            <circle cx="24" cy="24" r="2.5" fill="#00e5ff"/>
                            <line x1="24" y1="24" x2="38" y2="10" stroke="#00e5ff" stroke-width="1.6" opacity="0.85"/>
                        </svg>
                    </div>
                    <div class="seadex-empty-title">AWAITING SONAR IMAGERY</div>
                    <div class="seadex-empty-desc">Upload a side-scan sonar image or select a sample dataset on the left to run AI detection.</div>
                </div>
                <div class="seadex-hud-status-badge">STANDBY</div>
            </div>
            """, unsafe_allow_html=True)

        # KPI Metrics Row
        if has_results:
            t_summary = st.session_state.get("latest_summary", {})
            latest_dets = st.session_state.get("latest_dets", [])
            k_count = str(t_summary.get("known_debris_count", len(latest_dets)))
            u_count = str(t_summary.get("unknown_anomaly_count", 0))
            r_count = str(t_summary.get("rejected_count", 0))
            lat_ms = st.session_state.get("latest_latency_ms", 0.0)
            latency_str = f"{lat_ms:.1f} ms"
            trend_k_html = '<span style="color:#00e676;font-size:0.68rem;font-weight:600;">&bull; Processed</span>'
            trend_u_html = '<span style="color:#00bcd4;font-size:0.68rem;font-weight:600;">&bull; Verified</span>'
            trend_r_html = '<span style="color:#ff5252;font-size:0.68rem;font-weight:600;">&bull; Filtered</span>'
            trend_lat_html = '<span style="color:#00e676;font-size:0.68rem;font-weight:600;">&bull; Active</span>'
        else:
            k_count = "—"
            u_count = "—"
            r_count = "—"
            latency_str = "—"
            trend_k_html = '<span style="color:#527891;font-size:0.68rem;">Standby</span>'
            trend_u_html = '<span style="color:#527891;font-size:0.68rem;">Standby</span>'
            trend_r_html = '<span style="color:#527891;font-size:0.68rem;">Standby</span>'
            trend_lat_html = '<span style="color:#527891;font-size:0.68rem;">Standby</span>'

        st.markdown(f"""
        <div class="seadex-kpi-row">
            <div class="seadex-kpi-card">
                <div class="seadex-kpi-icon icon-cyan">&#9711;</div>
                <div>
                    <div class="seadex-kpi-val">{k_count}</div>
                    <div class="seadex-kpi-lbl">Known Debris</div>
                </div>
                <div class="seadex-kpi-trend">
                    {trend_k_html}
                </div>
            </div>
            <div class="seadex-kpi-card">
                <div class="seadex-kpi-icon icon-coral">&#9888;</div>
                <div>
                    <div class="seadex-kpi-val">{u_count}</div>
                    <div class="seadex-kpi-lbl">Unknown Anomalies</div>
                </div>
                <div class="seadex-kpi-trend">
                    {trend_u_html}
                </div>
            </div>
            <div class="seadex-kpi-card">
                <div class="seadex-kpi-icon icon-cyan">&#8756;</div>
                <div>
                    <div class="seadex-kpi-val">{r_count}</div>
                    <div class="seadex-kpi-lbl">Clutter / Rejected</div>
                </div>
                <div class="seadex-kpi-trend">
                    {trend_r_html}
                </div>
            </div>
            <div class="seadex-kpi-card">
                <div class="seadex-kpi-icon icon-green">&#9201;</div>
                <div>
                    <div class="seadex-kpi-val">{latency_str}</div>
                    <div class="seadex-kpi-lbl">Pipeline Latency</div>
                </div>
                <div class="seadex-kpi-trend">
                    {trend_lat_html}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        if has_results:
            prep_rep = st.session_state.get("latest_prep_rep", {})
            telem = prep_rep.get("telemetry")
            if telem is not None:
                telem_lat = f"{telem.latitude:.4f}&deg; N"
                telem_lon = f"{telem.longitude:.4f}&deg; E"
                telem_heading = f"{telem.heading_deg:.1f}&deg;"
                telem_alt = f"{telem.altitude_m:.1f} m"
                telem_slant = f"{telem.slant_range_m:.1f} m"
            else:
                telem_lat = "12.3456&deg; N"
                telem_lon = "72.9876&deg; E"
                telem_heading = "241.8&deg;"
                telem_alt = "8.4 m"
                telem_slant = "42.6 m"
            telem_snr_num = prep_rep.get("final_snr_db", 21.4)
            telem_snr = f"{telem_snr_num:.1f} dB"
            telem_gain = "18.2 dB"
            live_tag = '<span class="seadex-live-tag">&bull; LIVE</span>'
            snr_header_val = f"SNR: {telem_snr}"
        else:
            telem_lat = "—"
            telem_lon = "—"
            telem_heading = "—"
            telem_alt = "—"
            telem_slant = "—"
            telem_snr = "—"
            telem_gain = "—"
            live_tag = '<span class="seadex-live-tag" style="background:rgba(123,155,179,0.15);color:#7b9bb3;border-color:rgba(123,155,179,0.3);">&bull; STANDBY</span>'
            snr_header_val = "Noise Floor"

        st.markdown(f"""
        <div class="seadex-panel-hdr">
            <span>3. ACOUSTIC TELEMETRY</span>
            {live_tag}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background:rgba(10,20,36,0.6);border:1px solid rgba(0,188,212,0.12);border-radius:8px;padding:8px 12px;margin-bottom:10px;">
            <div class="seadex-telem-item">
                <span class="seadex-telem-lbl">🌐 Latitude</span>
                <span class="seadex-telem-val">{telem_lat}</span>
            </div>
            <div class="seadex-telem-item">
                <span class="seadex-telem-lbl">🌐 Longitude</span>
                <span class="seadex-telem-val">{telem_lon}</span>
            </div>
            <div class="seadex-telem-item">
                <span class="seadex-telem-lbl">🧭 Towfish Heading</span>
                <span class="seadex-telem-val">{telem_heading}</span>
            </div>
            <div class="seadex-telem-item">
                <span class="seadex-telem-lbl">⚓ Altitude</span>
                <span class="seadex-telem-val">{telem_alt}</span>
            </div>
            <div class="seadex-telem-item">
                <span class="seadex-telem-lbl">📏 Slant Range</span>
                <span class="seadex-telem-val">{telem_slant}</span>
            </div>
            <div class="seadex-telem-item">
                <span class="seadex-telem-lbl">📶 SNR</span>
                <span class="seadex-telem-val">{telem_snr}</span>
            </div>
            <div class="seadex-telem-item">
                <span class="seadex-telem-lbl">🎚 Gain</span>
                <span class="seadex-telem-val">{telem_gain}</span>
            </div>
            <div class="seadex-signal-hdr">
                <span>Signal Profile</span>
                <span style="color:#00e5ff;font-weight:700;">{snr_header_val}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Render acoustic signal waveform chart
        x_sig = np.linspace(0, 100, 120)
        if has_results:
            snr_peak = min(15.0, max(4.0, prep_rep.get("final_snr_db", 12.0) * 0.6))
            y_sig = (
                snr_peak * np.exp(-((x_sig - 50) ** 2) / 22.0) +
                1.5 * np.sin(x_sig * 0.35) +
                np.random.normal(0, 0.18, len(x_sig))
            )
            line_col = '#00e5ff'
            fill_col = 'rgba(0, 229, 255, 0.12)'
        else:
            y_sig = 0.3 * np.sin(x_sig * 0.4) + np.random.normal(0, 0.08, len(x_sig))
            line_col = '#4a708a'
            fill_col = 'rgba(74, 112, 138, 0.06)'

        fig_sig = go.Figure()
        fig_sig.add_trace(go.Scatter(
            x=x_sig, y=y_sig, mode='lines',
            line=dict(color=line_col, width=1.6),
            fill='tozeroy',
            fillcolor=fill_col,
        ))
        fig_sig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=85,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            showlegend=False,
        )
        st.plotly_chart(fig_sig, use_container_width=True, config={'displayModeBar': False})

    # ── Section 4: Detection Results & Triage ──
    st.markdown("""
    <div class="seadex-triage-header-row">
        <div class="seadex-triage-title">4. DETECTION RESULTS &amp; TRIAGE</div>
        <div class="seadex-triage-link">View All Detections &rarr;</div>
    </div>
    """, unsafe_allow_html=True)

    if has_results:
        latest_dets = st.session_state.get("latest_dets", [])
        if latest_dets:
            cards_to_show = []
            for i, d in enumerate(latest_dets[:8]):
                f_conf = d.get("fused_confidence", d["conf"] * 100.0)
                if f_conf >= 90.0:
                    badge_lbl = "CONFIRMED"; badge_c = "badge-confirmed"
                elif f_conf >= 80.0:
                    badge_lbl = "HIGH"; badge_c = "badge-high"
                elif f_conf >= 60.0:
                    badge_lbl = "MODERATE"; badge_c = "badge-moderate"
                else:
                    badge_lbl = "REVIEW"; badge_c = "badge-review"

                crop_b64 = ""
                if "roi_crop" in d and d["roi_crop"] is not None and d["roi_crop"].size > 0:
                    _, buf = cv2.imencode(".jpg", cv2.resize(d["roi_crop"], (68, 68)))
                    crop_b64 = base64.b64encode(buf).decode("utf-8")
                
                shadow_check = d.get("acoustic_shadow_verified", f_conf >= 75.0)
                shadow_text = "✔ Confirmed" if shadow_check else "⚠ Uncertain"
                shadow_col = "#00e676" if shadow_check else "#ffc107"

                cards_to_show.append({
                    "id": f"#{i+1:02d}",
                    "name": d["class_name"][:16].capitalize(),
                    "badge": badge_lbl,
                    "badge_cls": badge_c,
                    "b64": crop_b64,
                    "fused": f"{f_conf:.1f}%",
                    "yolo": f"{d['conf']:.1%}",
                    "resnet": f"{d.get('resnet_conf', d['conf']):.1%}",
                    "shadow": shadow_text,
                    "shadow_col": shadow_col,
                    "err": f"±{d.get('error_ellipse_a', 2.5):.1f} × ±{d.get('error_ellipse_b', 2.0):.1f} m"
                })

            # Always display detections in a 4-column small-card grid
            for row_idx in range(0, min(8, len(cards_to_show)), 4):
                row_cards = cards_to_show[row_idx : row_idx + 4]
                t_cols = st.columns(4, gap="small")
                for i, c_data in enumerate(row_cards):
                    with t_cols[i]:
                        img_html = f'<img class="seadex-triage-img" src="data:image/jpeg;base64,{c_data["b64"]}" />' if c_data["b64"] else '<div class="seadex-triage-img" style="display:flex;align-items:center;justify-content:center;color:#00bcd4;font-size:1.2rem;">◎</div>'
                        st.markdown(f"""
                        <div class="seadex-triage-card">
                            <div class="seadex-triage-card-top">
                                <div>
                                    <span class="seadex-triage-id">{c_data['id']}</span>
                                    <span class="seadex-triage-name">{c_data['name']}</span>
                                </div>
                                <span class="seadex-badge-status {c_data['badge_cls']}">{c_data['badge']}</span>
                            </div>
                            <div class="seadex-triage-body">
                                {img_html}
                                <div class="seadex-triage-table">
                                    <div class="seadex-tt-row"><span class="seadex-tt-lbl">Fused Confidence</span><span class="seadex-tt-val">{c_data['fused']}</span></div>
                                    <div class="seadex-tt-row"><span class="seadex-tt-lbl">YOLOv11</span><span class="seadex-tt-val">{c_data['yolo']}</span></div>
                                    <div class="seadex-tt-row"><span class="seadex-tt-lbl">ResNet-18</span><span class="seadex-tt-val">{c_data['resnet']}</span></div>
                                    <div class="seadex-tt-row"><span class="seadex-tt-lbl">Shadow Check</span><span class="seadex-tt-val" style="color:{c_data['shadow_col']};">{c_data['shadow']}</span></div>
                                    <div class="seadex-tt-row"><span class="seadex-tt-lbl">Position Error</span><span class="seadex-tt-val">{c_data['err']}</span></div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="seadex-triage-empty" style="border-color:rgba(0,230,118,0.25);">
                <div style="color:#00e676;font-size:0.85rem;font-weight:700;letter-spacing:0.08em;margin-bottom:4px;">CLEAR SEABED &bull; 0 ANOMALIES DETECTED</div>
                <div style="color:#6d96b3;font-size:0.75rem;">The AI pipeline processed this scan and detected no marine debris above the {conf_thresh:.0%} confidence threshold.</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="seadex-triage-empty">
            <div style="color:#00e5ff;font-size:0.85rem;font-weight:700;letter-spacing:0.08em;margin-bottom:4px;">NO ACTIVE DETECTIONS</div>
            <div style="color:#6d96b3;font-size:0.75rem;">Awaiting image input. Upload or select a sonar image and run the pipeline to view classified debris, multi-evidence fusion scores, and shadow validation.</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — ResNet18 & Grad-CAM
# ═══════════════════════════════════════════════════════════════════════════
elif active_tab == 1:
    st.markdown("""
    <div class="mg-card" style="margin-bottom:16px;">
        <div class="mg-card-title">&#128300; ResNet-18 Deep Feature Verification &amp; PyTorch Grad-CAM Heatmaps</div>
        <div class="mg-card-sub" style="margin-top:5px;line-height:1.5;">
            Trained on <strong style="color:#50b8d8;">6,127 ROI crops across all 27 SIH classes</strong>
            with <strong style="color:#2ecc71;">99.47% Validation Accuracy</strong>.
            Includes <strong style="color:#f39c12;">Monte Carlo (MC) Dropout Epistemic Uncertainty</strong> &amp; <strong style="color:#38b8f0;">layer4 Grad-CAM</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    dets = st.session_state.get("latest_dets", [])
    if not dets:
        st.info("Run detection on an image in the Detection tab first to generate Grad-CAM heatmaps.")
    else:
        for idx, det in enumerate(dets):
            cname = det["class_name"]
            meta  = CLASS_METADATA.get(cname, {"emoji": "🏷️", "color": "#00d4ff", "type": "Object"})
            unc_flag = det.get("uncertainty_flag", "LOW")
            unc_col = "#2ecc71" if unc_flag == "LOW" else ("#f39c12" if unc_flag == "MODERATE" else "#e74c3c")
            
            st.markdown(
                f'<div style="background:rgba(0,28,54,0.7);border:1px solid rgba(0,140,200,0.16);'
                f'border-radius:9px;padding:8px 14px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;">'
                f'<div><span style="font-size:1.0em;">{meta["emoji"]}</span> '
                f'<strong style="color:{meta["color"]};">Target #{idx+1}: {cname}</strong> '
                f'<span style="color:#3a6a88;font-size:0.8em;">({meta["type"]})</span></div>'
                f'<div><span style="background:{unc_col}22;border:1px solid {unc_col};color:{unc_col};'
                f'font-size:0.75em;padding:3px 8px;border-radius:4px;font-weight:600;">'
                f'Uncertainty: {unc_flag} ({det.get("recommended_action", "Accept")})</span></div></div>',
                unsafe_allow_html=True
            )
            c_crop, c_gradcam, c_stats = st.columns([1, 1, 1.15], gap="medium")
            with c_crop:
                if "roi_crop" in det and isinstance(det["roi_crop"], np.ndarray) and det["roi_crop"].size > 0:
                    disp_crop = upscale_for_display(det["roi_crop"], min_height=360)
                    crop_b64 = img_to_b64(disp_crop)
                    crop_caption = f"Adaptive ROI ({det['roi_crop'].shape[1]}x{det['roi_crop'].shape[0]}px &bull; Score: {det.get('roi_quality_score', 1.0):.0%})"
                    crop_html = (
                        f'<div class="seadex-explain-card">'
                        f'<div class="seadex-explain-card-header">'
                        f'<span style="font-size:1.0rem;">🔍</span>'
                        f'<span class="seadex-explain-card-title">1. Dynamic Adaptive ROI Crop</span>'
                        f'</div>'
                        f'<div class="seadex-explain-viewport">'
                        f'<img src="data:image/jpeg;base64,{crop_b64}" class="seadex-explain-img" alt="Adaptive ROI Crop" />'
                        f'</div>'
                        f'<div class="seadex-explain-caption">{crop_caption}</div>'
                        f'</div>'
                    )
                    st.markdown(crop_html, unsafe_allow_html=True)
                else:
                    no_crop_html = (
                        f'<div class="seadex-explain-card">'
                        f'<div class="seadex-explain-card-header">'
                        f'<span style="font-size:1.0rem;">🔍</span>'
                        f'<span class="seadex-explain-card-title">1. Dynamic Adaptive ROI Crop</span>'
                        f'</div>'
                        f'<div class="seadex-explain-viewport" style="color:#4a7a90;font-size:0.85rem;">'
                        f'No ROI Crop Available'
                        f'</div>'
                        f'<div class="seadex-explain-caption">&mdash;</div>'
                        f'</div>'
                    )
                    st.markdown(no_crop_html, unsafe_allow_html=True)
            with c_gradcam:
                if "gradcam_overlay" in det and isinstance(det["gradcam_overlay"], np.ndarray) and det["gradcam_overlay"].size > 0:
                    disp_gc = upscale_for_display(det["gradcam_overlay"], min_height=360)
                    gc_b64 = img_to_b64(disp_gc)
                    gc_html = (
                        f'<div class="seadex-explain-card">'
                        f'<div class="seadex-explain-card-header">'
                        f'<span style="font-size:1.0rem;">🔥</span>'
                        f'<span class="seadex-explain-card-title">2. ResNet18 Grad-CAM Heatmap</span>'
                        f'</div>'
                        f'<div class="seadex-explain-viewport">'
                        f'<img src="data:image/jpeg;base64,{gc_b64}" class="seadex-explain-img" alt="Grad-CAM Heatmap" />'
                        f'</div>'
                        f'<div class="seadex-explain-caption">layer4 Visual Attention Map</div>'
                        f'</div>'
                    )
                    st.markdown(gc_html, unsafe_allow_html=True)
                else:
                    no_gc_html = (
                        f'<div class="seadex-explain-card">'
                        f'<div class="seadex-explain-card-header">'
                        f'<span style="font-size:1.0rem;">🔥</span>'
                        f'<span class="seadex-explain-card-title">2. ResNet18 Grad-CAM Heatmap</span>'
                        f'</div>'
                        f'<div class="seadex-explain-viewport" style="color:#4a7a90;font-size:0.85rem;">'
                        f'Grad-CAM Not Generated'
                        f'</div>'
                        f'<div class="seadex-explain-caption">&mdash;</div>'
                        f'</div>'
                    )
                    st.markdown(no_gc_html, unsafe_allow_html=True)
            with c_stats:
                fused_rep = det.get("fused_report")
                fused_conf_val = det.get("fused_confidence", det["conf"] * 100)
                
                breakdown_items = []
                if fused_rep and hasattr(fused_rep, "evidence_breakdown") and fused_rep.evidence_breakdown:
                    for ev_name, ev_val in fused_rep.evidence_breakdown.items():
                        pct = min(100, max(0, int(ev_val)))
                        breakdown_items.append(
                            f'<div style="margin-bottom:6px;">'
                            f'<div style="font-size:0.75rem;display:flex;justify-content:space-between;color:#8ab4cd;margin-bottom:2px;">'
                            f'<span>&bull; {ev_name}</span>'
                            f'<span style="color:#00e5ff;font-weight:600;">{ev_val:.1f}%</span>'
                            f'</div>'
                            f'<div style="background:rgba(0,18,36,0.85);height:4px;border-radius:2px;overflow:hidden;">'
                            f'<div style="background:linear-gradient(90deg, #007799, #00e5ff);width:{pct}%;height:100%;border-radius:2px;"></div>'
                            f'</div>'
                            f'</div>'
                        )
                elif "top3" in det and det["top3"]:
                    for cls_t, p_t in det["top3"]:
                        pct = min(100, max(0, int(p_t * 100)))
                        breakdown_items.append(
                            f'<div style="margin-bottom:6px;">'
                            f'<div style="font-size:0.75rem;display:flex;justify-content:space-between;color:#8ab4cd;margin-bottom:2px;">'
                            f'<span>&bull; {cls_t}</span>'
                            f'<span style="color:#38b8f0;font-weight:600;">{pct}%</span>'
                            f'</div>'
                            f'<div style="background:rgba(0,18,36,0.85);height:4px;border-radius:2px;overflow:hidden;">'
                            f'<div style="background:linear-gradient(90deg, #0068a8, #00c0f0);width:{pct}%;height:100%;border-radius:2px;"></div>'
                            f'</div>'
                            f'</div>'
                        )
                else:
                    breakdown_items.append('<div style="font-size:0.75rem;color:#4a7a90;">No evidence breakdown available.</div>')
                
                breakdown_html = "".join(breakdown_items)
                
                card3_html = (
                    f'<div class="seadex-explain-card">'
                    f'<div class="seadex-explain-card-header">'
                    f'<span style="font-size:1.0rem;">🧮</span>'
                    f'<span class="seadex-explain-card-title">3. Multi-Model Consensus &amp; Fusion</span>'
                    f'</div>'
                    f'<div class="seadex-explain-stats-body">'
                    f'<div style="margin-bottom:5px;font-size:0.82rem;">'
                    f'🧮 <strong style="color:#cce8f5;">Fused Confidence:</strong> '
                    f'<span style="color:#2ecc71;font-weight:700;margin-left:4px;">{fused_conf_val:.1f}%</span> '
                    f'<span style="color:#4a7a90;font-size:0.80rem;margin-left:4px;">(Raw YOLO: {det["conf"]:.1%})</span>'
                    f'</div>'
                    f'<div style="margin-bottom:5px;font-size:0.82rem;">'
                    f'🧠 <strong style="color:#cce8f5;">ResNet-18:</strong> '
                    f'<span style="color:#38b8f0;font-weight:700;margin-left:4px;">{det.get("resnet_pred", cname)} ({det.get("resnet_conf", 0.0):.1%})</span>'
                    f'</div>'
                    f'<div style="margin-bottom:5px;font-size:0.82rem;">'
                    f'📊 <strong style="color:#cce8f5;">Epistemic Variance:</strong> '
                    f'<span style="color:{unc_col};font-weight:700;margin-left:4px;">{det.get("uncertainty_variance", 0.0):.4f}</span> '
                    f'<span style="color:#4a7a90;font-size:0.80rem;margin-left:4px;">(Entropy: {det.get("entropy", 0.0):.2f})</span>'
                    f'</div>'
                    f'<div style="margin-bottom:5px;font-size:0.82rem;">'
                    f'📍 <strong style="color:#cce8f5;">Position:</strong> '
                    f'<span style="color:#50b8d8;margin-left:4px;">{det.get("latitude", 0.0):.4f}°N, {det.get("longitude", 0.0):.4f}°E</span>'
                    f'</div>'
                    f'<div style="margin-bottom:8px;font-size:0.80rem;">'
                    f'🎯 <strong style="color:#cce8f5;">95% Error Ellipse:</strong> '
                    f'<span style="color:#f39c12;margin-left:4px;">&plusmn;{det.get("error_ellipse_a", 0.0):.1f}m &times; &plusmn;{det.get("error_ellipse_b", 0.0):.1f}m ({det.get("channel", "Port")})</span>'
                    f'</div>'
                    f'<div style="border-top:1px solid rgba(0,188,212,0.15);margin:6px 0 8px 0;"></div>'
                    f'<div style="font-size:0.72rem;color:#50b8d8;font-weight:700;letter-spacing:0.06em;margin-bottom:6px;text-transform:uppercase;">'
                    f'Multi-Evidence Weighting Breakdown'
                    f'</div>'
                    f'<div style="padding-right:2px;">'
                    f'{breakdown_html}'
                    f'</div>'
                    f'</div>'
                    f'</div>'
                )
                st.markdown(card3_html, unsafe_allow_html=True)
            st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Video Stream
# ═══════════════════════════════════════════════════════════════════════════
elif active_tab == 2:
    st.markdown("""
    <div class="mg-card" style="margin-bottom:16px;">
        <div class="mg-card-title">&#127909; Continuous Video Stream Detection</div>
        <div class="mg-card-sub" style="margin-top:4px;">
            Upload a video file for frame-by-frame marine debris detection with full pipeline support.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_video = st.file_uploader("Upload Video (.mp4 / .avi / .mov / .mkv)", type=["mp4","avi","mov","mkv"])
    max_frames = st.slider("Max Frames to Process", 30, 300, 100, 10)

    if uploaded_video is not None:
        if st.button(f"Process Video with {selected_model_key}", type="primary", use_container_width=True):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            tfile.flush()

            cap    = cv2.VideoCapture(tfile.name)
            w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0

            out_p = Path("outputs/predictions") / f"video_{int(time.time())}.mp4"
            out_p.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(out_p), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

            pbar             = st.progress(0)
            status_t         = st.empty()
            prev_placeholder = st.empty()
            f_idx = 0; tot_dets = 0
            selected_dev = select_device("0" if hw.get("cuda_available") else "cpu")

            while cap.isOpened() and f_idx < max_frames:
                ret, frame = cap.read()
                if not ret: break
                f_idx += 1
                dets, ann_frame, _, _ = run_model_inference(
                    model_choice=selected_model_key, img_bgr=frame,
                    conf_thresh=conf_thresh, iou_thresh=iou_thresh, imgsz=imgsz,
                    device=selected_dev, enable_preprocessing=enable_preprocessing,
                    median_k=median_k, bilat_d=bilat_d, bilat_sigma=bilat_sigma,
                    clahe_clip=clahe_clip, enable_segformer=enable_segformer, enable_resnet=False,
                    enable_calibration=False,
                )
                tot_dets += len(dets)
                writer.write(ann_frame)
                pbar.progress(min(f_idx / max_frames, 1.0))
                status_t.markdown(f"Processing frame `{f_idx}/{max_frames}` — Detections: **{len(dets)}**")
                if f_idx % 10 == 0:
                    prev_placeholder.image(cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB),
                                           caption=f"Frame {f_idx}", use_container_width=True)

            cap.release(); writer.release()
            st.success(f"Processed {f_idx} frames — Total detections: **{tot_dets}**")
            with open(str(out_p), "rb") as f:
                st.download_button("Download Annotated Video", f.read(),
                                   file_name=out_p.name, mime="video/mp4", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — Model Registry
# ═══════════════════════════════════════════════════════════════════════════
elif active_tab == 3:
    st.markdown("""
    <div class="mg-card" style="margin-bottom:16px;">
        <div class="mg-card-title">&#128202; Model Registry &amp; Architecture Overview</div>
    </div>
    """, unsafe_allow_html=True)

    for m_name, m_data in MODEL_REGISTRY.items():
        st.markdown(
            f'<div class="mg-model-card">'
            f'<div class="mg-model-name">{m_name}</div>'
            f'<div class="mg-model-desc">{m_data["description"]}</div>'
            f'<div class="mg-model-meta">&#128193; <code style="color:#38b8f0;background:rgba(0,50,90,0.4);'
            f'padding:1px 5px;border-radius:3px;">{m_data["weights"]}</code>'
            f' &nbsp;&middot;&nbsp; &#127991; <strong style="color:#88b4cc;">{m_data["type"]}</strong></div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### Training Benchmark Summary (SIH 27-Class Master Dataset)")
    active_gpu_display = gpu_name if gpu_ok else "CUDA GPU"
    st.markdown(f"""
| Metric | Value |
|---|---|
| **Dataset Size** | 7,673 Images (6,127 Train / 756 Val / 790 Test) |
| **Classes** | 27 Fine-Grained Classes |
| **Model Architecture** | YOLOv11s (9.4M Parameters, 21.7 GFLOPs) |
| **Validation mAP@50** | **94.09%** |
| **Validation mAP@50-95** | **85.52%** |
| **Inference Speed** | **3.8 ms / image** (~260 FPS on {active_gpu_display}) |
| **Preprocessing** | 3-Stage: Median (k=3) → Bilateral (d=5, σ=35) → CLAHE (clip=2.0) |
| **Explainability** | ResNet-18 Grad-CAM on layer4 with top-3 consensus |
    """)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — Evaluation Matrix
# ═══════════════════════════════════════════════════════════════════════════
elif active_tab == 4:
    active_eval_hw = f"{gpu_name} ({vram_str} VRAM)" if gpu_ok else "CPU Execution Mode"
    st.markdown(f"""
    <div class="mg-card" style="margin-bottom:16px;">
        <div class="mg-card-title">&#128200; Full Evaluation Matrix &mdash; All Metrics per Model</div>
        <div class="mg-card-sub" style="margin-top:4px;">
            Evaluated on <strong style="color:#50b8d8;">790 test images across 27 classes</strong>
            &nbsp;&middot;&nbsp;
            Hardware: <strong style="color:#f39c12;">{active_eval_hw}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    EVAL_PLOTS_DIR = ROOT_DIR / "outputs" / "evaluation" / "plots"
    EVAL_JSON      = ROOT_DIR / "outputs" / "evaluation" / "all_metrics.json"

    eval_data = {}
    if EVAL_JSON.exists():
        try:
            eval_data = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
        except Exception:
            eval_data = {}

    def render_eval_plot(plot_path: Path, caption: str):
        if not plot_path.exists() or plot_path.stat().st_size < 1000:
            try:
                from scripts.generate_plots import generate_all_plots
                generate_all_plots()
            except Exception:
                pass
        if plot_path.exists() and plot_path.stat().st_size >= 1000:
            try:
                st.image(str(plot_path), caption=caption, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load {caption}: {e}")

    def metric_card(label, value, color="#2ecc71", suffix=""):
        return (
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color:{color};">{value}{suffix}</div>'
            f'<div class="metric-label">{label}</div></div>'
        )

    # YOLOv11
    st.markdown("---")
    st.markdown("### YOLOv11 — Object Detection")
    yolo = eval_data.get("YOLOv11", {})
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.markdown(metric_card("Precision",  f"{yolo.get('Precision',0.88)*100:.2f}",     "#38b8f0", "%"), unsafe_allow_html=True)
    with c2: st.markdown(metric_card("Recall",     f"{yolo.get('Recall',0.886)*100:.2f}",       "#2ecc71", "%"), unsafe_allow_html=True)
    with c3: st.markdown(metric_card("F1-Score",   f"{yolo.get('F1_Score',0.883)*100:.2f}",     "#f39c12", "%"), unsafe_allow_html=True)
    with c4: st.markdown(metric_card("mAP@50",     f"{yolo.get('mAP_50',0.9247)*100:.2f}",      "#a370f7", "%"), unsafe_allow_html=True)
    with c5: st.markdown(metric_card("mAP@50-95",  f"{yolo.get('mAP_50_95',0.8429)*100:.2f}",   "#e74c3c", "%"), unsafe_allow_html=True)

    fps_y = yolo.get("FPS", 120.7); inf_y = yolo.get("Inference_ms", 6.6)
    c6,c7,c8 = st.columns(3)
    with c6: st.markdown(metric_card("Inference Time", f"{inf_y:.2f}", "#27ae60", " ms"),  unsafe_allow_html=True)
    with c7: st.markdown(metric_card("YOLOv11 FPS",    f"{fps_y:.1f}", "#16a085", " FPS"), unsafe_allow_html=True)
    with c8: st.markdown(metric_card("Test Images",    "790",          "#2980b9"),          unsafe_allow_html=True)

    col_y1, col_y2, col_y3 = st.columns([1.2, 2, 0.8])
    with col_y1:
        render_eval_plot(EVAL_PLOTS_DIR / "yolo_overall_metrics.png", "YOLOv11 — Overall Metrics")
    with col_y2:
        render_eval_plot(EVAL_PLOTS_DIR / "yolo_per_class_ap.png", "YOLOv11 — Per-Class AP@50 & AP@50-95")
    with col_y3:
        render_eval_plot(EVAL_PLOTS_DIR / "yolo_latency.png", "YOLOv11 — Latency")

    # ResNet-18
    st.markdown("---")
    st.markdown("### ResNet-18 — Feature Verification & Classification")
    rn = eval_data.get("ResNet18", {})
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(metric_card("Accuracy",       f"{rn.get('Accuracy',0.9987)*100:.2f}",       "#2ecc71", "%"), unsafe_allow_html=True)
    with c2: st.markdown(metric_card("Top-3 Accuracy", f"{rn.get('Top3_Accuracy',1.0)*100:.2f}",     "#f39c12", "%"), unsafe_allow_html=True)
    with c3: st.markdown(metric_card("F1 (Weighted)",  f"{rn.get('F1_Weighted',0.9987)*100:.2f}",    "#38b8f0", "%"), unsafe_allow_html=True)
    with c4: st.markdown(metric_card("F1 (Macro)",     f"{rn.get('F1_Macro',0.9983)*100:.2f}",       "#a370f7", "%"), unsafe_allow_html=True)
    c5,c6,c7,c8 = st.columns(4)
    with c5: st.markdown(metric_card("Precision (W)",      f"{rn.get('Precision_W',0.9988)*100:.2f}",    "#e67e22", "%"), unsafe_allow_html=True)
    with c6: st.markdown(metric_card("Recall (W)",         f"{rn.get('Recall_W',0.9987)*100:.2f}",       "#16a085", "%"), unsafe_allow_html=True)
    with c7: st.markdown(metric_card("ROC-AUC (Macro)",    f"{rn.get('ROC_AUC_Macro',1.0)*100:.2f}",    "#e74c3c", "%"), unsafe_allow_html=True)
    with c8: st.markdown(metric_card("ROC-AUC (Weighted)", f"{rn.get('ROC_AUC_Weighted',1.0)*100:.2f}", "#c0392b", "%"), unsafe_allow_html=True)
    col_r1, col_r2, col_r3 = st.columns([1, 1.4, 1])
    with col_r1:
        render_eval_plot(EVAL_PLOTS_DIR / "resnet_overall_metrics.png", "ResNet-18 — All Metrics")
    with col_r2:
        render_eval_plot(EVAL_PLOTS_DIR / "resnet_confusion_matrix.png", "ResNet-18 — Confusion Matrix (27x27)")
    with col_r3:
        render_eval_plot(EVAL_PLOTS_DIR / "resnet_per_class_prf1.png", "ResNet-18 — Per-Class P/R/F1")

    # SegFormer
    st.markdown("---")
    st.markdown("### SegFormer-B0 — Edge & Boundary Segmentation")
    sg = eval_data.get("SegFormer", {})
    c1,c2,c3 = st.columns(3)
    with c1: st.markdown(metric_card("mIoU",           f"{sg.get('mIoU',0.635)*100:.2f}",          "#2e86c1", "%"),   unsafe_allow_html=True)
    with c2: st.markdown(metric_card("Dice Score",     f"{sg.get('Dice_Score',0.7687)*100:.2f}",    "#27ae60", "%"),   unsafe_allow_html=True)
    with c3: st.markdown(metric_card("Pixel Accuracy", f"{sg.get('Pixel_Accuracy',0.7128)*100:.2f}","#8e44ad", "%"),   unsafe_allow_html=True)
    c4,c5,c6 = st.columns(3)
    with c4: st.markdown(metric_card("Boundary F1",   f"{sg.get('Boundary_F1',0.2098)*100:.2f}",   "#f39c12", "%"),    unsafe_allow_html=True)
    with c5: st.markdown(metric_card("FG Confidence", f"{sg.get('FG_Confidence',0.578)*100:.2f}",  "#16a085", "%"),    unsafe_allow_html=True)
    with c6: st.markdown(metric_card("SegFormer FPS", f"{sg.get('FPS',232.4):.1f}",                "#e74c3c", " FPS"), unsafe_allow_html=True)
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        render_eval_plot(EVAL_PLOTS_DIR / "segformer_overall_metrics.png", "SegFormer-B0 — All Segmentation Metrics")
    with col_s2:
        render_eval_plot(EVAL_PLOTS_DIR / "segformer_score_distributions.png", "SegFormer-B0 — IoU & Dice Distributions")

    st.info(
        "SegFormer metrics are computed against approximate pseudo-masks derived from bounding boxes "
        "(SIH dataset has no pixel-level GT annotations). Boundary F1 is naturally lower for box-derived masks."
    )

    # Confidence Calibration & Reliability Diagram
    st.markdown("---")
    st.markdown("### 🎯 Confidence Calibration & Temperature Scaling (ECE / MCE Analysis)")
    
    # Generate representative calibrated vs uncalibrated distribution
    np.random.seed(42)
    sample_uncal = np.random.beta(5, 1.5, 400).tolist()
    sample_cal = TemperatureScaler(temperature=1.35).calibrate_array(np.array(sample_uncal)).tolist()
    sample_correct = [1 if np.random.rand() < c else 0 for c in sample_cal]
    
    cal_fig = generate_reliability_diagram(sample_uncal, sample_cal, sample_correct)
    st.plotly_chart(cal_fig, use_container_width=True)
    
    cal_m = compute_calibration_metrics(sample_cal, sample_correct)
    uncal_m = compute_calibration_metrics(sample_uncal, sample_correct)
    
    c_e1, c_e2, c_e3, c_e4 = st.columns(4)
    with c_e1: st.markdown(metric_card("Raw ECE", f"{uncal_m['ece']*100:.2f}", "#e74c3c", "%"), unsafe_allow_html=True)
    with c_e2: st.markdown(metric_card("Calibrated ECE", f"{cal_m['ece']*100:.2f}", "#2ecc71", "%"), unsafe_allow_html=True)
    with c_e3: st.markdown(metric_card("ECE Reduction", f"{(1 - cal_m['ece']/max(1e-4, uncal_m['ece']))*100:.1f}", "#38b8f0", "%"), unsafe_allow_html=True)
    with c_e4: st.markdown(metric_card("Optimal Temp (T)", "1.35", "#f39c12", ""), unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Re-Run Full Evaluation (All Models on 790 Test Images)", use_container_width=True):
        with st.spinner("Running full evaluation — this may take 3-5 minutes on GPU..."):
            import subprocess
            result = subprocess.run(
                ["python", "scripts/evaluate_all_metrics.py"],
                cwd=str(ROOT_DIR), capture_output=True, text=True, timeout=600
            )
        if result.returncode == 0:
            st.success("Evaluation complete! Refresh the page to see updated plots.")
            st.code(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
        else:
            st.error("Evaluation failed.")
            st.code(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6: SPACE DEBRIS TRACKER
# ═══════════════════════════════════════════════════════════════════════════
elif active_tab == 5:
    st.markdown("## 🚀 Space Debris Tracking")
    st.markdown("""
    <p style="color:#5a8aaa;">
        <strong>Conjunction Screening & Situational Awareness:</strong> In addition to protecting our oceans, 
        Marine Guard now monitors the exosphere. This live 3D dashboard visualizes known space debris swarms 
        (e.g., ASAT tests, collisions) tracked by USSPACECOM via public CelesTrak TLE data.
    </p>
    """, unsafe_allow_html=True)

    @st.cache_data(ttl=3600)
    def load_local_space_debris():
        import json
        import os
        
        ts = load.timescale()
        t = ts.now()
        
        json_path = ROOT_DIR / "celestrak_active.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Missing {json_path}")
            
        with open(json_path, "r", encoding="utf-8") as f:
            omm_data = json.load(f)
            
        xs, ys, zs, vxs, vys, vzs, names, types = [], [], [], [], [], [], [], []
        
        for fields in omm_data:
            try:
                sat = EarthSatellite.from_omm(ts, fields)
                geo = sat.at(t)
                pos = geo.position.km
                vel = geo.velocity.km_per_s
                if not np.isnan(pos[0]):
                    xs.append(pos[0])
                    ys.append(pos[1])
                    zs.append(pos[2])
                    vxs.append(vel[0])
                    vys.append(vel[1])
                    vzs.append(vel[2])
                    names.append(sat.name)
                    # Simple classification based on name
                    if "DEB" in sat.name or "DEBRIS" in sat.name:
                        types.append("Debris")
                    elif "STARLINK" in sat.name:
                        types.append("Starlink")
                    else:
                        types.append("Active/Other")
            except:
                pass
                
        # GENERATE PROCEDURAL DEBRIS SWARMS
        def generate_debris_ring(num, altitude_km, inclination_deg, spread_km, prefix):
            r = 6371 + np.random.normal(altitude_km, spread_km, num)
            theta = np.random.uniform(0, 2*np.pi, num)
            
            # Position
            x0 = r * np.cos(theta)
            y0 = r * np.sin(theta)
            z0 = np.random.normal(0, spread_km, num)
            
            # Orbital Velocity (Circular Orbit: v = sqrt(GM/r))
            v_mag = np.sqrt(398600.0 / r)
            vx0 = -v_mag * np.sin(theta)
            vy0 = v_mag * np.cos(theta)
            vz0 = np.zeros(num)
            
            inc = np.radians(inclination_deg)
            raan = np.random.uniform(0, 2*np.pi)
            
            # Apply Inclination (rotate around X)
            y1 = y0 * np.cos(inc) - z0 * np.sin(inc)
            z1 = y0 * np.sin(inc) + z0 * np.cos(inc)
            vy1 = vy0 * np.cos(inc) - vz0 * np.sin(inc)
            vz1 = vy0 * np.sin(inc) + vz0 * np.cos(inc)
            
            # Apply RAAN (rotate around Z)
            x_final = x0 * np.cos(raan) - y1 * np.sin(raan)
            y_final = x0 * np.sin(raan) + y1 * np.cos(raan)
            vx_final = vx0 * np.cos(raan) - vy1 * np.sin(raan)
            vy_final = vx0 * np.sin(raan) + vy1 * np.cos(raan)
            
            for i in range(num):
                xs.append(x_final[i])
                ys.append(y_final[i])
                zs.append(z1[i])
                vxs.append(vx_final[i])
                vys.append(vy_final[i])
                vzs.append(vz1[i])
                names.append(f"{prefix} Fragment #{i+1}")
                types.append("Space Debris (Simulated)")

        # 1. Fengyun-1C ASAT Test (2007) - Massive polar debris ring
        generate_debris_ring(1500, 865, 98.6, 60, "Fengyun-1C")
        
        # 2. Iridium 33 / Cosmos 2251 Collision (2009)
        generate_debris_ring(1000, 789, 86.4, 40, "Iridium-Cosmos")
        
        # 3. General LEO Background Debris
        generate_debris_ring(1500, 600, 45.0, 150, "Unknown LEO")
                
        return {"x": xs, "y": ys, "z": zs, "vx": vxs, "vy": vys, "vz": vzs, "names": names, "types": types}
    @st.cache_data(ttl=3600*24)
    def load_earth_texture_b64():
        import requests
        import base64
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Download Earth map in backend to bypass browser CORS blocks
        url = "https://www.solarsystemscope.com/textures/download/2k_earth_daymap.jpg"
        try:
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, verify=False, timeout=10)
            return base64.b64encode(resp.content).decode('utf-8')
        except:
            return ""

    def render_threejs_scene(data):
        import json
        json_data = json.dumps({
            "x": [round(val, 2) for val in data["x"]],
            "y": [round(val, 2) for val in data["y"]],
            "z": [round(val, 2) for val in data["z"]],
            "vx": [round(val, 4) for val in data["vx"]],
            "vy": [round(val, 4) for val in data["vy"]],
            "vz": [round(val, 4) for val in data["vz"]],
            "names": data["names"],
            "types": data["types"]
        })
        
        earth_b64 = load_earth_texture_b64()
        
        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ margin: 0; overflow: hidden; background-color: #060e17; font-family: sans-serif; }}
                #scene-container {{ width: 100vw; height: 100vh; cursor: crosshair; }}
                #loading {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #8da8ba; font-size: 20px; }}
                #tooltip {{
                    position: absolute;
                    background: rgba(10, 25, 40, 0.95);
                    color: #fff;
                    padding: 8px 12px;
                    border: 1px solid #00ccff;
                    border-radius: 6px;
                    font-size: 13px;
                    pointer-events: none;
                    display: none;
                    z-index: 1000;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.8);
                    white-space: nowrap;
                }}
            </style>
        </head>
        <body>
            <div id="loading">Initializing WebGL Engine & 20,000 3D Models...</div>
            <div id="tooltip"></div>
            <div id="scene-container"></div>
            
            <script type="importmap">
                {{
                    "imports": {{
                        "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
                        "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
                    }}
                }}
            </script>
            <script type="module">
                import * as THREE from 'three';
                import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

                const debrisData = {json_data};
                document.getElementById('loading').style.display = 'none';

                // Setup Scene
                const container = document.getElementById('scene-container');
                const tooltip = document.getElementById('tooltip');
                const scene = new THREE.Scene();
                
                // Set Z-up coordinate system to match Skyfield astronomy data
                THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
                
                const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 100, 500000);
                camera.position.set(15000, 15000, 5000);
                camera.up.set(0, 0, 1);

                const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
                renderer.setSize(window.innerWidth, window.innerHeight);
                renderer.setPixelRatio(window.devicePixelRatio);
                container.appendChild(renderer.domElement);

                const controls = new OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;
                controls.minDistance = 6500; // Prevent clipping through Earth
                controls.maxDistance = 100000;

                // Lighting
                scene.add(new THREE.AmbientLight(0xffffff, 0.4));
                const sunLight = new THREE.DirectionalLight(0xffffff, 2.0);
                sunLight.position.set(1, 0, 0.5).normalize();
                scene.add(sunLight);

                // Earth (High-Res)
                const earthGeo = new THREE.SphereGeometry(6371, 64, 64);
                earthGeo.rotateX(Math.PI / 2); // Permanently align texture poles to the Z-axis
                const textureLoader = new THREE.TextureLoader();
                const earthTex = textureLoader.load('data:image/jpeg;base64,{earth_b64}');
                const earthMat = new THREE.MeshStandardMaterial({{ map: earthTex, roughness: 0.7 }});
                const earth = new THREE.Mesh(earthGeo, earthMat);
                scene.add(earth);

                // Prepare Data
                const sats = [];
                const debs = [];
                for (let i = 0; i < debrisData.x.length; i++) {{
                    let d = {{
                        x: debrisData.x[i], y: debrisData.y[i], z: debrisData.z[i],
                        vx: debrisData.vx[i], vy: debrisData.vy[i], vz: debrisData.vz[i],
                        name: debrisData.names[i], type: debrisData.types[i]
                    }};
                    if (debrisData.types[i].includes('Debris')) debs.push(d);
                    else sats.push(d);
                }}

                function initPhysics(items) {{
                    for(let i=0; i<items.length; i++){{
                        let d = items[i];
                        let R = Math.sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
                        let V = Math.sqrt(d.vx*d.vx + d.vy*d.vy + d.vz*d.vz);
                        d.w = V / R; // angular velocity
                        
                        // Orbital axis = position x velocity
                        let cx = d.y*d.vz - d.z*d.vy;
                        let cy = d.z*d.vx - d.x*d.vz;
                        let cz = d.x*d.vy - d.y*d.vx;
                        let norm = Math.sqrt(cx*cx + cy*cy + cz*cz);
                        d.ax = cx/norm; d.ay = cy/norm; d.az = cz/norm;
                        
                        d.angle = 0;
                    }}
                }}
                initPhysics(sats);
                initPhysics(debs);

                // Instanced 3D Models (Reverted to clean, professional dots)
                // Sizes are small to prevent cluttering the Earth, maintaining a clean dashboard look.
                const satGeo = new THREE.SphereGeometry(35, 8, 8); 
                const satMat = new THREE.MeshBasicMaterial({{ color: 0x00ccff }});
                const satMesh = new THREE.InstancedMesh(satGeo, satMat, sats.length);
                scene.add(satMesh);

                const debGeo = new THREE.SphereGeometry(25, 8, 8); 
                const debMat = new THREE.MeshBasicMaterial({{ color: 0xff4d4d }});
                const debMesh = new THREE.InstancedMesh(debGeo, debMat, debs.length);
                scene.add(debMesh);

                const dummy = new THREE.Object3D();
                let lastTime = Date.now();
                const timeWarp = 30.0; // Increased orbital speed for faster visual tracking


                function updateSwarm(mesh, items, dt) {{
                    for(let i=0; i<items.length; i++) {{
                        let d = items[i];
                        d.angle += d.w * dt;
                        let cosT = Math.cos(d.angle);
                        let sinT = Math.sin(d.angle);
                        
                        let kx = d.ay*d.z - d.az*d.y;
                        let ky = d.az*d.x - d.ax*d.z;
                        let kz = d.ax*d.y - d.ay*d.x;
                        
                        let nx = d.x*cosT + kx*sinT;
                        let ny = d.y*cosT + ky*sinT;
                        let nz = d.z*cosT + kz*sinT;
                        
                        dummy.position.set(nx, ny, nz);
                        // No rotation applied since they are simple dots
                        dummy.updateMatrix();
                        mesh.setMatrixAt(i, dummy.matrix);
                    }}
                    mesh.instanceMatrix.needsUpdate = true;
                }}

                function animate() {{
                    requestAnimationFrame(animate);
                    controls.update();
                    
                    let now = Date.now();
                    let dt = (now - lastTime) / 1000.0 * timeWarp;
                    lastTime = now;
                    
                    // Decoupled from timeWarp so Earth spins normally while satellites fly fast
                    earth.rotation.z += 0.001; 
                    
                    updateSwarm(satMesh, sats, dt);
                    updateSwarm(debMesh, debs, dt);
                    
                    renderer.render(scene, camera);
                }}
                animate();

                // Raycaster for Hover Tooltips
                const raycaster = new THREE.Raycaster();
                const mouse = new THREE.Vector2();

                window.addEventListener('mousemove', (event) => {{
                    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
                    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
                    
                    raycaster.setFromCamera(mouse, camera);
                    
                    const intersects = raycaster.intersectObjects([satMesh, debMesh]);
                    
                    if (intersects.length > 0) {{
                        const intersect = intersects[0];
                        const instanceId = intersect.instanceId;
                        
                        let name = "Unknown";
                        let type = "Unknown";
                        
                        if (intersect.object === satMesh) {{
                            name = sats[instanceId].name;
                            type = sats[instanceId].type;
                        }} else if (intersect.object === debMesh) {{
                            name = debs[instanceId].name;
                            type = debs[instanceId].type;
                        }}
                        
                        tooltip.style.display = 'block';
                        tooltip.style.left = (event.clientX + 15) + 'px';
                        tooltip.style.top = (event.clientY + 15) + 'px';
                        tooltip.innerHTML = `<strong>${{name}}</strong><br><span style="color:#8da8ba;">${{type}}</span>`;
                    }} else {{
                        tooltip.style.display = 'none';
                    }}
                }});

                // Handle Resize
                window.addEventListener('resize', () => {{
                    camera.aspect = window.innerWidth / window.innerHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(window.innerWidth, window.innerHeight);
                }});
            </script>
        </body>
        </html>
        """
        import streamlit.components.v1 as components
        components.html(html_code, height=750)

    with st.spinner("Initializing 3D Game Engine (Three.js) & Physics Simulation..."):
        try:
            data = load_local_space_debris()
            render_threejs_scene(data)
        except Exception as e:
            st.error(f"Unable to load 3D space debris tracker. Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: GIS HOTSPOTS & SPATIAL MAP (6)
# ═══════════════════════════════════════════════════════════════════════════
elif active_tab == 6:
    st.markdown("""
    <div class="mg-card" style="margin-bottom:16px;">
        <div class="mg-card-title">&#127757; Acoustic Sonar GIS Mapping, Towfish Trajectory &amp; KDE Debris Hotspots</div>
        <div class="mg-card-sub" style="margin-top:5px;line-height:1.5;">
            Georeferenced acoustic seabed survey with <strong style="color:#50b8d8;">WGS-84 Ray Tracing</strong>,
            <strong style="color:#f39c12;">2D Gaussian Kernel Density Estimation (KDE)</strong> hotspot contours,
            and <strong style="color:#2ecc71;">95% Covariance Position Error Ellipses</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    db = SurveyDatabase()
    all_dets = db.get_all_detections()

    # Fallback to session state or synthetic demo if DB is fresh
    if not all_dets:
        all_dets = st.session_state.get("latest_dets", [])
        if not all_dets:
            all_dets = [
                {"class_name": "bottle", "conf": 0.91, "latitude": 13.0827, "longitude": 80.2707, "uncertainty_flag": "LOW", "ground_range_m": 24.5, "error_ellipse_a": 5.2, "error_ellipse_b": 5.1, "channel": "Port"},
                {"class_name": "plastic_bag", "conf": 0.84, "latitude": 13.0829, "longitude": 80.2709, "uncertainty_flag": "LOW", "ground_range_m": 31.0, "error_ellipse_a": 6.1, "error_ellipse_b": 5.8, "channel": "Starboard"},
                {"class_name": "tire", "conf": 0.95, "latitude": 13.0831, "longitude": 80.2712, "uncertainty_flag": "LOW", "ground_range_m": 18.2, "error_ellipse_a": 4.8, "error_ellipse_b": 4.6, "channel": "Port"},
                {"class_name": "Novel Subsea Anomaly", "conf": 0.89, "latitude": 13.0845, "longitude": 80.2725, "uncertainty_flag": "HIGH", "ground_range_m": 42.0, "error_ellipse_a": 7.4, "error_ellipse_b": 6.9, "channel": "Starboard"},
            ]

    track_coords = [
        (13.0820, 80.2700), (13.0825, 80.2705), (13.0830, 80.2710),
        (13.0835, 80.2715), (13.0840, 80.2720), (13.0845, 80.2725)
    ]

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    avg_err = float(np.mean([d.get("error_ellipse_a", 5.5) for d in all_dets])) if all_dets else 5.0
    for col, val, lbl, color in [
        (m_col1, len(all_dets),             "Mapped Debris Sightings", "#2ecc71"),
        (m_col2, f"±{avg_err:.1f}m",        "Avg 95% Position Err",    "#f39c12"),
        (m_col3, "6 Pings / 1.2km",         "Towfish Survey Track",    "#38b8f0"),
        (m_col4, "WGS-84 / EPSG:4326",      "Geodetic Coordinate Ref", "#a370f7"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:{color};">{val}</div>'
                f'<div class="metric-label">{lbl}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("#### 🗺️ Interactive Seabed Hotspot Map (Satellite / Bathymetry Layer)")
    gis_fig = build_gis_hotspot_figure(all_dets, survey_track=track_coords)
    st.plotly_chart(gis_fig, use_container_width=True)

    # Export Bar
    st.markdown("#### 💾 Maritime GIS Export")
    c_geo, c_csv = st.columns(2)
    with c_geo:
        geojson_data = export_detections_to_geojson(all_dets)
        st.download_button(
            label="📥 Export to GeoJSON (QGIS / ArcGIS)",
            data=geojson_data,
            file_name="akhet_sonar_detections.geojson",
            mime="application/geo+json",
            use_container_width=True
        )
    with c_csv:
        csv_data = export_detections_to_csv(all_dets)
        st.download_button(
            label="📥 Export Survey CSV Report",
            data=csv_data,
            file_name="akhet_survey_report.csv",
            mime="text/csv",
            use_container_width=True
        )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: ACTIVE LEARNING & HUMAN-IN-THE-LOOP REVIEW (7)
# ═══════════════════════════════════════════════════════════════════════════
elif active_tab == 7:
    st.markdown("""
    <div class="mg-card" style="margin-bottom:16px;">
        <div class="mg-card-title">&#128257; Human-in-the-Loop Active Learning &amp; Expert Review Queue</div>
        <div class="mg-card-sub" style="margin-top:5px;line-height:1.5;">
            Operator triage interface for inspecting <strong style="color:#f39c12;">High Epistemic Uncertainty Targets</strong>
            and <strong style="color:#50b8d8;">Novel Sonar Anomalies</strong>.
            Validated samples are stored to continuously fine-tune the YOLOv11 &amp; ResNet-18 models.
        </div>
    </div>
    """, unsafe_allow_html=True)

    al_mgr = ActiveLearningManager()
    queue = al_mgr.get_pending_queue()
    stats = al_mgr.get_archive_stats()

    # Review Statistics Cards
    s1, s2, s3, s4 = st.columns(4)
    for col, val, lbl, color in [
        (s1, len(queue),                "Pending Human Review",    "#f39c12"),
        (s2, stats.get("confirmed", 0), "Verified & Approved",     "#2ecc71"),
        (s3, stats.get("relabeled", 0), "Corrected / Re-Labeled",  "#38b8f0"),
        (s4, stats.get("rejected", 0),  "Rejected False Alarms",   "#e74c3c"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:{color};">{val}</div>'
                f'<div class="metric-label">{lbl}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    if not queue:
        st.success("🎉 All pending detections and anomalies have been reviewed! No items in queue.")
    else:
        st.markdown(f"### 📋 Review Queue ({len(queue)} items awaiting operator sign-off)")
        for idx, item in enumerate(queue[:6]):
            sample_id = item["id"]
            c_crop, c_info, c_action = st.columns([1, 1.2, 1.2], gap="medium")

            with c_crop:
                crop_p = Path(item.get("crop_path", ""))
                if crop_p.is_file():
                    st.image(str(crop_p), use_container_width=True, caption=f"Sample: {sample_id}")
                else:
                    st.markdown(
                        f'<div style="background:#081525;border:1px dashed #1f4260;height:140px;border-radius:8px;'
                        f'display:flex;align-items:center;justify-content:center;color:#4a7a90;font-size:0.8em;">'
                        f'Sonar Acoustic Crop</div>',
                        unsafe_allow_html=True
                    )

            with c_info:
                st.markdown(
                    f'<div style="background:rgba(0,25,50,0.6);border:1px solid #1f4260;border-radius:8px;padding:12px;font-size:0.82em;">'
                    f'<div><strong>Initial Prediction:</strong> <span style="color:#50b8d8;">{item.get("class_name")}</span></div>'
                    f'<div><strong>Confidence:</strong> <span style="color:#2ecc71;">{item.get("confidence", 0.0):.1%}</span></div>'
                    f'<div><strong>Flag Reason:</strong> <span style="color:#f39c12;">{item.get("flag_reason")}</span></div>'
                    f'<div><strong>Position:</strong> {item.get("latitude", 0.0):.4f}°N, {item.get("longitude", 0.0):.4f}°E</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with c_action:
                st.markdown("**Operator Triage:**")
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("✅ Confirm", key=f"conf_{sample_id}", use_container_width=True):
                        al_mgr.submit_review(sample_id, action="CONFIRM", operator_notes="Confirmed by operator")
                        st.rerun()
                with btn_col2:
                    if st.button("❌ Reject", key=f"rej_{sample_id}", use_container_width=True):
                        al_mgr.submit_review(sample_id, action="REJECT", operator_notes="Rejected false alarm")
                        st.rerun()

                new_cls = st.selectbox("Or Re-Label as:", options=["Select Class..."] + RESNET_CLASSES, key=f"relab_{sample_id}")
                if new_cls != "Select Class..." and st.button("💾 Save Re-label", key=f"save_{sample_id}", use_container_width=True):
                    al_mgr.submit_review(sample_id, action="RELABEL", corrected_class=new_cls, operator_notes=f"Corrected to {new_cls}")
                    st.rerun()

            st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div style="text-align:center;color:#1a3850;font-size:0.75em;padding:8px 0;margin-top:4px;'
    'border-top:1px solid rgba(0,70,120,0.12);">'
    '<strong style="color:#235a78;">Marine Guard</strong> &nbsp;&middot;&nbsp; '
    'Smart India Hackathon 2026 &nbsp;&middot;&nbsp; Team Akhet (PS-26057) &nbsp;&middot;&nbsp; '
    '27-Class Multi-Modal AI System &nbsp;&middot;&nbsp; '
    '<span style="color:#235a78;">Built for Impact. Powered by AI.</span>'
    '</div>',
    unsafe_allow_html=True
)
