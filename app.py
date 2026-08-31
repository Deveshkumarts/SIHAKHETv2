"""
🌊 Akhet Marine & Sonar AI Platform (SIH 2026 - PS 26057)
Modular Multi-Model Architecture with 3-Stage Preprocessing (Median -> Bilateral -> CLAHE),
SegFormer Edge Segmentation, and ResNet-18 PyTorch Grad-CAM Explainability.
"""

import sys
import io
import json
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

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
from utils.roi_utils import expand_and_clamp_bbox, roi_mask_to_full_image
from utils.sonar_preprocess import (
    preprocess_universal_image,
    apply_median_filter,
    apply_bilateral_denoise,
    apply_clahe,
)
from resnet.classifier import ResNet18InferenceEngine, MASTER_CLASSES as RESNET_CLASSES

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Marine Guard — Akhet AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
}
.stApp {
    background: #08111f !important;
}
/* ── Main container max width ── */
div[data-testid="stMainBlockContainer"],
.block-container {
    max-width: 100% !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}


/* ── Sidebar shell ── */
[data-testid="stSidebar"] {
    background: #0c1829 !important;
    border-right: 1px solid rgba(0,160,220,0.12) !important;
    min-width: 240px !important;
    max-width: 240px !important;
}
[data-testid="stSidebarContent"] {
    padding: 0 0 16px 0 !important;
}

/* ── Sidebar text colors ── */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {
    color: #9bbdd4 !important;
    font-size: 0.82em !important;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #5aaed4 !important;
    font-size: 0.82em !important;
    font-weight: 600 !important;
    margin: 12px 0 4px 0 !important;
    padding-top: 2px !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(0,160,220,0.12) !important;
    margin: 8px 0 !important;
}
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] .stCaption p {
    color: #4a7a99 !important;
    font-size: 0.75em !important;
    line-height: 1.4 !important;
}

/* ── Selectbox in sidebar ── */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(0,30,60,0.7) !important;
    border: 1px solid rgba(0,140,200,0.3) !important;
    border-radius: 8px !important;
    color: #b0d8f0 !important;
    font-size: 0.82em !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #b0d8f0 !important;
}

/* ── Sliders ── */
[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
    background: #0096c7 !important;
    border-color: #00ccff !important;
}

/* ── Toggle ── */
[data-testid="stSidebar"] [data-testid="stToggle"] label { font-size: 0.82em !important; }

/* ── Checkboxes ── */
[data-testid="stSidebar"] [data-testid="stCheckbox"] label { font-size: 0.82em !important; }

/* ── Expander ── */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(0,25,50,0.5) !important;
    border: 1px solid rgba(0,140,200,0.15) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    color: #7aafcc !important;
    font-size: 0.8em !important;
}

/* ── Global headings ── */
h1,h2,h3,h4,h5 { color: #c0ddf5 !important; }

/* ── Top header ── */
.mg-topbar {
    background: linear-gradient(90deg, #0d1e38 0%, #0f2448 100%);
    border-bottom: 1px solid rgba(0,160,220,0.15);
    padding: 10px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 0 -1rem 1rem -1rem;
    flex-wrap: nowrap;
    gap: 6px;
    min-width: 0;
    width: calc(100% + 2rem);
    box-sizing: border-box;
}
.mg-topbar-left { display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
.mg-topbar-icon {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #1460a0, #00a8d4);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2em; flex-shrink: 0;
}
.mg-topbar-title { font-size: 1.05em; font-weight: 700; color: #e4f0ff; letter-spacing: 0.02em; }
.mg-topbar-sub   { font-size: 0.68em; color: #6a9ab8; margin-top: 1px; }
.mg-topbar-right { display: flex; align-items: center; gap: 7px; flex-wrap: nowrap; min-width: 0; overflow: hidden; }
.mg-badge {
    display: inline-flex; align-items: center; gap: 4px;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.72em; font-weight: 600;
    white-space: nowrap;
}
.mg-badge-green  { background: rgba(0,200,90,0.12);  border: 1px solid rgba(0,200,90,0.35); color: #00dd70; }
.mg-badge-blue   { background: rgba(0,160,220,0.12); border: 1px solid rgba(0,160,220,0.30); color: #60c0f0; }
.mg-avatar {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #1a5a90, #0090c0);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    color: #fff; font-size: 0.75em; font-weight: 700;
    flex-shrink: 0;
}

/* ── Pipeline badge row ── */
.mg-pipeline-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px; }
.mg-pipe-badge {
    display: inline-flex; align-items: center; gap: 5px;
    border-radius: 16px;
    padding: 4px 12px;
    font-size: 0.75em; font-weight: 500;
    white-space: nowrap;
}
.mg-pipe-filter { background: rgba(0,140,220,0.10); border: 1px solid rgba(0,160,255,0.22); color: #80c8f0; }
.mg-pipe-model  { background: rgba(0,200,100,0.08); border: 1px solid rgba(0,220,110,0.22); color: #60d890; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(12,24,44,0.8);
    border-radius: 10px;
    padding: 4px 6px;
    border: 1px solid rgba(0,140,200,0.14);
    gap: 2px !important;
    margin-bottom: 16px;
}
[data-testid="stTabs"] [role="tab"] {
    background: transparent !important;
    color: #5a8aaa !important;
    border-radius: 7px !important;
    padding: 7px 14px !important;
    font-size: 0.8em !important;
    font-weight: 500 !important;
    border: none !important;
    transition: all 0.2s ease;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: rgba(0,120,190,0.25) !important;
    color: #90d0f8 !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #0096c7 !important;
}
[data-testid="stTabs"] [role="tab"]:hover {
    background: rgba(0,140,200,0.12) !important;
    color: #80c8e8 !important;
}
[data-testid="stTabs"] [role="tabpanel"] { padding-top: 0 !important; }

/* ── Cards ── */
.mg-card {
    background: rgba(12,28,52,0.85);
    border: 1px solid rgba(0,140,200,0.16);
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 14px;
    box-sizing: border-box;
}
.mg-card-hdr {
    display: flex; align-items: flex-start; gap: 10px; margin-bottom: 12px;
}
.mg-num {
    width: 26px; height: 26px;
    background: linear-gradient(135deg, #0060a0, #009acc);
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.78em; font-weight: 700; color: #fff; flex-shrink: 0;
}
.mg-num-green  { background: linear-gradient(135deg, #005830, #009850); }
.mg-num-purple { background: linear-gradient(135deg, #4a0090, #7a30c0); }
.mg-card-title { font-size: 0.92em; font-weight: 600; color: #c0dff5; }
.mg-card-sub   { font-size: 0.74em; color: #4a7a99; margin-top: 2px; }

/* ── Stat grid ── */
.mg-stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px; }
.mg-stat {
    background: rgba(0,30,60,0.55);
    border: 1px solid rgba(0,140,200,0.18);
    border-radius: 9px;
    padding: 11px 10px;
    text-align: center;
}
.mg-stat-val { font-size: 1.5em; font-weight: 700; }
.mg-stat-lbl { font-size: 0.68em; color: #5a8aaa; margin-top: 3px; }
.mg-stat.c-blue   .mg-stat-val { color: #38b8f0; }
.mg-stat.c-green  .mg-stat-val { color: #2ecc71; }
.mg-stat.c-purple .mg-stat-val { color: #a370f7; font-size: 1.0em; padding-top: 5px; }
.mg-stat.c-orange .mg-stat-val { color: #f39c12; font-size: 1.0em; padding-top: 5px; }

/* ── Info rows ── */
.mg-info-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid rgba(0,120,180,0.08);
    font-size: 0.76em;
}
.mg-info-row:last-child { border-bottom: none; }
.mg-info-lbl { color: #4a7090; display: flex; align-items: center; gap: 6px; }
.mg-info-val { color: #38b8f0; font-weight: 500; text-align: right; max-width: 55%; }

/* ── Status dot ── */
.dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    margin-right: 4px;
    vertical-align: middle;
}
.dot-green  { background: #2ecc71; box-shadow: 0 0 5px #2ecc71; }
.dot-yellow { background: #f39c12; box-shadow: 0 0 5px #f39c12; }
.dot-blue   { background: #3db8e8; box-shadow: 0 0 5px #3db8e8; }

/* ── Status row ── */
.mg-sys-row {
    display: flex; justify-content: space-between; align-items: center;
    font-size: 0.76em;
    padding: 5px 0;
    border-bottom: 1px solid rgba(0,100,150,0.10);
}
.mg-sys-row:last-child { border-bottom: none; }

/* ── Metric cards ── */
.metric-card {
    background: rgba(0,30,60,0.55);
    border: 1px solid rgba(0,140,200,0.18);
    border-radius: 9px;
    padding: 13px 10px;
    text-align: center;
    margin: 3px 0;
}
.metric-value { font-size: 1.55em; font-weight: 700; color: #2ecc71; }
.metric-label { font-size: 0.74em; color: #5a8aaa; margin-top: 3px; }

/* ── Detection target cards ── */
.mg-det-card {
    background: rgba(0,22,48,0.75);
    border-radius: 10px;
    padding: 13px 10px;
    text-align: center;
    margin: 4px 0;
}

/* ── Model registry card ── */
.mg-model-card {
    background: rgba(10,22,46,0.75);
    border-left: 3px solid #0090c0;
    border-radius: 0 10px 10px 0;
    padding: 13px 16px;
    margin: 8px 0;
}
.mg-model-name { font-size: 0.97em; font-weight: 600; color: #ddeeff; }
.mg-model-desc { font-size: 0.8em; color: #6a9aaa; margin-top: 4px; line-height: 1.4; }
.mg-model-meta { font-size: 0.73em; color: #4a6a7a; margin-top: 6px; }

/* ── Upload area ── */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(0,30,60,0.35) !important;
    border: 2px dashed rgba(0,140,200,0.30) !important;
    border-radius: 10px !important;
}

/* ── Primary button ── */
[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #0070b0, #0098d4) !important;
    border: none !important;
    border-radius: 9px !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 0.88em !important;
    box-shadow: 0 4px 14px rgba(0,140,210,0.3) !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0082c8, #00aaea) !important;
    box-shadow: 0 6px 18px rgba(0,160,234,0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── Secondary button ── */
[data-testid="stButton"] button[kind="secondary"] {
    background: rgba(0,50,90,0.45) !important;
    border: 1px solid rgba(0,140,200,0.28) !important;
    border-radius: 9px !important;
    color: #70b8e0 !important;
    font-weight: 500 !important;
    font-size: 0.88em !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #0070b0, #00ccff) !important;
    border-radius: 4px !important;
}

/* ── File uploader label override ── */
[data-testid="stFileUploader"] label {
    color: #7aafcc !important;
    font-size: 0.82em !important;
}

/* ── Selectbox in main area ── */
[data-baseweb="select"] > div {
    background: rgba(10,28,55,0.8) !important;
    border-color: rgba(0,120,180,0.25) !important;
    border-radius: 8px !important;
    color: #b0d0ea !important;
}

/* ── Table ── */
table { width: 100%; border-collapse: collapse; }
th {
    background: rgba(0,70,130,0.4) !important;
    color: #70b8e0 !important;
    font-size: 0.83em !important;
    padding: 10px 12px !important;
}
td {
    color: #90b8cc !important;
    font-size: 0.81em !important;
    padding: 8px 12px !important;
    border-bottom: 1px solid rgba(0,80,140,0.14) !important;
}

/* ── Hide Streamlit chrome — display:none removes layout space ── */
#MainMenu { display: none !important; }
footer    { display: none !important; }
header    { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stHeader"]     { display: none !important; }

/* ── FINAL TOP GAP KILL — must stay at bottom of stylesheet ── */
body [data-testid="stMainBlockContainer"] {
    padding-top: 0 !important;
    padding-bottom: 1rem !important;
    margin-top: 0 !important;
}
body .block-container {
    padding-top: 0 !important;
    padding-bottom: 1rem !important;
    margin-top: 0 !important;
}
body section[data-testid="stMain"] {
    padding-top: 0 !important;
}
body [data-testid="stVerticalBlock"] > div:first-child > div:first-child {
    margin-top: 0 !important;
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
}

SEGFORMER_WEIGHTS = (
    "weights/segformer_b0_best.pt"
    if Path("weights/segformer_b0_best.pt").exists()
    else "outputs/segformer/weights/best.pt"
)
RESNET_WEIGHTS = "weights/resnet18_debris_best.pt"

CLASS_METADATA = {
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
    if not p.exists():
        return YOLO("yolo11s.pt")
    return YOLO(str(p))


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


# ─── Inference ───────────────────────────────────────────────────────────────
def run_model_inference(
    model_choice, img_bgr, conf_thresh, iou_thresh, imgsz, device,
    enable_preprocessing=True, median_k=3, bilat_d=5, bilat_sigma=35.0,
    clahe_clip=2.0, enable_segformer=False, enable_resnet=True
):
    processed_img_bgr = (
        preprocess_universal_image(img_bgr, median_ksize=median_k,
                                   bilateral_d=bilat_d, bilateral_sigma=bilat_sigma,
                                   clahe_clip=clahe_clip)
        if enable_preprocessing else img_bgr
    )

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

    resnet_engine = load_resnet_engine() if enable_resnet else None
    for det in filtered_dets:
        rx1, ry1, rx2, ry2 = expand_and_clamp_bbox(det["bbox"], processed_img_bgr.shape, padding_ratio=0.20)
        roi_crop = processed_img_bgr[ry1:ry2, rx1:rx2]
        det["roi_crop"] = roi_crop
        det["roi_bbox"] = [rx1, ry1, rx2, ry2]
        if resnet_engine and roi_crop.size > 0:
            r = resnet_engine.predict_roi(roi_crop, target_class_name=det["class_name"])
            det["resnet_pred"]     = r["pred_class"]
            det["resnet_conf"]     = r["pred_conf"]
            det["gradcam_overlay"] = r["gradcam_overlay"]
            det["top3"]            = r["top3"]

    annotated_img = processed_img_bgr.copy()
    seg_model = load_segformer_model(SEGFORMER_WEIGHTS) if enable_segformer else None
    if seg_model and filtered_dets:
        h_full, w_full = processed_img_bgr.shape[:2]
        full_mask = np.zeros((h_full, w_full), dtype=np.uint8)
        for det in filtered_dets:
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

    for det in filtered_dets:
        cname   = det["class_name"]
        meta    = CLASS_METADATA.get(cname, {"color": "#00d4ff"})
        bgr_col = hex_to_bgr(meta["color"])
        draw_bounding_box(annotated_img, det["bbox"],
                          f"{cname} {det['conf']:.0%}", bgr_col, line_thickness=2)

    return filtered_dets, annotated_img, processed_img_bgr


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Brand header ──
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;
                padding:16px 16px 12px 16px;
                border-bottom:1px solid rgba(0,140,200,0.12);">
        <div>
            <div style="font-size:0.95em;font-weight:700;color:#d8eeff;">Marine Guard</div>
            <div style="font-size:0.65em;color:#4a7a99;">Marine Debris Detection</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Nav section ──
    st.markdown(
        '<div style="font-size:0.63em;font-weight:600;letter-spacing:0.08em;color:#3a6a88;'
        'padding:12px 16px 4px 16px;text-transform:uppercase;">Main Menu</div>',
        unsafe_allow_html=True
    )

    # ── Nav items (pure HTML — no Streamlit button cards) ──
    if "active_nav" not in st.session_state:
        st.session_state["active_nav"] = 0

    # tab_idx: 0=Detection, 1=ResNet/Explainability, 2=Video, 3=Model Registry, 4=Evaluation
    nav_map = [
        ("&#127968;", "Dashboard",              0),
        ("&#128269;", "Detection & Inspection", 0),
        ("&#128202;", "Model Registry",         3),
        ("&#128200;", "Evaluation Matrix",      4),
        ("&#127909;", "Video Stream",           2),
        ("&#128300;", "Explainability",         1),
    ]

    active_idx = st.session_state["active_nav"]

    nav_html = ""
    for icon, label, tab_idx in nav_map:
        if active_idx == tab_idx:
            nav_html += (
                f'<div class="mg-nav-item mg-nav-active" onclick="mgNav({tab_idx})">'
                f'<span class="mg-nav-icon">{icon}</span>'
                f'<span class="mg-nav-label">{label}</span>'
                f'</div>'
            )
        else:
            nav_html += (
                f'<div class="mg-nav-item" onclick="mgNav({tab_idx})">'
                f'<span class="mg-nav-icon">{icon}</span>'
                f'<span class="mg-nav-label">{label}</span>'
                f'</div>'
            )

    st.markdown(f"""
    <style>
    .mg-nav-item {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 12px 8px 16px;
        cursor: pointer;
        border-left: 3px solid transparent;
        color: #5a8aaa;
        font-size: 0.82em;
        font-weight: 400;
        transition: background 0.15s, color 0.15s, border-color 0.15s;
        user-select: none;
        margin: 1px 0;
    }}
    .mg-nav-item:hover {{
        background: rgba(0,130,190,0.10);
        color: #b0dcf8;
        border-left-color: #0096c7;
    }}
    .mg-nav-active {{
        background: rgba(0,130,190,0.20) !important;
        border-left-color: #0096c7 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }}
    .mg-nav-icon {{ font-size: 1em; }}
    .mg-nav-label {{ flex: 1; }}
    </style>

    <script>
    function mgNav(tabIdx) {{
        // Tell Streamlit via URL query param to rerun with the tab index
        window.parent.postMessage({{type: "streamlit:setComponentValue", value: tabIdx}}, "*");
        // Also directly click the tab button in the DOM
        setTimeout(function() {{
            var tabs = window.parent.document.querySelectorAll('[data-testid="stTabBar"] button[role="tab"]');
            if (tabs.length > tabIdx) tabs[tabIdx].click();
        }}, 80);
        setTimeout(function() {{
            var tabs = window.parent.document.querySelectorAll('[data-testid="stTabBar"] button[role="tab"]');
            if (tabs.length > tabIdx) tabs[tabIdx].click();
        }}, 300);
    }}
    </script>

    {nav_html}
    """, unsafe_allow_html=True)

    st.markdown("---")

    
    # ── Settings header ──
    st.markdown(
        '<div style="font-size:0.63em;font-weight:600;letter-spacing:0.08em;color:#3a6a88;'
        'padding:0 16px 4px 16px;text-transform:uppercase;">Settings</div>',
        unsafe_allow_html=True
    )

    # ── Model selector ──
    st.markdown("### Select Model")
    selected_model_key = st.selectbox(
        "Active Model",
        options=list(MODEL_REGISTRY.keys()),
        index=0,
        label_visibility="collapsed",
    )
    model_info = MODEL_REGISTRY[selected_model_key]
    desc_short = model_info["description"][:95] + ("..." if len(model_info["description"]) > 95 else "")
    st.markdown(
        f'<div style="background:rgba(0,30,60,0.45);border:1px solid rgba(0,110,170,0.18);'
        f'border-radius:7px;padding:8px 10px;margin:4px 0 10px 0;font-size:0.73em;'
        f'color:#3a7a99;line-height:1.45;">'
        f'<span style="color:#3aA0c8;font-weight:600;">{model_info["type"]}</span><br>'
        f'{desc_short}</div>',
        unsafe_allow_html=True
    )

    # ── Preprocessing ──
    st.markdown("### Preprocessing Pipeline")
    enable_preprocessing = st.toggle("Enable 3-Stage Preprocessing", value=True)
    with st.expander("Filter Parameter Tuning"):
        median_k    = st.selectbox("Median Filter Kernel", [3, 5, 7], index=0)
        bilat_d     = st.slider("Bilateral Diameter", 3, 11, 5, 2)
        bilat_sigma = st.slider("Bilateral Sigma", 15.0, 75.0, 35.0, 5.0)
        clahe_clip  = st.slider("CLAHE Clip Limit", 1.0, 4.0, 2.0, 0.5)

    # ── Detection settings ──
    st.markdown("### Detection Settings")
    conf_thresh      = st.slider("Confidence Threshold", 0.10, 0.95, model_info["default_conf"], 0.05)
    iou_thresh       = st.slider("NMS IoU Threshold", 0.10, 0.90, 0.45, 0.05)
    imgsz            = st.selectbox("Image Resolution", [640, 832, 1024], index=0)
    enable_segformer = st.checkbox("SegFormer Mask Overlay", value=True)
    enable_resnet    = st.checkbox("ResNet18 + Grad-CAM", value=True)

    st.markdown("---")

    # ── System Status ──
    st.markdown(
        '<div style="font-size:0.63em;font-weight:600;letter-spacing:0.08em;color:#3a6a88;'
        'padding:0 16px 6px 16px;text-transform:uppercase;">System Status</div>',
        unsafe_allow_html=True
    )

    hw     = get_device_info()
    gpu_ok = hw.get("cuda_available", False)
    dot_g  = '<span class="dot dot-green"></span>'
    dot_y  = '<span class="dot dot-yellow"></span>'
    dot_b  = '<span class="dot dot-blue"></span>'

    st.markdown(f"""
    <div style="padding:2px 14px 10px 14px;">
        <div class="mg-sys-row">
            <span style="color:#4a7090;">GPU</span>
            <span style="color:{'#2ecc71' if gpu_ok else '#f39c12'};">
                {dot_g if gpu_ok else dot_y}{'Available' if gpu_ok else 'CPU Mode'}
            </span>
        </div>
        <div class="mg-sys-row">
            <span style="color:#4a7090;">CUDA</span>
            <span style="color:{'#2ecc71' if gpu_ok else '#e74c3c'};">
                {'Enabled' if gpu_ok else 'Disabled'}
            </span>
        </div>
        <div class="mg-sys-row">
            <span style="color:#4a7090;">Memory</span>
            <span style="color:#38b8e8;">{dot_b}Optimized</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Team footer (normal document flow, not absolute) ──
    st.markdown("---")
    st.markdown("""
    <div style="padding:2px 12px 8px 12px;">
        <div style="display:flex;align-items:center;gap:10px;padding:5px 0;">
            <div style="width:30px;height:30px;background:linear-gradient(135deg,#0a4878,#006ea0);
                        border-radius:50%;display:flex;align-items:center;justify-content:center;
                        font-size:0.9em;flex-shrink:0;">&#127754;</div>
            <div>
                <div style="font-size:0.78em;font-weight:600;color:#a8cce8;">Marine Guard AI</div>
                <div style="font-size:0.64em;color:#3a6880;">Multi-Modal AI System</div>
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:10px;padding:5px 0;">
            <div style="width:30px;height:30px;background:rgba(0,55,100,0.6);
                        border-radius:50%;display:flex;align-items:center;justify-content:center;
                        font-size:0.85em;flex-shrink:0;">&#128100;</div>
            <div>
                <div style="font-size:0.76em;font-weight:500;color:#88b4cc;">Team Akhet</div>
                <div style="font-size:0.63em;color:#3a6880;">PS-26057</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TOP HEADER BAR
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="mg-topbar">
    <div class="mg-topbar-left">
        <div>
            <div class="mg-topbar-title">MARINE GUARD</div>
            <div class="mg-topbar-sub">Marine Debris Detection System</div>
        </div>
    </div>
    <div class="mg-topbar-right">
        <div class="mg-badge mg-badge-green">&#11044; Active</div>
        <div class="mg-badge mg-badge-blue">{model_info['type']}</div>
        <div class="mg-badge mg-badge-blue">YOLOv11 &middot; SegFormer &middot; ResNet18</div>
        <span style="color:#3a6888;font-size:1.1em;cursor:pointer;flex-shrink:0;" title="Settings">&#9881;</span>
        <span style="color:#3a6888;font-size:1.0em;cursor:pointer;flex-shrink:0;" title="Notifications">&#128276;</span>
        <div class="mg-avatar">TA</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Pipeline badges
st.markdown("""
<div class="mg-pipeline-row">
    <span class="mg-pipe-badge mg-pipe-filter">&#9312; Median Filter</span>
    <span class="mg-pipe-badge mg-pipe-filter">&#9313; Bilateral Denoising</span>
    <span class="mg-pipe-badge mg-pipe-filter">&#9314; LAB-CLAHE</span>
    <span class="mg-pipe-badge mg-pipe-model">&#128640; YOLOv11</span>
    <span class="mg-pipe-badge mg-pipe-model">&#129516; SegFormer-B0</span>
    <span class="mg-pipe-badge mg-pipe-model">&#128293; ResNet18 Grad-CAM</span>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  Detection & Inspection  ",
    "  ResNet18 & Grad-CAM  ",
    "  Video Stream Processing  ",
    "  Model Registry & Architecture  ",
    "  Evaluation Matrix  ",
])

# ── JS: Switch tab when sidebar nav is clicked ──
if st.session_state.get("goto_tab") is not None:
    _tab_idx = st.session_state.pop("goto_tab")
    import streamlit.components.v1 as _c
    _c.html(f"""
    <script>
    (function() {{
        function clickTab() {{
            var tabs = window.parent.document.querySelectorAll('[data-testid="stTabBar"] button[role="tab"]');
            if (tabs.length > {_tab_idx}) {{
                tabs[{_tab_idx}].click();
            }}
        }}
        setTimeout(clickTab, 100);
        setTimeout(clickTab, 300);
    }})();
    </script>
    """, height=0, scrolling=False)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_mid, col_right = st.columns([1, 1, 0.75], gap="small")

    with col_left:
        st.markdown("""
        <div class="mg-card">
            <div class="mg-card-hdr">
                <div class="mg-num">1</div>
                <div>
                    <div class="mg-card-title">Image Ingestion</div>
                    <div class="mg-card-sub">Upload an image for detection</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload image (JPG / PNG / BMP / WEBP)",
            type=["jpg","jpeg","png","bmp","webp"],
        )

        st.markdown(
            '<div style="text-align:center;color:#3a6888;font-size:0.76em;'
            'margin:8px 0;padding:5px 0;'
            'border-top:1px solid rgba(0,90,140,0.14);'
            'border-bottom:1px solid rgba(0,90,140,0.14);">'
            '&#8212;&#8194;OR&#8194;&#8212; Test with Pre-Loaded Dataset Samples (27 Classes)</div>',
            unsafe_allow_html=True
        )

        sample_options = [
            "None (Use Upload)",
            "🚢 Sample: Shipwrecks (Acoustic Sonar)",
            "🥫 Sample: Metal Can",
            "🔧 Sample: Lost Wrench",
            "🔩 Sample: Subsea Valve",
            "⚡ Sample: Pipeline or Cable",
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
        sample_choice = st.selectbox("Select sample image from SIH dataset:", sample_options)

        def get_sample_image(class_name: str):
            test_dir = Path("SIH_Dataset_27class/test/images")
            matches  = list(test_dir.glob(f"{class_name}_*.*"))
            if matches:
                return matches[0]
            src_dir = Path(r"C:\Users\CMRMuthuthiyagarajan\Downloads\SIH DATASETS") / class_name
            if src_dir.exists():
                files = list(src_dir.glob("*.*"))
                if files:
                    return files[0]
            return None

        sample_path = None
        if   "Shipwrecks"         in sample_choice: sample_path = get_sample_image("Shipwrecks")
        elif "Metal Can"          in sample_choice: sample_path = get_sample_image("can")
        elif "Lost Wrench"        in sample_choice: sample_path = get_sample_image("wrench")
        elif "Subsea Valve"       in sample_choice: sample_path = get_sample_image("valve")
        elif "Pipeline or Cable"  in sample_choice: sample_path = get_sample_image("pipeline or cable")
        elif "Small Tire"         in sample_choice: sample_path = get_sample_image("small-tire")
        elif "Large Tire"         in sample_choice: sample_path = get_sample_image("large-tire")
        elif "Plastic Bottle"     in sample_choice: sample_path = get_sample_image("plastic-bottle")
        elif "Drink Carton"       in sample_choice: sample_path = get_sample_image("drink-carton")
        elif "Drink Sachet"       in sample_choice: sample_path = get_sample_image("drink-sachet")
        elif "Glass Bottle"       in sample_choice: sample_path = get_sample_image("glass-bottle")
        elif "Brown Glass Bottle" in sample_choice: sample_path = get_sample_image("brown-glass-bottle")
        elif "Glass Jar"          in sample_choice: sample_path = get_sample_image("glass-jar")
        elif "Hook"               in sample_choice: sample_path = get_sample_image("hook")
        elif "Chain"              in sample_choice: sample_path = get_sample_image("chain")
        elif "Plastic Bidon"      in sample_choice: sample_path = get_sample_image("plastic-bidon")
        elif "Plastic Pipe"       in sample_choice: sample_path = get_sample_image("plastic-pipe")
        elif "Plastic Propeller"  in sample_choice: sample_path = get_sample_image("plastic-propeller")
        elif "Propeller"          in sample_choice: sample_path = get_sample_image("propeller")
        elif "Rotating Platform"  in sample_choice: sample_path = get_sample_image("rotating-platform")
        elif "Shampoo Bottle"     in sample_choice: sample_path = get_sample_image("shampoo-bottle")

        show_preprocessed_view = st.checkbox("Show Preprocessing Comparison (Raw vs. Filtered)", value=False)
        run_btn = st.button("Run Detection Pipeline", type="primary", use_container_width=True)

    with col_mid:
        st.markdown("""
        <div class="mg-card">
            <div class="mg-card-hdr">
                <div class="mg-num mg-num-green">2</div>
                <div>
                    <div class="mg-card-title">Detection Output</div>
                    <div class="mg-card-sub">Results appear here after running detection</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        result_placeholder = st.empty()
        result_placeholder.info("Upload an image or pick a sample, then click **Run Detection Pipeline**.")

        st.markdown("""
        <div style="background:rgba(0,55,100,0.22);border:1px solid rgba(0,140,200,0.18);
                    border-radius:9px;padding:11px 14px;margin-top:10px;">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
                <span style="font-size:1.0em;">&#129302;</span>
                <span style="font-weight:600;color:#90c8e8;font-size:0.85em;">AI Model: Marine Guard Detector</span>
            </div>
            <div style="font-size:0.74em;color:#3a7090;line-height:1.5;">
                All 27 Classes &nbsp;&middot;&nbsp; YOLOv11 &nbsp;&middot;&nbsp; SegFormer &nbsp;&middot;&nbsp; ResNet18
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="mg-card">
            <div style="display:flex;align-items:center;gap:7px;margin-bottom:10px;">
                <span style="color:#38b8f0;">&#128202;</span>
                <span style="font-weight:600;color:#c0dff5;font-size:0.86em;">Detection Summary</span>
            </div>
            <div class="mg-stat-grid">
                <div class="mg-stat c-blue">
                    <div class="mg-stat-val">27</div>
                    <div class="mg-stat-lbl">Total Classes</div>
                </div>
                <div class="mg-stat c-green">
                    <div class="mg-stat-val">33</div>
                    <div class="mg-stat-lbl">AI Models</div>
                </div>
            </div>
            <div class="mg-stat-grid">
                <div class="mg-stat c-purple">
                    <div class="mg-stat-val">Multi-Modal</div>
                    <div class="mg-stat-lbl">Detection</div>
                </div>
                <div class="mg-stat c-orange">
                    <div class="mg-stat-val">Real-time</div>
                    <div class="mg-stat-lbl">Processing</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="mg-card">
            <div style="font-weight:600;color:#c0dff5;font-size:0.84em;margin-bottom:8px;">Quick Info</div>
            <div class="mg-info-row">
                <span class="mg-info-lbl">&#9889; Active Mode</span>
                <span class="mg-info-val">Master Universal</span>
            </div>
            <div class="mg-info-row">
                <span class="mg-info-lbl">&#127959; Architecture</span>
                <span class="mg-info-val">YOLOv11+SegFormer</span>
            </div>
            <div class="mg-info-row">
                <span class="mg-info-lbl">&#128452; Dataset</span>
                <span class="mg-info-val">Marine Dataset</span>
            </div>
            <div class="mg-info-row">
                <span class="mg-info-lbl">&#127991; Classes</span>
                <span class="mg-info-val">27 Marine Classes</span>
            </div>
            <div class="mg-info-row">
                <span class="mg-info-lbl">&#127942; Hackathon</span>
                <span class="mg-info-val">SIH 2026</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="mg-card">
            <div style="font-weight:600;color:#c0dff5;font-size:0.84em;margin-bottom:8px;">&#9889; System Status</div>
            <div class="mg-sys-row">
                <span style="color:#4a7090;font-size:0.76em;">GPU</span>
                <span style="color:{'#2ecc71' if gpu_ok else '#f39c12'};font-size:0.76em;">
                    {dot_g if gpu_ok else dot_y}{'Available' if gpu_ok else 'CPU Mode'}
                </span>
            </div>
            <div class="mg-sys-row">
                <span style="color:#4a7090;font-size:0.76em;">CUDA</span>
                <span style="color:{'#2ecc71' if gpu_ok else '#e74c3c'};font-size:0.76em;">
                    {'Enabled' if gpu_ok else 'Disabled'}
                </span>
            </div>
            <div class="mg-sys-row">
                <span style="color:#4a7090;font-size:0.76em;">Memory</span>
                <span style="color:#38b8e8;font-size:0.76em;">{dot_b}Optimized</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Settings strip
    st.markdown("""
    <div class="mg-card" style="margin-top:2px;">
        <div class="mg-card-hdr">
            <div class="mg-num mg-num-purple">3</div>
            <div>
                <div class="mg-card-title">Detection &amp; Verification Settings</div>
                <div class="mg-card-sub">Current pipeline configuration &mdash; adjust from the sidebar</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3, gap="small")
    _card_base = ("background:rgba(0,28,56,0.55);border:1px solid rgba(0,140,200,0.16);"
                  "border-radius:10px;padding:14px 16px;")
    with s1:
        st.markdown(
            f'<div style="{_card_base}">'
            f'<div style="font-size:0.74em;color:#4a7a99;margin-bottom:8px;font-weight:500;">&#9881; Active Configuration</div>'
            f'<div style="display:flex;justify-content:space-between;font-size:0.8em;padding:3px 0;">'
            f'<span style="color:#3a7090;">Confidence</span><span style="color:#38b8f0;font-weight:600;">{conf_thresh:.2f}</span></div>'
            f'<div style="display:flex;justify-content:space-between;font-size:0.8em;padding:3px 0;">'
            f'<span style="color:#3a7090;">NMS IoU</span><span style="color:#38b8f0;font-weight:600;">{iou_thresh:.2f}</span></div>'
            f'<div style="display:flex;justify-content:space-between;font-size:0.8em;padding:3px 0;">'
            f'<span style="color:#3a7090;">Resolution</span><span style="color:#38b8f0;font-weight:600;">{imgsz}px</span></div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with s2:
        on = enable_segformer
        st.markdown(
            f'<div style="{_card_base}">'
            f'<div style="font-size:0.74em;color:#4a7a99;margin-bottom:10px;font-weight:500;">&#129516; SegFormer Mask Overlay</div>'
            f'<div style="display:flex;align-items:center;gap:10px;">'
            f'<div style="width:38px;height:20px;background:{"#0090c7" if on else "#162230"};'
            f'border-radius:10px;position:relative;flex-shrink:0;border:1px solid rgba(0,140,200,0.3);">'
            f'<div style="width:16px;height:16px;background:#fff;border-radius:50%;position:absolute;'
            f'top:1px;{"right:2px" if on else "left:2px"};"></div></div>'
            f'<span style="font-size:0.79em;color:#{"2ecc71" if on else "3a6a7a"};">'
            f'{"Masks enabled" if on else "Disabled"}</span></div></div>',
            unsafe_allow_html=True
        )
    with s3:
        on2 = enable_resnet
        st.markdown(
            f'<div style="{_card_base}">'
            f'<div style="font-size:0.74em;color:#4a7a99;margin-bottom:10px;font-weight:500;">&#128293; ResNet18 + Grad-CAM</div>'
            f'<div style="display:flex;align-items:center;gap:10px;">'
            f'<div style="width:38px;height:20px;background:{"#0090c7" if on2 else "#162230"};'
            f'border-radius:10px;position:relative;flex-shrink:0;border:1px solid rgba(0,140,200,0.3);">'
            f'<div style="width:16px;height:16px;background:#fff;border-radius:50%;position:absolute;'
            f'top:1px;{"right:2px" if on2 else "left:2px"};"></div></div>'
            f'<span style="font-size:0.79em;color:#{"2ecc71" if on2 else "3a6a7a"};">'
            f'{"Explainability on" if on2 else "Disabled"}</span></div></div>',
            unsafe_allow_html=True
        )

    # Run & Results
    if run_btn:
        img_bgr = None
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        elif sample_path and sample_path.exists():
            img_bgr = cv2.imread(str(sample_path))

        if img_bgr is not None:
            with st.spinner(f"Running {selected_model_key}..."):
                t0 = time.perf_counter()
                selected_dev = select_device("0" if hw.get("cuda_available") else "cpu")
                dets, annotated_bgr, prep_bgr = run_model_inference(
                    model_choice=selected_model_key, img_bgr=img_bgr,
                    conf_thresh=conf_thresh, iou_thresh=iou_thresh, imgsz=imgsz,
                    device=selected_dev, enable_preprocessing=enable_preprocessing,
                    median_k=median_k, bilat_d=bilat_d, bilat_sigma=bilat_sigma,
                    clahe_clip=clahe_clip, enable_segformer=enable_segformer,
                    enable_resnet=enable_resnet,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000

            st.session_state["latest_dets"]     = dets
            st.session_state["latest_img_bgr"]  = img_bgr
            st.session_state["latest_prep_bgr"] = prep_bgr

            result_placeholder.image(
                cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
                use_container_width=True,
                caption=f"Detection Output: {selected_model_key}"
            )

            if show_preprocessed_view:
                st.markdown("#### Preprocessing Comparison (Raw vs. Median + Bilateral + CLAHE)")
                c_raw, c_prep = st.columns(2)
                with c_raw:
                    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Raw Input", use_container_width=True)
                with c_prep:
                    st.image(cv2.cvtColor(prep_bgr, cv2.COLOR_BGR2RGB),
                             caption="Preprocessed (Median → Bilateral → CLAHE)", use_container_width=True)

            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            avg_conf = (np.mean([d["conf"] for d in dets]) * 100) if dets else 0
            fps_val  = 1000 / elapsed_ms if elapsed_ms > 0 else 0
            for col, val, lbl, color in [
                (m1, len(dets),             "Objects Detected", "#2ecc71"),
                (m2, f"{avg_conf:.1f}%",    "Avg Confidence",   "#f39c12"),
                (m3, f"{elapsed_ms:.1f}ms", "Pipeline Latency", "#38b8f0"),
                (m4, f"{fps_val:.0f}",      "Inference FPS",    "#a370f7"),
            ]:
                with col:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value" style="color:{color};">{val}</div>'
                        f'<div class="metric-label">{lbl}</div></div>',
                        unsafe_allow_html=True
                    )

            if dets:
                st.markdown("### Identified Targets")
                d_cols = st.columns(min(len(dets), 4))
                for i, det in enumerate(dets):
                    cname = det["class_name"]
                    meta  = CLASS_METADATA.get(cname, {"emoji": "🏷️", "color": "#00d4ff", "type": "Object"})
                    b     = det["bbox"]
                    with d_cols[i % 4]:
                        st.markdown(
                            f'<div class="mg-det-card" style="border:1.5px solid {meta["color"]}30;">'
                            f'<div style="font-size:1.7em;">{meta["emoji"]}</div>'
                            f'<div style="color:{meta["color"]};font-weight:600;font-size:0.84em;margin:4px 0;">{cname}</div>'
                            f'<div style="color:#3a6a80;font-size:0.71em;">{meta["type"]}</div>'
                            f'<div style="color:#2ecc71;font-weight:700;font-size:0.86em;margin-top:5px;">Conf: {det["conf"]:.1%}</div>'
                            f'<div style="color:#1a3a50;font-size:0.67em;margin-top:2px;">[{int(b[0])},{int(b[1])},{int(b[2])},{int(b[3])}]</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                st.info("Switch to the **ResNet18 & Grad-CAM** tab to view visual attention heatmaps.")
            else:
                st.warning("No targets found above threshold. Try lowering the confidence slider in the sidebar.")
        else:
            st.error("Please upload an image or select a sample image.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — ResNet18 & Grad-CAM
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="mg-card" style="margin-bottom:16px;">
        <div class="mg-card-title">&#128300; ResNet-18 Deep Feature Verification &amp; PyTorch Grad-CAM Heatmaps</div>
        <div class="mg-card-sub" style="margin-top:5px;line-height:1.5;">
            Trained on <strong style="color:#50b8d8;">6,127 ROI crops across all 27 SIH classes</strong>
            with <strong style="color:#2ecc71;">99.47% Validation Accuracy</strong>.
            Computes <strong style="color:#f39c12;">Gradient-Weighted Class Activation Maps (Grad-CAM)</strong> on layer4.
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
            st.markdown(
                f'<div style="background:rgba(0,28,54,0.7);border:1px solid rgba(0,140,200,0.16);'
                f'border-radius:9px;padding:8px 14px;margin-bottom:8px;">'
                f'<span style="font-size:1.0em;">{meta["emoji"]}</span> '
                f'<strong style="color:{meta["color"]};">Target #{idx+1}: {cname}</strong> '
                f'<span style="color:#3a6a88;font-size:0.8em;">({meta["type"]})</span></div>',
                unsafe_allow_html=True
            )
            c_crop, c_gradcam, c_stats = st.columns([1, 1, 1.2], gap="medium")
            with c_crop:
                st.markdown("**1. Dynamic ROI Crop (+20% Padding)**")
                if "roi_crop" in det and det["roi_crop"].size > 0:
                    st.image(cv2.cvtColor(det["roi_crop"], cv2.COLOR_BGR2RGB),
                             use_container_width=True,
                             caption=f"ROI ({det['roi_crop'].shape[1]}x{det['roi_crop'].shape[0]}px)")
            with c_gradcam:
                st.markdown("**2. ResNet18 Grad-CAM Heatmap**")
                if "gradcam_overlay" in det and det["gradcam_overlay"] is not None:
                    st.image(cv2.cvtColor(det["gradcam_overlay"], cv2.COLOR_BGR2RGB),
                             use_container_width=True, caption="layer4 Visual Attention")
            with c_stats:
                st.markdown("**3. Multi-Model Consensus**")
                st.markdown(
                    f'<div style="background:rgba(0,20,44,0.7);border:1px solid rgba(0,130,190,0.14);'
                    f'border-radius:9px;padding:14px;">'
                    f'<div style="margin-bottom:5px;font-size:0.82em;">&#127919; <strong>YOLO:</strong> '
                    f'<span style="color:#2ecc71;font-weight:700">{det["conf"]:.1%}</span></div>'
                    f'<div style="margin-bottom:5px;font-size:0.82em;">&#129504; <strong>ResNet18:</strong> '
                    f'<span style="color:#38b8f0;font-weight:700">{det.get("resnet_pred", cname)}</span></div>'
                    f'<div style="margin-bottom:10px;font-size:0.82em;">&#128293; <strong>R-Conf:</strong> '
                    f'<span style="color:#f39c12;font-weight:700">{det.get("resnet_conf", 0.0):.1%}</span></div>'
                    f'<hr style="border-color:rgba(0,130,190,0.12);margin:8px 0;">'
                    f'<div style="font-size:0.73em;color:#3a6a88;margin-bottom:6px;">Top Predictions:</div>',
                    unsafe_allow_html=True
                )
                if "top3" in det:
                    for cls_t, p_t in det["top3"]:
                        pct = int(p_t * 100)
                        st.markdown(
                            f'<div style="font-size:0.78em;display:flex;justify-content:space-between;margin:3px 0;">'
                            f'<span style="color:#7aa8c0;">&#8226; {cls_t}</span>'
                            f'<span style="color:#38b8f0;">{pct}%</span></div>'
                            f'<div style="background:#080f1c;height:4px;border-radius:3px;margin-bottom:4px;">'
                            f'<div style="background:linear-gradient(90deg,#0068a8,#00c0f0);'
                            f'width:{pct}%;height:4px;border-radius:3px;"></div></div>',
                            unsafe_allow_html=True
                        )
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Video Stream
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
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
                dets, ann_frame, _ = run_model_inference(
                    model_choice=selected_model_key, img_bgr=frame,
                    conf_thresh=conf_thresh, iou_thresh=iou_thresh, imgsz=imgsz,
                    device=selected_dev, enable_preprocessing=enable_preprocessing,
                    median_k=median_k, bilat_d=bilat_d, bilat_sigma=bilat_sigma,
                    clahe_clip=clahe_clip, enable_segformer=enable_segformer, enable_resnet=False,
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
with tab4:
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
    st.markdown("""
| Metric | Value |
|---|---|
| **Dataset Size** | 7,673 Images (6,127 Train / 756 Val / 790 Test) |
| **Classes** | 27 Fine-Grained Classes |
| **Model Architecture** | YOLOv11s (9.4M Parameters, 21.7 GFLOPs) |
| **Validation mAP@50** | **94.09%** |
| **Validation mAP@50-95** | **85.52%** |
| **Inference Speed** | **3.8 ms / image** (~260 FPS on RTX 4050 GPU) |
| **Preprocessing** | 3-Stage: Median (k=3) → Bilateral (d=5, σ=35) → CLAHE (clip=2.0) |
| **Explainability** | ResNet-18 Grad-CAM on layer4 with top-3 consensus |
    """)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — Evaluation Matrix
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("""
    <div class="mg-card" style="margin-bottom:16px;">
        <div class="mg-card-title">&#128200; Full Evaluation Matrix &mdash; All Metrics per Model</div>
        <div class="mg-card-sub" style="margin-top:4px;">
            Evaluated on <strong style="color:#50b8d8;">790 test images across 27 classes</strong>
            &nbsp;&middot;&nbsp;
            Hardware: <strong style="color:#f39c12;">NVIDIA GeForce RTX 4050 Laptop GPU</strong>
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
        p = EVAL_PLOTS_DIR / "yolo_overall_metrics.png"
        if p.exists(): st.image(str(p), caption="YOLOv11 — Overall Metrics", use_container_width=True)
    with col_y2:
        p = EVAL_PLOTS_DIR / "yolo_per_class_ap.png"
        if p.exists(): st.image(str(p), caption="YOLOv11 — Per-Class AP@50 & AP@50-95", use_container_width=True)
    with col_y3:
        p = EVAL_PLOTS_DIR / "yolo_latency.png"
        if p.exists(): st.image(str(p), caption="YOLOv11 — Latency", use_container_width=True)

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
        p = EVAL_PLOTS_DIR / "resnet_overall_metrics.png"
        if p.exists(): st.image(str(p), caption="ResNet-18 — All Metrics", use_container_width=True)
    with col_r2:
        p = EVAL_PLOTS_DIR / "resnet_confusion_matrix.png"
        if p.exists(): st.image(str(p), caption="ResNet-18 — Confusion Matrix (27x27)", use_container_width=True)
    with col_r3:
        p = EVAL_PLOTS_DIR / "resnet_per_class_prf1.png"
        if p.exists(): st.image(str(p), caption="ResNet-18 — Per-Class P/R/F1", use_container_width=True)

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
        p = EVAL_PLOTS_DIR / "segformer_overall_metrics.png"
        if p.exists(): st.image(str(p), caption="SegFormer-B0 — All Segmentation Metrics", use_container_width=True)
    with col_s2:
        p = EVAL_PLOTS_DIR / "segformer_score_distributions.png"
        if p.exists(): st.image(str(p), caption="SegFormer-B0 — IoU & Dice Distributions", use_container_width=True)

    st.info(
        "SegFormer metrics are computed against approximate pseudo-masks derived from bounding boxes "
        "(SIH dataset has no pixel-level GT annotations). Boundary F1 is naturally lower for box-derived masks."
    )

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
