"""
MarineGuard pipeline dashboard (standalone Streamlit page; the existing Akhet dashboard in app.py is untouched).

    streamlit run frontend/dashboard/app.py --server.port 8502

Tabs: LIVE / ANALYSIS  |  GIS  |  MODEL EVALUATION
Runs the pipeline in-process (SonarPipeline) - no cloud connection needed. The same results are served over
REST by api.py (POST /process, GET /results/{id}, /export) for a React or other web front end.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from backend.config import load_config  # noqa: E402
from backend.database.repository import DetectionRepository  # noqa: E402
from backend.pipeline.runner import QCFailed, SonarPipeline  # noqa: E402
from frontend.gis.views import render_gis  # noqa: E402
from frontend.metrics.views import render_metrics  # noqa: E402
from frontend.sonar import views as sonar  # noqa: E402

st.set_page_config(page_title="MarineGuard Pipeline", layout="wide")
st.markdown("<style>.block-container{padding-top:1.2rem;}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models (YOLO11-Seg, autoencoder, LSTM)...")
def get_pipeline() -> SonarPipeline:
    return SonarPipeline(load_config())


@st.cache_resource
def get_repo() -> DetectionRepository:
    return DetectionRepository(load_config())


pipe = get_pipeline()
st.title("MarineGuard  |  side-scan sonar debris intelligence")
st.caption("XTF/SSS > QC > calibration > median/bilateral/CLAHE > SNR > SA-CFAR > [YOLO11-Seg | Conv-AE] > fusion > "
           "LSTM tracking > geolocation + SCQI > GIS/KDE > PostGIS > export")

with st.sidebar:
    st.header("Input")
    samples = sorted((ROOT / "samples" / "raw_xtf").glob("*.xtf"))
    choice = st.radio("Source", ["Upload file", "Sample XTF survey"] if samples else ["Upload file"])
    up = st.file_uploader("Sonar file", type=["xtf", "jsf", "png", "jpg", "jpeg", "bmp", "webp"]) if choice == "Upload file" else None
    survey_id = st.text_input("Survey ID (optional)")
    persist = st.checkbox("Save detections to database", True)
    go = st.button("Run pipeline", type="primary", use_container_width=True)
    st.divider()
    st.caption("Models")
    for k, v in pipe.model_version.items():
        st.caption(f"{k}: `{v}`")
    repo = get_repo()
    st.caption(f"Database: **{repo.backend}**" + (f" ({repo.fallback_reason})" if repo.fallback_reason else ""))

if go:
    try:
        if choice == "Upload file":
            if up is None:
                st.sidebar.error("Choose a file first.")
                st.stop()
            src = Path(tempfile.mkdtemp()) / Path(up.name).name
            src.write_bytes(up.getvalue())
        else:
            src = samples[0]
        with st.spinner("Running pipeline..."):
            res = pipe.run(src, survey_id=survey_id or None)
        if persist:
            repo.save_run({"survey_id": res.survey_id, "run_id": res.run_id, "source_file": res.source,
                           "scqi": res.scqi.get("overall_score"), "snr_db": res.snr.get("snr_db"),
                           "config_hash": res.config_hash, "model_version": res.model_version}, res.detections)
        st.session_state["result"] = res.to_dict()
    except QCFailed as e:
        st.error(f"Quality control failed: {e}")
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")

tab_live, tab_gis, tab_eval = st.tabs(["LIVE / ANALYSIS", "GIS", "MODEL EVALUATION"])
result = st.session_state.get("result")

with tab_live:
    if not result:
        st.info("Upload a sonar file (or pick the sample survey) and press **Run pipeline**.")
    else:
        c = st.columns(5)
        types = [d["detection_type"] for d in result["detections"]]
        c[0].metric("Known debris", types.count("known"))
        c[1].metric("Unknown anomalies", types.count("unknown"))
        c[2].metric("Tracks", len(result["tracks"]))
        c[3].metric("Frames", len(result["frames"]))
        c[4].metric("Survey", result["survey_id"])
        sonar.render_images(result)
        sonar.render_reliability(result)
        sonar.render_detections(result)
        sonar.render_tracks(result)
        with st.expander("Stage execution log (time + output of every stage)"):
            sonar.render_stage_timings(result)

with tab_gis:
    if not result:
        st.info("Run the pipeline to see the map.")
    else:
        render_gis(result)

with tab_eval:
    render_metrics()
