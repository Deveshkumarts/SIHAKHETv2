"""
MarineGuard pipeline endpoints (mounted on the existing FastAPI app in api.py; existing routes are unchanged).

    POST /upload            store a sonar file (XTF/JSF/image)               -> upload_id
    POST /process           run the full pipeline on an upload                -> run summary (+ persisted)
    POST /detect            one-shot: file in -> detections out
    POST /track             LSTM tracking over supplied per-frame detections  -> tracks
    GET  /results/{run_id}  full stored result of a run
    GET  /metrics           model evaluation metrics + versions + latest stage timings
    GET  /export            CSV / JSON / GeoJSON of a run's detections

Also available under /api/v1 (same handlers).
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel, Field

from backend.config import Config, load_config
from backend.database.repository import DetectionRepository
from backend.gis.export import to_csv, to_geojson, to_json
from backend.pipeline.base import file_version
from backend.pipeline.runner import PipelineResult, QCFailed, SonarPipeline
from backend.pipeline.tracking import make_tracker

router = APIRouter(tags=["pipeline"])


class _State:
    cfg: Optional[Config] = None
    pipeline: Optional[SonarPipeline] = None
    repo: Optional[DetectionRepository] = None
    results: Dict[str, Dict[str, Any]] = {}
    latest_run: Optional[str] = None


STATE = _State()


def get_cfg() -> Config:
    if STATE.cfg is None:
        STATE.cfg = load_config()
    return STATE.cfg


def get_pipeline() -> SonarPipeline:
    if STATE.pipeline is None:
        STATE.pipeline = SonarPipeline(get_cfg())
    return STATE.pipeline


def get_repo() -> DetectionRepository:
    if STATE.repo is None:
        STATE.repo = DetectionRepository(get_cfg())
    return STATE.repo


def _upload_dir(cfg: Config) -> Path:
    d = cfg.path("paths.runs_dir").parent / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


_SAFE = re.compile(r"[^A-Za-z0-9._-]")


# ------------------------------------------------------------------ upload
@router.post("/upload")
async def upload(file: UploadFile = File(...), cfg: Config = Depends(get_cfg)):
    name = _SAFE.sub("_", Path(file.filename or "upload.bin").name)
    ext = Path(name).suffix.lower()
    if ext not in set(cfg.ingestion.supported):
        raise HTTPException(415, f"Unsupported file type '{ext}'. Supported: {sorted(cfg.ingestion.supported)}")
    data = await file.read()
    if len(data) > int(cfg.ingestion.max_upload_mb) * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {cfg.ingestion.max_upload_mb} MB")
    if not data:
        raise HTTPException(400, "Empty file")
    upload_id = uuid.uuid4().hex[:12]
    d = _upload_dir(cfg) / upload_id
    d.mkdir(parents=True)
    (d / name).write_bytes(data)
    return {"upload_id": upload_id, "filename": name, "bytes": len(data), "kind": ext.lstrip(".")}


def _find_upload(cfg: Config, upload_id: str) -> Path:
    d = _upload_dir(cfg) / _SAFE.sub("_", upload_id)
    files = list(d.glob("*")) if d.is_dir() else []
    if not files:
        raise HTTPException(404, f"upload_id '{upload_id}' not found")
    return files[0]


# ------------------------------------------------------------------ process / detect
class ProcessRequest(BaseModel):
    upload_id: str
    survey_id: Optional[str] = None
    persist: bool = Field(True, description="write detections to the database")


def _execute(pipe: SonarPipeline, path: Path, survey_id: Optional[str], persist: bool, repo_dep) -> PipelineResult:
    try:
        res = pipe.run(path, filename=path.name, survey_id=survey_id)
    except QCFailed as e:
        raise HTTPException(422, f"Quality control failed: {e}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    STATE.results[res.run_id] = res.to_dict()
    STATE.latest_run = res.run_id
    if persist:
        repo = repo_dep()
        repo.save_run({"survey_id": res.survey_id, "run_id": res.run_id, "source_file": res.source,
                       "scqi": res.scqi.get("overall_score"), "snr_db": res.snr.get("snr_db"),
                       "config_hash": res.config_hash, "model_version": res.model_version},
                      [d for d in res.detections])
    return res


@router.post("/process")
def process(req: ProcessRequest, cfg: Config = Depends(get_cfg), pipe: SonarPipeline = Depends(get_pipeline)):
    path = _find_upload(cfg, req.upload_id)
    res = _execute(pipe, path, req.survey_id, req.persist, get_repo)
    return {**res.summary(), "database": get_repo().backend if req.persist else None, "artifacts": res.artifacts,
            "model_version": res.model_version, "config_hash": res.config_hash}


@router.post("/detect")
async def detect(file: UploadFile = File(...), cfg: Config = Depends(get_cfg),
                 pipe: SonarPipeline = Depends(get_pipeline)):
    """One-shot convenience: file in, detections out (not persisted)."""
    up = await upload(file, cfg)
    res = _execute(pipe, _find_upload(cfg, up["upload_id"]), None, False, get_repo)
    slim = [{k: v for k, v in d.items() if k not in ("mask_polygon_global", "segmentation_mask")} for d in res.detections]
    return {"run_id": res.run_id, "count": len(slim), "detections": slim, "warnings": res.warnings,
            "model_version": res.model_version}


# ------------------------------------------------------------------ track
class TrackDet(BaseModel):
    x: float
    y: float
    w: float = 20.0
    h: float = 20.0
    conf: float = 0.5
    class_name: str = "object"


class TrackRequest(BaseModel):
    frames: List[List[TrackDet]] = Field(..., description="detections per consecutive frame, oldest first")
    dt: float = 1.0


@router.post("/track")
def track(req: TrackRequest, cfg: Config = Depends(get_cfg)):
    if not req.frames:
        raise HTTPException(400, "frames must not be empty")
    tr = make_tracker(cfg)
    per_frame = []
    for i, dets in enumerate(req.frames):
        snap = tr.update([d.model_dump() for d in dets], t=i * req.dt)
        per_frame.append({"frame": i, "tracks": [{k: v for k, v in t.items() if k != "payload"} for t in snap]})
    return {"predictor": tr.predictor_name, "model_version": tr.version, "n_frames": len(req.frames),
            "final_tracks": per_frame[-1]["tracks"], "frames": per_frame}


# ------------------------------------------------------------------ results
def _load_result(run_id: str, cfg: Config) -> Dict[str, Any]:
    if run_id in STATE.results:
        return STATE.results[run_id]
    p = cfg.path("paths.runs_dir") / _SAFE.sub("_", run_id) / "result.json"
    if not p.exists():
        raise HTTPException(404, f"run '{run_id}' not found")
    STATE.results[run_id] = json.loads(p.read_text())
    return STATE.results[run_id]


@router.get("/results/{run_id}")
def results(run_id: str, view: str = Query("full", pattern="^(full|summary|detections)$"),
            cfg: Config = Depends(get_cfg)):
    r = _load_result(run_id, cfg)
    if view == "summary":
        keys = ("run_id", "survey_id", "source", "kind", "config_hash", "model_version", "qc", "snr", "scqi",
                "total_seconds", "warnings", "artifacts", "stages")
        return {k: r[k] for k in keys}
    if view == "detections":
        return {"run_id": r["run_id"], "count": len(r["detections"]), "detections": r["detections"], "tracks": r["tracks"]}
    return r


# ------------------------------------------------------------------ metrics
_METRIC_FILES = {"yolo11_seg": "yolo11_seg_metrics.json", "autoencoder": "autoencoder_metrics.json",
                 "lstm": "lstm_metrics.json", "pipeline_e2e": "pipeline_e2e_metrics.json"}


@router.get("/metrics")
def metrics(cfg: Config = Depends(get_cfg)):
    ev = cfg.path("paths.runs_dir").parent / "evaluation"
    out: Dict[str, Any] = {"models": {}, "versions": {
        "yolo11_seg": file_version(cfg.path("yolo_seg.weights"), "yolo11-seg"),
        "conv_autoencoder": file_version(cfg.path("autoencoder.weights"), "conv-ae"),
        "lstm_tracker": file_version(cfg.path("tracking.weights"), "lstm-tracker")}, "config_hash": cfg.hash}
    for name, fn in _METRIC_FILES.items():
        p = ev / fn
        out["models"][name] = json.loads(p.read_text()) if p.exists() else {"status": "not_evaluated", "expected_file": str(p)}
    if STATE.latest_run and STATE.latest_run in STATE.results:
        r = STATE.results[STATE.latest_run]
        out["latest_run"] = {"run_id": r["run_id"], "total_seconds": r["total_seconds"],
                             "stage_seconds": {s["stage"]: s["seconds"] for s in r["stages"]}}
    return out


# ------------------------------------------------------------------ export
@router.get("/export")
def export(run_id: Optional[str] = None, format: str = Query("geojson", pattern="^(csv|json|geojson)$"),
           detection_type: Optional[str] = Query(None, pattern="^(known|unknown)$"), cfg: Config = Depends(get_cfg)):
    rid = run_id or STATE.latest_run
    if not rid:
        raise HTTPException(404, "no runs available yet - POST /process first")
    recs = _load_result(rid, cfg)["detections"]
    if detection_type:
        recs = [r for r in recs if r["detection_type"] == detection_type]
    body, media, ext = {"csv": (to_csv, "text/csv", "csv"), "json": (to_json, "application/json", "json"),
                        "geojson": (to_geojson, "application/geo+json", "geojson")}[format]
    return Response(content=body(recs), media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="marineguard_{rid}.{ext}"'})
