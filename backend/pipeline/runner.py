"""
MarineGuard end-to-end pipeline orchestrator.

RAW XTF/SSS -> ingestion -> QC -> calibration -> median -> bilateral -> CLAHE -> SNR -> SA-CFAR
  -> [YOLO11-Seg | Conv-AE] -> decision fusion -> LSTM tracking -> geolocation + SCQI -> GIS/KDE -> records

Stages 1-5 run on the whole waterfall (calibration and SNR need the full range); detection stages run
per along-track frame and the tracker links objects across overlapping frames.
Every stage is timed + logged; every detection carries the model versions and config hash.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from backend.config import Config, load_config
from backend.gis.hotspots import kde_hotspots, resurvey_areas, scqi_heatmap
from backend.pipeline import render
from backend.pipeline.anomaly import load_anomaly_scorer
from backend.pipeline.base import RunLog, file_version
from backend.pipeline.calibration import run_calibration
from backend.pipeline.fusion import NOISE, run_fusion
from backend.pipeline.geolocation import geolocate, telemetry_for_row
from backend.pipeline.ingestion import ingest
from backend.pipeline.preprocessing import run_preprocessing
from backend.pipeline.quality_control import run_quality_control
from backend.pipeline.sa_cfar import run_sa_cfar
from backend.pipeline.scqi import run_scqi
from backend.pipeline.signature_verifier import SignatureVerifier, verify_objects
from backend.pipeline.snr import run_snr
from backend.pipeline.tracking import make_tracker, track_objects
from backend.pipeline.yolo_seg import load_yolo_seg
from utils.sonar_calibration import compute_snr_index


class QCFailed(RuntimeError):
    pass


class _OffsetSNR:
    """SNR tile lookup in a frame's local coordinates."""

    def __init__(self, snr, y0: int):
        self.snr, self.y0 = snr, y0

    def tile_quality_at(self, x: float, y: float) -> float:
        return self.snr.tile_quality_at(x, y + self.y0)


class _TimedScorer:
    """Proxy that accumulates the autoencoder's inference time so it is reported as its own stage."""

    def __init__(self, scorer):
        self._s, self.seconds, self.calls, self.rois = scorer, 0.0, 0, 0

    @property
    def available(self):
        return self._s.available

    def score_rois(self, rois):
        t0 = time.perf_counter()
        out = self._s.score_rois(rois)
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        self.rois += len(rois)
        return out


def make_frames(height: int, cfg: Config) -> List[Tuple[int, int]]:
    fr = cfg.framing
    if height <= fr.frame_pings:
        return [(0, height)]
    frames, y0 = [], 0
    while True:
        y1 = min(height, y0 + fr.frame_pings)
        frames.append((y0, y1))
        if y1 >= height:
            break
        y0 += fr.stride_pings
        if height - y0 < fr.min_frame_pings:              # tail too short: extend the last frame instead
            break
    return frames


@dataclass
class PipelineResult:
    run_id: str
    survey_id: str
    source: str
    kind: str
    config_hash: str
    model_version: Dict[str, str]
    qc: Dict[str, Any]
    snr: Dict[str, Any]
    scqi: Dict[str, Any]
    frames: List[Dict[str, Any]]
    detections: List[Dict[str, Any]]
    objects: List[Dict[str, Any]]                 # de-duplicated per track (best observation), used by GIS
    tracks: List[Dict[str, Any]]
    gis: Dict[str, Any]
    stages: List[Dict[str, Any]]
    total_seconds: float
    warnings: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    telemetry_synthetic: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self.__dict__, default=_json_default))

    def summary(self) -> Dict[str, Any]:
        types = [d["detection_type"] for d in self.detections]
        return {"run_id": self.run_id, "survey_id": self.survey_id, "frames": len(self.frames),
                "detections": len(self.detections), "known": types.count("known"), "unknown": types.count("unknown"),
                "tracks": len(self.tracks), "scqi": self.scqi.get("overall_score"), "snr_db": self.snr.get("snr_db"),
                "total_seconds": self.total_seconds, "warnings": self.warnings}


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "__dict__"):
        return {k: v for k, v in o.__dict__.items() if not k.startswith("_")}
    return str(o)


class SonarPipeline:
    def __init__(self, cfg: Optional[Config] = None, yolo=None, scorer=None):
        self.cfg = cfg or load_config()
        self.yolo = yolo if yolo is not None else load_yolo_seg(self.cfg)
        self.scorer = scorer if scorer is not None else load_anomaly_scorer(self.cfg)
        self.verifier = SignatureVerifier(self.cfg.path("signature.model")) if self.cfg.get("signature.enable", True) else SignatureVerifier("__none__")
        self.model_version = {
            "pipeline": self.cfg.pipeline.version,
            "yolo11_seg": self.yolo.version,
            "conv_autoencoder": self.scorer.version,
            "lstm_tracker": file_version(self.cfg.path("tracking.weights"), "lstm-tracker"),
            "acoustic_signature": file_version(self.cfg.path("signature.model"), "acoustic-signature") if self.verifier.available else "acoustic-signature@unavailable",
        }

    def run(self, source: Union[str, Path, bytes], filename: Optional[str] = None, survey_id: Optional[str] = None,
            run_id: Optional[str] = None, save_artifacts: bool = True) -> PipelineResult:
        cfg = self.cfg
        np.random.seed(cfg.pipeline.seed)
        cv2.setRNGSeed(int(cfg.pipeline.seed))              # K-Means clutter segmentation uses OpenCV's own RNG
        try:
            import torch
            torch.manual_seed(int(cfg.pipeline.seed))
        except Exception:
            pass
        run_id = run_id or uuid.uuid4().hex[:12]
        rl = RunLog(run_id, cfg.path("logging.dir"))
        warnings: List[str] = []
        if not self.yolo.available:
            warnings.append("YOLO11-Seg weights not found - known-debris branch is inactive")
        if not self.scorer.available:
            warnings.append("Autoencoder weights/calibration not found - unknown-anomaly branch is inactive")

        # ---------- whole-waterfall stages ----------
        with rl.stage("ingestion") as s:
            ing = ingest(source, filename, cfg)
            s.update(kind=ing.kind, shape=list(ing.image_bgr.shape), pings=len(ing.telemetry),
                     synthetic_telemetry=ing.synthetic_telemetry)
        if ing.synthetic_telemetry:
            warnings.append("Input has no navigation data: positions use SYNTHETIC telemetry (not a real location)")
        img, tel = ing.image_bgr, ing.telemetry
        survey_id = survey_id or f"survey_{run_id}"

        with rl.stage("quality_control") as s:
            qc = run_quality_control(img, tel, cfg)
            s.update(passed=qc.passed, failed=[c.name for c in qc.checks if not c.passed])
        warnings += [f"QC: {c.name}={c.value} (limit {c.limit})" for c in qc.checks if not c.passed]
        if not qc.passed and cfg.quality_control.abort_on_fail:
            raise QCFailed("data quality control failed: " + ", ".join(c.name for c in qc.checks if not c.passed))

        with rl.stage("calibration") as s:
            if ing.kind == "image" and not cfg.get("calibration.apply_to_images", False):
                # TVG / water-column removal / slant-range correction assume a raw side-scan waterfall; on a plain
                # image or cropped object they would delete the centre and warp the geometry.
                cal_img, cal_rep = img, {"applied": False, "reason": "plain image is not a raw waterfall"}
            else:
                cal_img, cal_rep = run_calibration(img, telemetry_for_row(tel, img.shape[0] / 2, img.shape[0]), cfg)
            s.update({k: v for k, v in cal_rep.items() if isinstance(v, (int, float, str, bool))})

        enhanced = run_preprocessing(cal_img, cfg, rl)              # logs median / bilateral / clahe separately

        with rl.stage("snr") as s:
            snr = run_snr(enhanced, cfg)
            s.update(snr.summary())

        # ---------- per-frame detection stages ----------
        frames = make_frames(enhanced.shape[0], cfg)
        tracker = make_tracker(cfg)
        timed = _TimedScorer(self.scorer)
        acc = {"sa_cfar": 0.0, "yolo11_seg": 0.0, "decision_fusion": 0.0, "acoustic_signature": 0.0, "tracking": 0.0, "geolocation": 0.0, "scqi_frames": 0.0}
        counts = {"candidates": 0, "yolo": 0, "known": 0, "unknown": 0, "noise": 0}
        sig_counts = {"CONFIRMED": 0, "UNCERTAIN": 0, "MISMATCH": 0, "N/A": 0}
        records: List[Dict[str, Any]] = []
        cand_global: List[Dict[str, Any]] = []
        frame_rows: List[Dict[str, Any]] = []
        last_snapshot: Dict[int, Dict[str, Any]] = {}

        def tick(name, t0):
            acc[name] += time.perf_counter() - t0

        for fi, (y0, y1) in enumerate(frames):
            frame = enhanced[y0:y1]
            t0 = time.perf_counter(); _, cands, _ = run_sa_cfar(frame, cfg, waterfall=ing.kind in ("xtf", "jsf")); tick("sa_cfar", t0)
            # YOLO can consume the calibrated image (as it was trained: raw crops) or the enhanced one; see yolo_seg.input
            yolo_frame = frame if cfg.get("yolo_seg.input", "enhanced") == "enhanced" else cal_img[y0:y1]
            t0 = time.perf_counter(); yolo_dets = self.yolo.detect(yolo_frame); tick("yolo11_seg", t0)
            t0 = time.perf_counter()
            ae_frame = frame if cfg.get("autoencoder.input", "enhanced") == "enhanced" else cal_img[y0:y1]
            objs = run_fusion(frame, yolo_dets, cands, timed, _OffsetSNR(snr, y0), cfg, waterfall=ing.kind in ("xtf", "jsf"),
                              ae_image=ae_frame)
            tick("decision_fusion", t0)
            for o in objs:
                o.frame_id = fi
            t0 = time.perf_counter()
            c_ = verify_objects(cal_img[y0:y1], objs, self.verifier, waterfall=ing.kind in ("xtf", "jsf"),
                                geometry={"altitude_m": float(telemetry_for_row(tel, (y0 + y1) / 2, enhanced.shape[0]).altitude_m),
                                          "range_per_px_m": float(telemetry_for_row(tel, (y0 + y1) / 2, enhanced.shape[0]).slant_range_m) / max(1, enhanced.shape[1] / 2),
                                          "range_to_object_m": 0.0})
            for k_, v_ in c_.items():
                sig_counts[k_] += v_
            tick("acoustic_signature", t0)
            t0 = time.perf_counter(); snapshot = track_objects(tracker, objs, y0, float(fi)); tick("tracking", t0)
            live = [o for o in objs if o.category != NOISE]
            t0 = time.perf_counter(); geolocate(live, tel, enhanced.shape, y0); tick("geolocation", t0)

            t0 = time.perf_counter()
            q = compute_snr_index(frame)
            fscqi = run_scqi(frame, tel[int(y0 / enhanced.shape[0] * len(tel)):max(1, int(y1 / enhanced.shape[0] * len(tel)))] or tel, q, cfg)
            mid = telemetry_for_row(tel, (y0 + y1) / 2, enhanced.shape[0])
            frame_rows.append({"frame_id": fi, "y0": y0, "y1": y1, "scqi": fscqi.overall_score, "grade": fscqi.grade,
                               "snr_db": float(q.snr_db), "latitude": float(mid.latitude), "longitude": float(mid.longitude),
                               "reasons": list(fscqi.resurvey_reasons), "resurvey": bool(fscqi.resurvey_recommended)})
            tick("scqi_frames", t0)

            counts["candidates"] += len(cands)
            counts["yolo"] += len(yolo_dets)
            for c in cands:
                cg = dict(c); cg["bbox"] = [c["bbox"][0], c["bbox"][1] + y0, c["bbox"][2], c["bbox"][3] + y0]
                cand_global.append(cg)
            for t in snapshot:
                last_snapshot[t["track_id"]] = t
            for o in objs:
                counts[{"known": "known", "unknown": "unknown", "noise": "noise"}[o.detection_type]] += 1
                if o.category == NOISE:
                    continue
                records.append(self._record(o, survey_id, run_id, y0, frame_rows[-1], enhanced.shape[0]))

        for name, secs in acc.items():
            rl.record(name, secs, {"frames": len(frames)})
        rl.record("autoencoder", timed.seconds, {"rois": timed.rois, "available": self.scorer.available})
        rl.record("detection_summary", 0.0, {**counts, **{f"signature_{k.lower().replace('/', '')}": v for k, v in sig_counts.items()}})

        # ---------- survey-level ----------
        with rl.stage("scqi") as s:
            overall = run_scqi(enhanced, tel, snr.metrics, cfg)
            s.update(overall_score=overall.overall_score, grade=overall.grade, resurvey=overall.resurvey_recommended)
        for r in records:
            r["scqi"] = frame_rows[r["frame_id"]]["scqi"]

        objects = self._unique_objects(records)
        with rl.stage("gis") as s:
            gis = {"kde": kde_hotspots(objects, cfg), "scqi_heatmap": scqi_heatmap(frame_rows, cfg),
                   "resurvey": resurvey_areas(frame_rows, cfg)}
            s.update(hotspots=len(gis["kde"]["hotspots"]), resurvey_areas=len(gis["resurvey"]["areas"]))

        tracks = [t for t in last_snapshot.values() if t["state"] == "confirmed" or len(frames) == 1]
        for t in tracks:
            t.pop("payload", None)

        result = PipelineResult(
            run_id=run_id, survey_id=survey_id, source=ing.meta.get("filename", str(filename or source)[:120]), kind=ing.kind,
            config_hash=cfg.hash, model_version=self.model_version, qc=qc.to_dict(),
            snr={**snr.summary(), "warnings": list(snr.metrics.warnings)},
            scqi=overall.to_dict() if hasattr(overall, "to_dict") else {}, frames=frame_rows, detections=records,
            objects=objects, tracks=tracks, gis=gis, stages=rl.to_list(), total_seconds=rl.total_seconds(),
            warnings=warnings, telemetry_synthetic=ing.synthetic_telemetry)
        if save_artifacts:
            result.artifacts = self._save(result, img, enhanced, cand_global, cfg)
        return result

    # ---------------- helpers ----------------
    def _record(self, o, survey_id, run_id, y0, frame_row, full_h) -> Dict[str, Any]:
        an = o.anomaly or {}
        gx = [o.bbox[0], o.bbox[1] + y0, o.bbox[2], o.bbox[3] + y0]
        return {
            "survey_id": survey_id, "run_id": run_id, "frame_id": o.frame_id, "timestamp": o.timestamp,
            "class_name": o.class_name, "detection_type": o.detection_type, "category": o.category,
            "confidence": o.confidence, "bbox": [round(v, 2) for v in o.bbox], "bbox_global": [round(v, 2) for v in gx],
            "segmentation_mask": [[round(a, 1), round(b, 1)] for a, b in o.mask_polygon] or None,
            "mask_polygon_global": [[round(a, 1), round(b + y0, 1)] for a, b in o.mask_polygon],
            "mask_area_px": o.mask_area_px,
            "centroid_px": [round(o.centroid[0], 1), round(o.centroid[1] + y0, 1)],
            "anomaly_score": an.get("anomaly_score"), "reconstruction_error": an.get("reconstruction_error"),
            "track_id": getattr(o, "track_id", None), "latitude": o.latitude, "longitude": o.longitude,
            "positional_uncertainty_m": o.uncertainty_m, "snr_db": o.tile_snr_db, "scqi": frame_row["scqi"],
            "model_version": self.model_version, "config_hash": self.cfg.hash,
            "evidence": {**o.evidence, "acoustic_signature": o.signature},
            "signature_verdict": (o.signature or {}).get("verdict", "N/A"),
            "reasons": o.reasons, "features": o.features, "frame_y0": y0,
        }

    @staticmethod
    def _unique_objects(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        best: Dict[Any, Dict[str, Any]] = {}
        for r in records:
            key = ("t", r["track_id"]) if r.get("track_id") is not None else ("r", id(r))
            if key not in best or r["confidence"] > best[key]["confidence"]:
                best[key] = r
        return list(best.values())

    def _save(self, res: PipelineResult, original, enhanced, cand_global, cfg) -> Dict[str, str]:
        d = cfg.path("paths.runs_dir") / res.run_id
        d.mkdir(parents=True, exist_ok=True)
        paths = {
            "original": d / "original.png", "enhanced": d / "enhanced.png",
            "candidates": d / "sa_cfar_candidates.png", "detections": d / "detections.png",
        }
        cv2.imwrite(str(paths["original"]), original)
        cv2.imwrite(str(paths["enhanced"]), enhanced)
        cv2.imwrite(str(paths["candidates"]), render.draw_candidates(enhanced, cand_global))
        cv2.imwrite(str(paths["detections"]), render.draw_detections(enhanced, res.detections, res.tracks))
        out = {k: str(v) for k, v in paths.items()}
        (d / "result.json").write_text(json.dumps(res.to_dict(), indent=1))
        out["result"] = str(d / "result.json")
        return out
