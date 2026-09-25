"""Tests: GIS analytics/exports, database repository, REST API, and the end-to-end pipeline on the sample XTF."""

import csv
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.database.repository import DetectionRepository
from backend.gis.export import CSV_COLUMNS, to_csv, to_geojson, to_json
from backend.gis.hotspots import kde_hotspots, resurvey_areas, scqi_heatmap
from backend.pipeline.runner import SonarPipeline, make_frames


# ============================== GIS ==============================
def _pts(center, n, spread=0.00005, seed=0):
    rng = np.random.default_rng(seed)
    return [{"latitude": center[0] + rng.normal(0, spread), "longitude": center[1] + rng.normal(0, spread)} for _ in range(n)]


def test_kde_finds_the_dense_cluster_and_ignores_empty_input(cfg):
    pts = _pts((13.0800, 80.2700), 40) + _pts((13.0900, 80.2900), 3, seed=1)
    out = kde_hotspots(pts, cfg)
    assert out["n_points"] == 43 and out["hotspots"]
    top = out["hotspots"][0]
    assert abs(top["latitude"] - 13.0800) < 0.002 and abs(top["longitude"] - 80.2700) < 0.002 and top["n_objects"] >= 20
    assert out["geojson"]["features"][0]["geometry"]["type"] == "Point"
    assert kde_hotspots([], cfg)["hotspots"] == [] and kde_hotspots([{"latitude": None}], cfg)["n_points"] == 0


def test_kde_with_a_single_point_returns_that_point(cfg):
    (h,) = kde_hotspots([{"latitude": 13.0, "longitude": 80.0}], cfg)["hotspots"]
    assert abs(h["latitude"] - 13.0) < 0.001


def test_scqi_heatmap_grids_and_averages(cfg):
    f = [{"latitude": 13.0, "longitude": 80.0, "scqi": 80}, {"latitude": 13.00001, "longitude": 80.00001, "scqi": 60},
         {"latitude": 13.05, "longitude": 80.05, "scqi": 30}]
    out = scqi_heatmap(f, cfg)
    assert len(out["cells"]) == 2 and sorted(c["scqi"] for c in out["cells"]) == [30, 70]
    assert out["geojson"]["features"][0]["geometry"]["type"] == "Polygon"


def test_resurvey_areas_merge_consecutive_low_quality_frames(cfg):
    mk = lambda i, q: {"frame_id": i, "latitude": 13 + i * 1e-4, "longitude": 80.0, "scqi": q, "reasons": ["altitude"] if q < 55 else []}
    out = resurvey_areas([mk(0, 90), mk(1, 40), mk(2, 30), mk(3, 90), mk(4, 20), mk(5, 90)], cfg)
    assert [a["frame_ids"] for a in out["areas"]] == [[1, 2], [4]] and out["areas"][0]["reasons"] == ["altitude"]
    assert resurvey_areas([mk(0, 90), mk(1, 95)], cfg)["areas"] == []


REC = {"survey_id": "s", "run_id": "r", "frame_id": 2, "timestamp": 1.5, "class_name": "tire", "detection_type": "known",
       "confidence": 0.9, "bbox": [1, 2, 3, 4], "mask_area_px": 30, "anomaly_score": 0.1, "reconstruction_error": 0.001,
       "track_id": 4, "latitude": 13.0, "longitude": 80.0, "positional_uncertainty_m": 3.0, "snr_db": 12.0, "scqi": 88.0,
       "model_version": {"yolo11_seg": "yolo11-seg@abc"}, "config_hash": "h"}


def test_exports_contain_every_required_field_and_valid_geojson():
    rows = list(csv.DictReader(io.StringIO(to_csv([REC]))))
    assert list(rows[0]) == CSV_COLUMNS and rows[0]["track_id"] == "4" and rows[0]["x2"] == "3"
    assert "yolo11-seg@abc" in rows[0]["model_version"]
    gj = json.loads(to_geojson([REC, {**REC, "latitude": None}]))
    assert len(gj["features"]) == 1 and gj["features"][0]["geometry"]["coordinates"] == [80.0, 13.0]   # [lon, lat]
    assert json.loads(to_json([REC]))["count"] == 1


# ============================== database ==============================
def _survey(**kw):
    return {"survey_id": "s1", "run_id": "r1", "source_file": "a.xtf", "scqi": 90.0, "snr_db": 14.0, "config_hash": "h",
            "model_version": {"x": "1"}, **kw}


def test_sqlite_roundtrip_preserves_json_fields_and_is_idempotent(cfg):
    repo = DetectionRepository(cfg)
    assert repo.backend == "sqlite" and repo.fallback_reason is None or "DSN" in (repo.fallback_reason or "")
    seg = [[0, 0], [5, 0], [5, 5]]
    recs = [{**REC, "segmentation_mask": seg, "evidence": {"yolo": 0.9}}, {**REC, "frame_id": 3, "detection_type": "unknown"}]
    assert repo.save_run(_survey(), recs) == 2
    repo.save_run(_survey(), recs)                                       # same survey again -> replaced, not duplicated
    got = repo.detections(survey_id="s1")
    assert len(got) == 2 and got[0]["bbox"] == [1, 2, 3, 4] and got[0]["segmentation_mask"] == seg
    assert got[0]["model_version"] == {"yolo11_seg": "yolo11-seg@abc"} and got[0]["config_hash"] == "h"
    assert [d["detection_type"] for d in repo.detections(detection_type="unknown")] == ["unknown"]
    assert repo.surveys()[0]["survey_id"] == "s1"


def test_every_required_detection_field_is_stored(cfg):
    from backend.database.repository import DET_COLUMNS
    need = {"survey_id", "frame_id", "timestamp", "class_name", "detection_type", "confidence", "bbox", "segmentation_mask",
            "mask_area_px", "anomaly_score", "reconstruction_error", "track_id", "latitude", "longitude",
            "positional_uncertainty_m", "snr_db", "scqi", "model_version"}
    assert need <= set(DET_COLUMNS)


def test_strict_postgis_fails_loudly_and_auto_falls_back(cfg, monkeypatch):
    monkeypatch.delenv(cfg.database.postgres_dsn_env, raising=False)
    with pytest.raises(RuntimeError):
        DetectionRepository(cfg, backend="postgis")
    auto = DetectionRepository(cfg, backend="auto")
    assert auto.backend == "sqlite" and "not set" in auto.fallback_reason


def test_postgis_schema_defines_required_columns_and_spatial_index():
    from pathlib import Path
    sql = (Path(__file__).resolve().parents[2] / "backend" / "database" / "schema.sql").read_text()
    sql = " ".join(sql.split())                                      # ignore column-alignment whitespace
    for token in ("CREATE EXTENSION IF NOT EXISTS postgis", "geometry(Point, 4326)", "USING GIST", "segmentation_mask",
                  "reconstruction_error", "positional_uncertainty_m", "model_version JSONB NOT NULL", "track_id"):
        assert token in sql


# ============================== framing ==============================
def test_make_frames_cover_the_waterfall_with_overlap(cfg):
    fr = make_frames(1000, cfg)
    assert fr[0][0] == 0 and fr[-1][1] == 1000 and all(b[0] < a[1] for a, b in zip(fr, fr[1:]))
    assert make_frames(100, cfg) == [(0, 100)]


# ============================== end-to-end ==============================
@pytest.fixture(scope="module")
def e2e(tmp_path_factory, base_cfg):
    tmp = tmp_path_factory.mktemp("e2e")
    from utils.sonar_raw_ingestion import generate_synthetic_xtf
    xtf = tmp / "survey.xtf"
    generate_synthetic_xtf(xtf, num_pings=160, samples_per_channel=400, seed=0)   # same params as scripts/generate_sample_xtf.py
    cfg = base_cfg.override(**{"paths.runs_dir": str(tmp / "runs"), "logging.dir": str(tmp / "logs"),
                               "database.sqlite_path": str(tmp / "db.sqlite"), "database.backend": "sqlite"})
    pipe = SonarPipeline(cfg)
    return {"cfg": cfg, "pipe": pipe, "xtf": xtf, "res": pipe.run(xtf), "tmp": tmp}


def test_e2e_runs_every_stage_in_order_and_logs_time_and_output(e2e):
    stages = [s["stage"] for s in e2e["res"].stages]
    order = ["ingestion", "quality_control", "calibration", "median", "bilateral", "clahe", "snr", "sa_cfar", "yolo11_seg",
             "decision_fusion", "acoustic_signature", "tracking", "geolocation", "scqi_frames", "autoencoder", "detection_summary", "scqi",
             "gis"]
    assert stages == order
    assert all(s["ok"] and s["seconds"] >= 0 for s in e2e["res"].stages) and e2e["res"].total_seconds > 0
    log = (e2e["tmp"] / "logs" / f"{e2e['res'].run_id}.jsonl").read_text().strip().splitlines()
    assert len(log) == len(order) and json.loads(log[0])["stage"] == "ingestion" and "summary" in json.loads(log[0])


def test_e2e_finds_the_injected_target_at_the_right_place(e2e):
    dets = e2e["res"].detections
    assert dets, "the synthetic survey has an injected target (pings 50-70) that must be detected"
    d = max(dets, key=lambda x: x["confidence"])
    assert 45 <= d["bbox_global"][1] and d["bbox_global"][3] <= 76 and d["detection_type"] in ("known", "unknown")
    assert 13.0 < d["latitude"] < 13.1 and 80.2 < d["longitude"] < 80.35 and d["positional_uncertainty_m"] > 0


def test_e2e_every_detection_has_all_required_fields_and_versions(e2e):
    from backend.database.repository import DET_COLUMNS
    res = e2e["res"]
    for d in res.detections:
        assert all(c in d for c in DET_COLUMNS)
        assert d["config_hash"] == res.config_hash and set(d["model_version"]) >= {"yolo11_seg", "conv_autoencoder", "lstm_tracker"}
        assert d["track_id"] is not None and d["scqi"] is not None and d["detection_type"] in ("known", "unknown")


def test_e2e_survey_level_outputs(e2e):
    res = e2e["res"]
    assert res.qc["passed"] and not res.telemetry_synthetic and 0 <= res.scqi["overall_score"] <= 100
    assert res.gis["kde"]["hotspots"] and "scqi_heatmap" in res.gis and "areas" in res.gis["resurvey"]
    assert res.tracks and all(len(t["predicted"]) == e2e["cfg"].tracking.horizon for t in res.tracks)
    for name in ("original", "enhanced", "candidates", "detections", "result"):
        assert (e2e["tmp"] / "runs" / res.run_id).exists() and res.artifacts[name]


def test_e2e_is_reproducible_from_the_config(e2e):
    again = e2e["pipe"].run(e2e["xtf"])
    strip = lambda r: [(d["class_name"], d["bbox"], round(d["confidence"], 6), d["track_id"]) for d in r.detections]
    assert strip(again) == strip(e2e["res"]) and again.config_hash == e2e["res"].config_hash
    other = SonarPipeline(e2e["cfg"].override(**{"fusion.noise_max_fused": 0.9})).run(e2e["xtf"])
    assert other.config_hash != e2e["res"].config_hash


def test_e2e_persists_and_exports(e2e):
    res, cfg = e2e["res"], e2e["cfg"]
    repo = DetectionRepository(cfg)
    repo.save_run({"survey_id": res.survey_id, "run_id": res.run_id, "config_hash": res.config_hash,
                   "model_version": res.model_version, "scqi": res.scqi["overall_score"]}, res.detections)
    stored = repo.detections(survey_id=res.survey_id)
    assert len(stored) == len(res.detections) and stored[0]["model_version"] == res.model_version
    assert len(list(csv.DictReader(io.StringIO(to_csv(res.detections))))) == len(res.detections)


def test_e2e_plain_image_uses_synthetic_telemetry_and_says_so(e2e, sample_image):
    import cv2
    ok, buf = cv2.imencode(".png", sample_image)
    res = e2e["pipe"].run(buf.tobytes(), filename="x.png")
    assert res.telemetry_synthetic and any("SYNTHETIC" in w for w in res.warnings) and res.detections


# ============================== API ==============================
@pytest.fixture(scope="module")
def client(e2e):
    import warnings
    warnings.filterwarnings("ignore")
    import api
    from backend.api import router as R
    R.STATE.cfg, R.STATE.pipeline, R.STATE.repo, R.STATE.results, R.STATE.latest_run = e2e["cfg"], e2e["pipe"], None, {}, None
    return TestClient(api.app)


def test_api_full_workflow_upload_process_results_export_metrics(client, e2e):
    up = client.post("/upload", files={"file": ("survey.xtf", e2e["xtf"].read_bytes())})
    assert up.status_code == 200 and up.json()["kind"] == "xtf"
    pr = client.post("/process", json={"upload_id": up.json()["upload_id"], "survey_id": "api_s"})
    assert pr.status_code == 200
    body = pr.json()
    assert body["database"] == "sqlite" and body["detections"] >= 1 and body["model_version"]["pipeline"]
    rid = body["run_id"]
    assert client.get(f"/results/{rid}?view=summary").json()["config_hash"] == body["config_hash"]
    assert client.get(f"/results/{rid}?view=detections").json()["count"] == body["detections"]
    for fmt, media in (("csv", "text/csv"), ("json", "application/json"), ("geojson", "application/geo+json")):
        r = client.get(f"/export?run_id={rid}&format={fmt}")
        assert r.status_code == 200 and media in r.headers["content-type"] and "attachment" in r.headers["content-disposition"]
    assert client.get("/export?format=csv").status_code == 200            # defaults to the latest run
    m = client.get("/metrics").json()
    assert {"yolo11_seg", "autoencoder", "lstm"} <= set(m["models"]) and m["latest_run"]["run_id"] == rid
    assert client.get(f"/api/v1/results/{rid}?view=summary").status_code == 200        # /api/v1 alias


def test_api_track_and_detect_endpoints(client, sample_image):
    fr = [[{"x": 100 + 5 * i, "y": 100, "w": 30, "h": 30, "conf": 0.8, "class_name": "tire"}] for i in range(8)]
    t = client.post("/track", json={"frames": fr}).json()
    assert t["predictor"] == "lstm" and len(t["final_tracks"]) == 1 and len(t["final_tracks"][0]["predicted"]) == 5
    import cv2
    ok, buf = cv2.imencode(".png", sample_image)
    d = client.post("/detect", files={"file": ("a.png", buf.tobytes())}).json()
    assert d["count"] >= 1 and all("confidence" in x for x in d["detections"])


def test_api_error_handling_and_old_routes_still_work(client):
    assert client.post("/upload", files={"file": ("x.exe", b"abc")}).status_code == 415
    assert client.post("/upload", files={"file": ("x.png", b"")}).status_code == 400
    assert client.post("/process", json={"upload_id": "nope"}).status_code == 404
    assert client.get("/results/does_not_exist").status_code == 404
    assert client.post("/track", json={"frames": []}).status_code == 400
    assert client.post("/upload", files={"file": ("../../evil.png", b"x")}).json()["filename"] == "evil.png"   # no path traversal
    assert client.get("/health").json()["status"] == "healthy"                                              # pre-existing
    assert client.get("/api/v1/detections").status_code == 200 and client.get("/api/v1/telemetry").status_code == 200
    assert client.post("/api/v1/scqi", json={"altitude_m": 10}).status_code == 200
