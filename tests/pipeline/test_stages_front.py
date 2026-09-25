"""Unit tests: config, ingestion, quality control, calibration, preprocessing, SNR, SA-CFAR."""

import cv2
import numpy as np
import pytest

from backend.config import load_config
from backend.pipeline.base import RunLog, file_version
from backend.pipeline.calibration import run_calibration
from backend.pipeline.ingestion import ingest
from backend.pipeline.preprocessing import apply_clahe_luminance, run_preprocessing
from backend.pipeline.quality_control import ping_dropout_ratio, run_quality_control
from backend.pipeline.sa_cfar import merge_candidates, run_sa_cfar
from backend.pipeline.snr import run_snr
from utils.telemetry_parser import generate_synthetic_telemetry


# ---------------- config / logging / versioning ----------------
def test_config_hash_is_stable_and_sensitive(base_cfg):
    assert base_cfg.hash == load_config().hash
    assert base_cfg.override(**{"yolo_seg.conf": 0.99}).hash != base_cfg.hash
    assert base_cfg.get("fusion.w_yolo") == base_cfg.fusion.w_yolo
    assert base_cfg.get("does.not.exist", "dflt") == "dflt"


def test_runlog_records_time_and_summary_and_failure(tmp_path):
    rl = RunLog("t1", tmp_path)
    with rl.stage("ok_stage") as s:
        s["n"] = 3
    with pytest.raises(ValueError):
        with rl.stage("bad_stage"):
            raise ValueError("boom")
    assert [r.stage for r in rl.records] == ["ok_stage", "bad_stage"]
    assert rl.records[0].ok and rl.records[0].summary == {"n": 3} and rl.records[0].seconds >= 0
    assert not rl.records[1].ok and "boom" in rl.records[1].error
    assert (tmp_path / "t1.jsonl").read_text().count("\n") == 2          # every stage logged to disk


def test_file_version_changes_with_content(tmp_path):
    a = tmp_path / "w.pt"
    a.write_bytes(b"one")
    v1 = file_version(a, "m")
    a.write_bytes(b"two")
    assert v1 != file_version(a, "m") and file_version(tmp_path / "missing.pt", "m") == "m@unavailable"


# ---------------- ingestion ----------------
def test_ingest_xtf_gives_real_navigation(cfg, sample_xtf):
    r = ingest(sample_xtf, cfg=cfg)
    assert r.kind == "xtf" and not r.synthetic_telemetry and len(r.telemetry) == 160
    ts = [t.timestamp for t in r.telemetry]
    assert all(b > a for a, b in zip(ts, ts[1:])), "ping times must come from the XTF header and be monotonic"
    assert abs((ts[1] - ts[0]) - 0.25) < 1e-3


def test_ingest_image_flags_synthetic_telemetry(cfg, sample_image):
    ok, buf = cv2.imencode(".png", sample_image)
    r = ingest(buf.tobytes(), filename="x.png", cfg=cfg)
    assert r.kind == "image" and r.synthetic_telemetry and r.image_bgr.shape == sample_image.shape


def test_ingest_rejects_unsupported_and_corrupt(cfg):
    with pytest.raises(ValueError):
        ingest(b"abc", filename="x.exe", cfg=cfg)
    with pytest.raises(ValueError):
        ingest(b"not an image", filename="x.png", cfg=cfg)


# ---------------- quality control ----------------
def test_qc_passes_on_good_data(cfg, sample_image):
    rep = run_quality_control(sample_image, generate_synthetic_telemetry(1), cfg)
    assert rep.passed and all(c.passed for c in rep.checks)


def test_qc_flags_dead_pings_saturation_and_bad_altitude(cfg):
    dead = np.zeros((100, 100, 3), np.uint8)
    dead[:20] = 120
    rep = run_quality_control(dead, generate_synthetic_telemetry(1, altitude_m=80.0), cfg)
    failed = {c.name for c in rep.checks if not c.passed}
    assert not rep.passed and {"zero_ratio", "ping_dropout", "altitude_m"} <= failed
    assert abs(ping_dropout_ratio(cv2.cvtColor(dead, cv2.COLOR_BGR2GRAY)) - 0.8) < 1e-6


def test_qc_detects_gps_teleport(cfg, sample_image):
    tel = generate_synthetic_telemetry(10)
    tel[5].latitude += 1.0
    rep = run_quality_control(sample_image, tel, cfg)
    assert "trajectory_consistency" in {c.name for c in rep.checks if not c.passed}


# ---------------- calibration / preprocessing ----------------
def test_calibration_returns_report_and_valid_image(cfg, sample_image):
    tel = generate_synthetic_telemetry(1)[0]
    out, rep = run_calibration(sample_image, tel, cfg)
    assert out.ndim == 3 and out.dtype == np.uint8 and rep["applied"] and rep["altitude_m"] == tel.altitude_m


def test_calibration_can_be_disabled(cfg, sample_image):
    off = cfg.override(**{"calibration.enable": False})
    out, rep = run_calibration(sample_image, None, off)
    assert rep == {"applied": False} and out is sample_image


def test_preprocessing_runs_three_timed_stages_in_order(cfg, sample_image):
    rl = RunLog("p")
    out = run_preprocessing(sample_image, cfg, rl)
    assert [r.stage for r in rl.records] == ["median", "bilateral", "clahe"]
    assert out.shape == sample_image.shape and out.dtype == np.uint8
    assert not np.array_equal(out, sample_image)


def test_clahe_increases_contrast_on_low_contrast_input():
    rng = np.random.default_rng(1)
    low = np.clip(rng.normal(120, 4, (128, 128)), 0, 255).astype(np.uint8)
    low = cv2.cvtColor(low, cv2.COLOR_GRAY2BGR)
    assert apply_clahe_luminance(low, 3.0).std() > low.std() * 1.5


# ---------------- SNR ----------------
def test_snr_tile_map_matches_shape_and_lookup(cfg, sample_image):
    r = run_snr(sample_image, cfg)
    t = cfg.snr.tile_size
    assert r.tile_snr_db.shape == (-(-256 // t), -(-256 // t))
    assert r.tile_quality_at(0, 0) == pytest.approx(float(r.tile_snr_db[0, 0]))
    assert r.tile_quality_at(9999, 9999) == pytest.approx(float(r.tile_snr_db[-1, -1]))   # clamps, no IndexError
    assert set(r.summary()) >= {"snr_db", "dynamic_range_db"}


# ---------------- SA-CFAR ----------------
def test_sa_cfar_finds_injected_highlight_and_adds_roi(cfg, sample_image):
    _, cands, _ = run_sa_cfar(sample_image, cfg, waterfall=False)
    assert cands, "the injected highlight must produce at least one candidate"
    hit = [c for c in cands if c["bbox"][0] < 130 and c["bbox"][2] > 120 and c["bbox"][1] < 108 and c["bbox"][3] > 100]
    assert hit, "a candidate must overlap the injected object"
    c = hit[0]
    assert c["roi"][0] <= c["bbox"][0] and c["roi"][2] >= c["bbox"][2] and "candidate_id" in c


def test_sa_cfar_nadir_suppression_only_applies_to_waterfalls(cfg, sample_image):
    """The object sits in the central 25% corridor. A waterfall may have nadir water column there (suppressed);
    a plain image has no nadir and the object must still be found."""
    assert run_sa_cfar(sample_image, cfg, waterfall=False)[1], "plain images must not suppress the centre"
    shifted = np.roll(sample_image, 60, axis=1)                    # move the object out of the central corridor
    assert run_sa_cfar(shifted, cfg, waterfall=True)[1], "off-centre objects are found in waterfalls"


def test_sa_cfar_on_empty_frame_returns_no_candidates(cfg):
    flat = np.full((128, 128, 3), 90, np.uint8)
    assert run_sa_cfar(flat, cfg)[1] == []


def _cand(b, area):
    return {"bbox": b, "area": area, "peak_intensity": 200.0, "contrast_ratio": 2.0, "centroid": (0, 0), "regime_name": "r"}


def test_merge_candidates_joins_fragments_and_keeps_distant_ones():
    out = merge_candidates([_cand([0, 0, 10, 10], 50), _cand([0, 14, 10, 24], 30), _cand([100, 100, 110, 110], 40)], 6)
    assert len(out) == 2
    big = max(out, key=lambda c: c["area"])
    assert big["bbox"] == [0, 0, 10, 24] and big["area"] == 80 and big["merged_fragments"] == 2
    assert merge_candidates([_cand([0, 0, 10, 10], 5)], 6)[0]["area"] == 5
