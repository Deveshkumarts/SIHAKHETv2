"""Unit tests: YOLO11-Seg wrapper, autoencoder, decision fusion, LSTM tracking, geolocation, SCQI."""

import numpy as np
import pytest

from backend.pipeline.fusion import KNOWN, NOISE, UNKNOWN, fuse_evidence, run_fusion, snr_evidence
from backend.pipeline.geolocation import geolocate, telemetry_for_row
from backend.pipeline.scqi import run_scqi
from backend.pipeline.tracking import make_tracker, track_objects
from models.autoencoder import metrics as AEM
from models.autoencoder.anomaly import AnomalyScorer, error_to_score
from models.autoencoder.conv_ae import PatchConvAE
from models.lstm.tracker import LSTMTracker
from models.yolo11_seg.inference import YOLOSegDetector, mask_stats
from utils.sonar_calibration import compute_snr_index
from utils.telemetry_parser import generate_synthetic_telemetry


# ======================= YOLO11-Seg =======================
def test_mask_stats_area_and_centroid():
    m = np.zeros((50, 50), np.uint8)
    m[10:20, 30:40] = 1
    s = mask_stats(m)
    assert s["area"] == 100 and s["centroid"] == pytest.approx((34.5, 14.5))
    assert mask_stats(np.zeros((5, 5), np.uint8)) == {"area": 0, "centroid": None}


def test_yolo_without_weights_reports_unavailable_and_returns_nothing(tmp_path, sample_image):
    d = YOLOSegDetector(tmp_path / "missing.pt")
    assert not d.available and d.version == "yolo11-seg@unavailable" and d.detect(sample_image) == []


def test_yolo_with_trained_weights_returns_full_instances(cfg, anoma_image_path):
    w = cfg.path("yolo_seg.weights")
    if not w.exists():
        pytest.skip("YOLO11-Seg weights not trained yet")
    import cv2
    from backend.pipeline.yolo_seg import load_yolo_seg
    det = load_yolo_seg(cfg.override(**{"yolo_seg.conf": 0.05}))
    dets = det.detect(cv2.imread(str(anoma_image_path)))
    for d in dets:                                        # every required output field is present and consistent
        assert len(d["bbox"]) == 4 and 0 <= d["conf"] <= 1 and d["class_name"]
        assert d["mask"].shape == cv2.imread(str(anoma_image_path)).shape[:2]
        assert d["mask_area_px"] == int(d["mask"].sum()) and len(d["centroid"]) == 2


# ======================= Autoencoder =======================
def test_error_to_score_is_half_at_threshold_and_monotonic():
    assert error_to_score(0.01, 0.01) == pytest.approx(0.5)
    xs = [error_to_score(e, 0.01) for e in (0.001, 0.005, 0.01, 0.02, 0.1)]
    assert xs == sorted(xs) and 0 < xs[0] < 0.5 < xs[-1] < 1


def test_autoencoder_bottleneck_and_shapes():
    m = PatchConvAE(patch=64, latent=32)
    x = np.random.default_rng(0).random((4, 1, 64, 64), dtype=np.float32)
    import torch
    z = m.encode(torch.from_numpy(x))
    assert z.shape == (4, 32) and m(torch.from_numpy(x)).shape == (4, 1, 64, 64)


def test_detection_metrics_perfect_and_random():
    perfect = AEM.detection_metrics(np.array([.1, .2, .3]), np.array([2., 3., 4.]), 1.0)
    assert perfect["roc_auc"] == 1.0 and perfect["f1"] == 1.0 and perfect["false_positive_rate"] == 0.0
    assert perfect["false_negative_rate"] == 0.0
    rng = np.random.default_rng(0)
    rand = AEM.detection_metrics(rng.random(500), rng.random(500), 0.5)
    assert 0.4 < rand["roc_auc"] < 0.6


def test_threshold_selection_uses_validation_errors_and_respects_fpr_cap():
    rng = np.random.default_rng(0)
    normal, anom = rng.normal(1.0, 0.2, 400), rng.normal(2.0, 0.4, 100)
    best = AEM.select_threshold(normal, anom, max_fpr=0.05)
    assert best["false_positive_rate"] <= 0.05 and best["f1"] > 0.8 and 1.2 < best["threshold"] < 2.0


def test_trained_scorer_separates_a_bright_blob_from_seabed(cfg):
    if not (cfg.path("autoencoder.weights").exists() and cfg.path("autoencoder.calibration").exists()):
        pytest.skip("autoencoder not trained")
    sc = AnomalyScorer(cfg.path("autoencoder.weights"), cfg.path("autoencoder.calibration"))
    assert sc.available and sc.threshold > 0 and sc.version.startswith("conv-ae@")
    out = sc.score_rois([np.full((40, 40), 100, np.uint8)])
    assert out[0]["available"] and 0 <= out[0]["anomaly_score"] <= 1
    assert AnomalyScorer("nope.pt", "nope.json").score_rois([np.zeros((8, 8), np.uint8)]) == [{"available": False}]


# ======================= Decision fusion =======================
class StubScorer:
    available = True

    def __init__(self, scores):
        self.scores = list(scores)

    def score_rois(self, rois):
        out = []
        for _ in rois:
            s = self.scores.pop(0) if self.scores else 0.0
            out.append({"available": True, "reconstruction_error": 0.01 * s, "reconstruction_mae": 0.01,
                        "anomaly_score": s, "threshold": 0.01, "is_anomaly": s >= 0.5})
        return out


class StubSNR:
    def __init__(self, db):
        self.db = db

    def tile_quality_at(self, x, y):
        return self.db


def _yolo(bbox, conf, name="tire"):
    x1, y1, x2, y2 = bbox
    return {"bbox": list(map(float, bbox)), "class_id": 1, "class_name": name, "conf": conf,
            "mask": np.ones((1, 1), bool), "mask_polygon": [[x1, y1], [x2, y1], [x2, y2]], "mask_area_px": 80,
            "centroid": ((x1 + x2) / 2, (y1 + y2) / 2), "source": "YOLO11-Seg"}


def _cand(bbox, contrast=2.0):
    return {"bbox": list(bbox), "contrast_ratio": contrast, "regime_name": "sand", "area": 80}


@pytest.fixture()
def fcfg(cfg):
    return cfg.override(**{"fusion.w_shadow": 0.0, "fusion.learned_model": ""})   # hand-set rule path, shadow cue off


@pytest.fixture()
def noise_img():
    return np.clip(np.random.default_rng(3).normal(100, 10, (200, 200, 3)), 0, 255).astype(np.uint8)


def test_high_conf_yolo_is_known_even_with_zero_anomaly(noise_img, fcfg):
    (o,) = run_fusion(noise_img, [_yolo([20, 20, 50, 50], 0.9)], [], StubScorer([0.0]), StubSNR(15), fcfg)
    assert o.category == KNOWN and o.class_name == "tire" and o.detection_type == "known"
    assert o.mask_area_px == 80 and o.yolo["conf"] == 0.9


def test_autoencoder_flag_without_yolo_is_unknown_anomaly(noise_img, fcfg):
    (o,) = run_fusion(noise_img, [], [_cand([100, 100, 120, 120])], StubScorer([0.9]), StubSNR(15), fcfg)
    assert o.category == UNKNOWN and o.class_name == "Unknown Anomaly"
    assert any("autoencoder" in r for r in o.reasons)


def test_low_anomaly_score_alone_never_rejects_an_object(noise_img, fcfg):
    """Strong SA-CFAR + good SNR but ~zero anomaly score must NOT be discarded as noise."""
    (o,) = run_fusion(noise_img, [], [_cand([100, 100, 120, 120], contrast=3.5)], StubScorer([0.02]), StubSNR(20), fcfg)
    assert o.category == UNKNOWN and any("did not veto" in r for r in o.reasons)


def test_weak_evidence_everywhere_is_noise(noise_img, fcfg):
    (o,) = run_fusion(noise_img, [], [_cand([100, 100, 120, 120], contrast=1.1)], StubScorer([0.0]), StubSNR(3), fcfg)
    assert o.category == NOISE and o.detection_type == "noise"


def test_low_conf_yolo_plus_anomaly_becomes_unknown_not_known(noise_img, fcfg):
    (o,) = run_fusion(noise_img, [_yolo([20, 20, 50, 50], 0.2)], [], StubScorer([0.95]), StubSNR(15), fcfg)
    assert o.category == UNKNOWN


def test_yolo_and_cfar_on_same_object_are_one_object(noise_img, fcfg):
    objs = run_fusion(noise_img, [_yolo([20, 20, 50, 50], 0.8)], [_cand([22, 22, 48, 48], 2.5)], StubScorer([0.1]),
                      StubSNR(15), fcfg)
    assert len(objs) == 1 and objs[0].cfar is not None and objs[0].evidence["cfar"] is not None


def test_unmatched_cfar_candidate_is_kept_as_separate_object(noise_img, fcfg):
    objs = run_fusion(noise_img, [_yolo([20, 20, 50, 50], 0.8)], [_cand([120, 120, 150, 150], 2.5)], StubScorer([0.1, 0.9]),
                      StubSNR(15), fcfg)
    assert len(objs) == 2


def test_missing_evidence_is_renormalised_not_penalised():
    w = {"cfar": .25, "yolo": .4, "anomaly": .25, "snr": .1, "shadow": 0}
    assert fuse_evidence({"cfar": 0.8, "yolo": None, "anomaly": None, "snr": None}, w) == pytest.approx(0.8)
    assert fuse_evidence({"cfar": None}, w) == 0.0
    assert snr_evidence(None) is None and snr_evidence(100) == 1.0 and snr_evidence(-5) == 0.0


def test_fusion_works_when_autoencoder_is_unavailable(noise_img, fcfg):
    class Off:
        available = False
    (o,) = run_fusion(noise_img, [_yolo([20, 20, 50, 50], 0.7)], [], Off(), StubSNR(10), fcfg)
    assert o.category == KNOWN and o.anomaly is None and o.evidence["anomaly"] is None


def test_shadow_direction_is_nadir_relative_for_waterfalls_and_free_for_plain_images():
    from backend.pipeline.fusion import shadow_any_direction
    from utils.decision_gate import verify_acoustic_shadow
    g = np.full((100, 200), 100, np.uint8)
    g[40:50, 30:40] = 250                                         # object on the PORT side (left of centre)
    g[40:50, 40:60] = 10                                          # shadow on the wrong side for a port object
    img = np.dstack([g] * 3)
    assert verify_acoustic_shadow(img, [30, 40, 40, 50])[0] is False        # waterfall rule: port shadow must fall LEFT
    has, ratio = shadow_any_direction(img, [30, 40, 40, 50])
    assert has and ratio < 0.3                                              # plain image: any side counts


def test_context_crop_is_square_inside_image_and_scales_with_object():
    from backend.pipeline.fusion import context_crop
    img = np.zeros((300, 500, 3), np.uint8)
    assert context_crop(img, [100, 100, 110, 110], 8, 0).shape[:2] == (26, 26)              # legacy tight ROI
    c = context_crop(img, [5, 5, 15, 15], 8, 128)                                            # near the corner: slid, not squeezed
    assert c.shape[:2] == (128, 128)
    assert context_crop(img, [100, 100, 300, 260], 8, 64).shape[0] >= 200 + 16               # big object widens the context
    assert context_crop(img, [10, 10, 20, 20], 8, 1000).shape[:2] == (300, 300)             # capped by the image


# ======================= LSTM tracking =======================
def _mk(cfg, **kw):
    return LSTMTracker(cfg.path("tracking.weights"), seq_len=cfg.tracking.seq_len, horizon=cfg.tracking.horizon,
                       max_age=cfg.tracking.max_age, match_dist_px=cfg.tracking.match_dist_px, min_hits=3, **kw)


def _det(x, y, name="tire"):
    return {"x": float(x), "y": float(y), "w": 30.0, "h": 30.0, "conf": 0.8, "class_name": name}


def test_ids_persist_for_moving_and_static_objects(cfg):
    tr = _mk(cfg)
    ids = {"static": set(), "moving": set()}
    for f in range(15):
        dets = [_det(100, 300), _det(200 + 6 * f, 100 + 9 * f)]
        tr.update(dets, t=float(f))
        ids["static"].add(dets[0]["track_id"])
        ids["moving"].add(dets[1]["track_id"])
    assert len(ids["static"]) == 1 and len(ids["moving"]) == 1 and ids["static"] != ids["moving"]


def test_prediction_shape_and_accuracy_on_straight_motion(cfg):
    tr = _mk(cfg)
    for f in range(12):
        snap = tr.update([_det(50 + 5 * f, 80 + 10 * f)], t=float(f))
    t = snap[0]
    assert len(t["predicted"]) == cfg.tracking.horizon and t["state"] == "confirmed" and len(t["trajectory"]) >= 8
    assert abs(t["predicted"][0][0] - (50 + 5 * 12)) < 6 and abs(t["predicted"][0][1] - (80 + 10 * 12)) < 6
    assert t["velocity"] == pytest.approx([5.0, 10.0], abs=0.6)


def test_track_survives_short_gap_then_dies_after_max_age(cfg):
    tr = _mk(cfg)
    for f in range(6):
        tr.update([_det(100 + 4 * f, 100)], t=float(f))
    tid = tr.tracks[0].id
    tr.update([], t=6.0)
    snap = tr.update([_det(100 + 4 * 7, 100)], t=7.0)                    # reappears after one missed frame
    assert snap[0]["track_id"] == tid
    for f in range(8, 8 + cfg.tracking.max_age + 2):
        snap = tr.update([], t=float(f))
    assert snap == []


def test_far_detection_starts_a_new_track_and_class_is_kept(cfg):
    tr = _mk(cfg)
    for f in range(4):
        tr.update([_det(100, 100)], t=float(f))
    snap = tr.update([_det(100, 100), _det(500, 500, "can")], t=4.0)
    assert len(snap) == 2 and {t["class_name"] for t in snap} == {"tire", "can"}


def test_constant_velocity_fallback_when_weights_missing(tmp_path):
    tr = LSTMTracker(tmp_path / "none.pt")
    assert tr.predictor_name == "constant_velocity" and tr.version == "lstm-tracker@unavailable"
    for f in range(5):
        snap = tr.update([_det(10 + 3 * f, 10)], t=float(f))
    assert snap[0]["predicted"][0][0] == pytest.approx(10 + 3 * 5, abs=0.5)


def test_track_objects_skips_noise_and_assigns_ids(cfg, noise_img, fcfg):
    objs = run_fusion(noise_img, [_yolo([20, 20, 50, 50], 0.9)], [_cand([120, 120, 130, 130], 1.05)], StubScorer([0.0, 0.0]),
                      StubSNR(2), fcfg)
    snap = track_objects(make_tracker(cfg), objs, 0.0, 0.0)
    live = [o for o in objs if o.category != NOISE]
    assert len(snap) == len(live) and all(o.track_id is not None for o in live)
    assert all(getattr(o, "track_id", None) is None for o in objs if o.category == NOISE)


# ======================= Geolocation / SCQI =======================
def test_geolocation_uses_ping_telemetry_and_gives_uncertainty(cfg, noise_img, fcfg):
    tel = generate_synthetic_telemetry(num_pings=50, base_lat=13.0, base_lon=80.0)
    objs = run_fusion(noise_img, [_yolo([20, 20, 60, 60], 0.9), _yolo([120, 150, 160, 190], 0.9)], [], StubScorer([0, 0]),
                      StubSNR(15), fcfg)
    geolocate(objs, tel, (400, 200), frame_offset_y=200.0)
    for o in objs:
        assert 12.9 < o.latitude < 13.1 and 79.9 < o.longitude < 80.1
        assert o.uncertainty_m > 0 and o.ground_range_m >= 0 and o.timestamp is not None
    assert telemetry_for_row(tel, 0, 400) is tel[0] and telemetry_for_row(tel, 399, 400) is tel[-1]
    assert telemetry_for_row(tel[:1], 123, 400) is tel[0]


def test_scqi_drops_for_bad_survey_conditions(cfg, sample_image):
    q = compute_snr_index(sample_image)
    good = run_scqi(sample_image, generate_synthetic_telemetry(1, altitude_m=12.0), q, cfg)
    bad = run_scqi(sample_image, generate_synthetic_telemetry(1, altitude_m=45.0, speed_knots=9.0), q, cfg)
    assert 0 <= bad.overall_score < good.overall_score <= 100 and good.grade
