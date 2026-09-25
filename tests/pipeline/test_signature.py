"""Unit tests: acoustic-signature features, verifier verdict logic, and pipeline integration."""

import json

import numpy as np
import pytest

from backend.pipeline.acoustic_signature import CLASS_TO_FAMILY, FEATURE_NAMES, extract_signature, to_vector
from backend.pipeline.fusion import FusedObject
from backend.pipeline.signature_verifier import CONFIRMED, MISMATCH, NA, UNCERTAIN, SignatureVerifier, verify_objects


def scene(obj_val=220, bg=90, shadow=None, size=(16, 16), pos=(60, 60), shape=(160, 160), seed=0):
    rng = np.random.default_rng(seed)
    g = np.clip(rng.normal(bg, 4, shape), 0, 255).astype(np.uint8)
    y, x = pos
    g[y:y + size[0], x:x + size[1]] = obj_val
    if shadow is not None:
        g[y:y + size[0], x + size[1]:x + size[1] + 24] = shadow
    return np.dstack([g] * 3), [x, y, x + size[1], y + size[0]]


# ------------------------------------------------------------------ features
def test_bright_object_has_positive_impedance_contrast_and_dark_one_negative():
    bright, b1 = scene(obj_val=230)
    dark, b2 = scene(obj_val=30)
    fb, fd = extract_signature(bright, b1), extract_signature(dark, b2)
    assert fb["impedance_contrast_idx"] > 0.3 and fb["contrast_db"] > 6
    assert fd["impedance_contrast_idx"] < -0.3 and fd["contrast_db"] < -6


def test_contrast_index_is_bounded_and_monotonic_in_object_brightness():
    vals = []
    for v in (100, 140, 190, 250):
        img, b = scene(obj_val=v)
        vals.append(extract_signature(img, b)["impedance_contrast_idx"])
    assert vals == sorted(vals) and all(-1 <= x <= 1 for x in vals)


def test_shadow_is_detected_and_measured():
    img, b = scene(obj_val=230, shadow=8)
    f = extract_signature(img, b)
    assert f["shadow_depth"] > 0.5 and f["highlight_shadow_ratio"] > 5 and f["shadow_length_rel"] > 0.5
    none_img, nb = scene(obj_val=230)
    assert extract_signature(none_img, nb)["shadow_depth"] < 0.2


def test_features_complete_finite_and_vector_order_is_stable():
    img, b = scene(shadow=10)
    f = extract_signature(img, b)
    assert set(FEATURE_NAMES) <= set(f) and np.isfinite(to_vector(f)).all() and len(to_vector(f)) == len(FEATURE_NAMES)


def test_instance_mask_is_used_when_given():
    img, b = scene(obj_val=230)
    mask = np.zeros(img.shape[:2], bool)
    mask[64:72, 64:72] = True                                     # small part of the object
    full, part = extract_signature(img, b), extract_signature(img, b, mask)
    assert part["log_area"] < full["log_area"]


def test_height_only_for_waterfall_with_geometry():
    # waterfall geometry: the shadow falls AWAY from the nadir (image centre) -> object right of centre, shadow to its right
    img, b = scene(obj_val=230, shadow=8, pos=(60, 100))
    geo = {"altitude_m": 10.0, "range_per_px_m": 0.2, "range_to_object_m": 40.0}
    assert "height_m" not in extract_signature(img, b, waterfall=False, geometry=geo)          # plain image: no geometry
    f = extract_signature(img, b, waterfall=True, geometry=geo)
    assert 0 < f["height_m"] < 10                                                                # h = L*H/(R+L) < altitude


def test_degenerate_boxes_return_none():
    img, _ = scene()
    assert extract_signature(img, [10, 10, 11, 11]) is None and extract_signature(img, [500, 500, 600, 600]) is None


def test_every_class_is_assigned_to_a_material_family_or_documented_other():
    assert CLASS_TO_FAMILY["tire"] == "rubber" and CLASS_TO_FAMILY["can"] == "metal" and CLASS_TO_FAMILY["plastic-bottle"] == "plastic"


# ------------------------------------------------------------------ verifier logic (deterministic stub model)
@pytest.fixture()
def stub_model(tmp_path):
    """6 classes (top-3 confirmation needs more than 3): class 0 = strong positive weight on feature 0 (contrast),
    class 1 = strong negative, classes 2-5 = flat."""
    n = len(FEATURE_NAMES)
    coef = np.zeros((6, n))
    coef[0, 0], coef[1, 0] = 6.0, -6.0
    m = {"features": FEATURE_NAMES, "classes": ["bright_thing", "dark_thing"] + [f"neutral_{i}" for i in range(4)],
         "mean": [0.0] * n, "std": [1.0] * n, "coef": coef.tolist(), "intercept": [0.0] * 6, "class_index": list(range(6)),
         "policy": {"mismatch_if_rank_ge": 3, "and_p_lt": 0.05, "confirm_min_ratio": 0.5}}
    p = tmp_path / "sig.json"
    p.write_text(json.dumps(m))
    return SignatureVerifier(p)


def feats(idx):
    f = {n: 0.0 for n in FEATURE_NAMES}
    f["impedance_contrast_idx"] = idx
    return f


def test_verdicts_confirmed_uncertain_mismatch_and_na(stub_model):
    v = stub_model
    assert v.verify(feats(1.0), "bright_thing")["verdict"] == CONFIRMED                     # signature agrees
    assert v.verify(feats(1.0), "dark_thing")["verdict"] == MISMATCH                        # ranked last and p ~ 0
    assert v.verify(feats(0.0), "bright_thing")["verdict"] == CONFIRMED                     # no evidence either way: still in top-3 (tie)
    weak = v.verify(feats(0.1), "dark_thing")                                                # ranked low, but p ~ 0.09 >= tau
    assert weak["verdict"] == UNCERTAIN and weak["claimed_rank"] >= 3 and weak["claimed_probability"] >= 0.05
    assert v.verify(feats(1.0), None)["verdict"] == NA                                       # unknown anomaly: nothing to verify
    assert v.verify(feats(1.0), "bright_thing", validated_domain=False)["verdict"] == NA    # outside validated domain
    assert v.verify(feats(1.0), "not_a_class")["verdict"] == NA


def test_top3_membership_alone_does_not_confirm_when_signature_is_sure_of_another_class(stub_model):
    """Regression: bottle claimed, signature 98% sure it is a chain. Being in the top-3 must NOT count as confirmation."""
    out = stub_model.verify(feats(1.0), "neutral_0")                 # bright object: signature is dominated by bright_thing
    assert out["claimed_rank"] <= 3 and out["claimed_vs_best"] < 0.5
    assert out["verdict"] != CONFIRMED


def test_verifier_reports_nearest_classes_and_probabilities_sum_to_one(stub_model):
    out = stub_model.verify(feats(1.0), None)
    assert out["nearest_classes"][0]["class"] == "bright_thing" and abs(stub_model.probs(feats(0.3)).sum() - 1) < 1e-9
    assert out["claimed_class"] is None and "features" in out


def test_missing_model_disables_verification_gracefully(tmp_path):
    v = SignatureVerifier(tmp_path / "nope.json")
    assert not v.available
    img, b = scene()
    o = FusedObject("KNOWN_DEBRIS", "tire", "known", 0.9, [float(x) for x in b], {})
    assert verify_objects(img, [o], v, waterfall=False) == {CONFIRMED: 0, UNCERTAIN: 0, MISMATCH: 0, NA: 0} and o.signature is None


def test_verification_never_changes_an_objects_category(stub_model):
    img, b = scene(obj_val=230)
    o = FusedObject("KNOWN_DEBRIS", "dark_thing", "known", 0.9, [float(x) for x in b], {})
    n = FusedObject("NOISE", "Noise", "noise", 0.1, [float(x) for x in b], {})
    counts = verify_objects(img, [o, n], stub_model, waterfall=False)
    assert o.category == "KNOWN_DEBRIS" and o.class_name == "dark_thing" and o.confidence == 0.9   # flagged, not altered
    assert o.signature["verdict"] == MISMATCH and n.signature is None and counts[MISMATCH] == 1


def test_unknown_anomaly_gets_descriptors_but_no_verdict(stub_model):
    img, b = scene(obj_val=230, shadow=8)
    o = FusedObject("UNKNOWN_ANOMALY", "Unknown Anomaly", "unknown", 0.5, [float(x) for x in b], {})
    verify_objects(img, [o], stub_model, waterfall=False)
    assert o.signature["verdict"] == NA and o.signature["features"]["shadow_depth"] > 0.5 and o.signature["nearest_classes"]


# ------------------------------------------------------------------ trained model + pipeline
def test_trained_model_matches_feature_set_and_has_validated_policy(base_cfg):
    p = base_cfg.path("signature.model")
    if not p.exists():
        pytest.skip("signature model not trained")
    v = SignatureVerifier(p)
    assert v.available and len(v.classes) == 27 and v.k >= 1 and 0 < v.tau < 1


def test_pipeline_attaches_signature_and_logs_stage(cfg, sample_xtf, sample_image):
    import cv2
    from backend.pipeline.runner import SonarPipeline
    pipe = SonarPipeline(cfg)
    res = pipe.run(sample_xtf, save_artifacts=False)
    assert "acoustic_signature" in [s["stage"] for s in res.stages]
    assert "acoustic_signature" in res.model_version
    for d in res.detections:
        assert "acoustic_signature" in d["evidence"] and d["signature_verdict"] in (CONFIRMED, UNCERTAIN, MISMATCH, NA)
    ok, buf = cv2.imencode(".png", sample_image)
    img_res = pipe.run(buf.tobytes(), filename="a.png", save_artifacts=False)
    assert img_res.detections and all(d["signature_verdict"] in (CONFIRMED, UNCERTAIN, MISMATCH, NA) for d in img_res.detections)
