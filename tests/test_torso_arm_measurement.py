"""전달된 팔 마스크만 사용한다. 모델 추론·GPU·다운로드 없이 경계와 대기를 검사한다."""
import hashlib
import io
import json

import numpy as np
import pytest

from genai_lab.identity_measurement import prepare_arm_region, measure_torso_with_arms, measure_pixels
from genai_lab.identity_report import compare_measurements
from test_identity_measurement import synthetic


def scene():
    silhouette = np.zeros((100, 101), bool)
    silhouette[5:95, 10:91] = True
    joints = {"neck": (50, 20), "right_hip": (40, 80), "left_hip": (60, 80)}
    arms = np.zeros_like(silhouette)
    arms[5:95, 10:21] = True
    arms[5:95, 80:91] = True
    bands = [{"side": side, "chain": [(x, 10), (x, 90)], "r": 3}
             for side, x in (("right", 15), ("left", 85))]
    return silhouette, joints, arms, bands


def measure(silhouette, joints, handwear, bands, *, skin=None):
    region = prepare_arm_region(silhouette, handwear, joints)
    midhip = (np.asarray(joints["right_hip"]) + joints["left_hip"]) / 2
    return measure_torso_with_arms(silhouette, silhouette if skin is None else skin,
                                   joints["neck"], midhip, .3, 20, bands, region, 20)


def test_two_arm_boundaries_use_first_hit_and_ignore_legacy_merge():
    silhouette, joints, arms, bands = scene()
    result, line = measure(silhouette, joints, arms, bands)
    assert result["w_px"] == 59.5 and result["sil_w_px"] == 80.5
    assert result["ends"] == [59, 60] and result["w"] == 2.975
    assert result["boundary"] == ["arm_part", "arm_part"]
    assert result["status"] == "measured" and "invalid" not in result
    assert line == ((20.5, 38.0), (80.0, 38.0), False)
    json.dumps(result, allow_nan=False)


def test_no_arm_overlap_keeps_original_silhouette_width():
    silhouette, joints, arms, _ = scene()
    result, _ = measure(silhouette, joints, np.zeros_like(arms), [])
    assert result["w_px"] == result["sil_w_px"] == 80.5
    assert result["boundary"] == ["silhouette", "silhouette"]
    assert result["bare"] is True and result["exposure"] == 1


def test_missing_boundary_on_only_one_side_is_not_a_success():
    silhouette, joints, arms, bands = scene()
    arms[:, 80:] = False
    result, line = measure(silhouette, joints, arms, bands)
    assert result["boundary"] == ["arm_part", "silhouette"]
    assert result["invalid"] == "arm_touch" and result["status"] == "unmeasurable"
    assert line[-1] is True
    report = compare_measurements({"B3": result}, {"B3": result})
    assert report["items"]["B3"]["measurable"] is False
    assert report["items"]["B3"]["difference"] is None


def test_center_overlap_rejects_mask_without_guessing_an_arm_boundary():
    silhouette, joints, arms, bands = scene()
    arms[50, 50] = True
    region = prepare_arm_region(silhouette, arms, joints)
    assert region.usable is False and region.reason == "torso_center_on_arm"
    result, _ = measure(silhouette, joints, arms, bands)
    assert result["w_px"] == 80.5 and result["invalid"] == "arm_touch"
    assert result["boundary"] == ["silhouette", "silhouette"]


def test_unusable_mask_without_band_hit_keeps_merge_protection():
    silhouette, joints, arms, _ = scene()
    arms[50, 50] = True
    result, _ = measure(silhouette, joints, arms, [])
    assert result["invalid"] == "arm_merge"


def test_arm_region_is_intersection_without_modifying_inputs():
    silhouette, joints, arms, _ = scene()
    arms[0, 0] = True
    before = arms.copy()
    region = prepare_arm_region(silhouette, arms, joints)
    assert not region.mask[0, 0] and region.record()["pixels"] == 1980
    assert np.array_equal(arms, before)
    region.mask[:] = False
    assert np.array_equal(arms, before)


def test_silhouette_gap_does_not_jump_to_an_arm_part():
    silhouette, joints, arms, _ = scene()
    silhouette[:, 30] = False
    bands = [{"side": "right", "chain": [(35, 10), (35, 90)], "r": 3}]
    result, _ = measure(silhouette, joints, arms, bands)
    assert result["ends"][0] == 38 and result["boundary"][0] == "silhouette"
    assert result["invalid"] == "arm_touch"


def test_exposure_does_not_sample_arm_pixels():
    silhouette, joints, arms, bands = scene()
    skin = silhouette & ~arms
    result, _ = measure(silhouette, joints, arms, bands, skin=skin)
    assert result["exposure"] == 1 and result["bare"] is True


@pytest.mark.parametrize("handwear", [np.zeros((1, 1)), np.full((100, 101), np.nan),
                                    np.full((100, 101), -1), np.full((100, 101), "arm")])
def test_bad_mask_fails_instead_of_resizing_or_treating_it_as_empty(handwear):
    silhouette, joints, _, _ = scene()
    with pytest.raises(ValueError):
        prepare_arm_region(silhouette, handwear, joints)


@pytest.mark.parametrize("neck", [None, (50, 80), (float("nan"), 20)])
def test_missing_or_invalid_axis_is_explicit(neck):
    silhouette, joints, arms, bands = scene()
    joints["neck"] = neck
    region = prepare_arm_region(silhouette, arms, joints)
    assert not region.usable and region.reason == "no_axis"
    result, line = measure(silhouette, joints, arms, bands)
    assert result is None and line is None


def test_requested_parts_without_mask_are_pending_not_legacy_fallback():
    image, pose, alpha, head = synthetic()
    old, old_art = measure_pixels(image, pose, alpha, head)
    result, art = measure_pixels(image, pose, alpha, head, use_arm_parts=True)
    try:
        assert result["arm_region"] == {"available": False, "usable": False,
                                          "reason": "arm_parts_pending", "pixels": 0}
        for key in ("B3", "B4", "B5"):
            assert result[key]["status"] == "pending" and result[key]["invalid"] == "arm_parts_pending"
            assert "w" not in result[key]
        for key in ("B1", "B2", "B6", "B7", "B8", "skin_lab"):
            assert result[key] == old[key]
        assert np.array_equal(art["hair_quantiles"], old_art["hair_quantiles"])
        report = compare_measurements(result, result)
        assert report["items"]["B3"]["reason"] == "arm_parts_pending"
        assert report["items"]["B3"]["difference"] is None
    finally:
        image.close(); art["overlay"].close(); old_art["overlay"].close()


def test_opted_mask_updates_only_torso_and_review_overlay():
    image, pose, alpha, head = synthetic()
    old, old_art = measure_pixels(image, pose, alpha, head)
    arms = np.zeros_like(alpha); arms[50:150, 40:45] = 255; arms[50:150, 75:80] = 255
    before = arms.copy()
    result, art = measure_pixels(image, pose, alpha, head, use_arm_parts=True, handwear_mask=arms)
    try:
        assert result["B3"]["w_px"] == 30.5
        assert result["B3"]["boundary"] == ["arm_part", "arm_part"]
        for key in ("B1", "B2", "B6", "B7", "B8", "skin_lab", "hair_skin_dE"):
            assert result[key] == old[key]
        assert np.array_equal(art["hair_quantiles"], old_art["hair_quantiles"])
        assert not np.array_equal(art["overlay"], old_art["overlay"])
        assert np.array_equal(arms, before)
    finally:
        image.close(); art["overlay"].close(); old_art["overlay"].close()


def test_mask_cannot_be_silently_ignored_when_mode_is_off():
    image, pose, alpha, head = synthetic()
    try:
        with pytest.raises(ValueError, match="명시"):
            measure_pixels(image, pose, alpha, head, handwear_mask=np.zeros_like(alpha))
    finally:
        image.close()


@pytest.mark.parametrize("has_arms,measurement_sha,overlay_sha", [
    (False, "9cf8c0dd234664bd6c14aa540a2165d6f2eee2c9344812a8fa1304362a9723d0",
     "5955517e1b31f458f105bea5de18c68bad91552cd4dd743fa178f3bce7d024d1"),
    (True, "e3eacf3051fb0b5786c068ae5495fcde6ae88374f10e3af1e3ceb7b1892affee",
     "f4d681c01d636699425ac959093548b73c714c0b8a669338f5040ccf56538f63"),
])
def test_default_path_matches_locked_prechange_pixels_and_records(has_arms, measurement_sha, overlay_sha):
    image, pose, alpha, head = synthetic()
    if has_arms:
        pose["joints"] += [dict(joint_name=n, x=x, y=y, confidence_score=1) for n, (x, y) in
                           {"right_elbow": (43, 92), "left_elbow": (77, 92),
                            "right_wrist": (42, 118), "left_wrist": (78, 118)}.items()]
    result, art = measure_pixels(image, pose, alpha, head)
    try:
        assert hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest() == measurement_sha
        buffer = io.BytesIO(); art["overlay"].save(buffer, format="PNG")
        assert hashlib.sha256(buffer.getvalue()).hexdigest() == overlay_sha
        assert hashlib.sha256(art["hair_quantiles"].tobytes()).hexdigest() == "9ab7b6018abc4783467341209148753a51e0eca2f65a12f18e2970b26b853d45"
        assert "arm_region" not in result
    finally:
        image.close(); art["overlay"].close()
