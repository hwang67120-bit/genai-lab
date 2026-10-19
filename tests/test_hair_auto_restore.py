import numpy as np


def palette(*clusters):
    return dict(palette=[dict(fraction=f, lab=list(lab)) for f, lab in clusters], ab_quantiles=dict(a=[0.0] * 101, b=[0.0] * 101))


def test_only_single_chromatic_hair_is_eligible():
    from genai_lab.hair_auto_restore import reference_eligibility
    blue = palette((0.5, (53, 4, -22)), (0.3, (42, 4.5, -19.5)), (0.2, (65, 2, -24)))          # M6 같은 청회색
    black = palette((0.6, (21, 3, 1)), (0.4, (39, 3, 5)))
    streak = palette((0.6, (21, 3, 1)), (0.3, (39, 3, 5)), (0.05, (36, 31, 13)))               # F5 같은 빨간 줄
    grey_pink = palette((0.4, (85, 7.5, -0.4)), (0.4, (68, 12, -2)), (0.2, (66, 33, -5)))      # N2 같은 끝 색
    assert reference_eligibility(blue)["eligible"]
    assert not reference_eligibility(black)["eligible"]
    assert not reference_eligibility(streak)["eligible"]
    assert not reference_eligibility(grey_pink)["eligible"]
    assert not reference_eligibility({})["eligible"]


def test_cast_test_separates_scene_lighting_from_hair_only_change():
    from genai_lab.hair_auto_restore import cast_test
    mask = np.ones((20, 20), bool)
    lab = np.zeros((20, 20, 3), np.float32)
    lab[..., 1], lab[..., 2] = 14, -10                       # 머리 a·b가 (4,−22) → (14,−10)로 이동
    same = cast_test((4, -22), (60, 10, 10), lab, mask, (60, 20, 22))     # 피부도 같은 방향
    opposite = cast_test((4, -22), (60, 10, 10), lab, mask, (60, 2, 0))   # 피부는 반대 방향
    small = cast_test((12, -12), (60, 10, 10), lab, mask, (60, 2, 0))
    assert same["status"] == "follows_lighting"
    assert opposite["status"] == "hair_only_shift"
    assert small["status"] == "small_shift"
    assert cast_test((4, -22), None, lab, mask, (60, 2, 0))["status"] == "unmeasurable"


def test_collateral_flags_changes_outside_hair_and_in_eyes():
    from genai_lab.hair_auto_restore import collateral
    raw = np.zeros((40, 40, 3), np.uint8)
    hair = np.zeros((40, 40), bool); hair[:10] = True
    eyes = np.zeros((40, 40), bool); eyes[20:22, 10:20] = True
    inside = raw.copy(); inside[:10] = 50
    assert collateral(raw, inside, hair, dict(eyes=eyes))["ok"]
    outside = inside.copy(); outside[30:] = 50
    assert collateral(raw, outside, hair, dict(eyes=eyes))["outside_hair"] > 0
    in_eyes = inside.copy(); in_eyes[20:22, 10:20] = 50
    assert not collateral(raw, in_eyes, hair, dict(eyes=eyes))["ok"]


def test_band_error_needs_two_measured_bands():
    from genai_lab.hair_auto_restore import band_error, band_medians
    lab = np.zeros((60, 10, 3), np.float32); lab[..., 1] = 5
    mask = np.ones((60, 10), bool)
    bands = band_medians(lab, mask)
    assert len(bands) == 3 and band_error(bands, bands) == 0
    assert band_error([None, None, bands[2]], bands) is None


def test_ineligible_reference_keeps_the_raw_result():
    from genai_lab.hair_auto_restore import decide
    features = dict(items=dict(hair_color=palette((1.0, (21, 3, 1)))))
    image = np.zeros((40, 40, 3), np.uint8)
    out = decide(features, image, dict(hair=np.ones((40, 40), bool)), image, dict(hair=np.ones((40, 40), bool)))
    assert out["decision"] == "not_applicable" and "restored" not in out
