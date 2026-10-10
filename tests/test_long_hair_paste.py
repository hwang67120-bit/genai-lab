"""긴 머리 경로(L1): 짧은 머리 경로 불변, 수염·배경·위치·생성 긴 머리 유지·측정을 GPU 없이 확인한다."""
import numpy as np
import pytest
from genai_lab import head_paste_rules as rules
from genai_lab.head_paste_rules import (TAGS, paste_canvas, paste_profile, prepare_original, shift_head,
    continuing_hair, figure_measurements, background_measure, rule_name)
from genai_lab.studio_generation import long_hair_notice
from test_head_paste_rules import sample_parts


@pytest.mark.parametrize("appearance,expected", [
    (["short_hair", "blue_hair"], None),
    (["short hair"], None),
    (["long_hair"], dict(rule="G3+H2+L1", facial_hair=False)),
    ([], dict(rule="G3+H2+L1", facial_hair=False)),
    (["long_hair", "facial_hair"], dict(rule="G3+H2+L1", facial_hair=True)),
    (["very_long_hair", "beard"], dict(rule="G3+H2+L1", facial_hair=True))])
def test_profile_keeps_short_hair_on_verified_rule(appearance, expected):
    assert paste_profile(appearance) == expected
    assert rule_name(paste_profile(appearance)) == ("G3+H2" if expected is None else "G3+H2+L1")


def scene(background=(255, 255, 255)):
    parts = sample_parts()
    hm = np.zeros((1024, 1024), bool); hm[100:600, 200:800] = True
    H = parts["hair"].copy()
    original = np.full((1024, 1024, 3), background, np.uint8)
    original[parts["face"]] = (200, 160, 130)
    original[H] = (50, 90, 130)
    generated_parts = sample_parts()
    generated = np.full((1024, 1024, 3), 200, np.uint8)
    generated[generated_parts["face"]] = (205, 165, 135)
    generated[generated_parts["hair"]] = (70, 80, 90)
    return original, hm, parts, H, generated, generated_parts


def test_beard_rule_only_with_facial_hair_tag():
    original, hm, parts, H, *_ = scene()
    original[hm & ~H] = (200, 160, 130)  # 윤곽 안 머리 밖은 피부(배경색 아님)
    legacy = prepare_original(original, hm, parts, H)
    bare = prepare_original(original, hm, parts, H, paste_profile(["long_hair"]))
    bearded = prepare_original(original, hm, parts, H, paste_profile(["long_hair", "facial_hair"]))
    assert legacy["beard"].any() and bearded["beard"].any()
    assert not bare["beard"].any()
    # 수염으로 빠졌던 얼굴 옆 머리가 긴 머리 경로에서는 원본 머리로 붙는다.
    assert bare["o_head"][legacy["beard"]].all()


def test_colored_plain_background_is_not_pasted_as_hair():
    beige = (240, 236, 225)
    original, hm, parts, H, *_ = scene(beige)
    hole = np.zeros((1024, 1024), bool); hole[420:450, 210:240] = True  # 윤곽 안, 분할 밖 배경
    original[hole] = beige
    for name in TAGS:
        parts[name][hole] = False
    H[hole] = False
    measured = background_measure(original, hm, parts)
    assert measured["white_distance"] > 10 and measured["plain_ratio"] == 1.0
    legacy = prepare_original(original, hm, parts, H)
    long_ = prepare_original(original, hm, parts, H, paste_profile(["long_hair"]))
    inner = np.zeros_like(hole); inner[428:442, 218:232] = True
    # 기존 경로는 윤곽 안 배경을 피부·전경(알파 1)으로 붙인다. 긴 머리 경로는 배경으로 뺀다.
    assert legacy["o_skin"][hole].all() and legacy["a3"][inner].min() == 1
    assert not long_["o_head"][hole].any() and not long_["o_skin"][hole].any()
    assert long_["near_bg"][hole].all() and long_["a3"][inner].max() == 0
    np.testing.assert_allclose(long_["background"], beige)
    # 배경색 기준으로 알파를 계산해 바깥 배경을 머리 경계에 섞지 않는다.
    assert long_["a3"][0:50, 0:50].max() == 0


def test_long_path_aligns_original_eyes_to_generated_eyes():
    original, hm, parts, H, generated, generated_parts = scene()
    moved = shift_head(original, hm, parts, H, 0, 0)
    assert all(np.array_equal(a, b) for a, b in zip(moved[:2], (original, hm)))
    shifted_parts = {name: np.roll(mask, 30, axis=1) for name, mask in generated_parts.items()}
    shifted_generated = np.roll(generated, 30, axis=1)
    _, _, _, measured = paste_canvas(original, hm, parts, H, shifted_generated, shifted_parts, paste_profile(["long_hair"]))
    assert measured["shift_px"] == [30, 0]


def test_large_face_offset_skips_instead_of_guessing():
    original, hm, parts, H, generated, generated_parts = scene()
    far = {name: np.roll(mask, 120, axis=1) for name, mask in generated_parts.items()}
    with pytest.raises(ValueError, match="위치 차이"):
        paste_canvas(original, hm, parts, H, np.roll(generated, 120, axis=1), far, paste_profile(["long_hair"]))


def test_generated_long_hair_leaving_crop_is_kept_but_floating_piece_is_not():
    g_hair = np.zeros((1024, 1024), bool)
    o_hair = np.zeros((1024, 1024), bool); o_hair[100:400, 300:700] = True
    g_hair[420:1024, 100:150] = True   # 자른 범위 아래로 이어지는 생성 긴 머리
    g_hair[420:500, 850:900] = True    # 원본에 없는 떠 있는 생성 머리
    g_hair[700:760, 1000:1024] = True  # 턱 아래 오른쪽 가장자리로 나가는 머리
    keep = continuing_hair(g_hair, o_hair, chin=600)
    assert keep[420:1024, 100:150].all() and keep[700:760, 1000:1024].all()
    assert not keep[420:500, 850:900].any()


def test_long_path_never_changes_protected_features_or_outside_mask():
    original, hm, parts, H, generated, generated_parts = scene()
    canvas, mask, protected, measured = paste_canvas(original, hm, parts, H, generated, generated_parts,
                                                     paste_profile(["long_hair"]))
    assert np.array_equal(canvas[protected], generated[protected])
    assert np.array_equal(canvas[~mask], generated[~mask])
    assert measured["rule"] == "G3+H2+L1" and measured["beard_px"] == 0
    assert set(measured["hair"]) >= {"hair_area_ratio", "overflow_px", "hair_color_dE", "hair_palette_difference",
                                     "generated_hair_on_original_face_px", "paste_on_generated_clothes_px"}


def test_short_path_measurements_unchanged():
    original, hm, parts, H, generated, generated_parts = scene()
    _, _, _, measured = paste_canvas(original, hm, parts, H, generated, generated_parts)
    assert set(measured) == {"skin_dE", "forehead_blend", "erase_fill", "beard_px", "erase_px"}


def test_figure_measurements_count_length_tails_and_color():
    rgb = np.full((1232, 736, 3), 255, np.uint8)
    o_hair = np.zeros((1232, 736), bool); o_hair[100:400, 200:500] = True
    o_hair[400:900, 200:240] = True; o_hair[400:900, 460:500] = True  # 트윈테일 두 꼬리
    g_hair = np.zeros((1232, 736), bool); g_hair[100:400, 200:500] = True
    g_hair[400:700, 200:240] = True                                  # 꼬리 하나, 짧음
    original, generated = rgb.copy(), rgb.copy()
    original[o_hair] = (40, 40, 60); generated[g_hair] = (200, 60, 60)
    result = figure_measurements(o_hair, g_hair, original, generated, head_bottom=400)
    assert result["tails_below_head"] == [2, 1]
    assert result["lowest_hair_row"] == [899, 699]
    assert 0 < result["hair_iou"] < 1 and result["hair_color_dE"] > 20


def test_long_hair_notice_is_reference_only():
    assert long_hair_notice(None) == ""
    text = long_hair_notice(dict(figure=dict(lowest_hair_row=[899, 699], tails_below_head=[2, 1], hair_color_dE=31.2)))
    assert "참고용" in text and "원본 2 · 생성 1" in text and "16%" in text


class SquareSegmenter:
    """정사각형 입력을 받는지 확인하고, 흰색이 아닌 픽셀을 머리로 돌려준다."""
    def __init__(self):
        self.calls = 0

    def parse(self, image):
        assert image.shape == (1024, 1024, 3)
        self.calls += 1
        parts = {name: np.zeros((1024, 1024), bool) for name in TAGS}
        parts["hair"] = image.min(axis=2) < 128
        return parts


def test_figure_hair_restores_original_coordinates_and_parses_source_once():
    from genai_lab import head_paste_worker as worker
    worker.FIGURE_PARTS.clear()
    rgb = np.full((1232, 736, 3), 255, np.uint8)
    rgb[100:300, 50:150] = 0
    segmenter = SquareSegmenter()
    first = worker.figure_hair(segmenter, rgb, "source-sha")
    second = worker.figure_hair(segmenter, rgb, "source-sha")
    assert segmenter.calls == 1 and first is second and first.shape == (1232, 736)
    ys, xs = np.nonzero(first)
    assert abs(ys.min() - 100) <= 2 and abs(ys.max() - 299) <= 2 and abs(xs.min() - 50) <= 2 and abs(xs.max() - 149) <= 2
    worker.figure_hair(segmenter, rgb)  # 생성 그림은 매번 분할한다.
    assert segmenter.calls == 2
    worker.FIGURE_PARTS.clear()


def test_long_hair_summary_only_for_long_rule():
    from genai_lab.head_paste_worker import long_hair_summary
    assert long_hair_summary(dict(skin_dE=1.0)) is None
    summary = long_hair_summary(dict(rule="G3+H2+L1", shift_px=[3, 0], hair={}, figure={"hair_iou": 0.5}))
    assert summary["shift_px"] == [3, 0] and summary["figure"] == {"hair_iou": 0.5}
