"""기준 캐릭터 특징표 추출(2단계 1번, D-118) — GPU 없이 합성 그림으로."""
import json

import numpy as np
import pytest

from genai_lab import character_features as cf


def scene():
    """흰 배경 위: 청록 머리(위), 살색 얼굴(가운데), 눈(흰자 + 파란 홍채 + 검은 동공)."""
    image = np.full((200, 160, 3), 255, np.uint8)
    parts = {k: np.zeros((200, 160), bool) for k in ("hair", "face", "eyes", "nose", "mouth", "headwear", "eyewear", "earwear")}
    image[10:70, 30:130] = (60, 200, 200); parts["hair"][10:70, 30:130] = True
    image[70:150, 50:110] = (240, 200, 180); parts["face"][70:150, 50:110] = True
    image[95:105, 60:80] = (250, 250, 250)                 # 흰자
    image[97:103, 66:74] = (40, 90, 220)                   # 홍채(파랑)
    image[99:101, 69:71] = (0, 0, 0)                       # 동공
    parts["eyes"][95:105, 60:80] = True
    parts["nose"][120:124, 78:82] = True
    return image, parts


def test_tags_above_threshold_are_kept_in_score_order():
    scores = dict(long_hair=0.9, very_long_hair=0.6, short_hair=0.1, twintails=0.8, bangs=0.55, ahoge=0.49)
    assert cf.tag_values(scores, cf.HAIR_LENGTH_TAGS) == [("long_hair", 0.9), ("very_long_hair", 0.6)]
    assert [n for n, _ in cf.tag_values(scores, cf.HAIR_STYLE_TAGS)] == ["twintails", "bangs"]


def test_hair_color_comes_from_pixels_and_eye_color_ignores_sclera_and_pupil():
    image, parts = scene()
    hair = cf.dominant_colors(image, parts["hair"])
    assert hair and hair[0]["fraction"] > 0.9
    r, g, b = hair[0]["rgb"]
    assert r < 100 and g > 170 and b > 170                   # 청록
    eye = cf.eye_color(image, parts["eyes"])
    r, g, b = eye["rgb"]
    assert b > 150 and r < 100                               # 파란 홍채만 (흰자 250·동공 0은 제외)
    skin = cf.skin_color(image, parts)
    assert skin and skin["rgb"][0] > 200 and skin["pixels"] > 0


def test_extract_marks_unmeasurable_items_unknown_and_never_guesses():
    image, parts = scene()
    scores = dict(long_hair=0.9, twintails=0.8, aqua_hair=0.7, blue_eyes=0.8, headphones=0.6, dark_skin=0.1)
    features = cf.extract_features(image, parts, scores)
    it = features["items"]
    assert it["hair_length"]["value"] == "long_hair" and it["hair_style"]["values"] == ["twintails"]
    assert it["hair_color"]["tags"] == ["aqua_hair"] and it["hair_color"]["colors"]
    assert it["eye_color"]["tags"] == ["blue_eyes"] and it["eye_color"]["color"]["rgb"][2] > 150
    assert it["head_accessories"]["values"] == ["headphones"] and it["head_accessories"]["mask_px"]["headwear"] == 0
    assert it["skin"]["tags"] == [] and it["skin"]["color"]
    assert it["body_parts"]["state"] == "unknown" and it["gender"]["state"] == "unknown"      # 확정 전 → unknown, 추측 없음
    assert set(features["unknown"]) == {"body_parts", "gender"}
    assert all(v["confirmed"] is False for v in it.values())
    # 눈 마스크가 비면 눈 색은 unknown (태그도 없을 때)
    empty = cf.extract_features(image, dict(parts, eyes=np.zeros_like(parts["eyes"])), dict(long_hair=0.9))
    assert empty["items"]["eye_color"]["state"] == "unknown" and empty["items"]["eye_color"]["color"] is None


def test_confirmed_gender_and_body_parts_are_carried_as_user_facts(tmp_path):
    image, parts = scene()
    features = cf.extract_features(image, parts, {}, gender="female", body_parts=("tail", "ears"),
                                   provenance=dict(gender="saved", body_parts="user_confirmed"))
    it = features["items"]
    assert it["gender"] == dict(state="measured", source="saved", confirmed=True, value="female")
    assert it["body_parts"]["values"] == ["ears", "tail"] and it["body_parts"]["source"] == "user_confirmed"
    path = cf.write_features(tmp_path / "character-features", features)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema"] == "character_features/1" and saved["items"]["gender"]["value"] == "female"
    assert "성별 female" in cf.summary_line(features) and "부위 ['ears', 'tail']" in cf.summary_line(features)


def test_weak_candidates_are_recorded_but_not_promoted_to_measured():
    image, parts = scene()
    scores = dict(short_hair=0.33, ahoge=0.36, white_hair=0.36, long_hair=0.9)
    it = cf.extract_features(image, parts, scores)["items"]
    assert it["hair_length"]["value"] == "long_hair" and it["hair_length"]["weak"] == []        # 0.33은 약한 후보도 아님
    assert it["hair_style"]["state"] == "weak" and it["hair_style"]["values"] == [] and it["hair_style"]["weak"] == [("ahoge", 0.36)]
    assert it["hair_color"]["state"] == "measured" and it["hair_color"]["tags"] == [] and it["hair_color"]["weak"] == [("white_hair", 0.36)]


def test_tag_features_text_round_trip_and_confirmed_values_override_tags():
    scores = dict(long_hair=0.8, very_long_hair=0.55, twintails=0.7, aqua_hair=0.6, blue_eyes=0.9, headphones=0.55, ahoge=0.4)
    items = cf.tag_features(scores)
    assert items["hair_length"]["values"] == ["long_hair"] and items["hair_style"] == dict(values=["twintails"], weak=["ahoge"])
    text = cf.features_text(items)
    assert text.splitlines()[0] == "머리 길이: long_hair" and "피부: 모름" in text and "ahoge" in cf.weak_note(items)
    parsed = cf.parse_features_text(text.replace("머리 장식: headphones", "머리 장식: headphones, black ribbon").replace("피부: 모름", "피부: 없음"))
    assert parsed["head_accessories"] == ["headphones", "black_ribbon"] and parsed["skin"] == []
    with pytest.raises(ValueError):
        cf.parse_features_text("머리 길이: long_hair")                    # 빠진 항목
    with pytest.raises(ValueError):
        cf.parse_features_text(text + "\n모자: x")                       # 모르는 라벨
    image, parts = scene()
    features = cf.extract_features(image, parts, {}, confirmed=parsed)
    it = features["items"]
    assert it["hair_length"] == dict(it["hair_length"], value="long_hair", source="user", confirmed=True, state="measured")
    assert it["head_accessories"]["values"] == ["headphones", "black_ribbon"] and it["hair_color"]["tags"] == ["aqua_hair"]
    assert it["hair_color"]["colors"]                                       # 픽셀 측정은 그대로
    assert it["skin"]["tags"] == [] and it["skin"]["confirmed"] is True     # '없음'으로 확정


def test_hair_colour_ignores_original_background_mixed_into_the_hair_mask():
    # N4 사례(D-115): 회색 배경 그림에서 분할기가 배경을 머리카락에 넣으면 머리색이 배경색으로 치우친다
    image, parts = scene()
    image[(image == 255).all(2)] = (200, 205, 215)                            # 흰 배경 → 회색 배경
    image[:8], image[-8:], image[:, :8], image[:, -8:] = 255, 255, 255, 255   # 캔버스 흰 여백(실제 입력과 같은 구조)
    leaky = parts["hair"].copy(); leaky[8:18, 8:-8] = True; leaky[70:110, 8:30] = True   # 배경을 머리 마스크에 섞음
    colors = cf.dominant_colors(image, cf.hair_mask_without_background(image, leaky))
    r, g, b = colors[0]["rgb"]
    assert colors[0]["fraction"] > 0.9 and r < 100 and g > 170               # 청록이 주색, 회색 배경은 제외


def test_unknown_and_none_are_different_in_the_feature_text():
    # 검토 5번: '없음' = 사람이 없다고 확정([]), '모름'·빈 칸 = 확인 불가(None). 두 확인 창이 같은 뜻으로 쓴다
    base = cf.features_text({k: dict(values=None) for k in cf.TAG_ITEMS})
    assert all(line.endswith(": 모름") for line in base.splitlines())
    text = base.replace("머리 장식: 모름", "머리 장식: 없음").replace("피부: 모름", "피부:")
    parsed = cf.parse_features_text(text)
    assert parsed["head_accessories"] == [] and parsed["skin"] is None and parsed["hair_length"] is None
    with pytest.raises(ValueError):
        cf.parse_features_text(base.replace("머리 장식: 모름", "머리 장식: 모름, hat"))
    image, parts = scene()
    it = cf.extract_features(image, parts, dict(headphones=0.9), confirmed=parsed)["items"]
    assert it["head_accessories"]["values"] == [] and it["head_accessories"]["confirmed"] is True     # 확정 없음
    assert it["skin"]["confirmed"] is False                                                           # 모름은 확정하지 않음
