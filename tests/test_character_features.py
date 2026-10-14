"""기준 캐릭터 특징표 추출(2단계 1번, D-118) — GPU 없이 합성 그림으로."""
import json

import numpy as np

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
