"""결과 검증기(2단계 ③, D-121) — GPU 없이."""
import numpy as np

from genai_lab import character_features as cf
from genai_lab import character_verification as cv


def scene():
    """test_character_features.scene와 같은 합성 그림(흰 배경, 청록 머리, 살색 얼굴, 파란 홍채)."""
    image = np.full((200, 160, 3), 255, np.uint8)
    parts = {k: np.zeros((200, 160), bool) for k in ("hair", "face", "eyes", "nose", "mouth", "headwear", "eyewear", "earwear")}
    image[10:70, 30:130] = (60, 200, 200); parts["hair"][10:70, 30:130] = True
    image[70:150, 50:110] = (240, 200, 180); parts["face"][70:150, 50:110] = True
    image[95:105, 60:80] = (250, 250, 250)
    image[97:103, 66:74] = (40, 90, 220)
    image[99:101, 69:71] = (0, 0, 0)
    parts["eyes"][95:105, 60:80] = True
    parts["nose"][120:124, 78:82] = True
    return image, parts


def reference(**overrides):
    image, parts = scene()
    scores = dict(long_hair=0.9, twintails=0.8, aqua_hair=0.7, blue_eyes=0.8, headphones=0.6)
    return cf.extract_features(image, parts, scores, gender="female", body_parts=("tail",), **overrides), image, parts


def test_tag_items_are_kept_changed_missing_or_added():
    features, image, parts = reference()
    same = dict(long_hair=0.9, twintails=0.6, aqua_hair=0.7, blue_eyes=0.8, headphones=0.4, cat_tail=0.8, **{"1girl": 0.9})
    report = cv.verify_features(features, same)
    assert report["items"]["hair_length"]["status"] == "kept" and report["items"]["head_accessories"]["status"] == "kept"   # 0.4는 약한 수준이라도 유지
    assert report["items"]["body_parts"]["status"] == "kept" and report["items"]["gender"]["status"] == "kept"
    assert report["summary"]["changed"] == [] and report["items"]["skin"]["status"] == "kept"      # 피부 태그 없음 → 없음 그대로 = 유지
    different = dict(short_hair=0.8, ponytail=0.7, pink_hair=0.9, red_eyes=0.8, cat_ears=0.8, **{"1boy": 0.8})
    report = cv.verify_features(features, different)
    it = report["items"]
    assert it["hair_length"]["status"] == "changed" and it["hair_style"]["status"] == "changed"
    assert it["hair_color"]["status"] == "changed" and it["eye_color"]["status"] == "changed"
    assert it["head_accessories"]["status"] == "missing" and it["head_accessories"]["lost"] == ["headphones"]
    assert it["body_parts"] == dict(it["body_parts"], status="missing", missing=["tail"], added=["ears"])
    assert it["gender"]["status"] == "changed"
    weakly = cv.verify_features(features, dict(different, **{"1boy": 0.55}))
    assert weakly["items"]["gender"]["status"] == "unknown"                 # 0.7 미만이면 바뀜이라 하지 않는다


def test_confirmed_none_becomes_added_when_result_grows_one():
    features, image, parts = reference(confirmed=dict(hair_length=["long_hair"], hair_style=["twintails"], hair_color=["aqua_hair"],
                                                      eye_color=["blue_eyes"], head_accessories=[], skin=[]))
    report = cv.verify_features(features, dict(long_hair=0.9, twintails=0.8, aqua_hair=0.7, blue_eyes=0.8, hat=0.7))
    assert report["items"]["head_accessories"]["status"] == "added" and report["items"]["head_accessories"]["result"] == ["hat"]


def test_pixel_colours_decide_when_tags_are_silent_and_override_kept_tags():
    features, image, parts = reference()
    # 결과: 같은 마스크 자리에 머리색이 분홍으로 바뀐 그림
    changed = image.copy()
    changed[parts["hair"]] = (240, 120, 200)
    report = cv.verify_features(features, dict(long_hair=0.9, aqua_hair=0.7), changed, parts)
    assert report["items"]["hair_color"]["pixels"]["status"] == "changed" and report["items"]["hair_color"]["status"] == "changed"
    assert report["items"]["hair_color"]["tags_status"] == "kept"           # 태그는 같았지만 픽셀이 우선
    assert report["items"]["eye_color"]["pixels"]["status"] == "kept"
    # 태그가 '없어졌다'고 해도 픽셀이 유지면 유지(태그 노이즈)
    report = cv.verify_features(features, dict(long_hair=0.9, pink_hair=0.9), image, parts)
    assert report["items"]["hair_color"]["status"] == "kept" and report["items"]["hair_color"]["tags_status"] == "changed"
    # 태그가 전혀 없으면 픽셀 비교가 상태를 정한다
    report = cv.verify_features(features, {}, image, parts)
    assert report["items"]["hair_color"]["status"] == "kept" and report["items"]["skin"]["status"] == "kept"
    assert report["items"]["hair_length"]["status"] == "missing"


def test_unknown_reference_never_counts_as_change():
    image, parts = scene()
    features = cf.extract_features(image, parts, {})          # 모두 unknown/픽셀만
    report = cv.verify_features(features, dict(short_hair=0.9, cat_ears=0.9, **{"1boy": 0.9}))
    assert report["items"]["hair_length"]["status"] == "unknown" and report["items"]["body_parts"]["status"] == "unknown"
    assert report["items"]["gender"]["status"] == "unknown" and report["summary"]["changed"] == []


def test_hair_colour_matches_the_nearest_reference_cluster():
    reference = [dict(lab=[82.0, 0.4, -6.6], rgb=[197, 204, 217], fraction=0.51), dict(lab=[97.0, -0.1, -0.8], rgb=[243, 247, 248], fraction=0.39),
                 dict(lab=[34.0, 0.1, -3.0], rgb=[76, 80, 86], fraction=0.1)]
    white = dict(lab=[96.5, -0.2, -0.5], rgb=[242, 246, 247])
    assert cv.compare_color(reference, white, cv.HAIR_DE)["status"] == "kept"          # N4: 흰 하이라이트 군집과 맞음
    dark = dict(lab=[36.0, 0.0, -2.0], rgb=[80, 84, 90])
    assert cv.compare_color(reference, dark, cv.HAIR_DE)["status"] == "changed"        # 10% 군집은 후보가 아니다
    assert cv.compare_color(reference[0], white, cv.HAIR_DE)["status"] == "changed"    # 단일 색 비교는 그대로
