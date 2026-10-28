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
    single = cv.compare_color(reference[0], white, cv.HAIR_DE)                          # 단일 색 비교는 그대로, ΔE 15.9
    assert single["status"] == "unknown" and single["reason"] == "borderline"            # 문턱~1.25배는 경계값(D-126)


def two_eyes(left, right, hair=(60, 200, 200), streak=None):
    """두 눈이 따로 있는 합성 그림(D-126). left/right: 홍채 RGB. streak: 머리에 섞을 색(새 머리색 검사)."""
    image, parts = scene()
    parts["eyes"][:] = False
    for x0, colour in ((56, left), (88, right)):
        image[95:105, x0:x0 + 16] = (250, 250, 250)
        image[96:104, x0 + 3:x0 + 13] = colour
        parts["eyes"][95:105, x0:x0 + 16] = True
    if streak is not None:
        image[10:20, 30:130] = streak
    return image, parts


def eye_reference(left, right, tags):
    image, parts = two_eyes(left, right)
    return cf.extract_features(image, parts, {t: 0.8 for t in tags}, gender="female", body_parts=())


def test_heterochromia_lost_is_a_confident_change_even_when_tags_only_lose_a_colour():
    # K-1: 청/적 오드아이 → 양쪽 파랑. 태그는 red_eyes·heterochromia만 잃어 missing, 두 눈 픽셀이 바뀜을 잡는다
    reference = eye_reference((40, 90, 220), (220, 40, 50), ("blue_eyes", "red_eyes", "heterochromia"))
    sample, sample_parts = two_eyes((40, 90, 220), (220, 40, 50))
    measured = cf.eye_colors(sample, sample_parts["eyes"])
    assert measured[0]["hue"] < -60 and cv.heterochromia(measured)
    image, parts = two_eyes((40, 90, 220), (50, 100, 230))
    report = cv.verify_features(reference, dict(blue_eyes=0.9), image, parts)
    eye = report["items"]["eye_color"]
    assert eye["tags_status"] == "missing" and eye["pixels"]["reason"] == "heterochromia_lost"
    assert eye["status"] == "changed" and report["retry"] == ["eye_color"]
    image, parts = two_eyes((220, 40, 50), (40, 90, 220))                       # 좌우가 바뀐 시점은 유지
    assert cv.verify_features(reference, dict(blue_eyes=0.9, red_eyes=0.8, heterochromia=0.7), image, parts)["items"]["eye_color"]["status"] == "kept"


def test_closed_eyes_and_tag_pixel_disagreement_are_unknown_not_retry():
    reference = eye_reference((40, 90, 220), (40, 90, 220), ("blue_eyes",))
    image, parts = two_eyes((200, 40, 60), (200, 40, 60))
    closed = cv.verify_features(reference, dict(closed_eyes=0.9), image, parts)       # K-2: 감은 눈을 색 변화로 세지 않는다
    assert closed["items"]["eye_color"]["status"] == "unknown" and "eyes hidden" in closed["items"]["eye_color"]["reason"]
    disagree = cv.verify_features(reference, dict(blue_eyes=0.9), image, parts)       # 태그는 유지, 픽셀은 바뀜 → 사람 확인
    assert disagree["items"]["eye_color"]["status"] == "unknown" and disagree["retry"] == []
    skipped = cv.verify_features(reference, dict(red_eyes=0.9), image, parts, {"eye_color": "요청과 충돌: 눈 감기"})
    assert skipped["items"]["eye_color"]["status"] == "unknown" and skipped["items"]["eye_color"]["skipped"] and skipped["retry"] == []


def test_saturation_shift_of_the_same_hue_is_kept():
    # N1·N2: Qwen이 분홍 눈을 더 진하게 그린다. 같은 색조면 유지(밝기·채도는 그림체로 바뀐다)
    reference = eye_reference((205, 150, 160), (205, 150, 160), ("pink_eyes",))
    image, parts = two_eyes((230, 110, 175), (230, 110, 175))
    assert cv.verify_features(reference, dict(pink_eyes=0.8), image, parts)["items"]["eye_color"]["status"] == "kept"


def test_new_saturated_hair_colour_is_added_and_retried():
    # N2-2: 기준에 없던 원색 줄무늬. 가장 큰 색은 그대로라 예전 비교는 kept였다
    image, parts = scene()
    reference = cf.extract_features(image, parts, dict(aqua_hair=0.8), gender="female", body_parts=())
    streaked, parts2 = two_eyes((40, 90, 220), (40, 90, 220), streak=(240, 220, 30))
    report = cv.verify_features(reference, dict(aqua_hair=0.8), streaked, parts2)
    hair = report["items"]["hair_color"]
    assert hair["pixels"]["status"] == "kept" and hair["status"] == "added" and hair["new_colors"]
    assert "hair_color" in report["retry"]
    same = cv.verify_features(reference, dict(aqua_hair=0.8), image, parts)
    assert same["items"]["hair_color"]["status"] == "kept" and same["items"]["hair_color"]["new_colors"] == []


def test_generic_tag_is_not_a_change_and_weak_replacement_is_not_retried():
    image, parts = scene()
    reference = cf.extract_features(image, parts, dict(x_hair_ornament=0.8, long_hair=0.9), gender="female", body_parts=())
    generic = cv.verify_features(reference, dict(hair_ornament=0.8, long_hair=0.9))       # N2-1: X 장식 → 일반 장식 태그
    assert generic["items"]["head_accessories"]["status"] == "unknown" and generic["items"]["head_accessories"]["reason"] == "kept_generic"
    weak = cv.verify_features(reference, dict(x_hair_ornament=0.8, very_long_hair=0.6))
    assert weak["items"]["hair_length"]["status"] == "changed" and "hair_length" not in weak["retry"]
    strong = cv.verify_features(reference, dict(x_hair_ornament=0.8, short_hair=0.9))
    assert "hair_length" in strong["retry"]


def test_reduced_coloured_hair_and_low_margin_are_marked_for_review():
    # D-127 (요청 21 N2-2): 분홍 비중이 줄었는데 대표색 차이는 문턱 안이라 '유지'였다 → 유지 + 확인 필요
    image, parts = scene()
    image[10:40, 30:130] = (230, 120, 170)                  # 머리 절반이 분홍
    reference = cf.extract_features(image, parts, {}, gender="female", body_parts=())
    assert reference["items"]["hair_color"]["colored"] and reference["items"]["hair_color"]["colored"][0]["share"] > 0.3
    faded = image.copy()
    faded[10:37, 30:130] = (60, 200, 200)                   # 분홍이 거의 사라짐(3줄만 남음)
    report = cv.verify_features(reference, {}, faded, parts)
    hair = report["items"]["hair_color"]
    assert hair["reduced_colors"] and "색 있는 머리 비중 감소" in hair["review"] and "hair_color" in report["summary"]["review"]
    assert "hair_color" not in report["retry"]                                     # 조명·배경 영향이 있어 재시도 근거가 아니다
    same = cv.verify_features(reference, {}, image, parts)
    assert same["items"]["hair_color"]["reduced_colors"] == [] and not same["items"]["hair_color"]["review"]


def test_full_bleed_background_no_longer_yields_nan_and_is_removed_from_hair():
    # D-127: 그림이 캔버스 끝까지 차면 테두리가 비어 NaN 경고 + 배경 빼기 생략이었다(눈길 배경)
    import warnings
    image = np.full((120, 100, 3), (150, 160, 170), np.uint8)                    # 끝까지 찬 회색 배경
    image[30:90, 30:70] = (230, 120, 170)
    hair = np.zeros((120, 100), bool)
    hair[20:100, 20:80] = True                                                      # 분할기가 배경 일부를 머리에 넣음
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cleaned = cf.hair_mask_without_background(image, hair)
    assert not cleaned[25, 25] and cleaned[50, 50]
    from genai_lab.outfit_edit import original_background
    assert not original_background(image).any()                                    # 잠근 편집 단계 동작은 그대로
