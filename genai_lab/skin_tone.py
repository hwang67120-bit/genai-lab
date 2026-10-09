"""피부 톤의 읽기 시작점: 사용자 선택으로 태그를 바꾸고, 양 볼 측정은 추천만 만든다.

replace_skin_tags: 기존 태그와 사용자 선택 → 변경된 태그와 선택 기록.
observe_skin_tone: 기존 이미지·얼굴 좌표 파일 → 볼 측정값 또는 측정 불가 사유.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import math

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SkinRecommendationSettings:
    lightness_min: float = 65
    # 밝은 피부까지 황갈색으로 추천하지 않도록 시험에서 정한 상한을 유지한다.
    lightness_max: float = 80
    hue_min: float = 30
    hue_max: float = 70

    def __post_init__(self):
        validate_recommendation_settings(self)


def validate_recommendation_settings(settings):
    """설정 범위를 검사한다. 추천 결과를 보고 기준을 자동 조정하지 않는다."""
    values = (settings.lightness_min, settings.lightness_max,
              settings.hue_min, settings.hue_max)
    finite = all(math.isfinite(value) for value in values)
    lightness_valid = 0 <= settings.lightness_min <= settings.lightness_max <= 100
    hue_valid = 0 <= settings.hue_min <= settings.hue_max <= 360

    if not (finite and lightness_valid and hue_valid):
        raise ValueError("피부 톤 추천 기준이 올바르지 않습니다.")


@dataclass(frozen=True)
class SkinToneChoice:
    value: str | None = None  # 선택이 없으면 기존 태그를 그대로 유지한다.
    measurement: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.value not in (None, "dark", "tan", "unspecified"):
            raise ValueError("피부 톤 선택값 오류")


def is_skin_tag(tag):
    value = tag.replace("_", " ").strip()
    return value in ("dark skin", "tan") or value.startswith("dark-skinned ")


def default_skin_choice(tags):
    """감지한 피부 태그로 기본 선택만 정한다. 태그 자체는 바꾸지 않는다."""
    found = [tag.replace("_", " ").strip() for tag in tags if is_skin_tag(tag)]
    if not found:
        return None
    if set(found) == {"tan"}:
        return "tan"
    return "dark"


def replace_skin_tags(tags, choice):
    """입력 태그와 명시적 선택을 받아, 결과 태그와 변경 기록을 반환한다."""
    tags = tuple(tags)
    default = default_skin_choice(tags)

    if default is None:
        return apply_measured_skin_choice(tags, choice)

    return apply_detected_skin_choice(tags, choice, default)


def selected_skin_tags(value):
    """사용자가 고른 피부 톤에 대응하는 태그만 반환한다."""
    if value == "unspecified":
        return ()
    if value == "tan":
        return ("tan",)
    return ("dark skin",)


def apply_measured_skin_choice(tags, choice):
    """피부 태그가 없는 경우: 측정 화면에서 직접 고른 값만 추가한다."""
    if choice.value is None:
        return tags, None
    if choice.measurement.get("trigger") != "measured_tan":
        raise ValueError("감지된 피부 톤 태그가 없습니다.")

    added = selected_skin_tags(choice.value)
    record = skin_choice_record(choice, choice.value, "unspecified", (), (), added)
    record["trigger"] = "measured_tan"
    return tags + added, record


def apply_detected_skin_choice(tags, choice, default):
    """피부 태그가 있는 경우: 기본 선택은 그대로 두고 다른 선택만 교체한다."""
    original = tuple(tag for tag in tags if is_skin_tag(tag))
    selected = choice.value or default
    if selected == default:
        record = skin_choice_record(choice, selected, default, original, (), ())
        return tags, record

    inserted = selected_skin_tags(selected)
    result = replace_detected_skin_terms(tags, inserted)
    record = skin_choice_record(choice, selected, default, original, original, inserted)
    return result, record


def replace_detected_skin_terms(tags, inserted):
    """첫 피부 태그 위치에 선택값을 넣고, 나머지 피부 태그만 제거한다."""
    position = next(index for index, tag in enumerate(tags) if is_skin_tag(tag))
    before = tags[:position]
    after = tuple(tag for tag in tags[position:] if not is_skin_tag(tag))
    return before + inserted + after


def skin_choice_record(choice, selected, default, detected, removed, inserted):
    """태그 처리 결과를 기록한다. 측정값으로 선택을 덮어쓰지 않는다."""
    return {
        "selected": selected,
        "default": default,
        "was_default": selected == default,
        "detected_tags": detected,
        "removed_tags": removed,
        "inserted_tags": inserted,
        "measurement": choice.measurement,
        "automatic_replacement": False,
    }


def cheek_centers(points, scores, size):
    """얼굴 좌표에서 양 볼 중심을 반환한다. 부족한 좌표를 코 위치로 대체하지 않는다."""
    points = np.asarray(points, float)
    scores = np.asarray(scores, float)
    validate_face_landmarks(points, scores)

    centers = []
    for outer, inner in ((2, 31), (14, 35)):
        center = checked_cheek_center(points, scores, outer, inner, size)
        centers.append(center)
    return centers


def validate_face_landmarks(points, scores):
    shape_valid = points.shape == (68, 2) and scores.shape == (68,)
    finite = np.isfinite(points).all() and np.isfinite(scores).all()
    if not (shape_valid and finite):
        raise ValueError("양 볼의 얼굴 관절 기록이 없습니다.")


def checked_cheek_center(points, scores, outer, inner, size):
    if min(scores[outer], scores[inner]) < .3:
        raise ValueError("양 볼 관절 신뢰도가 부족합니다.")

    width, height = size
    x, y = np.rint((points[outer] + points[inner]) / 2).astype(int)
    inside = 3 <= x < width - 3 and 3 <= y < height - 3
    if not inside:
        raise ValueError("양 볼 7×7 표본이 이미지 밖입니다.")
    return int(x), int(y)


def measure_cheeks(rgb, points, scores, settings=SkinRecommendationSettings()):
    """이미지·얼굴 좌표를 받아, 볼 색 측정값과 추천 여부를 반환한다."""
    size = (rgb.shape[1], rgb.shape[0])
    centers = cheek_centers(points, scores, size)
    sample = median_cheek_color(rgb, centers)
    return cheek_measurement_record(sample, centers, settings)


def median_cheek_color(rgb, centers):
    """시험과 같은 색 변환·양 볼 7×7 중앙값 계산을 유지한다."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255
    lab[..., 1:] -= 128
    patches = [lab[y-3:y+4, x-3:x+4].reshape(-1, 3) for x, y in centers]
    return np.median(np.concatenate(patches), axis=0)


def cheek_measurement_record(sample, centers, settings):
    hue = math.degrees(math.atan2(float(sample[2]), float(sample[1]))) % 360
    lightness_matches = settings.lightness_min <= sample[0] <= settings.lightness_max
    hue_matches = settings.hue_min <= hue <= settings.hue_max
    # 색 이름 대신 측정한 중앙값으로 견본을 만든다.
    swatch = cv2.cvtColor(sample.reshape(1, 1, 3), cv2.COLOR_LAB2RGB)[0, 0]

    return {
        "status": "measured", "centers": centers, "sample_size": [7, 7],
        "lab": sample.tolist(), "rgb": np.rint(swatch * 255).astype(int).tolist(),
        "hue": hue, "recommend_tan": bool(lightness_matches and hue_matches),
        "rules": asdict(settings), "provisional": True,
    }


def observe_skin_tone(directory, settings=SkinRecommendationSettings()):
    """기존 분석 파일을 받아 측정한다. 실패하면 추천만 생략하고 사유를 반환한다."""
    directory = Path(directory)
    try:
        rgb, pose = read_skin_observation(directory)
        result = measure_cheeks(rgb, pose.get("face_points", []),
                                pose.get("face_scores", []), settings)
        return attach_skin_sources(result, directory)
    except (OSError, ValueError, KeyError, TypeError) as error:
        # 선택 화면은 계속 사용할 수 있어야 하므로, 이 경계에서만 측정 실패를 결과로 바꾼다.
        return {"status": "unavailable", "reason": str(error),
                "recommend_tan": False, "rules": asdict(settings)}


def read_skin_observation(directory):
    pose = json.loads((directory / "pose.json").read_text(encoding="utf-8"))
    if pose.get("person_count") != 1:
        raise ValueError("한 명의 얼굴 관절이 필요합니다.")
    with Image.open(directory / "character.png") as image:
        rgb = np.asarray(image.convert("RGB"))
    return rgb, pose


def attach_skin_sources(result, directory):
    from genai_lab.proportion_inputs import sha
    result["source_sha256"] = sha(directory / "character.png")
    result["pose_sha256"] = sha(directory / "pose.json")
    return result
