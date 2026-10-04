"""부위 RGB 측정 결과에서 검토용 색상 문구를 제안한다. 색/그림자의 의미 판정은 아니다."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
import cv2
import numpy as np
from PIL import ImageColor

PART_LABELS = {
    "human_ears": ("human ears", "사람 귀"),
    "animal_ears": ("animal ears", "동물 귀"),
    "ears": ("animal ears", "동물 귀"),
    "tail": ("tail", "꼬리"),
}


@dataclass(frozen=True)
class PartColorDescription:
    part_name: str
    colors: tuple[str, ...]
    status: str
    reasons: tuple[str, ...]
    evidence_json: bytes
    policy_fingerprint: str

    def __post_init__(self):
        if (self.part_name not in PART_LABELS or not isinstance(self.colors, tuple)
                or not isinstance(self.reasons, tuple) or not isinstance(self.evidence_json, bytes)
                or any(not isinstance(c, str) or not re.fullmatch("(?:dark |light )?[a-z]+", c) for c in self.colors)
                or len(set(self.colors)) != len(self.colors)
                or self.status not in ("proposed", "unresolved")
                or bool(self.colors) != (self.status == "proposed")):
            raise ValueError("부위 색상 설명 계약 오류")

    @property
    def prompt_text(self):
        # 색상 단어와 대상 부위를 한 문구로 묶는다. 색 위치/무늬/성별은 추가하지 않는다.
        if not self.colors:
            return ""
        families = {color.split()[-1] for color in self.colors}
        if len(families) == 1:
            family = next(iter(families))
            tones = {color.split()[0] for color in self.colors if " " in color}
            # 밝은/어두운 같은 색은 범위로 표현한다. 무늬나 그라데이션을 뜻하지 않는다.
            phrase = "dark and light " + family if tones == {"dark", "light"} else " and ".join(self.colors)
        else:
            phrase = " and ".join(self.colors)
        return phrase + " " + PART_LABELS[self.part_name][0]

    def record(self):
        return {"part_name": self.part_name, "colors": self.colors, "status": self.status,
                "reasons": self.reasons, "prompt_text": self.prompt_text,
                "evidence": json.loads(self.evidence_json), "policy_fingerprint": self.policy_fingerprint,
                "semantic_accuracy_verified": False}


def description_records(descriptions):
    if not isinstance(descriptions, tuple) or not all(isinstance(d, PartColorDescription) for d in descriptions):
        raise ValueError("불변 부위 색상 설명 목록이 필요합니다.")
    if len({d.part_name for d in descriptions}) != len(descriptions):
        raise ValueError("부위 색상 설명 이름이 중복됩니다.")
    return tuple(d.record() for d in descriptions)


def color_prompt_tags(descriptions):
    description_records(descriptions)
    return tuple(d.prompt_text for d in descriptions if d.status == "proposed")


def bind_color_descriptions(character_tags, descriptions):
    tags = color_prompt_tags(descriptions)
    replaced = {PART_LABELS[d.part_name][0] for d in descriptions if d.status == "proposed"}
    retained = tuple(tag for tag in character_tags if tag not in replaced)
    return (*retained, *tags), tuple(tag for tag in character_tags if tag in replaced)


def select_color_descriptions(all_descriptions, selected_parts):
    """Return explicitly approved proposals only, preserving source order."""
    description_records(all_descriptions)
    if isinstance(selected_parts, str):
        raise TypeError("승인할 부위 이름 목록이 필요합니다.")
    selected = tuple(selected_parts)
    if (len(selected) != len(set(selected))
            or any(name not in PART_LABELS for name in selected)):
        raise ValueError("귀꼬리 색상 승인 부위가 잘못됐습니다.")
    return tuple(
        description for description in all_descriptions
        if description.part_name in selected and description.status == "proposed"
    )


def color_description_summary(descriptions):
    records = description_records(descriptions)
    if not records:
        return ""
    lines = [
        "귀·꼬리 색상 설명 — 부위 보정 전용 / 전체 생성 문구 미전달 "
        "(정답 판정 아님)"
    ]
    lines[0] = (
        "귀꼬리 색상 설명 제안  정답 판정 아님 / "
        "선택한 항목만 생성 조건에 전달"
    )
    for d, record in zip(descriptions, records):
        labels = record["evidence"].get("color_labels", {})
        value = " + ".join(labels.get(c, c) for c in d.colors)
        lines.append(f"{PART_LABELS[d.part_name][1]}: " + (value if d.colors else
                     "판단 보류 / 색상 문구 미전달 (" + ", ".join(d.reasons) + ")"))
    lines.append("색의 위치·줄무늬·그라데이션은 추정하지 않습니다. 잘못된 제안이면 취소하세요.")
    return "\n".join(lines)


class PartColorNamer:
    def __init__(self, vocabulary, *, max_distance=22., minimum_margin=1.5,
                 minimum_area_ratio=.10, minimum_coverage=.85, maximum_colors=3,
                 dark_lightness=45., light_lightness=80.):
        values = (max_distance, minimum_margin, minimum_area_ratio, minimum_coverage,
                  dark_lightness, light_lightness)
        if (not all(np.isfinite(v) for v in values) or max_distance <= 0 or minimum_margin <= 0
                or not 0 < minimum_area_ratio <= 1 or not 0 < minimum_coverage <= 1
                or type(maximum_colors) is not int or not 1 <= maximum_colors <= 8
                or not 0 < dark_lightness < light_lightness < 100):
            raise ValueError("색상 설명 정책 설정 오류")
        self.policy = dict(max_distance=max_distance, minimum_margin=minimum_margin,
                           minimum_area_ratio=minimum_area_ratio, minimum_coverage=minimum_coverage,
                           maximum_colors=maximum_colors, dark_lightness=dark_lightness,
                           light_lightness=light_lightness)
        families = vocabulary.get("families", {})
        if not 2 <= len(families) <= 32:
            raise ValueError("색상 사전은 2~32개 색상군이 필요합니다.")
        names, rgbs, labels = [], [], {}
        for name, spec in families.items():
            if not re.fullmatch("[a-z]+", name) or not isinstance(spec.get("samples"), list) or not 1 <= len(spec["samples"]) <= 32:
                raise ValueError("색상 사전 항목 오류")
            labels[name] = str(spec.get("ko", name))
            for sample in spec["samples"]:
                if sample not in ImageColor.colormap:
                    raise ValueError("등록된 CSS 색상 이름이 필요합니다.")
                names.append(name)
                rgbs.append(ImageColor.getrgb(sample))
        self.names, self.labels = tuple(names), dict(labels)
        self.family_names = tuple(labels)
        for name, label in labels.items():
            self.labels["dark " + name] = "어두운 " + label
            self.labels["light " + name] = "밝은 " + label
        self.lab = cv2.cvtColor(np.array([rgbs], dtype=np.float32)/255., cv2.COLOR_RGB2Lab)[0]
        self.fingerprint = hashlib.sha256(json.dumps(
            {"vocabulary": vocabulary, "policy": self.policy}, sort_keys=True,
            ensure_ascii=False, allow_nan=False).encode()).hexdigest()

    def describe(self, part_name, evidence):
        if part_name not in PART_LABELS:
            raise ValueError("이번 색상 연결 범위는 귀와 꼬리입니다.")
        details = {"policy": self.policy, "color_labels": self.labels, "clusters": [],
                   "coverage": 0., "method": "nearest CSS family in OpenCV Lab; not probability"}
        def result(colors=(), reason=""):
            return PartColorDescription(part_name, tuple(colors), "proposed" if colors else "unresolved",
                (reason,) if reason else (), json.dumps(details, ensure_ascii=False, allow_nan=False).encode(),
                self.fingerprint)
        if evidence is None:
            return result(reason="색상 측정 없음")
        palette = np.asarray(evidence.palette_rgb, dtype=float)
        ratios = np.asarray(evidence.area_ratios, dtype=float)
        if not evidence.pixel_count:
            return result(reason="유효 픽셀 없음")
        if (palette.ndim != 2 or palette.shape[1:] != (3,) or ratios.shape != (len(palette),)
                or not 1 <= len(palette) <= 8 or not np.isfinite(palette).all()
                or np.any((palette < 0) | (palette > 255)) or not np.isfinite(ratios).all()
                or np.any((ratios < 0) | (ratios > 1)) or not np.isclose(ratios.sum(), 1., atol=1e-5)):
            raise ValueError("부위 색상 측정값 형식 오류")
        lab = cv2.cvtColor(palette[None].astype(np.float32)/255., cv2.COLOR_RGB2Lab)[0]
        totals, significant_ambiguous = {}, False
        for rgb, ratio, color in zip(palette, ratios, lab):
            distances = np.linalg.norm(self.lab-color, axis=1)
            ranked = sorted((float(min(d for d, n in zip(distances, self.names) if n == name)), name)
                            for name in self.family_names)
            distance, name = ranked[0]
            gap = ranked[1][0]-distance
            accepted = distance <= self.policy["max_distance"] and gap >= self.policy["minimum_margin"]
            details["clusters"].append({"rgb": rgb.tolist(), "area_ratio": float(ratio),
                "candidate": name, "distance": distance, "runner_up": ranked[1][1], "margin": gap,
                "accepted": accepted})
            if accepted:
                tone = ("dark " if color[0] < self.policy["dark_lightness"] else
                        "light " if color[0] > self.policy["light_lightness"] else "")
                # 흑/백의 밝기 수식은 중복이다. 명암을 그림자라고 판정하지 않는다.
                label = (tone if name not in ("black", "white") else "") + name
                details["clusters"][-1]["description_color"] = label
                totals[label] = totals.get(label, 0.) + float(ratio)
            elif ratio >= self.policy["minimum_area_ratio"]:
                significant_ambiguous = True
        selected = sorted(((ratio, name) for name, ratio in totals.items()
                           if ratio >= self.policy["minimum_area_ratio"]), reverse=True)
        details["coverage"] = sum(ratio for ratio, name in selected)
        if significant_ambiguous:
            return result(reason="주요 색상 이름 경계가 모호함")
        if details["coverage"] < self.policy["minimum_coverage"]:
            return result(reason="색상 설명의 측정 영역 포함률 부족")
        if len(selected) > self.policy["maximum_colors"]:
            return result(reason="복합색을 짧은 설명으로 축약하기 어려움")
        return result(tuple(name for ratio, name in selected))


def describe_input_part_colors(color_evidence, settings, root):
    if not settings.get("enabled", False):
        return ()
    path = Path(settings.get("vocabulary_path", "configs/part_color_vocabulary.json"))
    vocabulary = json.loads((path if path.is_absolute() else Path(root)/path).read_text(encoding="utf-8"))
    keys = ("max_distance", "minimum_margin", "minimum_area_ratio", "minimum_coverage",
            "maximum_colors", "dark_lightness", "light_lightness")
    namer = PartColorNamer(vocabulary, **{k: settings[k] for k in keys if k in settings})
    part_names = tuple(settings.get("part_names", ("ears", "tail")))
    if (not part_names or len(part_names) != len(set(part_names))
            or any(name not in PART_LABELS for name in part_names)):
        raise ValueError("색상 설명 대상 부위 이름이 잘못됐습니다.")
    # 부재를 단정하지 않고, 측정 누락도 미전달 상태로 표시한다.
    return tuple(
        namer.describe(name, color_evidence.get(name))
        for name in part_names
    )
