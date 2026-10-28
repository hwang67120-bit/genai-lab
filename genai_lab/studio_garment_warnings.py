"""의상 검사는 참고용이다. 승인 태그나 생성 문구를 자동 수정하지 않는다."""
from dataclasses import dataclass
from pathlib import Path
import json
import re

DEFAULT_RULES = Path(__file__).resolve().parents[1] / "configs/studio_garment_warnings.json"


def normalized(text):
    return " ".join(text.lower().replace("_", " ").split())


def contains_term(tag, term):
    return re.search(r"(?<!\w)" + re.escape(normalized(term)) + r"(?!\w)", normalized(tag)) is not None


@dataclass(frozen=True)
class GarmentWarningSettings:
    negative_overlap: bool
    rules: tuple[dict, ...]

    @classmethod
    def load(cls, path=DEFAULT_RULES):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if set(data) != {"negative_overlap", "rules"} or type(data["negative_overlap"]) is not bool:
            raise ValueError("의상 경고 설정 형식이 잘못됐습니다.")
        identifiers = set()
        for rule in data["rules"]:
            terms = rule.get("terms")
            keys = {"id", "message", "terms"} if terms is not None else {"id", "message", "left", "right"}
            if set(rule) != keys or not isinstance(rule["id"], str) or rule["id"] in identifiers:
                raise ValueError("의상 경고 규칙이 잘못됐거나 중복됐습니다.")
            if not isinstance(rule["message"], str) or not rule["message"].strip():
                raise ValueError("의상 경고 안내가 비어 있습니다.")
            for key in keys - {"id", "message"}:
                values = rule[key]
                if not isinstance(values, list) or not values or any(not isinstance(t, str) or not t.strip() for t in values):
                    raise ValueError("의상 경고 단어 목록이 잘못됐습니다.")
            identifiers.add(rule["id"])
        return cls(data["negative_overlap"], tuple(data["rules"]))


def garment_warnings(tags, negative, settings=None):
    settings = settings or GarmentWarningSettings.load()
    warnings = []
    for rule in settings.rules:
        if "terms" in rule:
            matched = [tag for tag in tags if any(contains_term(tag, t) for t in rule["terms"])]
        else:
            left = [tag for tag in tags if any(contains_term(tag, t) for t in rule["left"])]
            right = [tag for tag in tags if any(contains_term(tag, t) for t in rule["right"])]
            matched = list(dict.fromkeys(left + right)) if left and right else []
        if matched:
            warnings.append({"rule": rule["id"], "tags": matched, "message": rule["message"]})
    if settings.negative_overlap:
        for term in dict.fromkeys(normalized(t) for t in negative.split(",") if t.strip()):
            matched = [tag for tag in tags if contains_term(tag, term)]
            if matched:
                warnings.append({"rule": "negative_overlap", "term": term, "tags": matched,
                    "message": f"부정 문구와 겹침: {term} — 의상이 제대로 적용되지 않을 수 있음"})
    return warnings


def garment_review(tags, negative, settings):
    return {"garment_tags": list(tags), "warnings": garment_warnings(tags, negative, settings),
            "negative": negative, "automatic_removal": False,
            "warning_settings": {"negative_overlap": settings.negative_overlap, "rules": settings.rules}}
