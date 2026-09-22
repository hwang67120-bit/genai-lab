"""Gate hair inpainting when one reference view lacks the required geometry."""

from dataclasses import dataclass, fields
from pathlib import Path


FRONT_TAGS = ("looking_at_viewer", "facing_viewer")
SIDE_TAGS = ("profile", "from_side", "facing_to_the_side")
BACK_TAGS = ("from_behind", "facing_away", "back_focus", "looking_back")


@dataclass(frozen=True)
class HairViewGateSettings:
    enabled: bool = True
    score_threshold: float = .35
    minimum_margin: float = .08

    def record(self):
        return {
            "version": "hair_view_compatibility_v1",
            **self.__dict__,
            "compatible_views": {
                "front": ["front", "front_three_quarter"],
                "front_three_quarter": ["front", "front_three_quarter"],
                "side": ["side"],
                "back": ["back"],
            },
            "unresolved_policy": "skip_inpaint_try_next_candidate",
        }


def classify_hair_view(raw_scores, settings):
    """Classify only explicit WD14 camera/head-view evidence."""
    scores = dict(raw_scores or ())

    def maximum(names):
        return max((float(scores.get(name, 0.0)) for name in names), default=0.0)

    grouped = {
        "front": maximum(FRONT_TAGS),
        "side": maximum(SIDE_TAGS),
        "back": maximum(BACK_TAGS),
    }
    threshold = settings.score_threshold
    margin = settings.minimum_margin
    front, side, back = grouped["front"], grouped["side"], grouped["back"]
    if back >= threshold and back >= max(front, side) + margin:
        view, reason = "back", None
    elif front >= threshold and side >= threshold:
        view, reason = "front_three_quarter", None
    elif front >= threshold and front >= max(side, back):
        view, reason = "front", None
    elif side >= threshold and side >= max(front, back) + margin:
        view, reason = "side", None
    else:
        view, reason = "unresolved", "view_scores_ambiguous_or_below_threshold"
    evidence = {
        name: float(scores.get(name, 0.0))
        for name in (*FRONT_TAGS, *SIDE_TAGS, *BACK_TAGS)
    }
    return {
        "view": view,
        "reason": reason,
        "group_scores": grouped,
        "evidence_scores": evidence,
    }


def compare_hair_views(reference_report, candidate_report, settings):
    reference_view = reference_report["view"]
    candidate_view = candidate_report["view"]
    allowed = settings.record()["compatible_views"].get(reference_view, [])
    if reference_view == "unresolved":
        status, reason = "unresolved", "reference_view_unresolved"
    elif candidate_view == "unresolved":
        status, reason = "unresolved", "candidate_view_unresolved"
    elif candidate_view not in allowed:
        status, reason = "incompatible", "unobserved_hair_geometry_required"
    else:
        status, reason = "compatible", None
    return {
        **settings.record(),
        "status": status,
        "reason": reason,
        "reference": reference_report,
        "candidate": candidate_report,
        "allowed_candidate_views": allowed,
        "inpaint_allowed": status == "compatible",
    }


def resolve_hair_view_gate(config):
    raw = config.get("reference_analysis", {}).get("hair_view_gate", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("헤어 시점 호환성 설정은 객체여야 합니다.")
    allowed = {field.name for field in fields(HairViewGateSettings)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"알 수 없는 헤어 시점 호환성 설정: {sorted(unknown)}")
    settings = HairViewGateSettings(**raw)
    for name, value in (
            ("score_threshold", settings.score_threshold),
            ("minimum_margin", settings.minimum_margin)):
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not 0 <= float(value) <= 1):
            raise ValueError(f"헤어 시점 {name}은 0~1 숫자여야 합니다.")
    return settings


class WdHairViewAnalyzer:
    """Reuse the configured WD14 model and retain relevant raw scores."""

    def __init__(self, config):
        from genai_lab.clothing_analysis import (
            ClothingDesignAnalysisSettings,
            WdTagSession,
        )
        configured = config.get("clothing_design_analysis", {})
        allowed = {field.name for field in fields(ClothingDesignAnalysisSettings)}
        values = {key: value for key, value in configured.items() if key in allowed}
        if "cache_dir" in values:
            values["cache_dir"] = Path(values["cache_dir"])
        values["execution_provider"] = "CPUExecutionProvider"
        self.session = WdTagSession(ClothingDesignAnalysisSettings(**values))

    def analyze_pair(self, reference_image, candidate_image, settings,
                     check_running=None):
        if check_running is not None:
            check_running()
        reference = self.session.analyze(reference_image)
        if check_running is not None:
            check_running()
        candidate = self.session.analyze(candidate_image)
        return compare_hair_views(
            classify_hair_view(reference.raw_general_scores, settings),
            classify_hair_view(candidate.raw_general_scores, settings),
            settings,
        )

    def close(self):
        self.session.close()
