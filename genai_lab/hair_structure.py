"""Advisory hair-part contracts derived from user-approved evidence.

The contract separates independently observable hair properties. Missing
evidence remains unknown, and diagnostics never reject or retry a candidate.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from genai_lab.hair_detail_analysis import (
    HAIR_ARRANGEMENT_MARKERS,
    hair_detail_group,
)
from genai_lab.reference_tag_policy import normalize_tag


PART_ORDER = (
    "length",
    "front",
    "side",
    "arrangement",
    "structure",
    "texture",
    "color",
)
PART_LABELS = {
    "length": "overall length",
    "front": "front hair",
    "side": "side hair",
    "arrangement": "tied or arranged hair",
    "structure": "overall silhouette",
    "texture": "hair texture",
    "color": "hair color",
}


def _approved_hair_tags(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        normalize_tag(value)
        for value in values
        if isinstance(value, str)
        and normalize_tag(value)
        and hair_detail_group(value) != "unresolved"
    ))


def resolve_hair_structure(
    approved_character_tags: Iterable[object],
    analysis_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a partial contract without interpreting missing tags as absence."""

    approved = _approved_hair_tags(approved_character_tags)
    grouped = {
        part: tuple(tag for tag in approved if hair_detail_group(tag) == part)
        for part in PART_ORDER
    }
    report = analysis_report if isinstance(analysis_report, Mapping) else {}
    view_names = {
        str(item.get("name"))
        for item in report.get("views", ())
        if isinstance(item, Mapping)
    }
    components = {
        part: {
            "state": "confirmed" if grouped[part] else "unknown",
            "approved_tags": list(grouped[part]),
            "source": (
                "user_approved_character_tags"
                if grouped[part] else "insufficient_evidence"
            ),
        }
        for part in PART_ORDER
    }
    rear_observed = "rear_silhouette_candidate" in view_names
    components["rear_silhouette"] = {
        "state": "observed_candidate" if rear_observed else "unknown",
        "approved_tags": [],
        "source": (
            "face_anchored_observation_roi"
            if rear_observed else "insufficient_evidence"
        ),
        "semantic_identity_confirmed": False,
    }
    confirmed = [part for part in PART_ORDER if grouped[part]]
    return {
        "version": "hair_structure_contract_v1",
        "status": "PARTIAL" if confirmed else "UNRESOLVED",
        "components": components,
        "confirmed_parts": confirmed,
        "approved_hair_tags": list(approved),
        "rear_silhouette_observation_only": True,
        "absence_is_not_failure": True,
        "source": "derived_from_user_approved_tags",
        "enforcement": "soft_guidance_and_diagnostic_only",
    }


def hair_structure_guidance(contract: Mapping[str, Any] | None) -> str:
    """Describe confirmed parts positively; unknown parts stay with the Base."""

    if not isinstance(contract, Mapping):
        return ""
    components = contract.get("components", {})
    if not isinstance(components, Mapping):
        return ""
    phrases = []
    for part in PART_ORDER:
        item = components.get(part, {})
        if not isinstance(item, Mapping) or item.get("state") != "confirmed":
            continue
        tags = [
            normalize_tag(tag)
            for tag in item.get("approved_tags", ())
            if isinstance(tag, str) and normalize_tag(tag)
        ]
        if tags:
            phrases.append(f"{PART_LABELS[part]} ({', '.join(tags)})")
    if not phrases:
        return ""
    return (
        "Preserve the approved hairstyle by part: "
        + "; ".join(phrases)
        + ". Use these as soft guidance and keep unlisted hair parts from "
          "the approved character Base. "
    )


def _compatible_expected(group: str, expected: tuple[str, ...], tag: str) -> bool:
    if tag in expected:
        return True
    if group == "front":
        return any("bangs" in value for value in expected) and "bangs" in tag
    if group == "side":
        return any("sidelock" in value for value in expected) and "sidelock" in tag
    if group == "arrangement":
        return any(
            marker in tag and any(marker in value for value in expected)
            for marker in HAIR_ARRANGEMENT_MARKERS
        )
    return False


def evaluate_hair_structure_diagnostic(
    contract: Mapping[str, Any] | None,
    raw_scores: Mapping[str, float],
    *,
    score_threshold: float = .35,
    conflict_threshold: float = .45,
    minimum_margin: float = .10,
) -> dict[str, Any]:
    """Compare WD evidence per hair part without enforcing an action."""

    if not isinstance(contract, Mapping) or not contract.get("confirmed_parts"):
        return {
            "status": "NOT_EVALUATED",
            "reason": "approved_hair_structure_unresolved",
            "enforced": False,
            "absence_is_not_failure": True,
        }

    scores = {
        normalize_tag(name): float(score)
        for name, score in raw_scores.items()
    }
    grouped_scores: dict[str, dict[str, float]] = {part: {} for part in PART_ORDER}
    for tag, score in scores.items():
        group = hair_detail_group(tag)
        if group in grouped_scores:
            grouped_scores[group][tag] = score

    checks = {}
    has_conflict = False
    has_unresolved = False
    components = contract.get("components", {})
    for part in contract.get("confirmed_parts", ()):
        item = components.get(part, {}) if isinstance(components, Mapping) else {}
        expected = tuple(
            normalize_tag(tag) for tag in item.get("approved_tags", ())
            if isinstance(tag, str)
        )
        observed = grouped_scores.get(part, {})
        compatible = {
            tag: score for tag, score in observed.items()
            if _compatible_expected(part, expected, tag)
        }
        conflicting = {
            tag: score for tag, score in observed.items()
            if tag not in compatible
        }
        expected_tag, expected_score = max(
            compatible.items(), key=lambda pair: pair[1], default=(None, 0.0)
        )
        conflict_tag, conflict_score = max(
            conflicting.items(), key=lambda pair: pair[1], default=(None, 0.0)
        )
        conflict = (
            conflict_score >= conflict_threshold
            and conflict_score >= expected_score + minimum_margin
        )
        unresolved = not conflict and expected_score < score_threshold
        status = "DIAGNOSTIC" if conflict else "UNRESOLVED" if unresolved else "PASS"
        has_conflict = has_conflict or conflict
        has_unresolved = has_unresolved or unresolved
        checks[part] = {
            "status": status,
            "approved_tags": list(expected),
            "strongest_expected": expected_tag,
            "expected_score": expected_score,
            "strongest_conflict": conflict_tag,
            "conflict_score": conflict_score,
        }

    status = "DIAGNOSTIC" if has_conflict else "UNRESOLVED" if has_unresolved else "PASS"
    reason = (
        "possible_hair_part_mismatch" if has_conflict
        else "hair_part_evidence_incomplete" if has_unresolved
        else "semantic_evidence_consistent"
    )
    return {
        "status": status,
        "reason": reason,
        "component_checks": checks,
        "rear_silhouette": {
            "status": "NOT_EVALUATED",
            "reason": "front_view_roi_is_not_semantic_rear_hair",
        },
        "enforced": False,
        "absence_is_not_failure": True,
    }
