"""Evidence/intent/target-plan separation for hairstyle preservation.

The reference analyzer is allowed to be uncertain. Its observations never
become generation conditions until user-approved character tags turn them into
intent. Target coordinates are created only from output-space masks.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from genai_lab.hair_detail_analysis import hair_detail_group
from genai_lab.hair_structure import PART_ORDER
from genai_lab.reference_tag_policy import normalize_tag


PASS1_STRUCTURE_GROUPS = frozenset((
    "length", "front", "side", "arrangement", "structure",
))
PASS1_APPEARANCE_GROUPS = frozenset(("texture", "color"))
PASS2_LOCAL_GROUPS = frozenset(("front", "side", "texture", "color"))
BASE_RETRY_GROUPS = frozenset(("length", "arrangement", "structure"))


def _approved_tags(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        normalize_tag(value)
        for value in values
        if isinstance(value, str)
        and normalize_tag(value)
        and hair_detail_group(value) != "unresolved"
    ))


def _observation_state(item: Mapping[str, Any]) -> str:
    status = str(item.get("status") or "").strip().lower()
    if status == "geometry_conflict":
        return "conflict"
    if status in {"model_candidate", "mask_geometry_candidate"}:
        return "observed_candidate"
    return "unresolved"


def build_hair_observed_evidence(
    analysis_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Preserve detector evidence without promoting it to generation intent."""

    report = analysis_report if isinstance(analysis_report, Mapping) else {}
    observations: dict[str, list[dict[str, Any]]] = {
        part: [] for part in PART_ORDER
    }
    for raw in report.get("evidence", ()):
        if not isinstance(raw, Mapping):
            continue
        tag = normalize_tag(raw.get("tag", ""))
        group = hair_detail_group(tag)
        if not tag or group not in observations:
            continue
        score = raw.get("max_score")
        observations[group].append({
            "tag": tag,
            "state": _observation_state(raw),
            "score": float(score) if isinstance(score, (int, float)) else None,
            "score_semantics": "classifier_signal_not_probability",
            "evidence_views": [
                str(value) for value in raw.get("evidence_views", ())
            ],
            "eligible_for_prompt_candidate": bool(
                raw.get("eligible_for_prompt", False)),
            "source": "hair_detail_analysis",
        })

    components = {}
    for part in PART_ORDER:
        values = observations[part]
        states = {value["state"] for value in values}
        state = (
            "conflict" if "conflict" in states
            else "observed_candidate"
            if "observed_candidate" in states
            else "unresolved"
        )
        components[part] = {
            "state": state,
            "observations": values,
            "absence_confirmed": False,
        }

    rear = report.get("rear_silhouette_observation", {})
    if not isinstance(rear, Mapping):
        rear = {}
    return {
        "version": "hair_observed_evidence_v1",
        "status": (
            "OBSERVED" if any(observations.values()) else "UNRESOLVED"
        ),
        "analysis_status": str(report.get("status") or "unavailable"),
        "components": components,
        "rear_silhouette": {
            "state": str(rear.get("status") or "unresolved"),
            "semantic_identity_confirmed": bool(
                rear.get("semantic_identity_confirmed", False)),
        },
        "candidate_is_not_intent": True,
        "missing_candidate_is_not_absence": True,
    }


def build_hair_intent(
    approved_character_tags: Iterable[object],
) -> dict[str, Any]:
    """Build generation intent only from explicit user-approved hair tags."""

    approved = _approved_tags(approved_character_tags)
    grouped = {
        part: [tag for tag in approved if hair_detail_group(tag) == part]
        for part in PART_ORDER
    }
    components = {
        part: {
            "state": "approved" if grouped[part] else "unspecified",
            "tags": grouped[part],
            "source": (
                "user_approved_character_tags"
                if grouped[part] else "no_user_approved_intent"
            ),
        }
        for part in PART_ORDER
    }
    return {
        "version": "hair_intent_v1",
        "status": "APPROVED" if approved else "UNRESOLVED",
        "components": components,
        "approved_tags": list(approved),
        "unapproved_observations_excluded": True,
    }


def _tags_for_groups(intent: Mapping[str, Any], groups) -> list[str]:
    components = intent.get("components", {})
    if not isinstance(components, Mapping):
        return []
    result = []
    for group in PART_ORDER:
        if group not in groups:
            continue
        item = components.get(group, {})
        if not isinstance(item, Mapping) or item.get("state") != "approved":
            continue
        result.extend(
            normalize_tag(tag) for tag in item.get("tags", ())
            if isinstance(tag, str) and normalize_tag(tag)
        )
    return list(dict.fromkeys(result))


def build_hair_transfer_contract(
    approved_character_tags: Iterable[object],
    analysis_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Separate observations, approved intent, and stage responsibilities."""

    evidence = build_hair_observed_evidence(analysis_report)
    intent = build_hair_intent(approved_character_tags)
    return {
        "version": "hair_transfer_contract_v1",
        "status": intent["status"],
        "observed_evidence": evidence,
        "intent": intent,
        "stage_plan": {
            "pass1_structure_tags": _tags_for_groups(
                intent, PASS1_STRUCTURE_GROUPS),
            "pass1_appearance_tags": _tags_for_groups(
                intent, PASS1_APPEARANCE_GROUPS),
            "pass2_local_eligible_tags": _tags_for_groups(
                intent, PASS2_LOCAL_GROUPS),
            "base_retry_on_mismatch_tags": _tags_for_groups(
                intent, BASE_RETRY_GROUPS),
        },
        "coordinate_policy": {
            "source_coordinates_are_target_coordinates": False,
            "target_anchor": "redetected_output_face_and_hair",
            "source_anchor": "reference_face_and_hair",
        },
        "conditioning_policy": {
            "one_visual_condition_mode_per_run": True,
            "default_visual_condition_mode": "ip_adapter",
            "reference_input": "isolated_hair_on_neutral_background",
            "non_hair_features_excluded": True,
        },
        "uncertainty_policy": {
            "candidate_requires_user_approval": True,
            "unresolved_is_not_absent": True,
            "occluded_is_not_inferred": True,
            "unobserved_is_not_scored_as_mismatch": True,
        },
    }


def build_target_hair_plan(
    transfer_contract: Mapping[str, Any] | None,
    scope_report: Mapping[str, Any] | None,
    partition_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Record a target-space action after output masks have been detected."""

    scope = scope_report if isinstance(scope_report, Mapping) else {}
    partition = (
        partition_report if isinstance(partition_report, Mapping) else {}
    )
    reason = scope.get("reason")
    if scope.get("status") == "SELECTED":
        action = "local_refine"
    elif reason == "pass1_structure_requires_base_regeneration":
        action = "regenerate_base"
    else:
        action = "diagnostic_only"
    return {
        "version": "target_hair_plan_v1",
        "status": "PLANNED" if action != "diagnostic_only" else "UNRESOLVED",
        "action": action,
        "coordinate_space": "generated_output",
        "anchor": "redetected_output_face_and_hair",
        "selected_regions": list(scope.get("selected_regions", ())),
        "deferred_to_base_retry": list(scope.get(
            "deferred_to_base_retry", ())),
        "partition_status": partition.get("status"),
        "partition_semantic_identity_confirmed": bool(
            partition.get("semantic_identity_confirmed", False)),
        "source_coordinates_reused": False,
        "automatic_action_authorized": False,
        "contract_version": (
            transfer_contract.get("version")
            if isinstance(transfer_contract, Mapping) else None
        ),
    }