"""Conservative garment component and connection contracts.

Topology is derived only from user-approved garment tags. Missing tags never
prove that a component is absent, and unresolved topology is never promoted to
a generation constraint.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from genai_lab.reference_tag_policy import normalize_tag


UPPER_TERMS = frozenset((
    "shirt", "blouse", "jacket", "blazer", "coat", "sweater",
    "cardigan", "hoodie", "vest", "top", "crop top", "bra", "bikini top",
))
LOWER_TERMS = frozenset((
    "skirt", "mini skirt", "microskirt", "pants", "trousers", "shorts",
    "leggings", "tights", "pantyhose", "stockings", "thighhighs",
    "underwear", "panties", "briefs", "bikini bottom",
))
ONE_PIECE_TERMS = frozenset((
    "dress", "gown", "bodysuit", "leotard", "one piece swimsuit",
    "one-piece swimsuit",
))
TWO_PIECE_TERMS = frozenset((
    "bikini", "two piece swimsuit", "two-piece swimsuit",
))
AMBIGUOUS_TERMS = frozenset(("swimsuit", "uniform", "suit"))
TOPOLOGIES = frozenset((
    "two_piece", "one_piece", "upper_only", "lower_only", "unresolved",
))
EXPOSED_REGIONS = frozenset((
    "abdomen", "midriff", "side_waist", "chest_cutout", "back",
))
CONNECTIONS = frozenset(("upper_to_lower",))


def _tags(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        normalize_tag(value) for value in values
        if isinstance(value, str) and normalize_tag(value)
    ))


def _matches(tag: str, terms: Iterable[str]) -> bool:
    return any(tag == term or tag.endswith(" " + term) for term in terms)


def _validated_reviewed_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    topology = value.get("topology")
    if topology not in TOPOLOGIES - {"unresolved"}:
        raise ValueError("승인 의상 구조 종류가 올바르지 않습니다.")
    components = value.get("components", ())
    exposed = value.get("required_exposed_regions", ())
    forbidden = value.get("forbidden_connections", ())
    if not isinstance(components, (list, tuple)) or not all(
        isinstance(item, str) and item.strip() for item in components
    ):
        raise ValueError("승인 의상 구성품은 비어 있지 않은 문자열 목록이어야 합니다.")
    if not isinstance(exposed, (list, tuple)) or any(
        item not in EXPOSED_REGIONS for item in exposed
    ):
        raise ValueError("승인 의상 노출 영역이 올바르지 않습니다.")
    if not isinstance(forbidden, (list, tuple)) or any(
        item not in CONNECTIONS for item in forbidden
    ):
        raise ValueError("승인 의상 금지 연결이 올바르지 않습니다.")
    if topology == "two_piece" and "upper_to_lower" not in forbidden:
        raise ValueError("분리 의상은 상의와 하의 연결 금지를 포함해야 합니다.")
    return {
        "version": "garment_topology_contract_v1",
        "status": "CONFIRMED",
        "topology": topology,
        "components": list(dict.fromkeys(normalize_tag(item) for item in components)),
        "component_presence": dict(value.get("component_presence", {})),
        "required_exposed_regions": list(dict.fromkeys(exposed)),
        "forbidden_connections": list(dict.fromkeys(forbidden)),
        "evidence_tags": [],
        "ambiguous_tags": [],
        "source": "explicit_user_review",
        "enforcement": "soft_guidance_and_diagnostic_only",
    }


def resolve_garment_topology(
    approved_tags: Iterable[object],
    approved_detail_tags: Iterable[object] = (),
    reviewed_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve structure without treating missing evidence as absence."""

    if reviewed_contract is not None:
        if not isinstance(reviewed_contract, Mapping):
            raise ValueError("승인 의상 구조는 객체여야 합니다.")
        return _validated_reviewed_contract(reviewed_contract)

    tags = _tags((*approved_tags, *approved_detail_tags))
    upper = tuple(tag for tag in tags if _matches(tag, UPPER_TERMS))
    lower = tuple(tag for tag in tags if _matches(tag, LOWER_TERMS))
    one_piece = tuple(tag for tag in tags if _matches(tag, ONE_PIECE_TERMS))
    two_piece = tuple(tag for tag in tags if _matches(tag, TWO_PIECE_TERMS))
    ambiguous = tuple(tag for tag in tags if _matches(tag, AMBIGUOUS_TERMS))

    topology = "unresolved"
    components: list[str] = []
    presence = {"upper": "unknown", "lower": "unknown", "one_piece": "unknown"}
    exposed: list[str] = []
    forbidden: list[str] = []
    evidence: tuple[str, ...] = ()

    if two_piece or (upper and lower and not one_piece):
        topology = "two_piece"
        components = ["upper", "lower"]
        presence.update(upper="present", lower="present")
        exposed = ["abdomen"] if two_piece else []
        forbidden = ["upper_to_lower"]
        evidence = tuple(dict.fromkeys((*two_piece, *upper, *lower)))
    elif one_piece and not upper and not lower:
        topology = "one_piece"
        components = ["one_piece"]
        presence["one_piece"] = "present"
        evidence = one_piece
    elif upper and not lower and not one_piece:
        topology = "upper_only"
        components = ["upper"]
        presence["upper"] = "present"
        evidence = upper
    elif lower and not upper and not one_piece:
        topology = "lower_only"
        components = ["lower"]
        presence["lower"] = "present"
        evidence = lower

    confirmed = topology != "unresolved"
    return {
        "version": "garment_topology_contract_v1",
        "status": "CONFIRMED" if confirmed else "UNRESOLVED",
        "topology": topology,
        "components": components,
        "component_presence": presence,
        "required_exposed_regions": exposed,
        "forbidden_connections": forbidden,
        "evidence_tags": list(evidence),
        "ambiguous_tags": list(ambiguous),
        "source": "derived_from_user_approved_tags",
        "enforcement": "soft_guidance_and_diagnostic_only",
    }


def topology_guidance(contract: Mapping[str, Any] | None) -> str:
    """Return positive structural guidance only for confirmed topology."""

    if not isinstance(contract, Mapping) or contract.get("status") != "CONFIRMED":
        return ""
    topology = contract.get("topology")
    exposed = set(contract.get("required_exposed_regions", ()))
    if topology == "two_piece":
        sentence = (
            "Keep the approved upper and lower garments as two visually "
            "separate components; do not join them into a one-piece garment."
        )
        if "abdomen" in exposed or "midriff" in exposed:
            sentence += " Keep the abdomen between them visibly uncovered."
        return sentence
    if topology == "one_piece":
        return "Keep the approved garment as one connected garment component."
    if topology == "upper_only":
        return "Apply only the approved upper garment component."
    if topology == "lower_only":
        return "Apply only the approved lower garment component."
    return ""


def evaluate_topology_diagnostic(
    contract: Mapping[str, Any] | None,
    raw_scores: Mapping[str, float],
) -> dict[str, Any]:
    """Advisory WD-tag evidence; it never rejects or retries a candidate."""

    topology = contract.get("topology") if isinstance(contract, Mapping) else None
    if not isinstance(contract, Mapping) or contract.get("status") != "CONFIRMED":
        return {
            "status": "NOT_EVALUATED",
            "reason": "approved_garment_topology_unresolved",
            "enforced": False,
        }
    scores = {normalize_tag(name): float(score) for name, score in raw_scores.items()}
    two_piece_score = max((scores.get(term, 0.0) for term in TWO_PIECE_TERMS), default=0.0)
    one_piece_score = max((scores.get(term, 0.0) for term in ONE_PIECE_TERMS), default=0.0)
    ambiguous_score = max((scores.get(term, 0.0) for term in AMBIGUOUS_TERMS), default=0.0)
    component_checks = {}
    if topology == "two_piece":
        upper_score = max(
            (scores.get(term, 0.0) for term in UPPER_TERMS), default=0.0)
        lower_score = max(
            (scores.get(term, 0.0) for term in LOWER_TERMS), default=0.0)
        component_checks = {
            "upper": {
                "status": "PASS" if upper_score >= 0.35 else "UNRESOLVED",
                "score": upper_score,
            },
            "lower": {
                "status": "PASS" if lower_score >= 0.35 else "UNRESOLVED",
                "score": lower_score,
            },
        }
        mismatch = (
            one_piece_score >= 0.45
            and one_piece_score >= two_piece_score + 0.10
        )
        missing_component_evidence = any(
            check["status"] == "UNRESOLVED"
            for check in component_checks.values()
        )
        unresolved = (
            not mismatch
            and (
                missing_component_evidence
                or (two_piece_score < 0.35 and ambiguous_score >= 0.35)
            )
        )
    elif topology == "one_piece":
        mismatch = (
            two_piece_score >= 0.45
            and two_piece_score >= one_piece_score + 0.10
        )
        unresolved = (
            not mismatch
            and one_piece_score < 0.35
            and ambiguous_score >= 0.35
        )
    else:
        return {
            "status": "NOT_EVALUATED",
            "reason": "component_topology_has_no_connection_rule",
            "expected_topology": topology,
            "enforced": False,
            "absence_is_not_failure": True,
        }
    reason = "semantic_evidence_consistent"
    if mismatch:
        reason = "possible_topology_mismatch"
    elif unresolved:
        reason = (
            "component_evidence_incomplete"
            if component_checks
            else "generic_garment_label_only"
        )
    return {
        "status": "DIAGNOSTIC" if mismatch else "UNRESOLVED" if unresolved else "PASS",
        "reason": reason,
        "expected_topology": topology,
        "component_checks": component_checks,
        "two_piece_score": two_piece_score,
        "one_piece_score": one_piece_score,
        "ambiguous_score": ambiguous_score,
        "enforced": False,
        "absence_is_not_failure": True,
    }
