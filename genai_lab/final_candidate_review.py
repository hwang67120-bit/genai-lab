"""Stage 8 comparison evidence and explicit user decision policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

EVIDENCE_VERSION = "final_candidate_review_evidence_v1"
DECISION_VERSION = "final_candidate_review_decision_v1"
APPROVED = "approved"
APPROVED_WITH_REFINEMENT = "approved_with_refinement"
REJECTED = "rejected"

REFINEMENT_TARGET_LABELS = {
    "character": "캐릭터 정체성", "hair": "헤어", "body": "체형·비율",
    "garment": "의상 형태", "color": "의상 색상",
    "animal_ears": "동물귀", "tail": "꼬리", "structure": "인체·공간 구조",
}

REJECTION_POLICIES = {
    "pixel_corruption": ("픽셀 손상 또는 이미지 깨짐", "retry_new_seed", True, False, None, False),
    "structure_failure": ("신체·공간 구조 붕괴", "retry_new_seed", True, False, None, False),
    "multiple_people": ("인물이 한 명이 아님", "retry_new_seed", True, False, None, False),
    "character_identity_mismatch": ("캐릭터·얼굴 정체성 불일치", "retry_same_seed_guidance", True, True, "character", False),
    "hair_mismatch": ("헤어 형태·색상 불일치", "retry_same_seed_guidance", True, True, "hair", False),
    "garment_silhouette_mismatch": ("의상 종류·실루엣 불일치", "retry_same_seed_guidance", True, True, "garment", False),
    "garment_color_mismatch": ("의상 색상·소재 불일치", "retry_same_seed_guidance", True, True, "color", False),
    "unexpected_optional_part": ("요청하지 않은 동물귀·꼬리·장식 생성", "return_to_input_review", True, True, None, True),
    "reference_analysis_error": ("참조 분석·태그·마스크 오류", "return_to_input_review", True, True, None, True),
    "aesthetic_preference": ("기술적 오류는 없지만 원하는 인상이 아님", "retry_new_seed", True, False, None, False),
    "stop_without_retry": ("현재 실행을 종료하고 재시도하지 않음", "stop", True, False, None, False),
}


class FinalCandidateReviewError(ValueError):
    """Stage 8 evidence or decision is incomplete."""


@dataclass(frozen=True)
class FinalReviewEvidence:
    candidate_id: str
    candidate_number: int
    result_kind: str
    selected_engine: str | None
    sources: dict[str, str | None]
    technical_safety: dict[str, Any]
    comparisons: tuple[dict[str, Any], ...]
    guidance: dict[str, Any]
    refinement_targets: tuple[str, ...]
    report_path: str
    output_directory: str
    created_at: str

    def record(self) -> dict[str, Any]:
        value = asdict(self)
        value["version"] = EVIDENCE_VERSION
        value["comparisons"] = [dict(item) for item in self.comparisons]
        value["refinement_targets"] = list(self.refinement_targets)
        return value


@dataclass(frozen=True)
class FinalReviewDecision:
    candidate_id: str
    decision: str
    reason_code: str | None
    reason_label_ko: str | None
    next_action: str
    preserve_base: bool
    preserve_seed: bool
    retry_scope: str | None
    requires_input_revision: bool
    refinement_targets: tuple[str, ...]
    eligible_for_save: bool
    excluded_from_final_selection: bool
    training_use_approved: bool
    automatic_retry: bool
    decided_at: str
    evidence_sha256: str

    def record(self) -> dict[str, Any]:
        value = asdict(self)
        value["version"] = DECISION_VERSION
        value["refinement_targets"] = list(self.refinement_targets)
        return value


def _percentage(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= result <= 1.0:
        result *= 100.0
    return round(max(0.0, min(100.0, result)), 2)


def _selected_attempt(report: Mapping[str, Any], engine: str | None) -> Mapping[str, Any]:
    attempts = report.get("attempts", {})
    if not isinstance(attempts, Mapping):
        return {}
    if engine and isinstance(attempts.get(engine), Mapping):
        return attempts[engine]
    return next((item for item in attempts.values() if isinstance(item, Mapping)), {})


def _guidance_trace(report: Mapping[str, Any]) -> dict[str, Any]:
    trace = report.get("guidance_trace")
    controls = ["temporal_guidance", "spatial_guidance", "staged_guidance"]
    if not isinstance(trace, Mapping):
        return {
            "status": "not_reported", "requested": [], "applied": [],
            "unsupported": [], "not_reported": controls,
            "note_ko": "1~7단계 유도 실행 보고서가 없어 적용된 것으로 간주하지 않습니다.",
        }

    def names(key: str) -> list[str]:
        value = trace.get(key, ())
        if isinstance(value, Mapping):
            return [str(name) for name, enabled in value.items() if enabled]
        if isinstance(value, str):
            return [value]
        return [str(name) for name in value] if isinstance(value, Iterable) else []

    return {
        "status": str(trace.get("status", "reported")),
        "requested": names("requested"), "applied": names("applied"),
        "unsupported": names("unsupported"), "not_reported": names("not_reported"),
        "profile": trace.get("profile"), "phases": trace.get("phases", []),
        "note_ko": trace.get("note_ko"),
    }


def build_final_review_evidence(
    result: Any, selection: Any, *, request: Any | None = None
) -> FinalReviewEvidence:
    report = getattr(result, "report", None)
    if not isinstance(report, Mapping):
        raise FinalCandidateReviewError("최종 정밀화 보고서가 없습니다.")
    engine = getattr(result, "selected_engine", None)
    attempt = _selected_attempt(report, engine)
    gate = attempt.get("gate", {})
    gate = gate if isinstance(gate, Mapping) else {}
    similarity = gate.get("similarity_percentages", attempt.get("similarity", {}))
    similarity = similarity if isinstance(similarity, Mapping) else {}
    color = gate.get("color_percentage")
    if color is None and isinstance(attempt.get("color"), Mapping):
        color = attempt["color"].get("result")
    values = {**similarity, "color": color}
    specs = (
        ("character", "캐릭터 정체성", "character_reference"),
        ("hair", "헤어 형태·색상", "character_reference"),
        ("garment", "의상 종류·실루엣", "garment_reference"),
        ("color", "의상 색상", "garment_reference"),
        ("vibe", "전체 인상", "combined_references"),
    )
    comparisons = tuple({
        "component": key, "label_ko": label, "reference": reference,
        "value": _percentage(values.get(key)),
        "unit": "similarity_index_percentage", "probability": False,
        "status": "measured" if _percentage(values.get(key)) is not None else "not_calculable",
    } for key, label, reference in specs)

    person = report.get("person_count_diagnostic", {})
    person = person if isinstance(person, Mapping) else {}
    checks = gate.get("checks", {})
    checks = checks if isinstance(checks, Mapping) else {}
    blocking = gate.get("blocking_checks", ())
    blocking = (blocking,) if isinstance(blocking, str) else blocking
    technical = {
        "status": "PASS" if getattr(result, "status", None) in {"PASS", "BASE_LOCKED"} else "FAIL",
        "native_status": getattr(result, "status", None),
        "gate_status": gate.get("status"), "checks": dict(checks),
        "blocking_checks": list(blocking or ()),
        "person_count": person.get("detected_count"),
        "person_count_status": person.get("status", "NOT_REPORTED"),
        "person_count_blocking": bool(person.get("return_blocked", False)),
    }

    smoke = attempt.get("smoke_report", {})
    inputs = smoke.get("inputs", {}) if isinstance(smoke, Mapping) else {}
    character_input = inputs.get("character", {}) if isinstance(inputs, Mapping) else {}
    garment_input = inputs.get("garment", {}) if isinstance(inputs, Mapping) else {}
    character_path = character_input.get("path") if isinstance(character_input, Mapping) else None
    garment_path = garment_input.get("path") if isinstance(garment_input, Mapping) else None
    sources = {
        "character_reference": str(getattr(request, "reference_image_name", "") or "") or None,
        "garment_reference": str(garment_path or getattr(selection, "garment_reference", "") or "") or None,
        "approved_base": str(character_path or getattr(selection, "base_image", "") or "") or None,
        "final_candidate": str(getattr(result, "selected_image_path", "") or "") or None,
    }
    targets = gate.get("refinement_targets", report.get("decision", {}).get("refinement_targets", ()))
    targets = (targets,) if isinstance(targets, str) else targets
    number = int(getattr(selection, "candidate_number", 0))
    return FinalReviewEvidence(
        candidate_id=f"candidate-{number}", candidate_number=number,
        result_kind="refined_candidate" if getattr(result, "status", None) == "PASS"
                    else "approved_base_fallback",
        selected_engine=engine, sources=sources, technical_safety=technical,
        comparisons=comparisons, guidance=_guidance_trace(report),
        refinement_targets=tuple(str(item) for item in (targets or ()) if str(item)),
        report_path=str(getattr(result, "report_path")),
        output_directory=str(getattr(result, "output_directory")),
        created_at=datetime.now().astimezone().isoformat(),
    )


def build_external_review_evidence(
    candidate: Any,
    *,
    character_reference: Path | str | None,
    garment_reference: Path | str | None,
    output_directory: Path | str,
) -> FinalReviewEvidence:
    """Create honest Stage 8 evidence for an imported, unverified image."""
    source = getattr(candidate, "design_reference_record", None)
    source = source if isinstance(source, Mapping) else {}
    source_name = str(source.get("source_file_name", "external-candidate"))
    identity = hashlib.sha256(source_name.encode("utf-8")).hexdigest()[:12]
    comparisons = tuple({
        "component": key,
        "label_ko": label,
        "reference": reference,
        "value": None,
        "unit": "similarity_index_percentage",
        "probability": False,
        "status": "not_calculable",
    } for key, label, reference in (
        ("character", "캐릭터 정체성", "character_reference"),
        ("hair", "헤어 형태·색상", "character_reference"),
        ("garment", "의상 종류·실루엣", "garment_reference"),
        ("color", "의상 색상", "garment_reference"),
        ("vibe", "전체 인상", "combined_references"),
    ))
    return FinalReviewEvidence(
        candidate_id=f"external-{identity}",
        candidate_number=0,
        result_kind="external_import_unverified",
        selected_engine=None,
        sources={
            "character_reference": str(character_reference) if character_reference else None,
            "garment_reference": str(garment_reference) if garment_reference else None,
            "approved_base": None,
            "final_candidate": source_name,
        },
        technical_safety={
            "status": "NOT_RUN",
            "native_status": "external_import_unverified",
            "gate_status": "NOT_RUN",
            "checks": {},
            "blocking_checks": [],
            "person_count": None,
            "person_count_status": "NOT_RUN",
            "person_count_blocking": False,
        },
        comparisons=comparisons,
        guidance={
            "status": "not_applicable",
            "requested": [],
            "applied": [],
            "unsupported": [],
            "not_reported": [],
            "note_ko": "외부 편집 후보이므로 로컬 유도 과정과 자동 게이트를 실행하지 않았습니다.",
        },
        refinement_targets=(),
        report_path="",
        output_directory=str(Path(output_directory) / f"external-{identity}"),
        created_at=datetime.now().astimezone().isoformat(),
    )


def evidence_sha256(evidence: FinalReviewEvidence | Mapping[str, Any]) -> str:
    record = evidence.record() if isinstance(evidence, FinalReviewEvidence) else dict(evidence)
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def create_final_review_decision(
    evidence: FinalReviewEvidence, decision: str, *,
    reason_code: str | None = None, refinement_targets: Iterable[str] = (),
) -> FinalReviewDecision:
    if decision not in {APPROVED, APPROVED_WITH_REFINEMENT, REJECTED}:
        raise FinalCandidateReviewError(f"지원하지 않는 최종 결정입니다: {decision}")
    targets = tuple(dict.fromkeys(str(item) for item in refinement_targets if str(item)))
    unknown = [item for item in targets if item not in REFINEMENT_TARGET_LABELS]
    if unknown:
        raise FinalCandidateReviewError(f"지원하지 않는 미세조정 대상입니다: {unknown}")

    reason_label = None
    if decision == REJECTED:
        if reason_code not in REJECTION_POLICIES:
            raise FinalCandidateReviewError("거절 사유를 선택해야 합니다.")
        reason_label, action, keep_base, keep_seed, scope, revise = REJECTION_POLICIES[str(reason_code)]
        targets = targets or ((scope,) if scope else ())
        eligible, excluded = False, True
    elif decision == APPROVED_WITH_REFINEMENT:
        targets = targets or evidence.refinement_targets
        if not targets:
            raise FinalCandidateReviewError("조건부 승인에는 미세조정 대상이 필요합니다.")
        reason_code, action, keep_base, keep_seed, scope, revise = (
            None, "save_or_queue_refinement", True, True, None, False
        )
        eligible, excluded = True, False
    else:
        reason_code, action, keep_base, keep_seed, scope, revise = (
            None, "save_result", True, True, None, False
        )
        targets, eligible, excluded = (), True, False

    return FinalReviewDecision(
        candidate_id=evidence.candidate_id, decision=decision,
        reason_code=reason_code, reason_label_ko=reason_label,
        next_action=action, preserve_base=keep_base, preserve_seed=keep_seed,
        retry_scope=scope, requires_input_revision=revise,
        refinement_targets=targets, eligible_for_save=eligible,
        excluded_from_final_selection=excluded, training_use_approved=False,
        automatic_retry=False, decided_at=datetime.now().astimezone().isoformat(),
        evidence_sha256=evidence_sha256(evidence),
    )


def _write(path: Path, record: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)
    return path


def write_final_review_evidence(evidence: FinalReviewEvidence, output_directory: Path | str) -> Path:
    return _write(Path(output_directory) / "stage8-review-evidence.json", evidence.record())


def write_final_review_decision(decision: FinalReviewDecision, output_directory: Path | str) -> Path:
    return _write(Path(output_directory) / "stage8-user-decision.json", decision.record())


def load_final_review_evidence(path: Path | str) -> FinalReviewEvidence:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("version") != EVIDENCE_VERSION:
        raise FinalCandidateReviewError("지원하지 않는 8단계 증거 버전입니다.")
    return FinalReviewEvidence(
        candidate_id=str(record["candidate_id"]),
        candidate_number=int(record["candidate_number"]),
        result_kind=str(record["result_kind"]),
        selected_engine=record.get("selected_engine"),
        sources=dict(record.get("sources", {})),
        technical_safety=dict(record.get("technical_safety", {})),
        comparisons=tuple(dict(item) for item in record.get("comparisons", ())),
        guidance=dict(record.get("guidance", {})),
        refinement_targets=tuple(record.get("refinement_targets", ())),
        report_path=str(record["report_path"]),
        output_directory=str(record["output_directory"]),
        created_at=str(record["created_at"]),
    )
