"""Pre-ranking semantic conflicts against the user-approved condition."""

from dataclasses import dataclass, fields
from pathlib import Path

from genai_lab.reference_tag_policy import normalize_tag
from genai_lab.garment_topology import evaluate_topology_diagnostic
from genai_lab.hair_structure import evaluate_hair_structure_diagnostic


ANIMAL_SPECIES = ("cat", "dog", "fox", "wolf", "raccoon", "bear", "rabbit")
LOWER_BODY_GARMENTS = (
    "pants", "trousers", "shorts", "leggings", "tights", "pantyhose",
    "thighhighs", "stockings",
)
LOWER_BODY_EQUIVALENTS = {
    "pants": frozenset(("pants", "trousers")),
    "trousers": frozenset(("pants", "trousers")),
}


def _approved_lower_body_labels(tags):
    """Map approved descriptive tags to detector labels without broadening scope."""

    approved = set()
    for value in tags:
        tag = normalize_tag(value)
        if not tag:
            continue
        for label in LOWER_BODY_GARMENTS:
            if tag == label or tag.endswith(" " + label):
                approved.update(LOWER_BODY_EQUIVALENTS.get(label, (label,)))
    return approved


@dataclass(frozen=True)
class CandidateSemanticGateSettings:
    # 기존/테스트 설정은 동작을 바꾸지 않는다. 실제 프로필에서 명시적으로 켠다.
    enabled: bool = False
    conflict_threshold: float = .45
    minimum_margin: float = .10
    novel_garment_threshold: float = .55
    # 성별 분류기는 외형 진단에만 사용한다. 후보 반환이나 재시드를 막지 않는다.
    enforce_gender: bool = False

    def record(self):
        return {
            "version": "candidate_semantic_gate_v1",
            **self.__dict__,
            "policy": "explicit_conflict_before_similarity_ranking",
            "unresolved_policy": "manual_review_not_verified",
        }


def resolve_candidate_semantic_gate(config):
    raw = config.get("candidate_semantic_gate", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("후보 의미 게이트 설정은 객체여야 합니다.")
    allowed = {field.name for field in fields(CandidateSemanticGateSettings)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"알 수 없는 후보 의미 게이트 설정: {sorted(unknown)}")
    settings = CandidateSemanticGateSettings(**raw)
    for name in ("conflict_threshold", "minimum_margin", "novel_garment_threshold"):
        value = getattr(settings, name)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not 0 <= float(value) <= 1):
            raise ValueError(f"후보 의미 게이트 {name}은 0~1 숫자여야 합니다.")
    if type(settings.enforce_gender) is not bool:
        raise ValueError("후보 의미 게이트 enforce_gender는 bool이어야 합니다.")
    return settings


def _normalized_scores(raw_scores):
    items = raw_scores.items() if isinstance(raw_scores, dict) else (raw_scores or ())
    return {
        normalize_tag(name): float(score)
        for name, score in items
    }


def _specific_species(tags, suffix):
    normalized = {normalize_tag(tag) for tag in tags}
    return next((species for species in ANIMAL_SPECIES
                 if f"{species} {suffix}" in normalized), None)


def compare_lower_body_transition(base_report, final_report):
    """Separate inherited Base contamination from FLUX-introduced garments."""

    def conflicts(report):
        check = (
            (report or {}).get("checks", {})
            .get("unapproved_lower_body_garment", {})
        )
        values = check.get("detected_conflicts", {})
        return {
            normalize_tag(name): float(score)
            for name, score in values.items()
        } if isinstance(values, dict) else {}

    base = conflicts(base_report)
    final = conflicts(final_report)
    inherited = {
        name: score for name, score in final.items()
        if name in base
    }
    introduced = {
        name: score for name, score in final.items()
        if name not in base
    }
    resolved = {
        name: score for name, score in base.items()
        if name not in final
    }
    status = (
        "PASS" if not final
        else "MIXED" if inherited and introduced
        else "INHERITED_FROM_BASE" if inherited
        else "INTRODUCED_BY_REFINEMENT"
    )
    return {
        "version": "lower_body_transition_v1",
        "status": status,
        "base_conflicts": base,
        "final_conflicts": final,
        "inherited_conflicts": inherited,
        "introduced_conflicts": introduced,
        "resolved_conflicts": resolved,
        "base_is_not_neutral_when_inherited": bool(inherited),
        "final_violation_remains": bool(final),
    }


def evaluate_candidate_semantics(approved_run, raw_scores, settings, *, stage='final'):
    """Reject only high-confidence contradictions; absence is not proof."""
    scores = _normalized_scores(raw_scores)
    violations = []
    unresolved = []
    diagnostics = []
    checks = {}

    gender = approved_run.get("gender", "unspecified")
    if gender in {"male", "female"}:
        expected = "1boy" if gender == "male" else "1girl"
        conflict = "1girl" if gender == "male" else "1boy"
        expected_score = scores.get(expected, 0.0)
        conflict_score = scores.get(conflict, 0.0)
        failed = (
            conflict_score >= settings.conflict_threshold
            and conflict_score >= expected_score + settings.minimum_margin
        )
        # 성별 분류 점수는 체형을 직접 측정하지 못하므로 진단으로만 남긴다.
        # 명시적으로 재활성화하지 않는 한 후보 차단이나 재시드 근거가 아니다.
        enforced = settings.enforce_gender
        checks["gender"] = {
            "status": ("FAIL" if failed and enforced else
                       "DIAGNOSTIC" if failed else "PASS"),
            "expected": expected,
            "expected_score": expected_score,
            "conflict": conflict,
            "conflict_score": conflict_score,
            "enforced": enforced,
        }
        if failed and enforced:
            violations.append(f"gender_conflict:{conflict}")
        elif failed:
            diagnostics.append(f"gender_conflict:{conflict}")

    character_tags = approved_run.get("approved_character_tags", ())
    for suffix in ("ears", "tail"):
        expected_species = _specific_species(character_tags, suffix)
        if expected_species is None:
            continue
        expected_tag = f"{expected_species} {suffix}"
        expected_score = scores.get(expected_tag, 0.0)
        conflicts = {
            f"{species} {suffix}": scores.get(f"{species} {suffix}", 0.0)
            for species in ANIMAL_SPECIES if species != expected_species
        }
        conflict_tag, conflict_score = max(
            conflicts.items(), key=lambda item: item[1], default=(None, 0.0))
        failed = (
            conflict_score >= settings.conflict_threshold
            and conflict_score >= expected_score + settings.minimum_margin
        )
        checks[suffix] = {
            "status": "FAIL" if failed else "PASS",
            "expected": expected_tag,
            "expected_score": expected_score,
            "strongest_conflict": conflict_tag,
            "conflict_score": conflict_score,
        }
        if failed:
            violations.append(f"{suffix}_species_conflict:{conflict_tag}")

    garment_tags = {
        normalize_tag(tag) for tag in approved_run.get("approved_tags", ())
    }
    approved_lower = _approved_lower_body_labels(garment_tags)
    novel_lower = {
        tag: scores.get(tag, 0.0)
        for tag in LOWER_BODY_GARMENTS
        if tag not in approved_lower
        and scores.get(tag, 0.0) >= settings.novel_garment_threshold
    }
    base_stage = stage == "base"
    checks["unapproved_lower_body_garment"] = {
        "status": (
            "DIAGNOSTIC" if novel_lower and base_stage
            else "FAIL" if novel_lower
            else "PASS"
        ),
        "approved": sorted(approved_lower),
        "detected_conflicts": novel_lower,
        "stage": stage,
    }
    if novel_lower and base_stage:
        diagnostics.extend(
            f"base_unapproved_lower_body_garment:{tag}"
            for tag in sorted(novel_lower)
        )
    elif novel_lower:
        violations.extend(
            f"unapproved_lower_body_garment:{tag}"
            for tag in sorted(novel_lower)
        )

    topology = evaluate_topology_diagnostic(
        approved_run.get("garment_topology"), scores
    )
    checks["garment_topology"] = topology
    if topology["status"] in {"DIAGNOSTIC", "UNRESOLVED"}:
        diagnostics.append(
            f"garment_topology:{topology['reason']}"
        )

    hair_structure = evaluate_hair_structure_diagnostic(
        approved_run.get("hair_structure"), scores
    )
    checks["hair_structure"] = hair_structure
    if hair_structure["status"] in {"DIAGNOSTIC", "UNRESOLVED"}:
        diagnostics.append(
            f"hair_structure:{hair_structure['reason']}"
        )

    action = "retry" if violations else ("review" if unresolved else "pass")
    expected_strength = sum(
        float(check.get("expected_score", 0.0)) for check in checks.values())
    conflict_strength = sum(
        float(check.get("conflict_score", 0.0)) for check in checks.values())
    conflict_strength += sum(novel_lower.values())
    return {
        **settings.record(),
        "status": action.upper(),
        "action": action,
        "violations": violations,
        "unresolved": unresolved,
        "diagnostics": diagnostics,
        "checks": checks,
        "fallback_quality_score": expected_strength - conflict_strength,
        "raw_score_source": "wd14_general_labels",
        "absence_is_not_failure": True,
    }


class CandidateSemanticAnalyzer:
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

    def analyze(self, image, approved_run, settings, *, stage='final'):
        result = self.session.analyze(image)
        return evaluate_candidate_semantics(
            approved_run, result.raw_general_scores, settings, stage=stage)

    def close(self):
        self.session.close()


def reset_candidate_runtime(pipeline, visual_inputs, identity_scale,
                            garment_scale, *, staged=False):
    """Reset mutable per-call state without rebuilding model weights."""
    import torch
    from genai_lab.reference_order import (
        adapter_references,
        adapter_references_for_stage,
        unmasked_adapter_scale,
    )

    report = {
        "version": "candidate_runtime_reset_v1",
        "scheduler_recreated": False,
        "interrupt_cleared": False,
        "adapter_scales_restored": False,
        "model_hooks_released": False,
    }
    if hasattr(pipeline, "maybe_free_model_hooks"):
        pipeline.maybe_free_model_hooks()
        report["model_hooks_released"] = True
    scheduler = getattr(pipeline, "scheduler", None)
    if scheduler is not None and hasattr(type(scheduler), "from_config"):
        pipeline.scheduler = type(scheduler).from_config(scheduler.config)
        report["scheduler_recreated"] = True
    if hasattr(pipeline, "_interrupt"):
        pipeline._interrupt = False
        report["interrupt_cleared"] = True
    entries = (
        adapter_references_for_stage(
            visual_inputs, 'base', identity_scale, garment_scale
        )
        if staged
        else adapter_references(
            visual_inputs, identity_scale, garment_scale
        )
    )
    scales = [float(entry.scale) for entry in entries]
    pipeline.set_ip_adapter_scale(
        unmasked_adapter_scale(scales) if staged else [scales]
    )
    report["adapter_scales_restored"] = True
    torch.cuda.empty_cache()
    report["status"] = "reset"
    return report
