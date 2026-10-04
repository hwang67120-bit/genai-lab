"""Adaptive candidate selection and one-winner refinement policy."""

from dataclasses import dataclass
from enum import Enum
import math
from collections import Counter


class CandidateAction(str, Enum):
    KEEP = "keep"
    REPAIR = "repair"
    RETRY = "retry"
    REVIEW = "manual_review_required"
    QUARANTINE = "quarantine"


@dataclass(frozen=True)
class CandidatePipelineSettings:
    enabled: bool
    target_valid_candidates: int
    maximum_attempts: int
    repair_best_candidate_only: bool
    reject_failed_post_audit: bool
    minimum_component_similarity: float
    require_all_similarity_scores: bool
    gender_retry_attempts: int
    quality_retry_attempts: int
    output_coordinate_similarity: bool
    base_similarity_components: tuple[str, ...]
    final_similarity_components: tuple[str, ...]
    return_policy: str

    def record(self):
        return {
            "version": "adaptive_candidate_pipeline_v4",
            "enabled": self.enabled,
            "target_valid_candidates": self.target_valid_candidates,
            "maximum_attempts": self.maximum_attempts,
            "repair_best_candidate_only": self.repair_best_candidate_only,
            "reject_failed_post_audit": self.reject_failed_post_audit,
            "minimum_component_similarity": self.minimum_component_similarity,
            "require_all_similarity_scores": self.require_all_similarity_scores,
            "gender_retry_attempts": self.gender_retry_attempts,
            "gender_retry_policy": (
                "disabled"
                if self.gender_retry_attempts == 0
                else "only_after_initial_pool_all_gender_conflicts"
            ),
            "quality_retry_attempts": self.quality_retry_attempts,
            "quality_retry_policy": (
                "after_any_non_gender_base_failure"),
            "output_coordinate_similarity": self.output_coordinate_similarity,
            "base_similarity_components": list(self.base_similarity_components),
            "final_similarity_components": list(self.final_similarity_components),
            "return_policy": self.return_policy,
            "quality_scores_block_return": self.return_policy == "strict_all",
            "ranking_policy": "maximin_then_mean_then_component_scores",
        }


@dataclass(frozen=True)
class CandidateDecision:
    action: CandidateAction
    reasons: tuple[str, ...]
    report: dict


def resolve_candidate_pipeline(config, configured_candidate_count):
    raw = config.get("candidate_pipeline", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("후보 파이프라인 설정은 객체여야 합니다.")
    if type(configured_candidate_count) is not int or not 2 <= configured_candidate_count <= 100:
        raise ValueError("후보 최대 시도 횟수는 2~100이어야 합니다.")
    enabled = bool(raw.get("enabled", False))
    target = raw.get("target_valid_candidates", configured_candidate_count)
    maximum = raw.get("maximum_attempts", configured_candidate_count)
    repair_best = bool(raw.get("repair_best_candidate_only", True))
    reject_failed = bool(raw.get("reject_failed_post_audit", True))
    minimum_similarity = raw.get("minimum_component_similarity", 0.0)
    require_all_scores = bool(raw.get("require_all_similarity_scores", False))
    gender_retry_attempts = raw.get("gender_retry_attempts", 0)
    quality_retry_attempts = raw.get("quality_retry_attempts", 0)
    output_coordinate_similarity = bool(
        raw.get("output_coordinate_similarity", False))
    return_policy = str(raw.get("return_policy", "strict_all"))
    if return_policy not in {"strict_all", "hard_safety_only"}:
        raise ValueError(
            "candidate_pipeline.return_policy는 strict_all 또는 "
            "hard_safety_only여야 합니다.")
    allowed_components = {"character", "hair", "garment", "vibe"}

    def components(name, default):
        value = raw.get(name, default)
        if (not isinstance(value, (tuple, list)) or not value
                or not all(isinstance(item, str) for item in value)):
            raise ValueError(f"{name} must be a nonempty string list")
        result = tuple(value)
        if len(set(result)) != len(result) or set(result) - allowed_components:
            raise ValueError(f"{name} contains duplicate or unknown components")
        return result

    base_components = components(
        "base_similarity_components", ("character", "garment", "vibe"))
    final_components = components(
        "final_similarity_components", ("character", "garment", "vibe"))
    if output_coordinate_similarity and "hair" in base_components:
        # A missing output hair mask must fail closed instead of falling back to
        # the source-image coordinates.
        require_all_scores = True
    if type(target) is not int or type(maximum) is not int:
        raise ValueError("유효 후보 수와 최대 시도 횟수는 정수여야 합니다.")
    if (type(gender_retry_attempts) is not int
            or not 0 <= gender_retry_attempts <= 10):
        raise ValueError("성별 충돌 추가 시도 횟수는 0~10 사이의 정수여야 합니다.")
    if (type(quality_retry_attempts) is not int
            or not 0 <= quality_retry_attempts <= 10):
        raise ValueError("일반 품질 추가 시도 횟수는 0~10 사이의 정수여야 합니다.")
    if not 1 <= target <= maximum <= configured_candidate_count:
        raise ValueError("유효 후보 수 <= 최대 시도 횟수 <= 승인 후보 수여야 합니다.")
    if (not isinstance(minimum_similarity, (int, float))
            or isinstance(minimum_similarity, bool)
            or not math.isfinite(float(minimum_similarity))
            or not 0 <= float(minimum_similarity) <= 1):
        raise ValueError("후보 부위별 최소 유사도는 0~1 사이의 유한한 수여야 합니다.")
    return CandidatePipelineSettings(
        enabled, target, maximum, repair_best, reject_failed,
        float(minimum_similarity), require_all_scores,
        gender_retry_attempts, quality_retry_attempts,
        output_coordinate_similarity, base_components, final_components,
        return_policy)


def select_retry_phase(initial_gender_conflicts,
                       initial_quality_failures, settings):
    """Choose one bounded retry phase from the complete initial pool."""
    gender = tuple(bool(value) for value in initial_gender_conflicts)
    quality = tuple(bool(value) for value in initial_quality_failures)
    if len(gender) != settings.maximum_attempts or len(quality) != len(gender):
        raise ValueError("초기 후보 전체 결과가 모이기 전에는 재시도 단계를 결정할 수 없습니다.")
    if all(gender) and settings.gender_retry_attempts:
        return "gender", settings.gender_retry_attempts
    if any(quality) and settings.quality_retry_attempts:
        return "quality", settings.quality_retry_attempts
    return None, 0


def summarize_candidate_failures(quarantined, settings, *, retry_phase=None):
    """Aggregate gate evidence into one machine-readable terminal diagnosis."""
    stage_counts = Counter()
    reason_counts = Counter()
    categories = Counter()
    for item in quarantined:
        stage = str(item.get("failure_stage") or "unknown")
        stage_counts[stage] += 1
        report = item.get("report") or {}
        reasons = list(report.get("reasons") or ())
        reasons += list(report.get("violations") or ())
        reasons += list(report.get("unresolved") or ())
        for reason in reasons:
            reason_counts[str(reason)] += 1
        if stage in {"decoded_rgb", "final_latent"}:
            categories["corrupted"] += 1
        elif stage == "candidate_structure_gate":
            categories["structure_invalid"] += 1
        elif stage == "candidate_similarity_gate":
            categories["similarity_weak"] += 1
        elif stage == "reference_base_mask_validation":
            categories["reference_not_measurable"] += 1
        elif stage == "candidate_semantic_gate":
            if any(str(reason).startswith("gender_conflict:") for reason in reasons):
                categories["gender_conflict"] += 1
            else:
                categories["semantic_invalid"] += 1
        else:
            categories["other"] += 1
    if retry_phase == "quality":
        next_action = "quality_retry_exhausted"
    elif retry_phase == "gender":
        next_action = "gender_retry_exhausted"
    elif categories.get("similarity_weak") and not settings.quality_retry_attempts:
        next_action = "enable_bounded_quality_retry"
    else:
        next_action = "no_safe_retry_available"
    return {
        "version": "candidate_failure_summary_v1",
        "status": "no_returnable_candidate",
        "failure_stage_counts": dict(sorted(stage_counts.items())),
        "failure_category_counts": dict(sorted(categories.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "retry_phase": retry_phase,
        "next_action": next_action,
    }


def candidate_rank_key(record):
    scores = (record or {}).get("similarity", {})

    def score(name):
        value = scores.get(name)
        return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else -2.0

    decision = (record or {}).get("candidate_decision", {})
    names = tuple(decision.get("gating_components") or (
        "character", "garment", "vibe"))
    components = tuple(score(name) for name in names)
    minimum = min(components)
    mean = sum(components) / len(components)
    return minimum, mean, *components


# 성별 분류는 외형 진단이며 반환 차단 조건이 아니다.
HARD_SEMANTIC_PREFIXES = ()
HARD_STRUCTURE_REASONS = {
    "large_detached_object_above_character",
    "tall_convex_enclosure_not_character",
    "abnormally_wide_object_above_body",
}


def similarity_percentages(scores, component_names):
    """Return display diagnostics; values are indicators, not probabilities."""
    result = {}
    for name in component_names:
        value = (scores or {}).get(name)
        if isinstance(value, (int, float)) and math.isfinite(value):
            result[name] = round(max(0.0, min(1.0, float(value))) * 100.0, 2)
        else:
            result[name] = None
    return result


def semantic_blocking_reasons(report, settings):
    violations = tuple(str(item) for item in (report or {}).get("violations", ()))
    unresolved = tuple(str(item) for item in (report or {}).get("unresolved", ()))
    if settings.return_policy == "strict_all":
        return violations + unresolved
    return tuple(
        reason for reason in violations
        if reason.startswith(HARD_SEMANTIC_PREFIXES)
    )


def semantic_refinement_reasons(report, settings):
    blocking = set(semantic_blocking_reasons(report, settings))
    soft_diagnostics = [
        str(item) for item in (report or {}).get("diagnostics", ())
        if str(item).startswith("gender_conflict:")
    ]
    reasons = [
        str(item) for item in (
            list((report or {}).get("violations", ()))
            + list((report or {}).get("unresolved", ()))
            + soft_diagnostics
        )
        if str(item) not in blocking
    ]
    return tuple(dict.fromkeys(reasons))


def structure_blocking_reasons(report, settings):
    violations = tuple(str(item) for item in (report or {}).get("violations", ()))
    unresolved = tuple(str(item) for item in (report or {}).get("unresolved", ()))
    if settings.return_policy == "strict_all":
        return violations + unresolved
    return tuple(reason for reason in violations if reason in HARD_STRUCTURE_REASONS)


def structure_refinement_reasons(report, settings):
    blocking = set(structure_blocking_reasons(report, settings))
    reasons = [
        str(item) for item in (
            list((report or {}).get("violations", ()))
            + list((report or {}).get("unresolved", ()))
        )
        if str(item) not in blocking
    ]
    return tuple(reasons)


def preliminary_candidate_decision(record, settings=None, *, stage=None):
    record = record or {}
    latent = record.get("final_latent_integrity", {})
    image = record.get("generated_image_integrity", {})
    integrity_reasons = []
    if latent and latent.get("status") != "passed":
        integrity_reasons.append("final_latent_integrity")
    if image and image.get("status") != "passed":
        integrity_reasons.append("generated_image_integrity")
    integrity_failed = bool(integrity_reasons)

    scores = record.get("similarity", {})
    diagnostic_reasons = []
    component_scores = {}
    component_names = ("character", "garment", "vibe")
    if settings is not None and stage == "base":
        component_names = settings.base_similarity_components
    elif settings is not None and stage == "final":
        component_names = settings.final_similarity_components
    if settings is not None and settings.enabled:
        for name in component_names:
            value = scores.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                component_scores[name] = None
                if settings.require_all_similarity_scores:
                    diagnostic_reasons.append(f"similarity_unresolved:{name}")
                continue
            component_scores[name] = float(value)
            if value < settings.minimum_component_similarity:
                diagnostic_reasons.append(f"similarity_below_threshold:{name}")

    strict_quality = (
        settings is not None
        and settings.enabled
        and settings.return_policy == "strict_all"
    )
    blocking_quality_reasons = diagnostic_reasons if strict_quality else []
    reasons = [*integrity_reasons, *blocking_quality_reasons]
    action = (
        CandidateAction.QUARANTINE if integrity_failed else
        CandidateAction.RETRY if blocking_quality_reasons else
        CandidateAction.KEEP
    )
    percentages = similarity_percentages(scores, component_names)
    return CandidateDecision(action, tuple(reasons), {
        "version": "candidate_preliminary_decision_v4",
        "action": action.value,
        "reasons": reasons,
        "similarity_stage": stage,
        "gating_components": list(component_names),
        "component_scores": component_scores,
        "similarity_percentages": percentages,
        "minimum_component_similarity": (
            settings.minimum_component_similarity if settings else None),
        "target_percentage": (
            round(settings.minimum_component_similarity * 100.0, 2)
            if settings else None),
        "quality_status": (
            "refinement_required" if diagnostic_reasons else "meets_target"),
        "refinement_required": bool(diagnostic_reasons),
        "refinement_targets": diagnostic_reasons,
        "quality_scores_block_return": strict_quality,
    })


def final_candidate_decision(audit, settings, image_integrity=None):
    status = (audit or {}).get("status", "UNRESOLVED")
    image_status = (image_integrity or {}).get("status", "passed")
    image_failed = image_status != "passed"
    checks = (audit or {}).get("checks", {})
    blocking_colors = (audit or {}).get("blocking_color_checks", {})
    relaxed = settings.return_policy == "hard_safety_only"

    hard_check_names = {"semantic_gate", "structure"}
    hard_audit_failures = {
        name: detail.get("reason")
        for name, detail in checks.items()
        if name in hard_check_names and detail.get("status") == "FAIL"
    }
    quality_failures = {
        name: detail.get("reason")
        for name, detail in checks.items()
        if name not in hard_check_names and detail.get("status") == "FAIL"
    }
    unresolved_checks = {
        name: detail.get("reason")
        for name, detail in checks.items()
        if detail.get("status") == "UNRESOLVED"
    }
    needs_refinement_checks = {
        name: detail.get("reason")
        for name, detail in checks.items()
        if detail.get("status") == "NEEDS_REFINEMENT"
    }
    for name, reason in blocking_colors.items():
        quality_failures.setdefault(name, reason)

    reasons = []
    if image_failed:
        reasons.append("post_repair_image_integrity")
    if relaxed:
        if hard_audit_failures:
            reasons.append("hard_safety_post_audit_failed")
        blocking = image_failed or bool(hard_audit_failures)
    else:
        if settings.reject_failed_post_audit and status == "FAIL":
            reasons.append("generated_condition_post_audit_failed")
        if blocking_colors:
            reasons.append("approved_part_color_audit_failed_or_unresolved")
        blocking = (
            image_failed
            or bool(blocking_colors)
            or (settings.reject_failed_post_audit and status == "FAIL")
        )

    refinement_required = bool(
        quality_failures or unresolved_checks or needs_refinement_checks)
    if blocking:
        action = CandidateAction.QUARANTINE
    elif refinement_required:
        action = CandidateAction.REVIEW
        reasons.append("quality_refinement_required")
    elif status == "PASS" or relaxed:
        action = CandidateAction.KEEP
    else:
        action = CandidateAction.REVIEW
        reasons.append("generated_condition_post_audit_not_accepted")

    return CandidateDecision(action, tuple(reasons), {
        "version": "candidate_final_decision_v3",
        "action": action.value,
        "reasons": reasons,
        "audit_status": status,
        "post_repair_image_status": image_status,
        "return_policy": settings.return_policy,
        "hard_audit_failures": hard_audit_failures,
        "quality_failures": quality_failures,
        "unresolved_checks": unresolved_checks,
        "needs_refinement_checks": needs_refinement_checks,
        "blocking_color_checks": dict(blocking_colors),
        "refinement_required": refinement_required,
        "refinement_targets": sorted({
            *quality_failures.keys(), *unresolved_checks.keys(),
            *needs_refinement_checks.keys()
        }),
        "unresolved_policy": "return_for_review_and_refinement",
    })
