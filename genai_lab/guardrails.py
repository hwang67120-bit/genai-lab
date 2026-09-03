"""보정 전 경고와 최종 실행 차단을 한 가지 계약으로 판정한다."""

from dataclasses import dataclass
from enum import Enum


class GuardSeverity(str, Enum):
    """가드레일 검사 결과가 승인 흐름에 미치는 영향."""

    PASS = "pass"
    WARNING = "warning"
    BLOCK = "block"


@dataclass(frozen=True)
class GuardResult:
    """한 검사의 코드·수치·기준·보정 결과를 보관하는 결정론적 결과."""

    code: str
    stage: str
    severity: GuardSeverity
    measured_value: int | float | None
    threshold_value: int | float | None
    unit: str
    corrected_value: int | float | None
    message_ko: str
    recovery_action_ko: str | None = None


@dataclass(frozen=True)
class GuardDecision:
    """사용자 승인 가능 여부와 그 근거를 함께 보관한다."""

    results: tuple[GuardResult, ...]
    blocking_results: tuple[GuardResult, ...]
    warning_results: tuple[GuardResult, ...]
    passed_results: tuple[GuardResult, ...]

    @property
    def approval_enabled(self) -> bool:
        """최종 BLOCK이 0개일 때만 승인할 수 있다."""
        return len(self.blocking_results) == 0


def evaluate_guard_results(
    results: tuple[GuardResult, ...],
) -> GuardDecision:
    """개별 검사 결과를 승인 결정으로 모으되 빈 검사는 허용하지 않는다."""
    if not results:
        raise ValueError("가드레일 결과가 0개라 승인 여부를 계산할 수 없습니다.")

    return GuardDecision(
        results=results,
        blocking_results=tuple(
            result
            for result in results
            if result.severity is GuardSeverity.BLOCK
        ),
        warning_results=tuple(
            result
            for result in results
            if result.severity is GuardSeverity.WARNING
        ),
        passed_results=tuple(
            result
            for result in results
            if result.severity is GuardSeverity.PASS
        ),
    )


def create_catvton_preflight_guard_results(
    *,
    processed_mask_pixel_count: int,
    model_mask_pixel_count: int,
    soft_overlap_pixel_count: int,
    hard_overlap_pixel_count: int,
    final_protected_overlap_pixel_count: int,
    final_outside_foreground_pixel_count: int,
) -> tuple[GuardResult, ...]:
    """Preflight 실행기와 GUI가 공유하는 최종 마스크 승인 기준."""
    return (
        _positive_required_result(
            code="PROCESSED_MASK_EMPTY",
            stage="catvton_preflight",
            measured_value=processed_mask_pixel_count,
            message_pass_ko=(
                f"처리 크기 마스크가 {processed_mask_pixel_count:,}픽셀입니다."
            ),
            message_block_ko="처리 크기 변환 후 마스크가 0픽셀입니다.",
            recovery_action_ko="원본 마스크와 처리 해상도를 확인하세요.",
        ),
        _positive_required_result(
            code="FINAL_MODEL_MASK_EMPTY",
            stage="catvton_preflight",
            measured_value=model_mask_pixel_count,
            message_pass_ko=(
                f"최종 model_mask가 {model_mask_pixel_count:,}픽셀입니다."
            ),
            message_block_ko="금지 영역 제한 후 model_mask가 0픽셀입니다.",
            recovery_action_ko="보호 영역과 의상 변경 영역을 다시 확인하세요.",
        ),
        _corrected_overlap_result(
            code="MASK_BLUR_SOFT_OVERLAP_REMOVED",
            measured_value=soft_overlap_pixel_count,
            label_ko="blur 직후 1~127 약한 침범",
        ),
        _corrected_overlap_result(
            code="MASK_BLUR_HARD_OVERLAP_REMOVED",
            measured_value=hard_overlap_pixel_count,
            label_ko="blur 직후 128~255 강한 침범",
        ),
        _zero_required_result(
            code="FINAL_PROTECTED_OVERLAP",
            stage="catvton_preflight",
            measured_value=final_protected_overlap_pixel_count,
            message_pass_ko="최종 model_mask의 보호 영역 침범이 0픽셀입니다.",
            message_block_ko=(
                "최종 model_mask의 보호 영역 침범이 "
                f"{final_protected_overlap_pixel_count:,}픽셀입니다."
            ),
            recovery_action_ko="최종 model_mask에서 보호 영역을 다시 제거하세요.",
        ),
        _zero_required_result(
            code="FINAL_OUTSIDE_FOREGROUND",
            stage="catvton_preflight",
            measured_value=final_outside_foreground_pixel_count,
            message_pass_ko="최종 model_mask의 캐릭터 외곽 밖 침범이 0픽셀입니다.",
            message_block_ko=(
                "최종 model_mask의 캐릭터 외곽 밖 침범이 "
                f"{final_outside_foreground_pixel_count:,}픽셀입니다."
            ),
            recovery_action_ko="최종 model_mask를 캐릭터 외곽 안으로 제한하세요.",
        ),
    )


def create_human_agnostic_guard_decision(
    *,
    remaining_clothing_pixel_count: int,
    removal_percent: float | None,
    input_change_mask_pixel_count: int,
    input_protected_overlap_pixel_count: int,
    input_outside_foreground_pixel_count: int,
    processed_mask_pixel_count: int,
    model_mask_pixel_count: int,
    soft_overlap_pixel_count: int,
    hard_overlap_pixel_count: int,
    final_protected_overlap_pixel_count: int,
    final_outside_foreground_pixel_count: int,
    removal_status: str = "covered",
    protected_clothing_overlap_pixel_count: int = 0,
) -> GuardDecision:
    """Human-Agnostic 승인 수치를 PASS/WARNING/BLOCK으로 변환한다."""
    removal_text = "계산 불가" if removal_percent is None else f"{removal_percent:.3f}%"
    results = [
        _zero_required_result(
            code="ORIGINAL_CLOTHING_REMAINING",
            stage="original_clothing_removal",
            measured_value=remaining_clothing_pixel_count,
            message_pass_ko=(
                "검증 대상 기존 의상 잔여가 0픽셀입니다. "
                f"포함률={removal_text}"
            ),
            message_block_ko=(
                "검증 대상 기존 의상 잔여가 "
                f"{remaining_clothing_pixel_count:,}픽셀입니다. "
                f"포함률={removal_text}"
            ),
            recovery_action_ko="기존 의상 마스크와 보호 영역을 다시 확인하세요.",
        ),
        _positive_required_result(
            code="INPUT_CHANGE_MASK_EMPTY",
            stage="catvton_input_snapshot",
            measured_value=input_change_mask_pixel_count,
            message_pass_ko=(
                f"전처리 전 변경 마스크가 {input_change_mask_pixel_count:,}픽셀입니다."
            ),
            message_block_ko="전처리 전 변경 마스크가 0픽셀입니다.",
            recovery_action_ko="변경 영역 마스크를 다시 생성하세요.",
        ),
        _zero_required_result(
            code="INPUT_PROTECTED_OVERLAP",
            stage="catvton_input_snapshot",
            measured_value=input_protected_overlap_pixel_count,
            message_pass_ko="전처리 전 보호 영역 침범이 0픽셀입니다.",
            message_block_ko=(
                "전처리 전 보호 영역 침범이 "
                f"{input_protected_overlap_pixel_count:,}픽셀입니다."
            ),
            recovery_action_ko="보호 영역을 변경 마스크에서 제외하세요.",
        ),
        _zero_required_result(
            code="INPUT_OUTSIDE_FOREGROUND",
            stage="catvton_input_snapshot",
            measured_value=input_outside_foreground_pixel_count,
            message_pass_ko="전처리 전 캐릭터 외곽 밖 침범이 0픽셀입니다.",
            message_block_ko=(
                "전처리 전 캐릭터 외곽 밖 침범이 "
                f"{input_outside_foreground_pixel_count:,}픽셀입니다."
            ),
            recovery_action_ko="변경 마스크를 캐릭터 외곽 안으로 제한하세요.",
        ),
    ]
    if removal_status == "not_evaluable" or removal_percent is None:
        results.append(GuardResult(
            code="ORIGINAL_CLOTHING_NOT_EVALUABLE", stage="original_clothing_removal",
            severity=GuardSeverity.BLOCK, measured_value=None,
            threshold_value=1, unit="px", corrected_value=None,
            message_ko="검사 대상이 없어 기존 의상 포함 여부를 계산할 수 없습니다.",
            recovery_action_ko="기준 캐릭터에서 교체할 기존 의상을 다시 선택하세요.",
        ))
    results.append(_zero_required_result(
        code="TARGET_CLOTHING_PROTECTION_CONFLICT", stage="original_clothing_removal",
        measured_value=protected_clothing_overlap_pixel_count,
        message_pass_ko="교체 의상과 보호 영역의 충돌이 없습니다.",
        message_block_ko=f"교체 의상·보호 충돌 {protected_clothing_overlap_pixel_count:,}px를 다시 확인하세요.",
        recovery_action_ko="충돌 미리보기를 보고 기존 의상·특수 보호 마스크를 재선택하세요.",
    ))
    results.extend(
        create_catvton_preflight_guard_results(
            processed_mask_pixel_count=processed_mask_pixel_count,
            model_mask_pixel_count=model_mask_pixel_count,
            soft_overlap_pixel_count=soft_overlap_pixel_count,
            hard_overlap_pixel_count=hard_overlap_pixel_count,
            final_protected_overlap_pixel_count=(
                final_protected_overlap_pixel_count
            ),
            final_outside_foreground_pixel_count=(
                final_outside_foreground_pixel_count
            ),
        )
    )
    return evaluate_guard_results(tuple(results))


def _positive_required_result(
    *,
    code: str,
    stage: str,
    measured_value: int,
    message_pass_ko: str,
    message_block_ko: str,
    recovery_action_ko: str,
) -> GuardResult:
    passed = measured_value > 0
    return GuardResult(
        code=code,
        stage=stage,
        severity=GuardSeverity.PASS if passed else GuardSeverity.BLOCK,
        measured_value=measured_value,
        threshold_value=1,
        unit="px",
        corrected_value=None,
        message_ko=message_pass_ko if passed else message_block_ko,
        recovery_action_ko=None if passed else recovery_action_ko,
    )


def _zero_required_result(
    *,
    code: str,
    stage: str,
    measured_value: int,
    message_pass_ko: str,
    message_block_ko: str,
    recovery_action_ko: str,
) -> GuardResult:
    passed = measured_value == 0
    return GuardResult(
        code=code,
        stage=stage,
        severity=GuardSeverity.PASS if passed else GuardSeverity.BLOCK,
        measured_value=measured_value,
        threshold_value=0,
        unit="px",
        corrected_value=None,
        message_ko=message_pass_ko if passed else message_block_ko,
        recovery_action_ko=None if passed else recovery_action_ko,
    )


def _corrected_overlap_result(
    *,
    code: str,
    measured_value: int,
    label_ko: str,
) -> GuardResult:
    has_overlap = measured_value > 0
    return GuardResult(
        code=code,
        stage="catvton_preflight",
        severity=(GuardSeverity.WARNING if has_overlap else GuardSeverity.PASS),
        measured_value=measured_value,
        threshold_value=0,
        unit="px",
        corrected_value=0,
        message_ko=(
            f"{label_ko} {measured_value:,}픽셀을 금지 영역에서 제거했습니다."
            if has_overlap
            else f"{label_ko}이 0픽셀입니다."
        ),
    )
