"""GUI 이미지 생성 단계의 자동 진행 상태를 관리한다."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class GenerationWorkflowStage(str, Enum):
    """사용자 입력부터 최종 검토까지의 실행·진단 상태."""

    INPUT_READY = "input_ready"
    REFERENCE_PREPARING = "reference_preparing"
    CLOTHING_MASKING = "clothing_masking"
    CLOTHING_ANALYZING = "clothing_analyzing"
    POSE_ESTIMATING = "pose_estimating"
    BASE_GENERATING = "base_generating"
    BODY_MASKING = "body_masking"
    GARMENT_GEOMETRY = "garment_geometry"
    GARMENT_LINEART = "garment_lineart"
    CLOTHING_COMPOSITING = "clothing_compositing"
    FINAL_REVIEW = "final_review"
    COMPLETED = "completed"
    FAILED = "failed"


WORKFLOW_STAGE_PROGRESS: dict[GenerationWorkflowStage, tuple[int, int]] = {
    GenerationWorkflowStage.INPUT_READY: (0, 8),
    GenerationWorkflowStage.REFERENCE_PREPARING: (1, 8),
    GenerationWorkflowStage.CLOTHING_MASKING: (2, 8),
    GenerationWorkflowStage.CLOTHING_ANALYZING: (3, 8),
    GenerationWorkflowStage.POSE_ESTIMATING: (4, 8),
    GenerationWorkflowStage.BASE_GENERATING: (5, 8),
    GenerationWorkflowStage.BODY_MASKING: (6, 8),
    # TPS·Lineart는 활성 자동 경로가 아닌 진단용 상태로 보존한다.
    GenerationWorkflowStage.GARMENT_GEOMETRY: (6, 8),
    GenerationWorkflowStage.GARMENT_LINEART: (6, 8),
    GenerationWorkflowStage.CLOTHING_COMPOSITING: (7, 8),
    GenerationWorkflowStage.FINAL_REVIEW: (8, 8),
    GenerationWorkflowStage.COMPLETED: (8, 8),
    GenerationWorkflowStage.FAILED: (0, 8),
}


@dataclass
class GenerationWorkflowContext:
    """선택 입력과 현재 자동 진행 위치를 보관하는 GUI 임시 상태."""

    character_image_path: Path
    clothing_image_path: Path | None
    pose_image_path: Path | None
    current_stage: GenerationWorkflowStage = GenerationWorkflowStage.INPUT_READY
    failed_stage: GenerationWorkflowStage | None = None
    retry_count: int = 0
    active: bool = True

    @property
    def progress(self) -> tuple[int, int]:
        """현재 단계 번호와 전체 활성 8단계를 반환한다."""
        if self.current_stage is GenerationWorkflowStage.FAILED:
            if self.failed_stage is None:
                return (0, 10)
            return WORKFLOW_STAGE_PROGRESS[self.failed_stage]
        return WORKFLOW_STAGE_PROGRESS[self.current_stage]

    def move_to(self, stage: GenerationWorkflowStage) -> None:
        """실패 정보를 지우고 다음 실행 단계로 이동한다."""
        self.current_stage = stage
        self.failed_stage = None
        self.active = stage not in {
            GenerationWorkflowStage.COMPLETED,
            GenerationWorkflowStage.FAILED,
        }

    def fail(self, stage: GenerationWorkflowStage) -> None:
        """실패 위치를 보존하고 자동 진행을 멈춘다."""
        self.failed_stage = stage
        self.current_stage = GenerationWorkflowStage.FAILED
        self.active = False

    def retry(self) -> None:
        """같은 입력으로 실패 지점부터 재시도할 수 있게 바꾼다."""
        retry_stage = self.failed_stage or GenerationWorkflowStage.INPUT_READY
        self.retry_count += 1
        self.current_stage = retry_stage
        self.failed_stage = None
        self.active = True
