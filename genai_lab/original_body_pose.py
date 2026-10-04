"""기준 캐릭터 자체에서 추출한 신체 복원용 DWPose 계약."""

from dataclasses import dataclass

from PIL import Image

from genai_lab.image_digest import calculate_image_pixel_sha256
from genai_lab.pose_estimation import (
    PoseEstimationApprovedInput,
    PoseEstimationReviewCandidate,
    PoseControlPreparedInput,
    approve_pose_estimation_candidate,
    prepare_pose_control_input,
)
from genai_lab.pose_fallback import PoseFallbackSettings, evaluate_pose_quality


class OriginalBodyPoseError(ValueError):
    """기준 캐릭터와 DWPose 결과의 동일성·품질 계약 위반."""


@dataclass(frozen=True)
class OriginalBodyPose:
    """외부 자세와 분리해 보관하는 기준 캐릭터 신체 복원용 자세."""

    source_image_sha256: str
    source_width: int
    source_height: int
    approved_pose: PoseEstimationApprovedInput
    required_group_pass_count: int
    required_group_count: int
    non_black_pixel_count: int
    elapsed_seconds: float

    def close(self) -> None:
        """독립 소유한 ControlNet 뼈대 지도를 해제한다."""
        self.approved_pose.close()


def approve_original_body_pose(
    source_image: Image.Image,
    review_candidate: PoseEstimationReviewCandidate,
    quality_settings: PoseFallbackSettings,
) -> OriginalBodyPose:
    """같은 기준 캐릭터에서 나온 품질 통과 DWPose 결과만 복사한다."""
    if review_candidate.source_image.size != source_image.size:
        raise OriginalBodyPoseError(
            "DWPose 원본 크기가 기준 캐릭터 크기와 다릅니다."
        )
    source_sha256 = calculate_image_pixel_sha256(source_image, "RGB")
    candidate_sha256 = calculate_image_pixel_sha256(
        review_candidate.source_image,
        "RGB",
    )
    if source_sha256 != candidate_sha256:
        raise OriginalBodyPoseError(
            "DWPose 원본 픽셀이 기준 캐릭터와 다릅니다."
        )
    quality = evaluate_pose_quality(review_candidate, quality_settings)
    if not quality.accepted:
        raise OriginalBodyPoseError(
            "기준 캐릭터 DWPose 품질 미달: "
            + "; ".join(quality.rejection_reasons)
        )
    return OriginalBodyPose(
        source_image_sha256=source_sha256,
        source_width=source_image.width,
        source_height=source_image.height,
        approved_pose=approve_pose_estimation_candidate(review_candidate),
        required_group_pass_count=quality.required_group_pass_count,
        required_group_count=quality.required_group_count,
        non_black_pixel_count=quality.non_black_pixel_count,
        elapsed_seconds=review_candidate.elapsed_seconds,
    )


def prepare_original_body_pose_control(
    original_body_pose: OriginalBodyPose,
    source_image: Image.Image,
) -> PoseControlPreparedInput:
    """승인 때와 같은 기준 캐릭터인지 재검사하고 ControlNet 입력을 만든다."""
    if source_image.size != (
        original_body_pose.source_width,
        original_body_pose.source_height,
    ):
        raise OriginalBodyPoseError(
            "신체 복원 직전 기준 캐릭터 크기가 DWPose 승인 원본과 다릅니다."
        )
    current_sha256 = calculate_image_pixel_sha256(source_image, "RGB")
    if current_sha256 != original_body_pose.source_image_sha256:
        raise OriginalBodyPoseError(
            "신체 복원 직전 기준 캐릭터 픽셀이 DWPose 승인 원본과 다릅니다."
        )
    return prepare_pose_control_input(
        original_body_pose.approved_pose,
        target_width=source_image.width,
        target_height=source_image.height,
    )
