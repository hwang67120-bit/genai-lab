"""요청이 소유하는 명시적 자세 선택이다. 기본값이나 참조 조건을 바꾸지 않는다."""
from dataclasses import dataclass
import hashlib
import io
import math
from genai_lab.pose_estimation import PoseEstimationApprovedInput


@dataclass(frozen=True)
class ReferencePoseOptions:
    approved_pose: PoseEstimationApprovedInput
    base_strength: float

    def __post_init__(self):
        if not isinstance(self.approved_pose, PoseEstimationApprovedInput):
            raise TypeError("An approved pose is required")
        if not math.isfinite(self.base_strength) or not 0 < self.base_strength <= 1:
            raise ValueError("Base strength must be finite and in (0, 1]")


def observe_reference_pose(config, option, prepared, strength):
    """난수나 모델 호출 없이 실제로 투영한 제어 이미지를 기록한다."""
    from genai_lab.provenance import recorder
    rec = recorder(config)
    if rec is None:
        return
    result = {"enabled": option is not None, "base_strength": strength,
              "control_image_sha256": None}
    if option is not None:
        image = prepared.control_map_image
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        result.update(control_image_sha256=hashlib.sha256(encoded.getvalue()).hexdigest(),
                      control_image_pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                      control_image_size=list(image.size), hash_encoding="PNG",
                      resize_scale=prepared.resize_scale,
                      padding_left=prepared.padding_left, padding_top=prepared.padding_top)
    rec.loaded["reference_pose"] = result
    rec.flush()
