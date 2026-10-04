"""생성된 기준 캐릭터 좌표에 묶인 교체 의상·특수 보호 승인 계약."""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from genai_lab.image_digest import calculate_image_pixel_sha256


@dataclass(frozen=True)
class ApprovedTargetMasks:
    source_sha256: str
    clothing_mask: Image.Image
    special_protection_mask: Image.Image
    use_automatic_base: bool = False
    excluded_mask: Image.Image | None = None
    layered_priority: bool = False

    def close(self) -> None:
        self.clothing_mask.close()
        self.special_protection_mask.close()
        if self.excluded_mask is not None:
            self.excluded_mask.close()

    def copy(self) -> "ApprovedTargetMasks":
        return ApprovedTargetMasks(
            self.source_sha256,
            self.clothing_mask.copy(),
            self.special_protection_mask.copy(),
            self.use_automatic_base,
            self.excluded_mask.copy() if self.excluded_mask is not None else None,
            self.layered_priority,
        )

    def validate_source(self, source: Image.Image) -> None:
        if any(mask.size != source.size for mask in (
            self.clothing_mask, self.special_protection_mask,
        )):
            raise ValueError("승인 의상·보호 마스크와 기준 캐릭터 크기가 다릅니다.")
        if calculate_image_pixel_sha256(source, "RGB") != self.source_sha256:
            raise ValueError("기준 캐릭터가 바뀌어 의상·보호 마스크 재승인이 필요합니다.")
        if self.excluded_mask is not None and self.excluded_mask.size != source.size:
            raise ValueError("사용자 제외 마스크 크기가 기준 캐릭터와 다릅니다.")


def approve_target_masks(
    source: Image.Image,
    clothing_mask: Image.Image,
    special_protection_mask: Image.Image,
    *,
    use_automatic_base: bool = False,
    excluded_mask: Image.Image | None = None,
    layered_priority: bool = False,
) -> ApprovedTargetMasks:
    """명시적 승인 이벤트에서 호출하며 크기·빈 의상·역할 충돌을 검증한다."""
    if clothing_mask.size != source.size or special_protection_mask.size != source.size:
        raise ValueError("교체 의상·특수 보호 마스크는 기준 캐릭터와 같은 크기여야 합니다.")
    with clothing_mask.convert("L") as clothes, special_protection_mask.convert("L") as special:
        required = np.asarray(clothes) >= 128
        protected = np.asarray(special) >= 128
        if excluded_mask is not None and excluded_mask.size != source.size:
            raise ValueError("사용자 제외 마스크 크기가 기준 캐릭터와 다릅니다.")
        if not np.any(required) and not use_automatic_base:
            raise ValueError("교체할 기존 의상 마스크가 비어 있습니다.")
        conflicts = int(np.count_nonzero(required & protected))
        if conflicts:
            raise ValueError(
                f"교체 의상과 특수 보호 영역이 {conflicts:,}px 겹칩니다. "
                "겹친 영역의 마스크 후보를 다시 선택하세요."
            )
        return ApprovedTargetMasks(
            calculate_image_pixel_sha256(source, "RGB"),
            Image.fromarray(required.astype(np.uint8) * 255),
            Image.fromarray(protected.astype(np.uint8) * 255),
            use_automatic_base,
            excluded_mask.convert("L") if excluded_mask is not None else None,
            layered_priority or use_automatic_base,
        )


def accumulate_mask(existing: Image.Image, incoming: Image.Image) -> Image.Image:
    """추가 선택은 교체하지 않고 소프트 알파의 픽셀별 최댓값으로 누적한다."""
    if existing.size != incoming.size:
        raise ValueError("누적할 마스크의 크기가 다릅니다.")
    with existing.convert("L") as old, incoming.convert("L") as new:
        return Image.fromarray(np.maximum(np.asarray(old), np.asarray(new)))
