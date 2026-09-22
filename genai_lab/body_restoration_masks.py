"""신체 복원에서 의상 제거 영역과 체형 힌트의 역할을 분리한다."""

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from PIL import Image


class GarmentThickness(str, Enum):
    """원본 캐릭터가 입은 의상의 체형 은폐 정도."""

    THIN = "thin"
    NORMAL = "normal"
    THICK = "thick"
    UNKNOWN = "unknown"

    @classmethod
    def parse(cls, value: "GarmentThickness | str") -> "GarmentThickness":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().casefold())
        except ValueError as error:
            supported = ", ".join(item.value for item in cls)
            raise ValueError(
                f"지원하지 않는 의상 두께입니다: {value!r}. 허용값={supported}"
            ) from error


@dataclass(frozen=True)
class BodyRestorationMaskRoles:
    """기존 의상 제거 마스크와 두꺼운 의상용 내부 체형 힌트."""

    clothing_removal_mask: Image.Image
    inner_body_hint: Image.Image | None
    garment_thickness: GarmentThickness
    contraction_applied: bool
    erosion_ratio: float
    inset_pixels: int
    removal_pixel_count: int
    inner_hint_pixel_count: int
    protected_overlap_pixel_count: int
    outside_foreground_pixel_count: int

    def close(self) -> None:
        self.clothing_removal_mask.close()
        if self.inner_body_hint is not None:
            self.inner_body_hint.close()


def _binary_array(image: Image.Image) -> np.ndarray:
    with image.convert("L") as gray:
        return np.asarray(gray, dtype=np.uint8).copy() >= 128


def _binary_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L")


def prepare_body_restoration_mask_roles(
    clothing_removal_mask: Image.Image,
    protected_mask: Image.Image,
    character_foreground_mask: Image.Image,
    garment_thickness: GarmentThickness | str = GarmentThickness.UNKNOWN,
    *,
    erosion_ratio: float = 0.15,
) -> BodyRestorationMaskRoles:
    """제거 영역은 보존하고 두꺼운 의상에만 수축 체형 힌트를 만든다.

    clothing_removal_mask는 기존 의상을 지우는 계약이므로 절대로
    수축하지 않는다. 거리 변환으로 얻은 내부 영역은 다음 단계의
    Body Proxy 폭 보정에만 쓰는 참고값이다.
    """
    if len({
        clothing_removal_mask.size,
        protected_mask.size,
        character_foreground_mask.size,
    }) != 1:
        raise ValueError("의상 제거·보호·캐릭터 외곽 마스크 크기가 다릅니다.")
    if not 0.0 < erosion_ratio <= 0.5:
        raise ValueError("두꺼운 의상 수축 비율은 0보다 크고 0.5 이하여야 합니다.")

    thickness = GarmentThickness.parse(garment_thickness)
    removal = _binary_array(clothing_removal_mask)
    protected = _binary_array(protected_mask)
    foreground = _binary_array(character_foreground_mask)
    removal_pixel_count = int(np.count_nonzero(removal))
    if removal_pixel_count == 0:
        raise ValueError("기존 의상 제거 마스크가 비었습니다.")

    protected_overlap = int(np.count_nonzero(removal & protected))
    outside_foreground = int(np.count_nonzero(removal & ~foreground))
    if protected_overlap or outside_foreground:
        raise ValueError(
            "기존 의상 제거 마스크의 안전 계약 위반: "
            f"보호 영역={protected_overlap:,}px, "
            f"캐릭터 외곽 밖={outside_foreground:,}px"
        )

    if thickness is not GarmentThickness.THICK:
        return BodyRestorationMaskRoles(
            clothing_removal_mask=_binary_image(removal),
            inner_body_hint=None,
            garment_thickness=thickness,
            contraction_applied=False,
            erosion_ratio=erosion_ratio,
            inset_pixels=0,
            removal_pixel_count=removal_pixel_count,
            inner_hint_pixel_count=0,
            protected_overlap_pixel_count=protected_overlap,
            outside_foreground_pixel_count=outside_foreground,
        )

    removal_u8 = removal.astype(np.uint8)
    points = cv2.findNonZero(removal_u8)
    if points is None:
        raise ValueError("두꺼운 의상 내부 힌트를 계산할 픽셀이 없습니다.")
    _, _, garment_width, garment_height = cv2.boundingRect(points)
    total_contraction_pixels = max(
        2,
        round(min(garment_width, garment_height) * erosion_ratio),
    )
    inset_pixels = max(1, round(total_contraction_pixels / 2.0))

    distance_from_background = cv2.distanceTransform(
        removal_u8,
        cv2.DIST_L2,
        5,
    )
    inner_hint = distance_from_background > float(inset_pixels)
    inner_hint &= foreground & ~protected
    inner_hint_pixel_count = int(np.count_nonzero(inner_hint))
    if inner_hint_pixel_count == 0:
        raise ValueError(
            "두꺼운 의상 수축 뒤 내부 체형 힌트가 비었습니다. "
            "수축 비율을 낮추거나 의상 마스크를 다시 확인하세요."
        )

    return BodyRestorationMaskRoles(
        clothing_removal_mask=_binary_image(removal),
        inner_body_hint=_binary_image(inner_hint),
        garment_thickness=thickness,
        contraction_applied=True,
        erosion_ratio=erosion_ratio,
        inset_pixels=inset_pixels,
        removal_pixel_count=removal_pixel_count,
        inner_hint_pixel_count=inner_hint_pixel_count,
        protected_overlap_pixel_count=protected_overlap,
        outside_foreground_pixel_count=outside_foreground,
    )
