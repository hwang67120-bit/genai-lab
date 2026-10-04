"""CatVTON 원시 출력과 보호 합성 결과의 실제 효과를 수치로 측정한다."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TryOnEffectMetricsResult:
    """가상 착의가 모델 출력과 최종 후보에 남긴 픽셀 증거."""

    raw_changed_inside_model_mask: int
    final_changed_inside_approved_mask: int
    discarded_by_protection_pixels: int
    mean_rgb_l1_inside: float
    mask_leakage_pixels: int
    no_effect: bool
    clip_similarity: float | None = None
    dinov2_distance: float | None = None
    color_histogram_corr: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """로그와 저장 메타데이터에 사용할 직렬화 값을 반환한다."""
        return asdict(self)


def measure_try_on_effect_metrics(
    base_person: Image.Image,
    raw_try_on_image: Image.Image,
    final_person: Image.Image,
    approved_change_mask: Image.Image,
    model_mask: Image.Image,
) -> TryOnEffectMetricsResult:
    """원시 모델 출력과 최종 보호 합성의 변경 픽셀을 정확히 센다.

    `model_mask`는 CatVTON 처리 좌표일 수 있으므로 원본 좌표로 최근접
    투영한다. CatVTON이 0.5에서 마스크를 이분화하므로 128 이상만 실제
    모델 변경 영역으로 계산한다.
    """
    expected_size = base_person.size
    for image_name, image in (
        ("CatVTON 원시 출력", raw_try_on_image),
        ("최종 보호 합성", final_person),
        ("승인 변경 마스크", approved_change_mask),
    ):
        if image.size != expected_size:
            raise ValueError(
                f"기준 이미지와 {image_name} 크기가 다릅니다: "
                f"기준={expected_size}, 입력={image.size}"
            )

    projected_model_mask = model_mask.convert("L").resize(
        expected_size,
        Image.Resampling.NEAREST,
    )
    base_rgb = base_person.convert("RGB")
    raw_rgb = raw_try_on_image.convert("RGB")
    final_rgb = final_person.convert("RGB")
    approved_mask_l = approved_change_mask.convert("L")
    try:
        base_array = np.asarray(base_rgb, dtype=np.int16)
        raw_array = np.asarray(raw_rgb, dtype=np.int16)
        final_array = np.asarray(final_rgb, dtype=np.int16)
        approved_mask_array = np.asarray(
            approved_mask_l,
            dtype=np.uint8,
        )
        model_mask_array = np.asarray(projected_model_mask, dtype=np.uint8)

        raw_l1 = np.abs(base_array - raw_array).sum(axis=-1)
        final_l1 = np.abs(base_array - final_array).sum(axis=-1)
        raw_changed = raw_l1 > 0
        final_changed = final_l1 > 0
        approved_area = approved_mask_array > 0
        model_area = model_mask_array >= 128

        raw_changed_inside = int(np.count_nonzero(raw_changed & model_area))
        final_changed_inside = int(
            np.count_nonzero(final_changed & approved_area)
        )
        discarded_inside = int(
            np.count_nonzero(raw_changed & approved_area & ~final_changed)
        )
        leakage_pixels = int(
            np.count_nonzero(final_changed & ~approved_area)
        )
        approved_pixel_count = int(np.count_nonzero(approved_area))
        mean_rgb_l1_inside = (
            float(final_l1[approved_area].mean())
            if approved_pixel_count > 0
            else 0.0
        )
    finally:
        projected_model_mask.close()
        base_rgb.close()
        raw_rgb.close()
        final_rgb.close()
        approved_mask_l.close()

    return TryOnEffectMetricsResult(
        raw_changed_inside_model_mask=raw_changed_inside,
        final_changed_inside_approved_mask=final_changed_inside,
        discarded_by_protection_pixels=discarded_inside,
        mean_rgb_l1_inside=round(mean_rgb_l1_inside, 4),
        mask_leakage_pixels=leakage_pixels,
        no_effect=final_changed_inside == 0,
    )


def create_try_on_difference_image(
    base_person: Image.Image,
    final_person: Image.Image,
    amplification: int = 4,
) -> Image.Image:
    """최종 변경량을 4배 밝힌 RGB 차이맵을 메모리 이미지로 만든다."""
    if base_person.size != final_person.size:
        raise ValueError(
            "차이맵 입력 크기가 다릅니다: "
            f"기준={base_person.size}, 결과={final_person.size}"
        )
    if amplification < 1:
        raise ValueError("차이맵 증폭 배수는 1 이상이어야 합니다.")

    base_rgb = base_person.convert("RGB")
    final_rgb = final_person.convert("RGB")
    try:
        base_array = np.asarray(base_rgb, dtype=np.int16)
        final_array = np.asarray(final_rgb, dtype=np.int16)
        difference_array = np.abs(base_array - final_array)
        visible_difference = np.clip(
            difference_array * amplification,
            0,
            255,
        ).astype(np.uint8)
        return Image.fromarray(visible_difference)
    finally:
        base_rgb.close()
        final_rgb.close()
