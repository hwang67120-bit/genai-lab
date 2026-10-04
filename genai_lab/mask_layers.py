"""자동 의상 후보 + 사용자 추가/제외 + 보호 우선순위의 단일 계산."""

from dataclasses import dataclass
import numpy as np
from PIL import Image

from genai_lab.removal_diagnostics import trace_removal


@dataclass
class MaskLayerResult:
    images: dict[str, Image.Image]
    metrics: dict[str, int]

    def close(self):
        for image in self.images.values():
            image.close()


def _binary(image):
    with image.convert("L") as gray:
        return np.asarray(gray).copy() >= 128


@trace_removal("mask_layer_priority")
def synthesize_mask_layers(
    source: Image.Image,
    auto_clothing: Image.Image,
    auto_protection: Image.Image,
    face_hair: Image.Image,
    user_add: Image.Image,
    user_keep: Image.Image,
    user_exclude: Image.Image,
    foreground: Image.Image,
    use_automatic_base: bool = True,
) -> MaskLayerResult:
    masks = (auto_clothing, auto_protection, face_hair, user_add,
             user_keep, user_exclude, foreground)
    if any(mask.size != source.size for mask in masks):
        raise ValueError("마스크 레이어와 기준 이미지 크기가 다릅니다.")
    auto, protection, identity, manual, keep, exclude, fg = map(_binary, masks)
    fixed = identity | keep | exclude
    effective = fixed | (protection & ~manual)
    requested = (auto if use_automatic_base else np.zeros_like(auto)) | manual
    final = requested & fg & ~effective
    conflict = manual & fixed & ~exclude
    overridden = manual & protection & ~fixed & fg
    outside = manual & ~fg
    excluded = requested & exclude
    # 색은 진단용 복사본에만 입힌다. 원본/실행 입력은 변경하지 않는다.
    with source.convert("RGB") as rgb:
        overlay = np.array(rgb)
    for region, color in (
        (final, (255, 60, 60)), (fixed, (50, 100, 255)),
        (overridden, (40, 220, 80)), (conflict, (255, 220, 0)),
        (outside, (255, 70, 220)),
    ):
        overlay[region] = np.rint(overlay[region] * .4 + np.array(color) * .6).astype(np.uint8)
    arrays = {
        "final_mask": final, "requested_mask": requested,
        "effective_protection": effective, "fixed_protection": fixed,
        "face_hair_reference": identity,
        "manual_conflict": conflict, "manual_override": overridden,
        "manual_outside": outside, "user_excluded": excluded,
        "user_add": manual, "auto_clothing": auto,
    }
    return MaskLayerResult(
        {**{name: Image.fromarray(array.astype(np.uint8) * 255)
            for name, array in arrays.items()}, "overlay": Image.fromarray(overlay)},
        {name + "_pixels": int(array.sum()) for name, array in arrays.items()},
    )
