"""특수 부위의 검증된 원본 크롭과 생성 영향 영역을 연결한다."""
from dataclasses import dataclass
from typing import Protocol
import re
import numpy as np
from PIL import Image
from genai_lab.reference_regions import read_binary_mask
from genai_lab.scene_reference import PartReference


@dataclass
class DetectedPart:
    name: str
    source_mask: Image.Image
    target_region: Image.Image | None
    scale: float

    def close(self):
        self.source_mask.close()
        if self.target_region is not None:
            self.target_region.close()


class AdditionalPartAnalyzer(Protocol):
    def analyze(self, source, output_size, *, cancelled, deadline) -> list[DetectedPart]:
        """자동 검출 구현체가 소유권을 넘긴 결과만 반환한다.

        미검출을 부재로 확정하지 않는다. 생성 위치가 불명확하면
        target_region=None으로 반환하여 승인 전에 중단한다.
        기존 얼굴·의상 영역과의 가려짐은 구현체가 검토해야 한다.
        """
        ...

    def close(self):
        """분석기가 소유한 모델·텐서를 해제한다."""
        ...


def valid_part_name(name):
    # 부위 종류 목록 대신 파일 경로/내부 예약 이름 충돌만 제한한다.
    return (isinstance(name, str)
            and re.fullmatch(r"[a-z][a-z0-9_]{0,47}", name) is not None
            and re.fullmatch(r"(con|prn|aux|nul|com[1-9]|lpt[1-9])", name) is None
            and name not in {"identity", "garment", "hair", "source", "foreground", "initial", "vibe"})


def crop_reference(source, mask, padding_ratio=.15):
    if not np.isfinite(padding_ratio) or not 0 <= padding_ratio <= .25:
        raise ValueError("참조 여백 설정 오류")
    selected = read_binary_mask(mask, source.size)
    if not selected.any():
        raise ValueError("부위 참조 영역이 비었습니다.")
    with Image.fromarray(selected.astype(np.uint8) * 255) as hard:
        box = hard.getbbox()
        with source.convert("RGBA") as rgba:
            with Image.new("RGBA", source.size, "white") as white:
                white.alpha_composite(rgba)
                with white.convert("RGB") as rgb, Image.new("RGB", source.size, "white") as background:
                    with Image.composite(rgb, background, hard) as isolated:
                        crop = isolated.crop(box)
    side = max(crop.size)
    side += 2 * round(side * padding_ratio)
    result = Image.new("RGB", (side, side), "white")
    result.paste(crop, ((side-crop.width)//2, (side-crop.height)//2))
    crop.close()
    return result


def prepare_extra_references(source, detected_parts, occupied_masks):
    """빈 영역·좌표 불일치·가려짐 불확실성을 숨겨서 대체하지 않는다."""
    occupied = np.zeros((source.height, source.width), dtype=bool)
    for mask in occupied_masks:
        occupied |= read_binary_mask(mask, source.size)
    result, names = [], set()
    try:
        if len(detected_parts) > 7:
            raise ValueError("추가 부위 참조는 최대 7개입니다.")
        for part in detected_parts:
            if not valid_part_name(part.name) or part.name in names:
                raise ValueError("부위 참조 이름 오류 또는 중복")
            names.add(part.name)
            if part.target_region is None:
                raise ValueError(f"{part.name}: 생성 위치 판단 보류")
            region = read_binary_mask(part.target_region, source.size)
            if not region.any() or np.any(region & occupied):
                raise ValueError(f"{part.name}: 빈 영역 또는 가려짐 검토 필요")
            if not np.isfinite(part.scale) or not 0 <= part.scale <= 1:
                raise ValueError("부위 참조 강도 오류")
            occupied |= region
            result.append(PartReference(part.name, crop_reference(source, part.source_mask),
                                        part.target_region.copy(), part.scale))
        return tuple(result)
    except BaseException:
        for reference in result:
            reference.close()
        raise
