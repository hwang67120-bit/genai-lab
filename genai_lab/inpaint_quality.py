"""중립화 색의 잔여 후보를 표시한다. 의상 의미나 생성 성공을 판정하지 않는다."""

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class NeutralResidualDiagnostic:
    mask: Image.Image
    evaluated_pixel_count: int
    suspected_pixel_count: int
    suspected_percent: float | None

    def close(self) -> None:
        self.mask.close()


def inspect_neutral_residual(
    initial: Image.Image,
    output: Image.Image,
    approved_mask: Image.Image,
    neutral_rgb: tuple[int, int, int] = (127, 127, 127),
    tolerance: int = 8,
) -> NeutralResidualDiagnostic:
    """승인 영역 중 시작·결과 모두 중립색 인접인 픽셀을 경고 후보로 센다."""
    if initial.size != output.size or initial.size != approved_mask.size:
        raise ValueError("잔여 진단 입력 크기가 다릅니다.")
    if (len(neutral_rgb) != 3
            or any(type(value) is not int or not 0 <= value <= 255 for value in neutral_rgb)
            or type(tolerance) is not int or not 0 <= tolerance <= 255):
        raise ValueError("중립색 RGB와 허용 오차는 0~255 정수여야 합니다.")
    color = np.asarray(neutral_rgb, dtype=np.int16)
    with initial.convert("RGB") as start, output.convert("RGB") as final, approved_mask.convert("L") as mask:
        initially_neutral = np.all(np.abs(np.asarray(start, dtype=np.int16) - color) <= tolerance, axis=2)
        finally_neutral = np.all(np.abs(np.asarray(final, dtype=np.int16) - color) <= tolerance, axis=2)
        evaluated = initially_neutral & (np.asarray(mask) > 0)
        suspected = evaluated & finally_neutral
    evaluated_count = int(np.count_nonzero(evaluated))
    suspected_count = int(np.count_nonzero(suspected))
    return NeutralResidualDiagnostic(
        mask=Image.fromarray(suspected.astype(np.uint8) * 255),
        evaluated_pixel_count=evaluated_count,
        suspected_pixel_count=suspected_count,
        suspected_percent=(suspected_count / evaluated_count * 100 if evaluated_count else None),
    )
