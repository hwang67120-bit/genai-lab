"""저장 결과 비교(D-152): 저장 파일에는 AI 생성 표시(PNG 파일 정보)가 들어가 바이트가 달라지므로 픽셀과 표시로 비교한다."""
import io
from pathlib import Path

import numpy as np
from PIL import Image

from genai_lab.ai_label import read_label


def saved_as(saved, expected):
    """saved(경로)가 expected(경로 또는 PNG 바이트)와 픽셀·크기·모드가 같고 AI 생성 표시가 있으면 True."""
    data = Path(expected).read_bytes() if isinstance(expected, (str, Path)) else expected
    with Image.open(saved) as a, Image.open(io.BytesIO(data)) as b:
        same = a.mode == b.mode and a.size == b.size and np.array_equal(np.asarray(a), np.asarray(b))
    return same and read_label(saved) is not None
