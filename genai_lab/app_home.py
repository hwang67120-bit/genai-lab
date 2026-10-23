"""배포판(설치형) 모드(D-174): 실행기가 GENAI_APP_HOME에 설치 폴더를 알려 주면 앱은 그 안의 런타임을 쓰고,
개발 PC의 기본 경로(D:/genai-cache 등)로 넘어가지 않는다.

설치 폴더 배치(INSTALLER_PACKAGING_PREP 4-1):
  runtime-main/  주 Python(작업실·분석·Qwen 작업 프로세스)
  runtime-pose/  관절 분석 Python(DWPose, 버전이 달라 분리)
  app/           앱 코드·configs·docs
"""
import os
from pathlib import Path

ENV = "GENAI_APP_HOME"


def app_home():
    """배포판이면 설치 폴더(절대 경로 폴더), 아니면 None(개발 실행)."""
    value = os.environ.get(ENV, "").strip()
    if not value:
        return None
    home = Path(value)
    if not home.is_absolute() or not home.is_dir():
        raise ValueError(f"설치 폴더({ENV})는 이미 있는 절대 경로 폴더여야 합니다: {value}")
    return home


def runtime_python(kind, *, windowed=False):
    """배포판 런타임의 python.exe(kind: main|pose). 개발 실행이면 None."""
    home = app_home()
    if home is None:
        return None
    return home / f"runtime-{kind}" / ("pythonw.exe" if windowed else "python.exe")
