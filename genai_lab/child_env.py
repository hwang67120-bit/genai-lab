"""제품 하위 프로세스(분석·관절·머리 분할·Qwen 작업 프로세스)에 넘길 환경 변수(D-170).

부모 환경을 통째로 넘기지 않고, 실행에 필요한 이름만 골라 넘긴 뒤 제품 고정값(오프라인 등)을 덮어쓴다. 전부 지우지도 않는다:
- Windows·사용자 폴더·임시 폴더·실행 경로(PATH): Python·torch·onnxruntime이 DLL·임시 파일·사용자 폴더를 찾는 데 쓴다.
- GPU·스레드 설정(CUDA_*·PYTORCH_*·NVIDIA_*·OMP_NUM_THREADS 등): 사용자가 정했으면 그대로 따른다.
- 앱 설정(GENAI_*)·허깅페이스 캐시 위치(HF_HOME·HF_HUB_CACHE).
넘기지 않는 예: PYTHONPATH·PYTHONHOME·PYTHONSTARTUP·VIRTUAL_ENV·CONDA_* — 다른 Python 설치·개발 환경이 하위 프로세스에 섞이는 것을 막는다.
관찰 도구처럼 개발 중에 더 넘겨야 하면 GENAI_CHILD_ENV_EXTRA에 이름을 쉼표로 적는다(명시적으로만).
"""
import os

SYSTEM = ("SYSTEMROOT", "WINDIR", "SYSTEMDRIVE", "COMSPEC", "PATHEXT", "PATH", "TEMP", "TMP", "USERPROFILE", "HOMEDRIVE",
          "HOMEPATH", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMW6432",
          "COMMONPROGRAMFILES", "COMMONPROGRAMFILES(X86)", "COMMONPROGRAMW6432", "ALLUSERSPROFILE", "PUBLIC",
          "USERNAME", "USERDOMAIN", "COMPUTERNAME", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
          "PROCESSOR_IDENTIFIER", "PROCESSOR_LEVEL", "PROCESSOR_REVISION", "OS", "HOME", "LANG", "LC_ALL", "LC_CTYPE")
APP = ("GENAI_APP_HOME", "GENAI_MODEL_HOME", "GENAI_CHARACTER_MEMORY", "GENAI_STUDIO_RUNTIME", "GENAI_QWEN_SETTINGS", "GENAI_VISION_SETTINGS",
       "GENAI_STUDIO_DEVELOPER", "HF_HOME", "HF_HUB_CACHE")
THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
GPU_PREFIXES = ("CUDA_", "PYTORCH_", "NVIDIA_", "CUBLAS_", "CUDNN_")
FIXED = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
EXTRA = "GENAI_CHILD_ENV_EXTRA"
HF_CREDENTIALS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def hf_token_path(source=None):
    """huggingface_hub가 토큰을 찾는 파일 위치를 앱 데이터의 빈 경로로(D-172). 이 파일은 만들지 않는다.
    오프라인 설정만으로는 막히지 않는다 — 라이브러리가 요청 전에 사용자 토큰 파일(~/.cache/huggingface/token)을 먼저 읽는다."""
    import tempfile
    source = os.environ if source is None else source
    base = source.get("APPDATA") or source.get("LOCALAPPDATA") or tempfile.gettempdir()
    return os.path.join(base, "GenAILab", "hf-token-disabled", "token")


def isolate_hf_credentials(environ=None):
    """앱 프로세스 시작 때 한 번: 사용자 허깅페이스 인증정보를 읽지 않게 한다. 사용자 토큰 파일은 읽지도 고치지도 않는다."""
    import sys
    environ = os.environ if environ is None else environ
    for name in HF_CREDENTIALS:
        environ.pop(name, None)
    environ["HF_TOKEN_PATH"] = hf_token_path(environ)
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None:                        # 이미 불러왔으면 계산된 상수도 바꾼다
        constants.HF_TOKEN_PATH = environ["HF_TOKEN_PATH"]
    return environ["HF_TOKEN_PATH"]


def child_environment(source=None, **overrides):
    """하위 프로세스 환경 변수. overrides는 마지막에 덮어쓴다(예: CUDA_VISIBLE_DEVICES="" — CPU 분석)."""
    source = os.environ if source is None else source
    keep = {name.upper() for name in SYSTEM + APP + THREADS}
    extra = {name.strip().upper() for name in source.get(EXTRA, "").split(",") if name.strip()}
    env = {key: value for key, value in source.items()
           if key.upper() in keep or key.upper().startswith(GPU_PREFIXES) or key.upper() in extra}
    if extra:
        env[EXTRA] = source[EXTRA]
    env.update(FIXED)
    env["HF_TOKEN_PATH"] = hf_token_path(source)   # 하위 프로세스도 사용자 토큰 파일을 읽지 않는다(D-172)
    env.update({key: str(value) for key, value in overrides.items()})
    return env
