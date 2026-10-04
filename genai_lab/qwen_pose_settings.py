"""Explicit local runtime settings. No drive, environment name, install or model load."""
from dataclasses import asdict, dataclass
import importlib.metadata
import json
import math
from pathlib import Path

from genai_lab.qwen_preservation import file_sha, require_sha

EXPECTED_VERSIONS = {"torch": "2.9.1+cu128", "diffusers": "0.38.0", "transformers": "4.57.6",
                     "bitsandbytes": "0.50.2", "gguf": "0.19.0"}


def check_runtime_versions(version=importlib.metadata.version):
    actual, errors = {}, []
    for name, expected in EXPECTED_VERSIONS.items():
        try:
            actual[name] = version(name)
        except importlib.metadata.PackageNotFoundError:
            actual[name] = None
        if actual[name] != expected:
            errors.append(f"{name}: {actual[name]} != {expected}")
    if errors:
        raise RuntimeError("Qwen 환경 버전 불일치 (생성 중단): " + "; ".join(errors))
    return actual


@dataclass(frozen=True)
class QwenPoseSettings:
    python_executable: str
    model_root: str
    gguf_file: str
    artifact_manifest: str
    artifact_manifest_sha256: str
    analysis_cache_dir: str = ""
    offload: str = "group_block1"
    model_revision: str = "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
    gguf_revision: str = "0d33d9692b4b26212297240d87b0d4719aa4fd06"
    gguf_sha256: str = "8677bac90627adbbc11efab87b1870e701c4eb3689ee865a3de8ab81b705a723"
    steps: int = 40
    true_cfg_scale: float = 4.0
    guidance_scale: float = 1.0
    default_seed: int = 209212001

    def __post_init__(self):
        if not all((self.python_executable, self.model_root, self.gguf_file, self.artifact_manifest)):
            raise ValueError("Qwen 실행·모델·해시 명세 경로 설정이 필요합니다.")
        if self.offload != "group_block1":
            raise ValueError("검증된 group_block1 외 오프로드는 지원하지 않습니다.")
        if (self.steps, self.true_cfg_scale, self.guidance_scale) != (40, 4.0, 1.0):
            raise ValueError("검증된 40단계/CFG 4.0/guidance 1.0을 사용해야 합니다.")
        for value in (self.artifact_manifest_sha256, self.gguf_sha256):
            require_sha(value)
        if type(self.default_seed) is not int or not 0 <= self.default_seed < 2**63:
            raise ValueError("seed 형식 오류")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8-sig")))

    def record(self):
        return asdict(self)


def output_dimensions(width, height):
    """Diffusers 0.38 calculate_dimensions(1024**2, ratio), without importing it."""
    if min(width, height) <= 0:
        raise ValueError("잘못된 이미지 크기")
    ratio = width / height
    w = math.sqrt(1024 * 1024 * ratio)
    h = w / ratio
    return round(w / 32) * 32, round(h / 32) * 32


def validate_model_files(settings):
    """Hash every provisioned model artifact before imports/loads; no network."""
    manifest = Path(settings.artifact_manifest)
    if file_sha(manifest) != settings.artifact_manifest_sha256:
        raise ValueError("모델 파일 명세 SHA 불일치")
    records = json.loads(manifest.read_text(encoding="utf-8"))
    root = Path(settings.model_root).resolve()
    entries = records["files"]
    required = {"model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json",
                "vae/config.json", "transformer/config.json"}
    if not required <= set(entries):
        raise ValueError("모델 설정 파일 해시 누락")
    for group in ("text_encoder/", "vae/"):
        if not any(x.startswith(group) and x.endswith((".safetensors", ".bin")) for x in entries):
            raise ValueError("모델 가중치 해시 누락: " + group)
    for group in ("tokenizer/", "processor/"):
        if not any(x.startswith(group) for x in entries):
            raise ValueError("모델 전처리 파일 해시 누락: " + group)
    # Detect additions/edits not covered by the lock, including shard indexes.
    for p in root.rglob("*"):
        if p.is_file() and ".cache" not in p.relative_to(root).parts:
            if p.relative_to(root).as_posix() not in entries:
                raise ValueError("해시 명세 밖 모델 파일: " + str(p))
    for relative, expected in entries.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("모델 명세 경로가 루트 밖입니다.")
        require_sha(expected)
        if file_sha(path) != expected:
            raise ValueError("모델 SHA 불일치: " + relative)
    if file_sha(settings.gguf_file) != settings.gguf_sha256:
        raise ValueError("GGUF SHA 불일치")
    return {"manifest_sha256": settings.artifact_manifest_sha256, "files_checked": len(entries),
            "gguf_sha256": settings.gguf_sha256}
