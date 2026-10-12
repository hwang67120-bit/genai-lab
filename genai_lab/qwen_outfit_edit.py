"""원본 편집형 의상 교체의 Qwen 요청(D-094 2단계): 원본 캐릭터(768×1024 흰 캔버스) + 의상 그림 → 기존 Qwen 작업 프로세스.
메인 환경과 Qwen 작업 프로세스가 같은 검증을 쓴다. 자동 승인·재시도는 없다. 머리 합성은 메인 환경에서 따로 한다.
메모리: 6.5GiB 할당 상한, 문장 인코더 lm_head CPU, VAE 타일(시험 09~11에서 확인한 구성).
"""
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

from PIL import Image

from genai_lab.outfit_edit import SIZE, outfit_prompt
from genai_lab.qwen_preservation import file_sha, json_sha, require_sha
from genai_lab.qwen_pose_settings import QwenPoseSettings, output_dimensions

OUTFIT_SEED = 42            # 시험 08~11과 같은 seed
OUTFIT_SEEDS = (42, 43)     # D-105: 검사에 걸리면 두 번째 seed로 한 번만 자동 재시도
RESERVED_LIMIT_GIB = 6.5
MEMORY_POLICY = {"reserved_limit_gib": RESERVED_LIMIT_GIB, "vae_tiling": True, "text_encoder_lm_head": "cpu"}
LIMITATIONS = ("한 장에 약 30분이 걸립니다. 검사(머리 위치·형태, 팔)에 걸리면 자동으로 한 번 더 그려 최대 약 1시간이 걸립니다. "
               "원본 자세·체형은 유지되지 않을 수 있습니다. 두 번 모두 걸리면 원본 머리를 붙이지 않고 그 사실을 알려 드립니다.")


@dataclass(frozen=True)
class OutfitEditSpec:
    image_path: str           # 원본 캐릭터를 맞춘 768×1024 흰 캔버스(_run_image_edit의 기준 이미지)
    image_sha256: str
    garment_path: str
    garment_sha256: str
    head_accessories: str     # keep | remove (머리 장식에만 적용)
    confirmed: bool = False
    garment_terms: tuple = ()  # 의상 그림 자동 분석에서 사용자가 확인한 옷 이름(정리 후)
    remove_props: bool = True  # False는 원인 확인 시험 전용(D-096)
    gender: str = "unspecified"  # 사용자가 확인한 캐릭터 성별(D-106)

    def record(self):
        return dict(asdict(self), garment_terms=list(self.garment_terms))

    @property
    def sha256(self):
        return json_sha(self.record())

    def verify_image(self):
        for value in (self.image_sha256, self.garment_sha256):
            require_sha(value)
        if self.confirmed is not True:
            raise ValueError("캐릭터·의상과 머리 장식 선택을 확인해 주세요.")
        if type(self.remove_props) is not bool:
            raise ValueError("소품 제거 선택 형식 오류")
        outfit_prompt(self.head_accessories, self.garment_terms, self.remove_props, self.gender)
        if file_sha(self.image_path) != self.image_sha256:
            raise ValueError("준비한 캐릭터 그림이 변경됐습니다.")
        if file_sha(self.garment_path) != self.garment_sha256:
            raise ValueError("의상 그림이 변경됐습니다.")
        with Image.open(self.image_path) as im:
            if im.mode != "RGB" or im.size != SIZE:
                raise ValueError("캐릭터 그림은 768×1024 RGB 캔버스여야 합니다.")


def prepare_outfit_spec(canvas_path, garment_path, *, head_accessories, confirmed, directory, garment_terms=(),
                        remove_props=True, gender="unspecified"):
    """두 입력을 실행 폴더로 복사해 고정한다."""
    if confirmed is not True:
        raise ValueError("캐릭터·의상과 머리 장식 선택을 확인해 주세요.")
    from genai_lab.outfit_edit import clean_garment_terms
    garment_terms = clean_garment_terms(garment_terms)
    outfit_prompt(head_accessories, garment_terms)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    character, garment = Path(canvas_path).read_bytes(), Path(garment_path).read_bytes()
    (directory / "character.png").write_bytes(character)
    suffix = Path(garment_path).suffix.lower() or ".png"
    (directory / f"garment{suffix}").write_bytes(garment)
    spec = OutfitEditSpec(str(directory / "character.png"), hashlib.sha256(character).hexdigest(),
                          str(directory / f"garment{suffix}"), hashlib.sha256(garment).hexdigest(),
                          head_accessories, confirmed, garment_terms, remove_props, gender)
    spec.verify_image()
    return spec


def make_outfit_request(spec, *, settings, seed=OUTFIT_SEED):
    spec.verify_image()
    positive = outfit_prompt(spec.head_accessories, spec.garment_terms, spec.remove_props, spec.gender)
    return {"schema_version": 1, "task_kind": "outfit_edit", "images": 2, "outfit_spec": spec.record(),
            "prompt": {"positive": positive, "negative": " ",
                       "positive_sha256": hashlib.sha256(positive.encode("utf-8")).hexdigest()},
            "settings": settings.record(), "seed": seed, "memory": dict(MEMORY_POLICY),
            "basis_size": list(SIZE), "output_size": list(output_dimensions(*SIZE))}


def validate_outfit_request(request):
    if request.get("task_kind") != "outfit_edit" or request.get("schema_version") != 1 or request.get("images") != 2:
        raise ValueError("의상 편집 요청 형식 오류")
    record = dict(request["outfit_spec"])
    spec = OutfitEditSpec(**dict(record, garment_terms=tuple(record.get("garment_terms", ()))))
    spec.verify_image()
    QwenPoseSettings(**request["settings"])
    if request.get("seed") not in OUTFIT_SEEDS or type(request.get("seed")) is not int:
        raise ValueError("의상 편집 seed 계약 오류")
    if request.get("memory") != MEMORY_POLICY:
        raise ValueError("의상 편집 메모리 계약 오류(6.5GiB 상한·VAE 타일·lm_head CPU)")
    if request["prompt"]["positive"] != outfit_prompt(spec.head_accessories, spec.garment_terms, spec.remove_props,
                                                      spec.gender) or request["prompt"]["negative"] != " ":
        raise ValueError("확인한 머리 장식 선택과 지시문이 다릅니다.")
    if request.get("basis_size") != list(SIZE) or request.get("output_size") != list(output_dimensions(*SIZE)):
        raise ValueError("의상 편집 이미지 크기 계약 오류")
    return spec


def load_qwen_settings(path):
    """작업실 기본 Qwen 실행 설정을 읽는다. 사용자가 파일을 고르지 않는다. 없으면 무엇이 없는지 알려 준다."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"옷 바꾸기 실행 설정 파일이 없습니다: {path}")
    settings = QwenPoseSettings.load(path)
    for label, target, check in (("Qwen 실행 환경", settings.python_executable, Path.is_file),
                                 ("Qwen 모델 폴더", settings.model_root, Path.is_dir),
                                 ("Qwen 모델 파일", settings.gguf_file, Path.is_file),
                                 ("모델 파일 명세", settings.artifact_manifest, Path.is_file)):
        if not check(Path(target)):
            raise ValueError(f"{label}을(를) 찾지 못했습니다: {target}\n설정 파일: {path}")
    return settings


class OutfitEditWorkflow:
    """확인한 입력을 고정한 뒤 실제 GPU 점유를 검사하고 Qwen을 시작한다."""

    def __init__(self, spec, *, gpu_probe=None):
        from genai_lab.qwen_pose_edit import assert_parent_gpu_released
        spec.verify_image()
        self.spec_sha256 = spec.sha256
        self.gpu_probe = gpu_probe or assert_parent_gpu_released
        self.events = []
        self.phase = "confirmed"

    def before_launch(self, spec):
        if self.phase != "confirmed" or spec.sha256 != self.spec_sha256:
            raise ValueError("현재 의상 편집 확인이 필요합니다.")
        spec.verify_image()
        self.events.append({"stage": "before_qwen", "probe": self.gpu_probe()})
        self.phase = "qwen_running"


def run_outfit_edit(request, directory, workflow, **kwargs):
    from genai_lab.qwen_pose_edit import _run_image_edit
    spec = validate_outfit_request(request)
    return _run_image_edit(request, directory, workflow, spec, **kwargs)
