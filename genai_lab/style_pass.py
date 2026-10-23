"""그림체 후처리 한 번(D-161 시험 A, 시험 전용 — 제품·사용자 화면에 없음).

이미 만든 결과 그림 한 장을 같은 장면 그대로 그림체만 바꾼다. 지시문은 그림체 LoRA 제작자가 적은 트리거 문장 하나뿐이다
(새 자세·배경·특징 지시 없음). 출력 크기는 입력과 같아 칸끼리 겹쳐 비교할 수 있다. 메모리 정책은 편집과 같다(6.5GiB).
apply_lora=False는 같은 문장을 LoRA 없이 보내는 대조 칸이다(사용자가 고를 때만).
"""
from dataclasses import asdict, dataclass
import hashlib
import io
from pathlib import Path

from PIL import Image

from genai_lab.qwen_outfit_edit import MEMORY_POLICY, OutfitEditWorkflow
from genai_lab.qwen_pose_settings import QwenPoseSettings, output_dimensions
from genai_lab.qwen_preservation import file_sha, json_sha

SEED = 42


@dataclass(frozen=True)
class StylePassSpec:
    image_path: str             # 사용자가 판정한 결과 그림을 실행 폴더로 복사한 것
    image_sha256: str
    style_lora: str             # 트리거 문장과 LoRA 파일을 정하는 등록 이름(style_loras)
    apply_lora: bool = True
    style_strength: float = 1.0
    confirmed: bool = True

    images = 1

    def record(self):
        return asdict(self)

    @property
    def sha256(self):
        return json_sha(self.record())

    def verify_image(self):
        from genai_lab.style_loras import check_strength, style_lora
        if self.confirmed is not True or type(self.apply_lora) is not bool:
            raise ValueError("후처리 입력 확인이 필요합니다.")
        if style_lora(self.style_lora) is None:
            raise ValueError("그림체 LoRA 이름이 필요합니다(트리거 문장).")
        if self.apply_lora:
            check_strength(self.style_strength)
        elif self.style_strength != 0.0:
            raise ValueError("LoRA 없는 대조 칸은 강도 0이어야 합니다.")
        if file_sha(self.image_path) != self.image_sha256:
            raise ValueError("후처리 입력 SHA 불일치")

    def prompt(self):
        from genai_lab.style_loras import style_lora
        return style_lora(self.style_lora)["trigger"]

    @property
    def lora_name(self):
        """작업 프로세스가 얹을 LoRA. 대조 칸이면 없음."""
        return self.style_lora if self.apply_lora else ""


def prepare_style_spec(image_path, *, directory, style_lora, apply_lora=True, strength=1.0):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    data = Path(image_path).read_bytes()
    (directory / "input.png").write_bytes(data)
    spec = StylePassSpec(str(directory / "input.png"), hashlib.sha256(data).hexdigest(), style_lora,
                         apply_lora=apply_lora, style_strength=float(strength) if apply_lora else 0.0)
    spec.verify_image()
    return spec


def make_style_request(spec, *, settings):
    spec.verify_image()
    with Image.open(io.BytesIO(Path(spec.image_path).read_bytes())) as image:
        size = image.size
    positive = spec.prompt()
    return {"schema_version": 1, "task_kind": "style_pass", "images": 1, "style_spec": spec.record(),
            "prompt": {"positive": positive, "negative": " ", "positive_sha256": hashlib.sha256(positive.encode("utf-8")).hexdigest()},
            "settings": settings.record(), "seed": SEED, "memory": dict(MEMORY_POLICY),
            "basis_size": list(size), "output_size": list(output_dimensions(*size))}


def validate_style_request(request):
    if request.get("task_kind") != "style_pass" or request.get("schema_version") != 1 or request.get("images") != 1:
        raise ValueError("그림체 후처리 요청 형식 오류")
    spec = StylePassSpec(**request["style_spec"])
    spec.verify_image()
    QwenPoseSettings(**request["settings"])
    if request.get("seed") != SEED or type(request.get("seed")) is not int:
        raise ValueError("그림체 후처리 seed 계약 오류")
    if request.get("memory") != MEMORY_POLICY:
        raise ValueError("그림체 후처리 메모리 계약 오류(6.5GiB 상한·VAE 타일·lm_head CPU)")
    positive = spec.prompt()
    if request["prompt"] != {"positive": positive, "negative": " ",
                             "positive_sha256": hashlib.sha256(positive.encode("utf-8")).hexdigest()}:
        raise ValueError("그림체 후처리 지시문은 트리거 문장 하나뿐이어야 합니다.")
    with Image.open(spec.image_path) as image:
        size = image.size
    if request.get("basis_size") != list(size) or request.get("output_size") != list(output_dimensions(*size)):
        raise ValueError("그림체 후처리 크기 계약 오류")
    return spec


class StylePassWorkflow(OutfitEditWorkflow):
    """편집과 같은 GPU 점유 검사·단계 기록."""


def run_style_pass_edit(request, directory, workflow, **kwargs):
    from genai_lab.qwen_pose_edit import _run_image_edit
    spec = validate_style_request(request)
    return _run_image_edit(request, directory, workflow, spec, **kwargs)
