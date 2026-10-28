"""자유 생성의 Qwen 요청(D-123): 기준 이미지(768×1024 캔버스) + 머리 크롭(시각 참조) → 기존 Qwen 작업 프로세스(task_kind free_generation).
메인 환경과 작업 프로세스가 같은 검증을 쓴다(지시문은 명세 값에서 다시 계산해 대조). 메모리 정책은 편집과 같다.
"""
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

from PIL import Image

from genai_lab.free_generation import SEEDS, free_prompt
from genai_lab.outfit_edit import SIZE
from genai_lab.qwen_outfit_edit import MEMORY_POLICY, OutfitEditWorkflow
from genai_lab.qwen_pose_settings import QwenPoseSettings, output_dimensions
from genai_lab.qwen_preservation import file_sha, json_sha


@dataclass(frozen=True)
class FreeGenerationSpec:
    image_path: str            # 기준 이미지(편집 승인 결과 또는 원본)를 맞춘 768×1024 캔버스
    image_sha256: str
    reference_path: str        # 머리 크롭(시각 참조, image 2)
    reference_sha256: str
    request_text: str          # 사용자 요청(연출). 비어 있으면 무작위 자세
    gender: str = "unspecified"
    body_parts: tuple = None
    features_text: str = ""    # 확정 특징표에서 만든 문장(free_generation.features_sentence)
    base_kind: str = "edited"  # edited | original
    confirmed: bool = False
    selection_seed: int = 42  # 연출 선택 seed. 생성 재시도 seed와 독립
    reference_box: tuple = (0, 0, 768, 1024)
    reference_scope: str = "head"   # head | wide — 넓은 크롭이면 지시문이 head close-up이라고 부르지 않는다(D-126 ③)
    skeleton_path: str = ""         # D-131 자세 골격 그림(image 3). 비면 그림 2장(기존과 같음)
    skeleton_sha256: str = ""
    skeleton_joints: tuple = ()     # 골격을 그린 관절 기록 — 작업 프로세스가 다시 그려 대조한다
    pose_key: str = ""
    absent_parts_negative: bool = False   # 요청 28 시험 B(기본 꺼짐): 없는 부위를 지시문 대신 부정 지시문으로
    accelerator: str = ""                 # D-141 시험용 가속(qwen_accelerators). 비면 검증된 40단계/CFG 4.0
    style_lora: str = ""                  # D-159 그림체 LoRA 맛보기(style_loras, 시험 전용). 비면 없음
    style_strength: float = 0.0

    @property
    def images(self):
        return 3 if self.skeleton_path else 2

    def record(self):
        return dict(asdict(self), body_parts=None if self.body_parts is None else list(self.body_parts))

    @property
    def sha256(self):
        return json_sha(self.record())

    def verify_image(self):
        if self.confirmed is not True or self.base_kind not in ("edited", "original"):
            raise ValueError("기준 종류와 사용자 확인이 필요합니다.")
        if self.gender not in ("male", "female", "unspecified"):
            raise ValueError("성별 형식 오류")
        if self.body_parts is None or not set(self.body_parts) <= {"ears", "tail", "wings"}:
            raise ValueError("부위를 확인해 주세요(없으면 빈 목록).")
        if type(self.selection_seed) is not int or not 0 <= self.selection_seed < 2**63:
            raise ValueError("연출 선택 seed 계약 오류")
        from genai_lab.free_generation import check_request
        checked = check_request(self.request_text)
        if not checked["ok"]:
            raise ValueError(checked["reason"])
        for path, expected in ((self.image_path, self.image_sha256), (self.reference_path, self.reference_sha256)):
            if file_sha(path) != expected:
                raise ValueError("자유 생성 입력 그림이 변경됐습니다.")
        with Image.open(self.image_path) as im:
            if im.mode != "RGB" or im.size != SIZE:
                raise ValueError("기준 그림은 768×1024 RGB 캔버스여야 합니다.")
            box = self.reference_box
            if len(box) != 4 or any(type(x) is not int for x in box) or not (0 <= box[0] < box[2] <= SIZE[0] and 0 <= box[1] < box[3] <= SIZE[1]):
                raise ValueError("머리 참조 좌표 오류")
            if self.reference_scope not in ("head", "wide"):
                raise ValueError("참조 범위 오류")
            with Image.open(self.reference_path) as reference:
                if reference.mode != "RGB" or reference.size != (box[2]-box[0], box[3]-box[1]) or reference.tobytes() != im.crop(box).tobytes():
                    raise ValueError("추가 참조는 같은 기준 이미지의 크롭이어야 합니다.")
        if self.skeleton_path:
            import numpy as np
            from genai_lab.pose_guidance import POSE_TEMPLATES, draw_skeleton
            if self.pose_key not in POSE_TEMPLATES or len(self.skeleton_joints) != 18:
                raise ValueError("지원 자세 골격 계약 오류")
            if file_sha(self.skeleton_path) != self.skeleton_sha256:
                raise ValueError("자세 골격 그림이 변경됐습니다.")
            records = [dict(joint_name=r[0], x=float(r[1]), y=float(r[2]), detected=bool(r[3])) for r in self.skeleton_joints]
            with Image.open(self.skeleton_path) as skeleton:
                if skeleton.mode != "RGB" or skeleton.size != SIZE or not np.array_equal(np.asarray(skeleton), draw_skeleton(records, SIZE)):
                    raise ValueError("골격 그림은 기록된 관절로 코드가 그린 그림이어야 합니다.")
        elif self.skeleton_sha256 or self.skeleton_joints or self.pose_key:
            raise ValueError("골격 없이 골격 기록만 있습니다.")
        if self.accelerator:
            from genai_lab.qwen_accelerators import accelerator
            accelerator(self.accelerator)          # 알 수 없는 이름은 ValueError
        if self.style_lora:
            from genai_lab.style_loras import check_strength, style_lora
            style_lora(self.style_lora)
            check_strength(self.style_strength)
        elif self.style_strength:
            raise ValueError("그림체 LoRA 없이 강도만 있습니다.")

    def prompt(self, seed):
        text = free_prompt(self.features_text, self.gender, self.body_parts, self.request_text, self.selection_seed,
                           reference_scope=self.reference_scope, skeleton=bool(self.skeleton_path),
                           absent_parts_negative=self.absent_parts_negative)
        if self.style_lora:                 # 그림체 LoRA의 지시 문장(제작자 표기)을 맨 앞에
            from genai_lab.style_loras import style_lora
            text = style_lora(self.style_lora)["trigger"] + " " + text
        return text

    def negative(self):
        from genai_lab.free_generation import absent_parts_negative_prompt
        return absent_parts_negative_prompt(self.body_parts) if self.absent_parts_negative else " "


def prepare_free_spec(canvas_path, reference_path, *, directory, request_text, gender="unspecified", body_parts=None,
                      features_text="", base_kind="edited", confirmed=True, reference_box=(0, 0, 768, 1024), selection_seed=42,
                      reference_scope="head", skeleton_path=None, skeleton_joints=(), pose_key="", absent_parts_negative=False,
                      accelerator="", style_lora="", style_strength=0.0):
    """두 입력을 실행 폴더로 복사해 고정한다."""
    if confirmed is not True:
        raise ValueError("기준 그림과 요청을 확인해 주세요.")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    base, reference = Path(canvas_path).read_bytes(), Path(reference_path).read_bytes()
    (directory / "base.png").write_bytes(base)
    (directory / "reference.png").write_bytes(reference)
    skeleton = dict()
    if skeleton_path:
        data = Path(skeleton_path).read_bytes()
        (directory / "skeleton.png").write_bytes(data)
        skeleton = dict(skeleton_path=str(directory / "skeleton.png"), skeleton_sha256=hashlib.sha256(data).hexdigest(),
                        skeleton_joints=tuple((r["joint_name"], r["x"], r["y"], bool(r["detected"])) for r in skeleton_joints),
                        pose_key=pose_key)
    spec = FreeGenerationSpec(str(directory / "base.png"), hashlib.sha256(base).hexdigest(),
                              str(directory / "reference.png"), hashlib.sha256(reference).hexdigest(),
                              (request_text or "").strip(), gender, None if body_parts is None else tuple(body_parts),
                              features_text or "", base_kind, confirmed, selection_seed=selection_seed, reference_box=tuple(reference_box),
                              reference_scope=reference_scope, absent_parts_negative=absent_parts_negative, accelerator=accelerator,
                              style_lora=style_lora, style_strength=float(style_strength), **skeleton)
    spec.verify_image()
    spec.prompt(SEEDS[0])          # 요청 문장 검사(외형·의상·안전) — 여기서 걸리면 ValueError
    return spec


def make_free_request(spec, *, settings, seed=SEEDS[0]):
    spec.verify_image()
    positive = spec.prompt(seed)
    return {"schema_version": 1, "task_kind": "free_generation", "images": spec.images, "free_spec": spec.record(),
            "prompt": {"positive": positive, "negative": spec.negative(),
                       "positive_sha256": hashlib.sha256(positive.encode("utf-8")).hexdigest()},
            "settings": settings.record(), "seed": seed, "memory": dict(MEMORY_POLICY),
            "basis_size": list(SIZE), "output_size": list(output_dimensions(*SIZE))}


def validate_free_request(request):
    if request.get("task_kind") != "free_generation" or request.get("schema_version") != 1:
        raise ValueError("자유 생성 요청 형식 오류")
    record = dict(request["free_spec"])
    parts = record.get("body_parts")
    spec = FreeGenerationSpec(**dict(record, body_parts=None if parts is None else tuple(parts),
                                     skeleton_joints=tuple(tuple(r) for r in record.get("skeleton_joints") or ())))
    if request.get("images") != spec.images:
        raise ValueError("자유 생성 그림 수 계약 오류")
    spec.verify_image()
    QwenPoseSettings(**request["settings"])
    if request.get("seed") not in SEEDS or type(request.get("seed")) is not int:
        raise ValueError("자유 생성 seed 계약 오류")
    if request.get("memory") != MEMORY_POLICY:
        raise ValueError("자유 생성 메모리 계약 오류(6.5GiB 상한·VAE 타일·lm_head CPU)")
    if request["prompt"].get("positive_sha256") != hashlib.sha256(spec.prompt(request["seed"]).encode("utf-8")).hexdigest():
        raise ValueError("자유 생성 지시문 SHA 오류")
    if request["prompt"]["positive"] != spec.prompt(request["seed"]) or request["prompt"]["negative"] != spec.negative():
        raise ValueError("확인한 요청과 지시문이 다릅니다.")
    if request.get("basis_size") != list(SIZE) or request.get("output_size") != list(output_dimensions(*SIZE)):
        raise ValueError("자유 생성 이미지 크기 계약 오류")
    return spec


class FreeGenerationWorkflow(OutfitEditWorkflow):
    """편집과 같은 GPU 점유 검사·단계 기록."""


def run_free_generation_edit(request, directory, workflow, **kwargs):
    from genai_lab.qwen_pose_edit import _run_image_edit
    spec = validate_free_request(request)
    return _run_image_edit(request, directory, workflow, spec, **kwargs)
