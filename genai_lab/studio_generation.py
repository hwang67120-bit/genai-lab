"""Desktop one-pass lifecycle. Raw generation records remain unapproved.

Only explicit input confirmation can build a request. Only an explicitly reviewed,
unchanged candidate can be exported. No legacy Base/refinement fallback is used.
"""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time

from genai_lab.onepass_generation import (OnePassRequest, OnePassCancelled,
    prepare_onepass_inputs, generate_onepass_request)
from genai_lab.onepass_generation_settings import OnePassGenerationSettings
from genai_lab.onepass_prompt_settings import OnePassPromptSettings
from genai_lab.onepass_prompt import CharacterTagGroups, AppearanceOverrides
from genai_lab.onepass_prompt_tokenizers import load_onepass_tokenizers
from genai_lab.onepass_pose import InputDecision
from genai_lab.onepass_gender import prepare_onepass_gender
from genai_lab.studio_garment_warnings import GarmentWarningSettings, garment_review
from genai_lab.studio_background import read_background


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class StudioRuntime:
    # Deployment paths are centralized. No package installation or model download.
    pose_python: Path = Path("D:/genai-cache/catvton-venv/Scripts/python.exe")
    pose_models: Path = Path("D:/genai-cache/huggingface/easy-dwpose/checkpoints")
    model_cache: Path = Path("D:/genai-cache/huggingface")
    # Existing provisioned head detector cache; deployment can override without copying models.
    head_cache: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] /
        "outputs/pose-identity-20260927/identity-v2/model-cache/hub")
    analysis_python: Path = field(default_factory=lambda: Path(sys.executable))
    generation: OnePassGenerationSettings = field(default_factory=OnePassGenerationSettings)
    prompt: OnePassPromptSettings = field(default_factory=OnePassPromptSettings)
    analysis_timeout: float = 600.0
    proportion_sketch_root: Path = Path("G:/genai-cache/models/t2i-adapter-sketch-sdxl-1.0")
    garment_warnings: GarmentWarningSettings = field(default_factory=GarmentWarningSettings.load)

    @classmethod
    def from_environment(cls):
        config = os.environ.get("GENAI_STUDIO_RUNTIME")
        if not config:
            return cls()
        values = json.loads(Path(config).read_text(encoding="utf-8"))
        allowed = {"pose_python", "pose_models", "model_cache", "analysis_python", "head_cache", "garment_warning_rules", "proportion_sketch_root"}
        if set(values) - allowed:
            raise ValueError("작업실 실행 설정에 알 수 없는 항목이 있습니다.")
        warnings = GarmentWarningSettings.load(values.pop("garment_warning_rules")) if "garment_warning_rules" in values else GarmentWarningSettings.load()
        return cls(**{key: Path(value) for key, value in values.items()}, garment_warnings=warnings)


def cpu_process(command, log, cancelled, timeout):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", ONNX_MODE="cpu",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    with Path(log).open("wb") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        start = time.monotonic()
        try:
            while process.poll() is None:
                if cancelled():
                    raise OnePassCancelled("분석을 취소했습니다.")
                if time.monotonic() - start > timeout:
                    raise RuntimeError("이미지 분석 시간이 초과되었습니다. 실행 기록을 확인해 주세요.")
                time.sleep(.1)
            if process.returncode:
                detail_path = Path(log).parent / "analysis-error.json"
                detail = "이미지 분석을 마치지 못했습니다."
                if detail_path.is_file():
                    detail = json.loads(detail_path.read_text(encoding="utf-8")).get("message", detail)
                raise RuntimeError(f"{detail}\n상세 기록: {log}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def save_analysis_reference(image, destination):
    """Flatten actual transparency on white; retain the opaque conversion path."""
    from PIL import Image
    transparent = False
    if "A" in image.getbands() or "transparency" in image.info:
        with image.convert("RGBA") as rgba:
            with rgba.getchannel("A") as alpha:
                transparent = alpha.getextrema()[0] < 255
            if transparent:
                with Image.new("RGBA", rgba.size, (255, 255, 255, 255)) as white:
                    with Image.alpha_composite(white, rgba) as composite:
                        with composite.convert("RGB") as rgb:
                            rgb.save(destination)
    if not transparent:
        with image.convert("RGB") as rgb:
            rgb.save(destination)
    return {"alpha_composited": transparent,
            "background_rgb": [255, 255, 255] if transparent else None,
            "analysis_sha256": digest(destination)}


def analyze_inputs(character, garment, directory, runtime, *, cancelled=lambda: False):
    """Snapshot references; isolate CPU models from generation and UI preferences."""
    from PIL import Image
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    references = {}
    for name, source in (("character", character), ("garment", garment)):
        source = Path(source)
        data = source.read_bytes()
        references[name] = {"source": str(source), "sha256": hashlib.sha256(data).hexdigest()}
        # Decode the same bytes we hashed; never re-open a changed reference for analysis.
        import io
        with Image.open(io.BytesIO(data)) as im:
            references[name]["preprocessing"] = save_analysis_reference(
                im, directory / f"{name}.png")
    record(directory / "references.json", references)
    script = Path(__file__).with_name("studio_analysis.py")
    for interpreter, mode in ((runtime.pose_python, "pose"), (runtime.analysis_python, "features")):
        if not interpreter.is_file():
            raise RuntimeError(f"분석 실행 환경을 찾을 수 없습니다: {interpreter}")
        cpu_process([str(interpreter), str(script), mode, str(directory),
                     str(runtime.pose_models), str(runtime.model_cache), str(runtime.head_cache)],
                    directory / f"{mode}.log", cancelled, runtime.analysis_timeout)
    analysis = json.loads((directory / "analysis.json").read_text(encoding="utf-8"))
    analysis.update(directory=str(directory), source=str(character), references=references)
    return analysis


def build_request(analysis, garment_tags, gender, *, confirmed, runtime, preferences=None,
                  tokenizers=None, seeds=None, appearance=AppearanceOverrides(), proportion_pose=None):
    """Preferences are written only by a confirmed explicit user choice."""
    if not confirmed:
        raise ValueError("입력 이미지 확인이 필요합니다.")
    if not garment_tags:
        raise ValueError("입을 옷을 확인해 주세요. 옷을 인식하지 못했으면 다른 이미지를 선택하세요.")
    from genai_lab.character_preferences import save_character_gender
    from genai_lab.reference_tag_policy import excluded_garment_tag
    if any(excluded_garment_tag(t) for t in garment_tags):
        raise ValueError("의상 설명에는 캐릭터 성별이나 체형을 넣을 수 없습니다.")
    source = analysis["source"]
    save_character_gender(source, gender, settings=preferences)
    groups = CharacterTagGroups(**{k: tuple(analysis["groups"][k]) for k in ("appearance", "body", "fixed")})
    terms = (*groups.appearance, *groups.body, *groups.fixed)
    conditions = prepare_onepass_gender(source, terms, runtime.prompt.negative_template, settings=preferences)
    # Classifications must partition the gender-resolved terms, using its normalization.
    groups = CharacterTagGroups(**{k: tuple(t.replace("_", " ") for t in getattr(groups, k))
                                   for k in ("appearance", "body", "fixed")})
    directory = Path(analysis["directory"])
    face = directory / "face.png"
    if digest(face) != analysis["face_sha256"]:
        raise ValueError("확인한 얼굴 이미지가 변경됐습니다. 다시 확인해 주세요.")
    pose_args = {}
    choice = "proceed_without_pose"
    decision = InputDecision("not_requested", (), (), ("proceed_without_pose",))
    if proportion_pose is not None:
        from genai_lab.studio_proportion import verify_pose
        decision = verify_pose(proportion_pose)
        if Path(proportion_pose["source_file"]).resolve() != (directory / "character.png").resolve():
            raise ValueError("다른 캐릭터에서 준비한 비율 입력입니다.")
        choice = "proceed_with_pose"
        pose_args = dict(control_file=Path(proportion_pose["control_file"]),
                        control_sha256=proportion_pose["control_sha256"], ip_early=proportion_pose["ip_early"])
    inputs = prepare_onepass_inputs(choice=choice, decision=decision, **pose_args,
        gender=conditions, groups=groups, garment_tags=tuple(garment_tags), pose_tags=(),
        slim=analysis["slim"], tokenizers=tokenizers or load_onepass_tokenizers(runtime.prompt),
        face_file=face, face_sha256=analysis["face_sha256"], prompt_settings=runtime.prompt, appearance=appearance)
    start = secrets.randbelow(2**62)
    request = OnePassRequest(inputs, seeds or tuple(start+i for i in range(4)), runtime.generation)
    record(directory / "approval.json", {"approved": True, "gender": gender,
        "garment_tags": list(garment_tags), "face_sha256": analysis["face_sha256"],
        "garment_review": garment_review(garment_tags, inputs.prompt.negative, runtime.garment_warnings),
        "references": analysis["references"], "pose_mode": inputs.pose_mode,
        "proportion_mode": "two_pass" if proportion_pose is not None else "off",
        "prompt": inputs.prompt.positive, "negative": inputs.prompt.negative,
        "appendage_appearance": inputs.prompt.rules["appendage_appearance"],
        "appendage_review_required": inputs.prompt.rules["appendage_review_required"]})
    return request


class StudioResults:
    """Separate human-review/export ledger; never falsify raw gate evidence."""
    def __init__(self, batch, *, appendage_review_required=False):
        if not batch.candidates:
            raise ValueError("완료된 후보가 없습니다.")
        self.batch = batch
        from genai_lab.proportion_generation import ProportionBatch
        self.two_pass = isinstance(batch, ProportionBatch)
        self.product_records = tuple(json.loads(c.record_path.read_text(encoding="utf-8"))
                                     for c in batch.candidates) if self.two_pass else ()
        self.backgrounds = tuple(None if self.two_pass else read_background(c) for c in batch.candidates)
        self.appendage_review_required = appendage_review_required or any(
            c.record.get("prompt", {}).get("rules", {}).get("appendage_review_required", False)
            for c in batch.candidates)
        self.selected = 0
        self.tail_edit = None
        self.approved_sha = None
        self.checks = {}
        self.status = "awaiting_user_review"
        self.persist()

    def persist(self, **extra):
        record(self.batch.directory / "user-review.json", {
            "status": self.status, "selected": self.selected,
            "approved_sha256": self.approved_sha, "automatic_gates_executed": False,
            "reviewer": "user", "checks": self.checks, "appendage_review_required": self.appendage_review_required,
            "background": self.backgrounds[self.selected],
            "proportion_mode": "two_pass" if self.two_pass else "off",
            **({"tail_edit": self.tail_edit} if self.tail_edit else {}), **extra})

    def select(self, index):
        if not 0 <= index < len(self.batch.candidates):
            raise ValueError("후보 번호 오류")
        self.selected = index
        self.tail_edit = None
        self.approved_sha = None
        self.checks = {}
        self.status = "awaiting_user_review"
        self.persist()

    @property
    def candidate(self):
        return self.batch.candidates[self.selected]

    @property
    def raw_path(self):
        return self.candidate.raw.path if self.two_pass else self.candidate.path

    @property
    def basis_path(self):
        if self.two_pass:
            return self.candidate.path
        background = self.backgrounds[self.selected]
        if background and background["status"] == "completed":
            return self.candidate.path.parent / "product.png"
        return self.candidate.path

    @property
    def background_notice(self):
        if self.two_pass:
            return "2단계 비율 생성·흰 배경 정리 완료 · 비율과 의상 색을 직접 확인해 주세요."
        background = self.backgrounds[self.selected]
        if not background or background["status"] != "completed":
            return "배경 정리 안 됨 · 생성 원본을 표시합니다."
        return "흰 배경 정리 완료 · 몸에 붙은 장식·줄은 남을 수 있습니다."

    @property
    def current_path(self):
        return Path(self.tail_edit["directory"]) / "product.png" if self.tail_edit else self.basis_path

    def verify_basis(self):
        raw_sha = self.verify_original()
        if self.two_pass:
            data = json.loads(self.candidate.record_path.read_text(encoding="utf-8"))
            if (data != self.product_records[self.selected] or not data.get("valid") or not data.get("completed")
                    or data.get("raw_sha256") != raw_sha or data.get("product_sha256") != digest(self.basis_path)
                    or Path(data["raw_file"]).resolve() != self.raw_path.resolve()
                    or Path(data["product_file"]).resolve() != self.basis_path.resolve()):
                raise ValueError("2단계 생성 결과나 기록이 변경됐습니다.")
            return data["product_sha256"]
        background = self.backgrounds[self.selected]
        if read_background(self.candidate) != background:
            raise ValueError("배경 정리 기록이 변경됐습니다.")
        if background:
            if background["raw_sha256"] != raw_sha:
                raise ValueError("배경 정리 기준 원본이 다릅니다.")
            if background["status"] == "completed":
                if digest(self.basis_path) != background["product_sha256"]:
                    raise ValueError("배경 정리 결과가 변경됐습니다.")
                return background["product_sha256"]
        return raw_sha

    def verify_original(self):
        candidate = self.candidate.raw if self.two_pass else self.candidate
        data = json.loads(candidate.record_path.read_text(encoding="utf-8"))
        if not data.get("valid") or not data.get("completed") or digest(candidate.path) != data.get("raw_sha256"):
            raise ValueError("생성 기록이 정상 완료 상태가 아니거나 이미지가 변경됐습니다.")
        return data["raw_sha256"]

    def verify_current(self):
        original_sha = self.verify_basis()
        if self.tail_edit:
            from genai_lab.qwen_tail_edit import tail_product_info
            if tail_product_info(self.tail_edit["directory"], original_sha) != self.tail_edit:
                raise ValueError("채택한 꼬리 편집 기록이 변경됐습니다.")
            return self.tail_edit["product_sha256"]
        return original_sha

    def adopt_tail_edit(self, directory):
        from genai_lab.qwen_tail_edit import tail_product_info
        info = tail_product_info(directory, self.verify_basis())
        self.tail_edit = info
        self.approved_sha = None
        self.checks = {}
        self.status = "awaiting_user_review"
        self.persist()

    def approve(self, checks):
        required = {"character", "garment", "exposure"}
        if self.appendage_review_required:
            required.add("appendages")
        if set(checks) != required or not all(checks.values()):
            raise ValueError("캐릭터·의상·노출 및 해당하는 꼬리·귀 확인을 완료해 주세요.")
        self.approved_sha = self.verify_current()
        self.status = "user_approved"
        self.checks = dict(checks)
        self.persist()

    def export(self, destination):
        if self.status != "user_approved" or self.verify_current() != self.approved_sha:
            raise ValueError("현재 결과를 먼저 확인해 주세요.")
        destination = Path(destination)
        if destination.suffix.lower() != ".png":
            raise ValueError("PNG 파일로 저장해 주세요.")
        sidecar = destination.with_suffix(".review.json")
        if destination.exists() or sidecar.exists():
            raise FileExistsError("같은 이름의 파일이 있습니다. 다른 이름을 선택하세요.")
        data = self.current_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != self.approved_sha:
            raise ValueError("확인한 이미지가 변경됐습니다.")
        metadata = {"source_run": str(self.candidate.record_path), "sha256": self.approved_sha,
                    "approval_record": str(self.batch.directory / "user-review.json"),
                    "reviewer": "user", "automatic_gates_executed": False}
        metadata["raw_sha256"] = self.verify_original()
        metadata["proportion_mode"] = "two_pass" if self.two_pass else "off"
        metadata["background"] = self.backgrounds[self.selected]
        if self.tail_edit:
            metadata["tail_edit"] = dict(self.tail_edit)
        created = []
        try:
            with sidecar.open("x", encoding="utf-8") as f:
                created.append(sidecar)
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            with destination.open("xb") as f:
                created.append(destination)
                f.write(data)
        except Exception:
            for path in created:
                path.unlink()  # Only files exclusively created by this export.
            raise
        self.status = "saved"
        self.persist(destination=str(destination))
        return destination
