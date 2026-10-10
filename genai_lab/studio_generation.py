"""데스크톱 1회 생성의 수명을 관리한다. 원본 생성 기록은 미승인 상태다. 명시적 입력 확인만 요청을 만들 수 있고, 검토받고 변경되지 않은 후보만 내보낼 수
있다. 구형 기준 생성·정밀화로의 대체는 없다.
"""
from dataclasses import dataclass, field, replace
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
from genai_lab.skin_tone import SkinRecommendationSettings


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class StudioRuntime:
    # 배포 경로는 한곳에서 관리한다. 패키지 설치나 모델 다운로드는 하지 않는다.
    pose_python: Path = Path("D:/genai-cache/catvton-venv/Scripts/python.exe")
    pose_models: Path = Path("D:/genai-cache/huggingface/easy-dwpose/checkpoints")
    model_cache: Path = Path("D:/genai-cache/huggingface")
    # 기존 머리 검출 캐시를 사용한다. 배포에서 모델 복사 없이 경로를 바꿀 수 있다.
    head_cache: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] /
        "outputs/pose-identity-20260927/identity-v2/model-cache/hub")
    analysis_python: Path = field(default_factory=lambda: Path(sys.executable))
    generation: OnePassGenerationSettings = field(default_factory=OnePassGenerationSettings)
    prompt: OnePassPromptSettings = field(default_factory=OnePassPromptSettings)
    quality_tags: bool = False  # 최종 기본값 결정 전까지 기존 프롬프트를 유지한다.
    finishing: bool = False
    finishing_face_reference: bool = False
    analysis_timeout: float = 600.0
    proportion_sketch_root: Path = Path("G:/genai-cache/models/t2i-adapter-sketch-sdxl-1.0")
    head_semantic_repo: Path = Path("D:/genai-cache/tools/see-through")
    head_semantic_checkpoint: Path = Path("D:/genai-cache/huggingface/models--24yearsold--l2d_sam_iter2/snapshots/0b0608310fb7a89e32ecd7397147a249540cf1e7/checkpoint-18000.pt")
    head_paste_timeout: float = 1200.0
    skin_recommendation: SkinRecommendationSettings = field(default_factory=SkinRecommendationSettings)
    garment_warnings: GarmentWarningSettings = field(default_factory=GarmentWarningSettings.load)

    def __post_init__(self):
        import math
        if not isinstance(self.head_paste_timeout, (int, float)) or not math.isfinite(self.head_paste_timeout) or self.head_paste_timeout <= 0:
            raise ValueError("머리 작업 시간 한도는 양의 유한값이어야 합니다.")
        if any(type(value) is not bool for value in (self.quality_tags, self.finishing, self.finishing_face_reference)):
            raise ValueError("품질 태그·마무리 설정은 bool이어야 합니다.")

    @classmethod
    def from_environment(cls):
        config = os.environ.get("GENAI_STUDIO_RUNTIME")
        if not config:
            return cls()
        values = json.loads(Path(config).read_text(encoding="utf-8"))
        allowed = {"pose_python", "pose_models", "model_cache", "analysis_python", "head_cache", "garment_warning_rules", "proportion_sketch_root", "quality_tags", "finishing", "finishing_face_reference", "skin_recommendation", "head_semantic_repo", "head_semantic_checkpoint", "head_paste_timeout"}
        if set(values) - allowed:
            raise ValueError("작업실 실행 설정에 알 수 없는 항목이 있습니다.")
        skin = SkinRecommendationSettings(**values.pop("skin_recommendation", {}))
        warnings = GarmentWarningSettings.load(values.pop("garment_warning_rules")) if "garment_warning_rules" in values else GarmentWarningSettings.load()
        return cls(**{key: value if key in ("quality_tags", "finishing", "finishing_face_reference", "head_paste_timeout") else Path(value)
                      for key, value in values.items()}, garment_warnings=warnings, skin_recommendation=skin)


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
    """실제 투명 영역만 흰 배경에 합성한다. 불투명 입력의 변환 경로는 유지한다."""
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
    """참조를 고정하고 CPU 모델을 생성과 화면 선택 저장소에서 분리한다."""
    from PIL import Image
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    references = {}
    for name, source in (("character", character), ("garment", garment)):
        source = Path(source)
        data = source.read_bytes()
        references[name] = {"source": str(source), "sha256": hashlib.sha256(data).hexdigest()}
        # 해시를 계산한 같은 바이트를 읽는다. 변경된 참조 파일을 다시 열어 분석하지 않는다.
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
    """사용자가 명시적으로 확인한 선택만 저장한다."""
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
    # 성별이 정리된 태그를 같은 정규화 규칙으로 빠짐없이 분류해야 한다.
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
    tokens = tokenizers or load_onepass_tokenizers(runtime.prompt)
    inputs = prepare_onepass_inputs(choice=choice, decision=decision, **pose_args,
        gender=conditions, groups=groups, garment_tags=tuple(garment_tags), pose_tags=(),
        slim=analysis["slim"], tokenizers=tokens,
        face_file=face, face_sha256=analysis["face_sha256"], prompt_settings=runtime.prompt, appearance=appearance)
    if appearance.skin_tone.value is not None:
        # 유효한 문구에 선택값을 적용한 뒤 저장한다. 다음에는 이 파일의 선택값을 불러온다.
        from genai_lab.character_preferences import save_character_skin_tone
        save_character_skin_tone(source, appearance.skin_tone.value, settings=preferences)
    if runtime.quality_tags and proportion_pose is None:
        from genai_lab.finishing_prompt import apply_quality_format
        inputs = replace(inputs, prompt=apply_quality_format(inputs.prompt, tokens))
    start = secrets.randbelow(2**62)
    request = OnePassRequest(inputs, seeds or tuple(start+i for i in range(4)), runtime.generation)
    record(directory / "approval.json", {"approved": True, "gender": gender,
        "garment_tags": list(garment_tags), "face_sha256": analysis["face_sha256"],
        "garment_review": garment_review(garment_tags, inputs.prompt.negative, runtime.garment_warnings),
        "references": analysis["references"], "pose_mode": inputs.pose_mode,
        "proportion_mode": "two_pass" if proportion_pose is not None else "off",
        "prompt": inputs.prompt.positive, "negative": inputs.prompt.negative,
        **({"skin_tone": inputs.prompt.rules["skin_tone"]} if "skin_tone" in inputs.prompt.rules else {}),
        "appendage_appearance": inputs.prompt.rules["appendage_appearance"],
        "appendage_review_required": inputs.prompt.rules["appendage_review_required"],
        **({"quality_format":"diagnostic_2b", "quality_default_decision":"pending"}
           if runtime.quality_tags and proportion_pose is None else {})})
    return request


class StudioResults:
    """사용자 검토·내보내기 기록을 분리한다. 원본 검사 근거를 바꾸지 않는다."""
    def __init__(self, batch, *, appendage_review_required=False):
        if not batch.candidates:
            raise ValueError("완료된 후보가 없습니다.")
        self.batch = batch
        from genai_lab.proportion_generation import ProportionBatch
        self.two_pass = isinstance(batch, ProportionBatch)
        self.product_records = tuple(json.loads(c.record_path.read_text(encoding="utf-8"))
                                     for c in batch.candidates) if self.two_pass else ()
        self.backgrounds = tuple(None if self.two_pass else read_background(c) for c in batch.candidates)
        from genai_lab.studio_finishing import read_finishing
        self.finishings = tuple(None if self.two_pass else read_finishing(c) for c in batch.candidates)
        self.appendage_review_required = appendage_review_required or any(
            c.record.get("prompt", {}).get("rules", {}).get("appendage_review_required", False)
            for c in batch.candidates)
        self.auto_head = getattr(batch, "auto_head", None)
        self.head_records = ()
        if self.auto_head:
            if len(self.auto_head["items"]) != len(batch.candidates):
                raise ValueError("자동 머리 결과 수가 다릅니다.")
            self.head_records = tuple(json.loads((Path(item["directory"]) / "result.json").read_text(encoding="utf-8"))
                if item["status"] == "completed" else None for item in self.auto_head["items"])
        self.show_before_head = False
        self.rejections = []
        self.identity_reports = {}
        self.selected = 0
        self.tail_edit = None
        self.optional_finishing = None
        self.approved_sha = None
        self.checks = {}
        self.status = "awaiting_user_review"
        self.persist()

    def persist(self, **extra):
        record(self.batch.directory / "user-review.json", {
            "status": self.status, "selected": self.selected,
            "head_view": self.head_view, "auto_head": self.auto_head, "rejections": self.rejections,
            "approved_sha256": self.approved_sha, "automatic_gates_executed": False,
            "reviewer": "user", "identity_report": self.identity_report_record, "checks": self.checks, "appendage_review_required": self.appendage_review_required,
            "background": self.active_background,
            "finishing_applied": bool(self.active_finishing),
            **({"finishing": self.active_finishing} if self.active_finishing else {}),
            **({"optional_finishing": self.optional_finishing} if self.optional_finishing else {}),
            "proportion_mode": "two_pass" if self.two_pass else "off",
            **({"tail_edit": self.tail_edit} if self.tail_edit else {}), **extra})

    @property
    def identity_report_record(self):
        """측정값이 없거나 바뀌어도 사용자 승인과 이미지 저장은 막지 않는다."""
        try:
            info = self.identity_reports.get(digest(self.current_path))
            if info is None:
                return {"status": "pending_or_unavailable", "informational_only": True}
            if digest(info["path"]) != info["sha256"]:
                return {"status": "changed", "informational_only": True}
            return {"status": "completed", "informational_only": True, **info}
        except OSError:
            return {"status": "unavailable", "informational_only": True}

    def select(self, index):
        if not 0 <= index < len(self.batch.candidates):
            raise ValueError("후보 번호 오류")
        self.selected = index
        self.tail_edit = None
        self.optional_finishing = None
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
        if self.optional_finishing:
            return Path(self.optional_finishing["product_file"])
        if self.active_head and not self.show_before_head:
            return Path(self.active_head["product_file"])
        return self.original_basis_path

    @property
    def original_basis_path(self):
        if self.two_pass:
            return self.candidate.path
        background = self.backgrounds[self.selected]
        if background and background["status"] == "completed":
            return self.candidate.path.parent / "product.png"
        if self.finishings[self.selected]:
            return self.candidate.path.parent / "finished.png"
        return self.candidate.path

    @property
    def active_head(self):
        """화면에서는 고정한 제품 정보를 읽는다. 전체 모델 검사는 승인·저장 전에 수행한다."""
        return self.head_records[self.selected] if self.auto_head else None

    @property
    def head_view(self):
        return "before" if self.show_before_head or not self.active_head else "pasted"

    @property
    def head_notice(self):
        if not self.auto_head:
            return ""
        item = self.auto_head["items"][self.selected]
        if item["status"] != "completed":
            return "머리 붙이기 건너뜀: " + item["reason"]
        return ("붙이기 전 결과" if self.show_before_head else "원본 머리 붙이기 완료") + " · 테두리·작은 조각은 남을 수 있습니다."

    def toggle_head_view(self):
        """두 보기를 한 후보 안에서 바꾼다. 보기를 바꾸면 이전 승인은 해제한다."""
        if not self.active_head:
            return
        self.show_before_head = not self.show_before_head
        self.select(self.selected)

    def reject_and_next(self):
        """현재 보기의 미승인을 기록하고 같은 보기 방식으로 다음 후보로 이동한다."""
        self.rejections.append(dict(selected=self.selected, seed=self.candidate.seed,
            head_view=self.head_view, status="not_approved", sha256=self.verify_current()))
        self.select((self.selected + 1) % len(self.batch.candidates))
        return self.selected

    @property
    def head_paste_unavailable_reason(self):
        if not self.two_pass:
            return "원본 비율 참고로 만든 결과에서만 사용할 수 있습니다."
        if self.auto_head:
            return "머리는 생성 뒤 자동으로 처리됩니다."
        if self.tail_edit or self.optional_finishing:
            return "머리 붙이기는 꼬리 편집·마무리 전에 적용해 주세요."
        if self.selected >= len(self.product_records):
            return "완료된 비율 생성 기록이 없습니다."
        if self.product_records[self.selected].get("head_paste"):
            return "이미 원본 머리를 붙인 후보입니다. 원래 후보를 골라 주세요."
        return ""

    def add_head_candidate(self, directory):
        """검사된 파생 후보를 추가한다. 선택·저장은 기존 사용자 승인 흐름을 따른다."""
        from genai_lab.studio_head_paste import head_product_info
        from genai_lab.proportion_generation import ProportionCandidate
        from genai_lab.proportion_inputs import require
        require(not self.head_paste_unavailable_reason, self.head_paste_unavailable_reason)
        info = head_product_info(directory, self.verify_original())
        require(info["before_sha256"] == self.verify_current(), "머리 붙이기의 기준 후보가 다릅니다.")
        data = {**self.product_records[self.selected], "product_file": info["product_file"],
                "product_sha256": info["product_sha256"], "head_paste": info}
        path = Path(directory) / "candidate.json"
        record(path, data)
        candidate = ProportionCandidate(self.candidate.seed, self.candidate.raw, Path(info["product_file"]), path, data)
        previous = (self.batch, self.product_records, self.backgrounds, self.finishings, self.selected,
                    self.tail_edit, self.optional_finishing, self.approved_sha, self.checks, self.status)
        try:
            self.batch = replace(self.batch, candidates=(*self.batch.candidates, candidate))
            self.product_records = (*self.product_records, data)
            self.backgrounds = (*self.backgrounds, None)
            self.finishings = (*self.finishings, None)
            self.select(len(self.batch.candidates) - 1)
        except Exception:
            (self.batch, self.product_records, self.backgrounds, self.finishings, self.selected,
             self.tail_edit, self.optional_finishing, self.approved_sha, self.checks, self.status) = previous
            raise

    @property
    def finishing_unavailable_reason(self):
        if self.tail_edit:
            return "꼬리를 고친 뒤에는 고화질 마무리를 적용할 수 없습니다."
        if self.two_pass:
            return "원본 비율 참고로 만든 결과에는 이번 마무리를 적용하지 않습니다."
        if self.optional_finishing or self.finishings[self.selected]:
            return "이미 마무리한 결과입니다. 중복 마무리는 지원하지 않습니다."
        return ""

    @property
    def active_finishing(self):
        return self.optional_finishing["finishing"] if self.optional_finishing else self.finishings[self.selected]

    @property
    def active_background(self):
        return self.optional_finishing["background"] if self.optional_finishing else self.backgrounds[self.selected]

    def adopt_finishing(self, directory):
        from genai_lab.studio_optional_finishing import finishing_info
        if self.finishing_unavailable_reason:
            raise ValueError(self.finishing_unavailable_reason)
        info = finishing_info(directory, self.verify_original(), self.verify_basis())
        previous = (self.optional_finishing, self.approved_sha, self.checks, self.status)
        try:
            self.optional_finishing = info
            self.approved_sha = None
            self.checks = {}
            self.status = "awaiting_user_review"
            self.persist()
        except Exception:
            self.optional_finishing, self.approved_sha, self.checks, self.status = previous
            raise

    @property
    def background_notice(self):
        if self.optional_finishing:
            return "고화질 마무리·흰 배경 정리 완료 · 눈색과 장식을 직접 확인해 주세요."
        if self.two_pass:
            return "2단계 비율 생성·흰 배경 정리 완료 · 비율과 의상 색을 직접 확인해 주세요."
        background = self.backgrounds[self.selected]
        if not background or background["status"] != "completed":
            return "배경 정리 안 됨 · 마무리 결과를 표시합니다." if self.finishings[self.selected] else "배경 정리 안 됨 · 생성 원본을 표시합니다."
        return "흰 배경 정리 완료 · 몸에 붙은 장식·줄은 남을 수 있습니다."

    @property
    def current_path(self):
        return Path(self.tail_edit["directory"]) / "product.png" if self.tail_edit else self.basis_path

    def verify_basis(self):
        before_sha = self.verify_original_basis()
        if self.active_head and not self.show_before_head:
            from genai_lab.studio_head_paste import head_product_info
            info = head_product_info(self.active_head["directory"], self.verify_original())
            if info != self.active_head or info["before_sha256"] != before_sha:
                raise ValueError("자동 머리 붙이기의 기준 결과가 다릅니다.")
            before_sha = info["product_sha256"]
        if self.optional_finishing:
            from genai_lab.studio_optional_finishing import finishing_info
            info = finishing_info(self.optional_finishing["directory"], self.verify_original(), before_sha)
            if info != self.optional_finishing:
                raise ValueError("선택한 마무리 기록이 변경됐습니다.")
            return info["product_sha256"]
        return before_sha

    def verify_original_basis(self):
        raw_sha = self.verify_original()
        if self.two_pass:
            data = json.loads(self.candidate.record_path.read_text(encoding="utf-8"))
            if (data != self.product_records[self.selected] or not data.get("valid") or not data.get("completed")
                    or data.get("raw_sha256") != raw_sha or data.get("product_sha256") != digest(self.original_basis_path)
                    or Path(data["raw_file"]).resolve() != self.raw_path.resolve()
                    or Path(data["product_file"]).resolve() != self.original_basis_path.resolve()):
                raise ValueError("2단계 생성 결과나 기록이 변경됐습니다.")
            if data.get("head_paste"):
                from genai_lab.studio_head_paste import head_product_info
                if head_product_info(data["head_paste"]["directory"], raw_sha) != data["head_paste"]:
                    raise ValueError("머리 붙이기 기록이 변경됐습니다.")
            return data["product_sha256"]
        if self.finishings[self.selected]:
            from genai_lab.studio_finishing import finishing_source
            source, finishing = finishing_source(self.candidate)
            if finishing != self.finishings[self.selected]:
                raise ValueError("마무리 기록이 변경됐습니다.")
        background = self.backgrounds[self.selected]
        if read_background(self.candidate) != background:
            raise ValueError("배경 정리 기록이 변경됐습니다.")
        if background:
            if background["raw_sha256"] != raw_sha:
                raise ValueError("배경 정리 기준 원본이 다릅니다.")
            if self.finishings[self.selected] and (background.get("source_sha256") != finishing["finished_sha256"]
                    or Path(background.get("source_file", "")).resolve() != source.resolve()):
                raise ValueError("배경 정리와 마무리 결과의 연결이 다릅니다.")
            if background["status"] == "completed":
                if digest(self.original_basis_path) != background["product_sha256"]:
                    raise ValueError("배경 정리 결과가 변경됐습니다.")
                return background["product_sha256"]
        return self.finishings[self.selected]["finished_sha256"] if self.finishings[self.selected] else raw_sha

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
                    "head_view": self.head_view,
                    "approval_record": str(self.batch.directory / "user-review.json"),
                    "reviewer": "user", "automatic_gates_executed": False}
        if self.two_pass and self.product_records[self.selected].get("head_paste"):
            metadata["head_paste"] = self.product_records[self.selected]["head_paste"]
        if self.active_head and not self.show_before_head:
            metadata["head_paste"] = self.active_head
        if self.auto_head:
            metadata["auto_head_status"] = self.auto_head["items"][self.selected]
        metadata["identity_report"] = self.identity_report_record
        metadata["raw_sha256"] = self.verify_original()
        metadata["proportion_mode"] = "two_pass" if self.two_pass else "off"
        metadata["background"] = self.active_background
        metadata["finishing_applied"] = bool(self.active_finishing)
        if self.active_finishing:
            metadata["finishing"] = self.active_finishing
        if self.optional_finishing:
            metadata["optional_finishing"] = self.optional_finishing
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
                path.unlink()  # 이번 내보내기에서 독점적으로 만든 파일만 처리한다.
            raise
        self.status = "saved"
        self.persist(destination=str(destination))
        return destination
