"""Optional tail correction: human crop/choices -> fixed prompt -> existing Qwen runner.

No detection, automatic approval, retry, or new model settings. Source images and
initial generated candidates remain immutable. Crop coordinates are original pixels.
"""
from dataclasses import asdict, dataclass
import hashlib
import io
from pathlib import Path
import re

from PIL import Image
from genai_lab.qwen_preservation import file_sha, json_sha, require_sha
from genai_lab.qwen_pose_settings import QwenPoseSettings, output_dimensions
from genai_lab.reference_tag_policy import excluded_appendage_tag
from genai_lab.onepass_garment_vocabulary import GARMENT_VOCABULARY

TAIL_SEED = 209212001

PREFIX = ("Picture 1 is the character image to edit. Picture 2 is a close-up of the reference tail. "
          "Edit only the tail in Picture 1: the character must have exactly one tail, attached at "
          "the lower back, with no extra or disconnected tail parts. ")
SUFFIX = ("Keep everything else in Picture 1 unchanged: the same face, hair, ears, body, pose, "
          "outfit, garment colors, illustration style, framing, and plain white background. "
          "Do not copy the outfit, accessories, or background from Picture 2, and do not increase exposure.")
SPIRAL_EXAMPLE = "The tip of the tail curls into a tight spiral, as in Picture 2."
LIMITATIONS = ("시험에서는 한 장에 약 30분이 걸렸습니다. 큰 여분 꼬리나 굵기 차이가 남을 수 있습니다. "
               "캐릭터 3명의 시험 결과이며, 얼굴·옷 등 다른 부분도 반드시 비교해 주세요.")
# Finite lexical checks, not a natural-language semantic classifier.
SPECIES = frozenset("raccoon fox wolf cat dog rabbit chameleon crocodile alligator snake lizard "
                    "feline canine dragon tiger lion leopard cheetah panther squirrel mouse rat "
                    "monkey horse pony cow deer bird fish shark mermaid kitsune tanuki nekomata".split())
EXTRA_FORBIDDEN = frozenset("petite loli child childlike skinny outfit clothing clothes garment "
                          "costume naked exposed topless bottomless shoes boots".split())


def validate_tip(text):
    if not isinstance(text, str) or not text.isascii() or any(ord(c) < 32 or ord(c) == 127 for c in text):
        raise ValueError("끝 모양은 한 줄의 영어 문장으로 입력해 주세요.")
    if not text:
        return
    if text != text.strip() or len(text) > 300 or re.fullmatch(r"[A-Za-z0-9 ,'-]+[.]?", text) is None:
        raise ValueError("끝 모양은 300자 이내의 영어 한 문장으로 입력해 주세요.")
    words = re.findall(r"[a-z0-9]+", text.lower())
    phrases = {" ".join(words[i:j]) for i in range(len(words)) for j in range(i+1, min(i+5, len(words))+1)}
    blocked = SPECIES | EXTRA_FORBIDDEN | {n.name for n in GARMENT_VOCABULARY}
    if any(excluded_appendage_tag(t) or t in blocked or (t.endswith('s') and t[:-1] in SPECIES) for t in phrases):
        raise ValueError("끝 모양에는 종 이름·성별·체형·노출·의상 지시를 넣을 수 없습니다.")


def assemble_tail_prompt(pattern, tip=""):
    if type(pattern) is not bool:
        raise ValueError("무늬 있음 또는 없음을 직접 선택해 주세요.")
    validate_tip(tip)
    match = ("Make that single tail match the tail in Picture 2 in shape, thickness, "
             + ("color, and pattern. " if pattern else "and color. "))
    positive = PREFIX + match + (tip + " " if tip else "") + SUFFIX
    return {"positive": positive, "negative": " ",
            "positive_sha256": hashlib.sha256(positive.encode("utf-8")).hexdigest()}


def validate_box(box, size):
    if not isinstance(box, (tuple, list)) or len(box) != 4 or any(type(x) is not int for x in box):
        raise ValueError("꼬리 영역은 원본 좌표 정수 4개가 필요합니다.")
    x0, y0, x1, y1 = box
    w, h = size
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("꼬리 영역이 비어 있거나 이미지 밖입니다.")
    if tuple(box) == (0, 0, w, h):
        raise ValueError("원본 전체 대신 꼬리만 감싸 주세요.")


def crop_bytes(source, box, expected_sha):
    data = Path(source).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("원본 캐릭터가 변경됐습니다. 다시 확인해 주세요.")
    with Image.open(io.BytesIO(data)) as im:
        validate_box(box, im.size)
        with im.convert("RGBA") as rgba, Image.new("RGBA", im.size, "white") as white:
            with Image.alpha_composite(white, rgba) as composite, composite.convert("RGB") as rgb:
                with rgb.crop(box) as crop:
                    output = io.BytesIO()
                    crop.save(output, format="PNG")
                    return output.getvalue()


@dataclass(frozen=True)
class TailEditSpec:
    image_path: str
    image_sha256: str
    source_path: str
    source_sha256: str
    crop_path: str
    crop_sha256: str
    box: tuple[int, int, int, int]
    pattern: bool
    tip: str = ""
    confirmed: bool = False

    def record(self):
        return asdict(self)

    @property
    def sha256(self):
        return json_sha(self.record())

    def verify_image(self):
        for value in (self.image_sha256, self.source_sha256, self.crop_sha256):
            require_sha(value)
        if self.confirmed is not True:
            raise ValueError("꼬리 영역과 편집 조건을 확인해 주세요.")
        assemble_tail_prompt(self.pattern, self.tip)
        if file_sha(self.image_path) != self.image_sha256:
            raise ValueError("선택한 생성 결과가 변경됐습니다.")
        expected = crop_bytes(self.source_path, self.box, self.source_sha256)
        if hashlib.sha256(expected).hexdigest() != self.crop_sha256 or file_sha(self.crop_path) != self.crop_sha256:
            raise ValueError("꼬리 크롭이 확인한 원본 영역과 다릅니다.")
        with Image.open(self.image_path) as im:
            if im.mode != "RGB":
                raise ValueError("선택한 결과는 RGB 이미지여야 합니다.")


def prepare_tail_spec(image_path, source_path, box, *, pattern, tip="", confirmed=False,
                      directory, source_sha256, image_sha256):
    """Snapshot both inputs. A caller cannot substitute a full image for Picture 2."""
    assemble_tail_prompt(pattern, tip)
    if confirmed is not True:
        raise ValueError("꼬리 영역과 편집 조건을 확인해 주세요.")
    cropped = crop_bytes(source_path, box, source_sha256)
    basis = Path(image_path).read_bytes()
    source = Path(source_path).read_bytes()
    if hashlib.sha256(basis).hexdigest() != image_sha256 or hashlib.sha256(source).hexdigest() != source_sha256:
        raise ValueError("확인 중 입력 이미지가 변경됐습니다.")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "basis.png").write_bytes(basis)
    (directory / "character-source.png").write_bytes(source)
    (directory / "tail.png").write_bytes(cropped)
    spec = TailEditSpec(str(directory / "basis.png"), image_sha256,
                        str(directory / "character-source.png"), source_sha256,
                        str(directory / "tail.png"), hashlib.sha256(cropped).hexdigest(),
                        tuple(box), pattern, tip, confirmed)
    spec.verify_image()
    from genai_lab.qwen_record_io import write_json
    write_json(directory / "selection.json", {"spec": spec.record(), "original_source": str(source_path),
               "original_basis": str(image_path), "coordinates": "original_pixels", "background": [255,255,255]})
    return spec


def make_tail_request(spec, *, settings):
    if settings.default_seed != TAIL_SEED:
        raise ValueError("꼬리 편집은 시험 seed 209212001을 사용합니다.")
    spec.verify_image()
    with Image.open(spec.image_path) as im:
        size = im.size
    return {"schema_version": 1, "task_kind": "tail_edit", "images": 2,
            "tail_spec": spec.record(), "prompt": assemble_tail_prompt(spec.pattern, spec.tip),
            "settings": settings.record(), "seed": settings.default_seed,
            "basis_size": list(size), "output_size": list(output_dimensions(*size))}


def validate_tail_request(request):
    if request.get("task_kind") != "tail_edit" or request.get("schema_version") != 1 or request.get("images") != 2:
        raise ValueError("꼬리 편집 요청 형식 오류")
    spec = TailEditSpec(**request["tail_spec"])
    spec.verify_image()
    settings = QwenPoseSettings(**request["settings"])
    if settings.default_seed != TAIL_SEED or request.get("seed") != TAIL_SEED or type(request.get("seed")) is not int:
        raise ValueError("꼬리 편집 seed 계약 오류")
    if request["prompt"] != assemble_tail_prompt(spec.pattern, spec.tip):
        raise ValueError("확인한 꼬리 조건과 지시문이 다릅니다.")
    with Image.open(spec.image_path) as im:
        if request.get("basis_size") != list(im.size) or request.get("output_size") != list(output_dimensions(*im.size)):
            raise ValueError("꼬리 편집 이미지 크기 계약 오류")
    return spec


class TailEditWorkflow:
    """No automatic analysis: bind human crop approval, then check GPU ownership."""
    def __init__(self, spec, *, gpu_probe=None):
        from genai_lab.qwen_pose_edit import assert_parent_gpu_released
        spec.verify_image()
        self.spec_sha256 = spec.sha256
        self.gpu_probe = gpu_probe or assert_parent_gpu_released
        self.events = []
        self.phase = "confirmed"

    def before_launch(self, spec):
        if self.phase != "confirmed" or spec.sha256 != self.spec_sha256:
            raise ValueError("현재 꼬리 선택에 대한 확인이 필요합니다.")
        spec.verify_image()
        self.events.append({"stage": "before_qwen", "probe": self.gpu_probe()})
        self.phase = "qwen_running"


def run_tail_edit(request, directory, workflow, **kwargs):
    from genai_lab.qwen_pose_edit import _run_image_edit
    spec = validate_tail_request(request)
    return _run_image_edit(request, directory, workflow, spec, **kwargs)


def tail_product_info(directory, basis_sha256):
    """Verify completed raw, derived preview and its exact request before adoption/export."""
    import json
    directory = Path(directory)
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    spec = validate_tail_request(request)
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    launch = json.loads((directory / "launcher.json").read_text(encoding="utf-8"))
    if (spec.image_sha256 != basis_sha256 or run.get("status") != "completed"
            or launch.get("status") != "awaiting_review"
            or run.get("request_sha256") != json_sha(request)
            or launch.get("request_sha256") != json_sha(request)
            or launch.get("basis_sha256") != basis_sha256
            or file_sha(directory / "raw.png") != run.get("raw_sha256")
            or launch.get("raw_sha256") != run.get("raw_sha256")
            or file_sha(directory / "product.png") != launch.get("product_sha256")):
        raise ValueError("현재 후보의 완료된 꼬리 편집 결과가 아닙니다.")
    with Image.open(directory / "product.png") as product:
        if list(product.size) != request["basis_size"]:
            raise ValueError("꼬리 편집 미리보기 크기가 다릅니다.")
    return {"directory": str(directory.resolve()), "basis_sha256": basis_sha256,
            "raw_sha256": run["raw_sha256"], "product_sha256": launch["product_sha256"],
            "request_sha256": json_sha(request)}
