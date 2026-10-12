"""보존형 의상 교체 비교 B(Qwen 전용 환경 D:/genai-cache/qwen-test-venv에서 실행): Qwen-Image-Edit-2511(GGUF Q4_K_M, 로컬)로
원본 캐릭터 + 의상 그림을 편집한 뒤 원본과 합성한다. genai_lab을 import하지 않는다. 다운로드하지 않는다(로컬 파일만).
--paste mask: CatVTON 시험(tryon-01)에서 사용자가 확인한 의상 마스크 안에서만 합성한다(시험 08). 마스크 밖·보존 영역 픽셀이 바뀌면 실패.
--paste head: 머리(머리카락·얼굴·머리 장식 등 보존 영역)만 원본 픽셀이고 나머지는 Qwen 결과를 쓴다(시험 09). Qwen이 소품을 빼고
              흰 배경으로 그리므로 원본 배경·소품·신발이 남지 않고, 다리와 신발이 한 그림에서 나와 발목이 꺾이지 않는다.
              머리 가장자리 2px는 Qwen 머리를 써서 원본 배경색이 번지지 않게 한다.
머리 정렬(두 방식 공통): 원본 머리 주변과 Qwen 결과 전체에서 특징점(SIFT)을 맞춰 위치·크기·기울기를 잰다. 일치점이 적으면
붙이지 않는다(unmeasured). 차이가 허용 범위 안이면 Qwen 결과 전체를 원본 좌표로 되돌린 뒤(크기·기울기·위치) 다시 재서
거의 일치할 때만 붙인다. 범위를 넘거나 다시 재도 어긋나면 붙이지 않는다(rejected). 붙이지 않은 장은 raw만 남긴다.
--paste head에서는 되돌린 그림이 화면 밖으로 나가지 않도록 흰 여백으로 화면을 넓힌다(시험 11: 고정 768×1024에 넣어 발끝이 잘림).
원본 머리는 넓힌 화면에서도 같은 크기·자리에 그대로 붙는다. 넓힌 양은 canvas_pad(왼쪽·위·오른쪽·아래)로 기록한다.
              --head-accessories keep: 머리 장식(머리띠·모자·헤드셋·귀걸이·안경)을 원본 그대로 유지, remove: 지운다.
              이 선택은 머리 장식에만 적용된다. 무기 등 다른 소품은 어느 쪽이든 지운다.
메모리: --reserved-limit-gib를 PyTorch 할당 상한으로 걸어 그 이상은 할당 자체가 실패한다(공유 메모리로 넘기지 않음).
단계(문장 인코딩·모델 적재·첫 단계(조건 그림 VAE 인코딩 포함)·각 단계·VAE 디코딩)마다 현재값과 그 단계 최고값을 memory.json에 남긴다.
--vae-tiling(VAE 타일 처리), --area(실제 생성 픽셀 수)는 메모리 절감 선택지다. 기본값은 시험 08과 같다."""
import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SIZE = (768, 1024)
FEATHER = 2
HEAD_EDGE = 2
MATCH_RATIO = 0.75
RANSAC_PX = 3.0
MIN_INLIERS = 12
SCALE_RANGE = (0.75, 1.35)      # 되돌릴 수 있는 크기 차이
ROTATE_LIMIT = 10.0             # 되돌릴 수 있는 기울기(도)
RESIDUAL_SHIFT = 2.0            # 되돌린 뒤 남아도 되는 위치 차이(px)
RESIDUAL_SCALE = 0.02
RESIDUAL_ROTATE = 1.0
CONTENT_DIFF = 25               # 흰색과 채널 차이가 이보다 크면 그림 내용으로 본다
HEAD_WINDOW = 15
BORDER = 32
PROMPT = ("Dress the character from image 1 in the outfit shown in image 2: a long dark brown coat, a white shirt with a black "
          "ribbon tie, suspenders, dark high-waisted trousers and black shoes. Keep the character's face, hairstyle, hair color, "
          "body shape, body proportions, pose, hands, accessories and background exactly the same as in image 1. "
          "Do not change the camera angle, framing or the character's size or position.")
HEAD_ACCESSORIES = ("headwear", "earwear", "eyewear")  # 사용자 선택 대상은 머리 장식뿐이다
HEAD_KEEP = ("hair", "face", "eyes", "nose", "mouth", "ears", "tail", "wings")
OUTFIT_HEAD = ("Dress the character from image 1 in the outfit shown in image 2: a long dark brown coat, a white shirt with a black "
               "ribbon tie, suspenders, dark high-waisted trousers and black shoes. Remove the weapon and every carried or held prop. "
               "Place the character alone on a plain pure white background with nothing else. ")
PROMPT_HEAD = {
    "keep": OUTFIT_HEAD + ("Keep the character's face, hairstyle, hair color, head accessories and body proportions exactly the "
                           "same as in image 1. "),
    # 시험 10: 장식 삭제 지시에서 다른 인물(갈색 머리 남성)이 나왔다. 같은 인물(성별·나이·얼굴·머리색·머리 모양) 유지를 먼저 못 박는다.
    "remove": OUTFIT_HEAD + ("The character must stay the same person as in image 1: same gender, age, face, hair color and "
                             "hairstyle. Only take off the head accessories: remove every head accessory (headphones, headbands, "
                             "hair ornaments, hats, earrings and glasses) and show the character's own hair where they were. "
                             "Keep body proportions exactly the same as in image 1. "),
}
FRAMING = "Do not change the camera angle, framing or the character's size or position."
PROMPT_HEAD = {k: v + FRAMING for k, v in PROMPT_HEAD.items()}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path, mode="RGB"):
    with Image.open(path) as image:
        return np.asarray(image.convert(mode))


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def match_head(person, edited, window):
    """원본 머리 주변(window)의 특징점을 Qwen 결과 전체에서 찾아 원본→편집 닮음 변환(위치·크기·기울기)을 잰다."""
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray(person), window.astype(np.uint8) * 255)
    kp2, des2 = sift.detectAndCompute(gray(edited), None)
    record = dict(source_features=len(kp1), target_features=len(kp2), matches=0, inliers=0)
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return dict(record, status="unmeasured")
    pairs = [m[0] for m in cv2.BFMatcher().knnMatch(des1, des2, k=2) if len(m) == 2 and m[0].distance < MATCH_RATIO * m[1].distance]
    record["matches"] = len(pairs)
    if len(pairs) < MIN_INLIERS:
        return dict(record, status="unmeasured")
    src = np.float32([kp1[m.queryIdx].pt for m in pairs])
    dst = np.float32([kp2[m.trainIdx].pt for m in pairs])
    matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX)
    record["inliers"] = int(inliers.sum()) if inliers is not None else 0
    if matrix is None or record["inliers"] < MIN_INLIERS:
        return dict(record, status="unmeasured")
    a, b = float(matrix[0, 0]), float(matrix[1, 0])
    ys, xs = np.nonzero(window)
    center = np.array([xs.mean(), ys.mean(), 1.0])
    moved = matrix @ center
    return dict(record, status="measured", matrix=matrix.tolist(), scale=float(np.hypot(a, b)),
                rotate_deg=float(np.degrees(np.arctan2(b, a))), head_shift_px=[float(moved[0] - center[0]), float(moved[1] - center[1])])


def near_identity(match):
    return (match["status"] == "measured" and abs(match["scale"] - 1) <= RESIDUAL_SCALE
            and abs(match["rotate_deg"]) <= RESIDUAL_ROTATE and max(map(abs, match["head_shift_px"])) <= RESIDUAL_SHIFT)


def content_mask(image):
    """흰 바탕이 아닌 그림 내용. 점 잡음은 3×3 열림으로 지운다."""
    content = (np.abs(255 - image.astype(np.int16)).max(axis=2) > CONTENT_DIFF).astype(np.uint8)
    return cv2.morphologyEx(content, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0


def outside_frame(points, width, height):
    x, y = points
    return (x < 0) | (y < 0) | (x > width - 1) | (y > height - 1)


def pad_image(image, pad, value):
    left, top, right, bottom = pad
    widths = ((top, bottom), (left, right)) + (((0, 0),) if image.ndim == 3 else ())
    return np.pad(image, widths, constant_values=value)


def realign(person, edited, window, expand=False):
    """머리가 원본과 같은 자리·크기가 되도록 Qwen 결과 전체를 되돌린다. 확인되지 않으면 None(붙이지 않음).
    expand=True면 되돌린 그림 내용이 잘리지 않게 화면을 흰 여백으로 넓힌다(원본 좌표는 pad만큼 옮겨진다)."""
    first = match_head(person, edited, window)
    report = dict(first=first, canvas_pad=[0, 0, 0, 0])
    if first["status"] != "measured":
        return None, dict(report, decision="unmeasured")
    if not (SCALE_RANGE[0] <= first["scale"] <= SCALE_RANGE[1] and abs(first["rotate_deg"]) <= ROTATE_LIMIT):
        return None, dict(report, decision="rejected", reason="크기·기울기 차이가 범위를 넘음")
    height, width = person.shape[:2]
    inverse = cv2.invertAffineTransform(np.float32(first["matrix"])).astype(np.float64)
    ys, xs = np.nonzero(content_mask(edited))
    mapped = inverse @ np.stack([xs, ys, np.ones_like(xs)]).astype(np.float64)
    pad = [0, 0, 0, 0]
    if expand and xs.size:
        pad = [max(0, int(np.ceil(-mapped[0].min()))), max(0, int(np.ceil(-mapped[1].min()))),
               max(0, int(np.ceil(mapped[0].max() - (width - 1)))), max(0, int(np.ceil(mapped[1].max() - (height - 1))))]
    inverse[0, 2] += pad[0]
    inverse[1, 2] += pad[1]
    size = (width + pad[0] + pad[2], height + pad[1] + pad[3])
    lost = int(outside_frame(mapped + np.array([[pad[0]], [pad[1]]]), *size).sum()) if xs.size else 0
    back = cv2.warpAffine(edited, inverse, size, flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    second = match_head(pad_image(person, pad, 255), back, pad_image(window, pad, False))
    report.update(second=second, canvas_pad=pad, content_outside_px=lost)
    if not near_identity(second):
        return None, dict(report, decision="rejected", reason="되돌린 뒤에도 머리가 맞지 않음")
    return back, dict(report, decision="aligned")


def fit_white(image, size):
    """비교 그림용: 넓어진 화면을 비율 유지로 size 안에 줄여 흰 바탕 가운데 놓는다(결과 파일은 줄이지 않는다)."""
    width, height = size
    if image.shape[1] == width and image.shape[0] == height:
        return image
    scale = min(width / image.shape[1], height / image.shape[0])
    small = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    out = np.full((height, width, 3), 255, np.uint8)
    y, x = (height - small.shape[0]) // 2, (width - small.shape[1]) // 2
    out[y:y + small.shape[0], x:x + small.shape[1]] = small
    return out


def composite(person, edited, change):
    weight = cv2.GaussianBlur(change.astype(np.float32), (0, 0), FEATHER)
    weight[~change] = 0
    return np.clip(np.rint(person * (1 - weight[..., None]) + edited * weight[..., None]), 0, 255).astype(np.uint8)


def head_protect(parts, protect_png, accessories):
    """머리 붙이기 영역. 확인된 protect.png(머리 장식 포함)와 1단계 분할이 일치하는지 확인한 뒤,
    remove면 머리 장식만 뺀다. 머리 장식 외의 영역은 바꾸지 않는다."""
    base = np.logical_or.reduce([parts[k] for k in HEAD_KEEP])
    worn = np.logical_or.reduce([parts[k] for k in HEAD_ACCESSORIES])
    require(np.array_equal(base | worn, protect_png), "1단계 분할과 protect.png가 다릅니다.")
    return base | worn if accessories == "keep" else base & ~worn


def head_weight(protect):
    """머리 붙이기 가중치: 보존 영역을 2px 줄인 뒤 가장자리만 부드럽게 한다. 보존 영역 밖 가중치는 0(원본 배경이 따라오지 않음).
    core는 가중치가 사실상 1인 안쪽(8σ 줄임)으로, 원본 픽셀과 정확히 같아야 한다."""
    inner = cv2.erode(protect.astype(np.uint8), np.ones((2 * HEAD_EDGE + 1,) * 2, np.uint8)) > 0
    weight = cv2.GaussianBlur(inner.astype(np.float32), (0, 0), FEATHER)
    weight[~inner] = 0
    core = cv2.erode(inner.astype(np.uint8), np.ones((8 * FEATHER + 1,) * 2, np.uint8)) > 0
    return weight, core


def head_composite(person, edited, protect):
    weight, core = head_weight(protect)
    result = np.clip(np.rint(person * weight[..., None] + edited * (1 - weight[..., None])), 0, 255).astype(np.uint8)
    return result, core


def head_window(protect):
    return cv2.dilate(protect.astype(np.uint8), np.ones((2 * HEAD_WINDOW + 1,) * 2, np.uint8)) > 0


def border_white_ratio(image):
    """가장자리 32px 띠에서 거의 흰색(세 채널 모두 235 이상)인 비율. 배경이 남았는지 보는 기록용 값."""
    frame = np.ones(image.shape[:2], bool)
    frame[BORDER:-BORDER, BORDER:-BORDER] = False
    return float((image[frame].min(axis=1) >= 235).mean())


class MemoryGuard:
    """단계별 PyTorch 메모리 기록. 각 단계 최고값은 직전 기록 뒤부터 잰다(단계마다 최고값을 초기화).
    한도는 apply_cap()으로 할당 상한을 걸어 지키고, check()는 넘었는지 다시 확인해 멈춘다."""

    def __init__(self, cuda, limit_gib, log_path=None):
        self.cuda, self.limit, self.log_path, self.stages, self.stopped = cuda, limit_gib, log_path, [], None
        self.current = "시작"

    def begin(self, stage):
        """다음 작업 이름을 먼저 남긴다. 그 작업 중 할당이 실패해도 어느 단계였는지 남는다."""
        self.current = stage
        self.write()

    def apply_cap(self, device=0):
        if self.limit is None:
            return None
        total = self.cuda.get_device_properties(device).total_memory
        fraction = min(1.0, self.limit * 2 ** 30 / total)
        self.cuda.set_per_process_memory_fraction(fraction, device)
        return fraction

    def write(self):
        if self.log_path is not None:  # 한도로 멈춰도 어느 단계까지 썼는지 남는다
            write_json(self.log_path, dict(limit_gib=self.limit, current=self.current, stages=self.stages,
                                           stopped=self.stopped))

    def check(self, stage):
        self.current = stage
        peak = self.cuda.max_memory_reserved() / 2 ** 30
        self.stages.append(dict(stage=stage, allocated_gib=round(self.cuda.memory_allocated() / 2 ** 30, 3),
                                reserved_gib=round(self.cuda.memory_reserved() / 2 ** 30, 3),
                                stage_peak_reserved_gib=round(peak, 3)))
        self.cuda.reset_peak_memory_stats()
        self.write()
        if self.limit is not None and peak > self.limit:
            self.stopped = f"{stage}: 단계 최고 reserved {peak:.2f}GiB > 한도 {self.limit}GiB"
            self.write()
            raise MemoryError(self.stopped)

    def fail(self, error):
        try:
            at_failure = dict(allocated_gib=round(self.cuda.memory_allocated() / 2 ** 30, 3),
                              reserved_gib=round(self.cuda.memory_reserved() / 2 ** 30, 3),
                              stage_peak_reserved_gib=round(self.cuda.max_memory_reserved() / 2 ** 30, 3))
        except Exception as reason:  # 수집 실패는 미측정으로 남긴다
            at_failure = dict(unmeasured=str(reason))
        self.stages.append(dict(stage=f"{self.current}(실패)", **at_failure))
        self.stopped = f"'{self.current}' 중 할당 실패: {error}"
        self.write()

    def step_callback(self, name):
        def callback(pipe, step, timestep, kwargs):
            self.check(f"{name}/step{step}" + ("(조건 그림 VAE 인코딩 포함)" if step == 0 else ""))
            last = step + 1 == pipe.num_timesteps if hasattr(pipe, "num_timesteps") else False
            self.current = f"{name}/decode" if last else f"{name}/step{step + 1}"
            return kwargs
        return callback


def text_encoder_device_map():
    """문장 인코딩은 마지막 은닉 상태만 쓰고 lm_head(어휘 152,064 × 3,584, bf16 1.02GiB) 출력은 쓰지 않는다.
    그래서 lm_head는 GPU에 올리지 않고 CPU에 둔 뒤 적재 후 Identity로 바꿔 계산도 하지 않는다. 인코딩 결과는 같다."""
    return {"model": 0, "lm_head": "cpu"}


def load_pipeline(model_root, gguf_file):
    import torch
    from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel, GGUFQuantizationConfig
    from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
    te = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(Path(model_root) / "text_encoder"), torch_dtype=torch.bfloat16,
        local_files_only=True, device_map=text_encoder_device_map(), quantization_config=BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["visual", "lm_head"],
        llm_int8_enable_fp32_cpu_offload=True))
    te.lm_head = torch.nn.Identity()  # 쓰지 않는 어휘 출력(전체 토큰 × 152,064) 계산을 없앤다
    gc.collect()
    encoder = QwenImageEditPlusPipeline.from_pretrained(model_root, transformer=None, vae=None, text_encoder=te,
                                                        torch_dtype=torch.bfloat16, local_files_only=True)
    return encoder, te


def run(args):
    import torch
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_dimensions
    lock_file = args.prepared / "mask-lock.json"
    confirmation = json.loads((args.prepared / "user-confirmation.json").read_text(encoding="utf-8"))
    require(confirmation["reviewer"] == "user" and confirmation["review_sha256"] == sha(lock_file), "사용자가 확인한 마스크가 아닙니다.")
    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    require(sha(lock["garment"]) == lock["garment_sha256"], "의상 이미지가 바뀌었습니다.")
    cases = [c for c in lock["cases"] if not args.cases or c["name"] in args.cases]
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    garment = Image.open(lock["garment"]).convert("RGB")
    started = time.perf_counter()
    width, height = calculate_dimensions(args.area, SIZE[0] / SIZE[1])
    prompt = PROMPT_HEAD[args.head_accessories] if args.paste == "head" else PROMPT
    guard = MemoryGuard(torch.cuda, args.reserved_limit_gib, out / "memory.json")
    cap_fraction = guard.apply_cap()
    torch.cuda.reset_peak_memory_stats()
    try:
        generate(args, cases, lock, lock_file, garment, prompt, guard, width, height, out, started, cap_fraction)
    except torch.OutOfMemoryError as error:
        guard.fail(str(error).splitlines()[0])
        raise


def generate(args, cases, lock, lock_file, garment, prompt, guard, width, height, out, started, cap_fraction):
    import torch
    from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel, GGUFQuantizationConfig
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_dimensions, CONDITION_IMAGE_SIZE
    results, rows = [], []
    # 1) 문장·그림 인코딩(4비트 텍스트 인코더)을 장마다 먼저 하고 내린다.
    guard.begin("text_encoder_load")
    encoder, te = load_pipeline(args.model_root, args.gguf_file)
    guard.check("text_encoder_loaded")
    encoded = {}
    for case in cases:
        folder = args.prepared / case["name"]
        require(sha(folder / "change-mask.png") == case["change_mask_sha256"], f"{case['name']}: 확정 마스크가 바뀌었습니다.")
        person = Image.open(folder / "person.png").convert("RGB")
        cond = []
        for image in (person, garment):
            cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, image.width / image.height)
            cond.append(encoder.image_processor.resize(image, ch, cw))
        guard.begin(f"{case['name']}/encode")
        with torch.no_grad():
            pe, pm = encoder.encode_prompt(prompt=prompt, image=cond, device=torch.device("cuda"))
            ne, nm = encoder.encode_prompt(prompt=" ", image=cond, device=torch.device("cuda"))
        encoded[case["name"]] = tuple(v.to("cpu") if v is not None else None for v in (pe, pm, ne, nm))
        guard.check(f"{case['name']}/encode")
    del encoder, te
    gc.collect()
    torch.cuda.empty_cache()
    # 2) GGUF 변환기를 블록 단위로 GPU에 올려 생성한다(기존 Qwen 작업 프로세스와 같은 메모리 방식).
    guard.begin("transformer_load")
    tr = QwenImageTransformer2DModel.from_single_file(args.gguf_file, quantization_config=GGUFQuantizationConfig(
        compute_dtype=torch.bfloat16), torch_dtype=torch.bfloat16, config=args.model_root, subfolder="transformer",
        local_files_only=True)
    require(getattr(tr.config, "zero_cond_t", None) is True, "zero_cond_t=True 계약 불일치")
    pipe = QwenImageEditPlusPipeline.from_pretrained(args.model_root, transformer=tr, text_encoder=None, tokenizer=None,
                                                     processor=None, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.vae.to("cuda")
    if args.vae_tiling:
        require(hasattr(pipe.vae, "enable_tiling"), "이 VAE는 타일 처리를 지원하지 않습니다.")
        pipe.vae.enable_tiling()
    tr.enable_group_offload(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
                            offload_type="block_level", num_blocks_per_group=1)
    guard.check("transformer_loaded")
    for case in cases:
        folder = args.prepared / case["name"]
        person_img = Image.open(folder / "person.png").convert("RGB")
        person = np.asarray(person_img)
        change = load(folder / "change-mask.png", "L") > 127
        protect = load(folder / "protect.png", "L") > 127
        if args.paste == "head":
            with np.load(folder / "parts.npz") as parts:
                protect = head_protect(parts, protect, args.head_accessories)
        pe, pm, ne, nm = encoded[case["name"]]
        t0 = time.perf_counter()
        guard.begin(f"{case['name']}/step0(조건 그림 VAE 인코딩 포함)")
        image = pipe(image=[person_img, garment], prompt_embeds=pe.to("cuda"), prompt_embeds_mask=pm.to("cuda") if pm is not None else None,
                     negative_prompt_embeds=ne.to("cuda"), negative_prompt_embeds_mask=nm.to("cuda") if nm is not None else None,
                     true_cfg_scale=args.true_cfg, guidance_scale=1.0, num_inference_steps=args.steps, height=height, width=width,
                     generator=torch.Generator("cpu").manual_seed(args.seed),
                     callback_on_step_end=guard.step_callback(case["name"])).images[0]
        seconds = time.perf_counter() - t0
        guard.check(f"{case['name']}/decode")
        name = f"{case['name']}_{args.seed}"
        image.save(out / f"{name}_raw.png")
        edited = np.asarray(image.convert("RGB").resize(SIZE, Image.LANCZOS))
        window = head_window(protect)
        aligned, alignment = realign(person, edited, window, expand=args.paste == "head")
        pad = alignment["canvas_pad"]
        record = dict(case=case["name"], seed=args.seed, paste=args.paste, seconds=round(seconds, 1),
                      head_accessories=args.head_accessories if args.paste == "head" else None,
                      alignment=alignment)
        if aligned is None:  # 머리를 확인할 수 없거나 어긋나면 붙이지 않는다
            record.update(composited=False)
            results.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            rows.append(np.concatenate([person, edited, np.full_like(person, 255)], 1))
            write_json(out / "results.json", dict(results=results, memory=guard.stages))
            continue
        record.update(composited=True)
        if args.paste == "head":
            person_c, protect_c = pad_image(person, pad, 255), pad_image(protect, pad, False)
            result, core = head_composite(person_c, aligned, protect_c)
            record.update(head_core_equal=bool(np.array_equal(result[core], person_c[core])), head_core_px=int(core.sum()),
                          head_mean_abs_diff=round(float(np.abs(aligned.astype(np.float32) - person_c)[protect_c].mean()), 2),
                          output_size=[int(result.shape[1]), int(result.shape[0])],
                          border_white_ratio=round(border_white_ratio(result), 3))
            ok = record["head_core_equal"]
        else:
            result = composite(person, aligned, change)
            record.update(outside_mean_abs_diff=round(float(np.abs(aligned.astype(np.float32) - person)[~change].mean()), 2),
                          outside_equal=bool(np.array_equal(result[~change], person[~change])),
                          protect_equal=bool(np.array_equal(result[protect], person[protect])))
            ok = record["outside_equal"] and record["protect_equal"]
        Image.fromarray(aligned).save(out / f"{name}_aligned.png")
        Image.fromarray(result).save(out / f"{name}.png")
        record["result_sha256"] = sha(out / f"{name}.png")
        results.append(record)
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
        require(ok, f"{name}: 보존해야 할 원본 픽셀이 바뀌었습니다.")
        rows.append(np.concatenate([person, edited, fit_white(result, SIZE)], 1))
        write_json(out / "results.json", dict(results=results, memory=guard.stages))
    Image.fromarray(np.concatenate([np.asarray(Image.fromarray(r).resize((1152, 512))) for r in rows], 0)).save(out / "before_raw_after.png")
    write_json(out / "results.json", dict(results=results, seconds=time.perf_counter() - started, prompt=prompt, paste=args.paste, head_accessories=args.head_accessories,
        reserved_limit_gib=args.reserved_limit_gib, cap_fraction=cap_fraction, vae_tiling=args.vae_tiling, area=args.area,
        memory=guard.stages, steps=args.steps, true_cfg=args.true_cfg, output_size=[width, height], gguf=str(args.gguf_file),
        gguf_sha256=sha(args.gguf_file), mask_lock_sha256=sha(lock_file), code_sha256=sha(Path(__file__)),
        max_stage_peak_reserved_gib=max(s["stage_peak_reserved_gib"] for s in guard.stages)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True, help="사용자가 마스크를 확인한 tryon 폴더")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="*", help="생략하면 잠금의 전체 캐릭터")
    parser.add_argument("--model-root", default="D:/genai-cache/models/Qwen-Image-Edit-2511")
    parser.add_argument("--gguf-file", default="D:/genai-cache/models/Qwen-Image-Edit-2511-GGUF/qwen-image-edit-2511-Q4_K_M.gguf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--true-cfg", type=float, default=4.0)
    parser.add_argument("--paste", choices=("mask", "head"), default="mask")
    parser.add_argument("--vae-tiling", action="store_true", help="VAE를 타일로 나눠 처리(메모리 절감)")
    parser.add_argument("--area", type=int, default=1024 * 1024, help="실제 생성 픽셀 수(기본 1024×1024 → 896×1184)")
    parser.add_argument("--head-accessories", choices=("keep", "remove"), default="keep",
                        help="--paste head에서 머리 장식 유지/삭제(사용자 체크 항목). 머리 장식에만 적용")
    parser.add_argument("--reserved-limit-gib", type=float, default=6.5,
                        help="PyTorch reserved 상한. 이 이상은 할당이 실패해 멈춘다. 사용자 승인 없이 올리지 않는다")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
