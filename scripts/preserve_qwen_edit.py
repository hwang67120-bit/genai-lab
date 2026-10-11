"""보존형 의상 교체 비교 B(Qwen 전용 환경 D:/genai-cache/qwen-test-venv에서 실행): Qwen-Image-Edit-2511(GGUF Q4_K_M, 로컬)로
원본 캐릭터 + 의상 그림을 편집한 뒤 원본과 합성한다. genai_lab을 import하지 않는다. 다운로드하지 않는다(로컬 파일만).
--paste mask: CatVTON 시험(tryon-01)에서 사용자가 확인한 의상 마스크 안에서만 합성한다(시험 08). 마스크 밖·보존 영역 픽셀이 바뀌면 실패.
--paste head: 머리(머리카락·얼굴·머리 장식 등 보존 영역)만 원본 픽셀이고 나머지는 Qwen 결과를 쓴다(시험 09). Qwen이 소품을 빼고
              흰 배경으로 그리므로 원본 배경·소품·신발이 남지 않고, 다리와 신발이 한 그림에서 나와 발목이 꺾이지 않는다.
              머리 위치는 머리 주변에서 재서 맞춘다. 머리 가장자리 2px는 Qwen 머리를 써서 원본 배경색이 번지지 않게 한다.
--reserved-limit-gib를 넘으면 그 단계에서 멈춘다(문장 인코딩 뒤, 모델 적재 뒤, 생성 매 단계)."""
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
MAX_SHIFT = 24
HEAD_EDGE = 2
HEAD_WINDOW = 15
BORDER = 32
PROMPT = ("Dress the character from image 1 in the outfit shown in image 2: a long dark brown coat, a white shirt with a black "
          "ribbon tie, suspenders, dark high-waisted trousers and black shoes. Keep the character's face, hairstyle, hair color, "
          "body shape, body proportions, pose, hands, accessories and background exactly the same as in image 1. "
          "Do not change the camera angle, framing or the character's size or position.")
PROMPT_HEAD = ("Dress the character from image 1 in the outfit shown in image 2: a long dark brown coat, a white shirt with a black "
               "ribbon tie, suspenders, dark high-waisted trousers and black shoes. Remove the weapon and every carried or held prop. "
               "Place the character alone on a plain pure white background with nothing else. Keep the character's face, hairstyle, "
               "hair color, head accessories and body proportions exactly the same as in image 1. Do not change the camera angle, "
               "framing or the character's size or position.")


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


def estimate_shift(original, edited, reference_area):
    """보존해야 할 영역(마스크 밖)에서 편집 결과가 원본에서 얼마나 평행 이동했는지 잰다(위상 상관)."""
    window = reference_area.astype(np.float32)
    a = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32) * window
    b = cv2.cvtColor(edited, cv2.COLOR_RGB2GRAY).astype(np.float32) * window
    (dx, dy), response = cv2.phaseCorrelate(a, b)
    return float(dx), float(dy), float(response)


def align(edited, dx, dy):
    matrix = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(edited, matrix, (edited.shape[1], edited.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def composite(person, edited, change):
    weight = cv2.GaussianBlur(change.astype(np.float32), (0, 0), FEATHER)
    weight[~change] = 0
    return np.clip(np.rint(person * (1 - weight[..., None]) + edited * weight[..., None]), 0, 255).astype(np.uint8)


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
    """단계별 PyTorch allocated/reserved 기록. 한도를 넘으면 그 단계에서 멈춘다."""

    def __init__(self, cuda, limit_gib, log_path=None):
        self.cuda, self.limit, self.log_path, self.stages = cuda, limit_gib, log_path, []

    def check(self, stage):
        reserved = self.cuda.max_memory_reserved() / 2 ** 30
        self.stages.append(dict(stage=stage, allocated_gib=round(self.cuda.memory_allocated() / 2 ** 30, 3),
                                max_reserved_gib=round(reserved, 3)))
        if self.log_path is not None:  # 한도로 멈춰도 어느 단계까지 썼는지 남는다
            write_json(self.log_path, dict(limit_gib=self.limit, stages=self.stages))
        if self.limit is not None and reserved > self.limit:
            raise MemoryError(f"{stage}: reserved {reserved:.2f}GiB > 한도 {self.limit}GiB")

    def step_callback(self, name):
        def callback(pipe, step, timestep, kwargs):
            self.check(f"{name}/step{step}")
            return kwargs
        return callback


def load_pipeline(model_root, gguf_file):
    import torch
    from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel, GGUFQuantizationConfig
    from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
    te = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(Path(model_root) / "text_encoder"), torch_dtype=torch.bfloat16,
        local_files_only=True, device_map={"": 0}, quantization_config=BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["visual", "lm_head"]))
    encoder = QwenImageEditPlusPipeline.from_pretrained(model_root, transformer=None, vae=None, text_encoder=te,
                                                        torch_dtype=torch.bfloat16, local_files_only=True)
    return encoder, te


def run(args):
    import torch
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel, GGUFQuantizationConfig
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_dimensions, CONDITION_IMAGE_SIZE
    lock_file = args.prepared / "mask-lock.json"
    confirmation = json.loads((args.prepared / "user-confirmation.json").read_text(encoding="utf-8"))
    require(confirmation["reviewer"] == "user" and confirmation["review_sha256"] == sha(lock_file), "사용자가 확인한 마스크가 아닙니다.")
    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    require(sha(lock["garment"]) == lock["garment_sha256"], "의상 이미지가 바뀌었습니다.")
    cases = [c for c in lock["cases"] if not args.cases or c["name"] in args.cases]
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    garment = Image.open(lock["garment"]).convert("RGB")
    results, rows, started = [], [], time.perf_counter()
    width, height = calculate_dimensions(1024 * 1024, SIZE[0] / SIZE[1])
    prompt = PROMPT_HEAD if args.paste == "head" else PROMPT
    guard = MemoryGuard(torch.cuda, args.reserved_limit_gib, out / "memory.json")
    torch.cuda.reset_peak_memory_stats()
    # 1) 문장·그림 인코딩(4비트 텍스트 인코더)을 장마다 먼저 하고 내린다.
    encoder, te = load_pipeline(args.model_root, args.gguf_file)
    encoded = {}
    for case in cases:
        folder = args.prepared / case["name"]
        require(sha(folder / "change-mask.png") == case["change_mask_sha256"], f"{case['name']}: 확정 마스크가 바뀌었습니다.")
        person = Image.open(folder / "person.png").convert("RGB")
        cond = []
        for image in (person, garment):
            cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, image.width / image.height)
            cond.append(encoder.image_processor.resize(image, ch, cw))
        with torch.no_grad():
            pe, pm = encoder.encode_prompt(prompt=prompt, image=cond, device=torch.device("cuda"))
            ne, nm = encoder.encode_prompt(prompt=" ", image=cond, device=torch.device("cuda"))
        encoded[case["name"]] = tuple(v.to("cpu") if v is not None else None for v in (pe, pm, ne, nm))
        guard.check(f"{case['name']}/encode")
    del encoder, te
    gc.collect()
    torch.cuda.empty_cache()
    # 2) GGUF 변환기를 블록 단위로 GPU에 올려 생성한다(기존 Qwen 작업 프로세스와 같은 메모리 방식).
    tr = QwenImageTransformer2DModel.from_single_file(args.gguf_file, quantization_config=GGUFQuantizationConfig(
        compute_dtype=torch.bfloat16), torch_dtype=torch.bfloat16, config=args.model_root, subfolder="transformer",
        local_files_only=True)
    require(getattr(tr.config, "zero_cond_t", None) is True, "zero_cond_t=True 계약 불일치")
    pipe = QwenImageEditPlusPipeline.from_pretrained(args.model_root, transformer=tr, text_encoder=None, tokenizer=None,
                                                     processor=None, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.vae.to("cuda")
    tr.enable_group_offload(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
                            offload_type="block_level", num_blocks_per_group=1)
    guard.check("transformer_loaded")
    for case in cases:
        folder = args.prepared / case["name"]
        person_img = Image.open(folder / "person.png").convert("RGB")
        person = np.asarray(person_img)
        change = load(folder / "change-mask.png", "L") > 127
        protect = load(folder / "protect.png", "L") > 127
        pe, pm, ne, nm = encoded[case["name"]]
        t0 = time.perf_counter()
        image = pipe(image=[person_img, garment], prompt_embeds=pe.to("cuda"), prompt_embeds_mask=pm.to("cuda") if pm is not None else None,
                     negative_prompt_embeds=ne.to("cuda"), negative_prompt_embeds_mask=nm.to("cuda") if nm is not None else None,
                     true_cfg_scale=args.true_cfg, guidance_scale=1.0, num_inference_steps=args.steps, height=height, width=width,
                     generator=torch.Generator("cpu").manual_seed(args.seed),
                     callback_on_step_end=guard.step_callback(case["name"])).images[0]
        seconds = time.perf_counter() - t0
        name = f"{case['name']}_{args.seed}"
        image.save(out / f"{name}_raw.png")
        edited = np.asarray(image.convert("RGB").resize(SIZE, Image.LANCZOS))
        if args.paste == "head":
            dx, dy, response = estimate_shift(person, edited, head_window(protect))
        else:
            dx, dy, response = estimate_shift(person, edited, ~change & ~protect)
        shifted = abs(dx) <= MAX_SHIFT and abs(dy) <= MAX_SHIFT
        aligned = align(edited, dx, dy) if shifted else edited
        record = dict(case=case["name"], seed=args.seed, paste=args.paste, seconds=round(seconds, 1),
                      shift_px=[round(dx, 2), round(dy, 2)], shift_response=round(response, 3), aligned=shifted)
        if args.paste == "head":
            result, core = head_composite(person, aligned, protect)
            record.update(head_core_equal=bool(np.array_equal(result[core], person[core])), head_core_px=int(core.sum()),
                          head_mean_abs_diff=round(float(np.abs(aligned.astype(np.float32) - person)[protect].mean()), 2),
                          border_white_ratio=round(border_white_ratio(result), 3))
            ok = record["head_core_equal"]
        else:
            result = composite(person, aligned, change)
            record.update(outside_mean_abs_diff=round(float(np.abs(aligned.astype(np.float32) - person)[~change].mean()), 2),
                          outside_equal=bool(np.array_equal(result[~change], person[~change])),
                          protect_equal=bool(np.array_equal(result[protect], person[protect])))
            ok = record["outside_equal"] and record["protect_equal"]
        Image.fromarray(result).save(out / f"{name}.png")
        record["result_sha256"] = sha(out / f"{name}.png")
        results.append(record)
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
        require(shifted or args.paste == "mask", f"{name}: 머리 위치 차이가 {MAX_SHIFT}px를 넘습니다.")
        require(ok, f"{name}: 보존해야 할 원본 픽셀이 바뀌었습니다.")
        rows.append(np.concatenate([person, edited, result], 1))
        write_json(out / "results.json", dict(results=results, memory=guard.stages))
    Image.fromarray(np.concatenate([np.asarray(Image.fromarray(r).resize((1152, 512))) for r in rows], 0)).save(out / "before_raw_after.png")
    write_json(out / "results.json", dict(results=results, seconds=time.perf_counter() - started, prompt=prompt, paste=args.paste,
        reserved_limit_gib=args.reserved_limit_gib, memory=guard.stages, steps=args.steps, true_cfg=args.true_cfg, output_size=[width, height], gguf=str(args.gguf_file),
        gguf_sha256=sha(args.gguf_file), mask_lock_sha256=sha(lock_file), code_sha256=sha(Path(__file__)),
        max_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30))


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
    parser.add_argument("--reserved-limit-gib", type=float, default=6.5,
                        help="PyTorch 최고 reserved 한도. 넘으면 그 단계에서 멈춘다. 사용자 승인 없이 올리지 않는다")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
