"""보존형 의상 교체 비교 B(Qwen 전용 환경 D:/genai-cache/qwen-test-venv에서 실행): Qwen-Image-Edit-2511(GGUF Q4_K_M, 로컬)로
원본 캐릭터 + 의상 그림을 편집한 뒤 원본과 합성한다. genai_lab을 import하지 않는다. 다운로드하지 않는다(로컬 파일만).
--paste mask: CatVTON 시험(tryon-01)에서 사용자가 확인한 의상 마스크 안에서만 합성한다(시험 08). 마스크 밖·보존 영역 픽셀이 바뀌면 실패.
--paste head: 머리(머리카락·얼굴·머리 장식 등 보존 영역)만 원본 픽셀이고 나머지는 Qwen 결과를 쓴다(시험 09). Qwen이 소품을 빼고
              흰 배경으로 그리므로 원본 배경·소품·신발이 남지 않고, 다리와 신발이 한 그림에서 나와 발목이 꺾이지 않는다.
              머리 가장자리 2px는 Qwen 머리를 써서 원본 배경색이 번지지 않게 한다.
머리 정렬(두 방식 공통): 원본 머리 주변과 Qwen 결과 전체에서 특징점(SIFT)을 맞춰 위치·크기·기울기를 잰다. 일치점이 적으면
붙이지 않는다(unmeasured). 차이가 허용 범위 안이면 Qwen 결과 전체를 원본 좌표로 되돌린 뒤(크기·기울기·위치) 다시 재서
거의 일치할 때만 붙인다. 범위를 넘거나 다시 재도 어긋나면 붙이지 않는다(rejected). 붙이지 않은 장은 raw만 남긴다.
위치 검사는 얼굴에 붙은 일치점(얼굴 경계 25px 안)이 8개 이상일 때만 믿는다(긴 머리·장식 일치점만으로 위치를 정하지 않음).
--paste head는 위치가 맞아도 머리 형태·연결을 따로 본다(시험 11 W3: Qwen이 양갈래를 짧은 머리로 바꿈).
  지지율: 얼굴에서 40px 넘게 떨어진 원본 머리를 Qwen 그림의 같은 머리색이 받쳐 주는 비율. 0.5 미만이면 거부.
  떠 있는 조각: 붙인 원본 머리 중 (원본 머리 ∪ Qwen 머리색)을 따라 얼굴과 이어지지 않는 비율. 0.2 초과면 거부.
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
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from genai_lab.outfit_edit import *  # noqa: F401,F403  공통 정렬·합성(동작은 시험 08~11과 같음)
from genai_lab import outfit_edit

PROMPT = ("Dress the character from image 1 in the outfit shown in image 2: a long dark brown coat, a white shirt with a black "
          "ribbon tie, suspenders, dark high-waisted trousers and black shoes. Keep the character's face, hairstyle, hair color, "
          "body shape, body proportions, pose, hands, accessories and background exactly the same as in image 1. "
          "Do not change the camera angle, framing or the character's size or position.")
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
        with np.load(folder / "parts.npz") as parts:
            face = np.logical_or.reduce([parts[k] for k in ("face", "eyes", "nose", "mouth")])
            hair = parts["hair"].copy()
        aligned, alignment = realign(person, edited, window, expand=args.paste == "head", face=face)
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
        if args.paste == "head":
            shape = hair_shape(pad_image(person, pad, 255), aligned, pad_image(hair, pad, False),
                               pad_image(face, pad, False), pad_image(protect, pad, False))
            record.update(hair_shape=shape)
            if shape["status"] != "ok":  # 위치가 맞아도 머리 형태가 바뀌었으면 붙이지 않는다
                aligned = None
        if aligned is None:
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
