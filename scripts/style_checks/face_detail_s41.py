"""S41 얼굴 다듬기 시안(사용자 11/1 "남은 크레딧 다 쓰고 이미지 생성 마무리" — 전신 구도에서 얼굴이 작아 눈·얼굴 세부가 뭉개짐):
후보 그림에서 머리를 찾아(deepghs anime_head_detection, 앱과 같은 캐시) 머리 둘레 정사각형을 잘라 1024로 키우고, 같은 FLUX + 그림체 LoRA로
'같은 그림을 더 세밀하게' 다시 그린다(그림 1 = 자른 머리, 그림 2 = 원본 캐릭터). 원래 크기로 줄여 위상상관으로 위치를 맞춘 뒤
가장자리를 부드럽게 한 타원으로 붙인다. 어긋남이 크면 붙이지 않는다. 끝으로 애니메이션 업스케일러(deepghs/cdc_anime_onnx MIT, 4배 → 2배로 줄임).
사용: python face_detail_s41.py <출력 폴더> <이름>=<앱 작업 폴더(flux)> ...
"""
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
import gc, json, sys, time
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab"); sys.path.insert(0, r"D:\Code\genai-lab\genai_lab")
import numpy as np
from PIL import Image, ImageDraw

CROP_SCALE = 1.9            # 머리 상자의 1.9배 정사각형(귀·머리카락 끝까지)
FACE_SIZE = 1024
MAX_SHIFT = 12.0            # 원래 크기에서 이보다 어긋나면 붙이지 않음
FEATHER = 0.18              # 타원 가장자리 부드럽게(자른 변 대비)
STRENGTH = 0.5              # 2판: 원래 얼굴에서 출발해 4단계 중 뒤쪽 2단계만 다시(flux_img2img) — 1판(편집으로 새로 그림)은 구도·표정이
                            # 바뀌어 6장 모두 어긋남 12px 초과로 못 붙임. 0.75는 세부가 더 바뀜.
FACE_PROMPT = ("Redraw image 1, a close-up of the character's head, as the same picture with much finer detail: keep exactly the same "
               "face, expression, head angle, hair shape, hair colour, eye colour, ears, accessories, lighting and framing as image 1. "
               "The character is the one shown in image 2. Sharp detailed eyes with clear reflections and eyelashes, fine individual "
               "hair strands, clean crisp line art, smooth soft shading. No text, no new objects.")


def head_box(rgb, head_cache):
    import huggingface_hub.constants as hub
    hub.HF_HUB_CACHE = str(head_cache)
    from imgutils.detect import detect_heads
    found = detect_heads(Image.fromarray(rgb))
    if not found:
        return None
    (x0, y0, x1, y1), _, score = max(found, key=lambda d: d[2])
    return [int(x0), int(y0), int(x1), int(y1)], float(score)


def square(box, w, h):
    x0, y0, x1, y1 = box
    cx, cy, side = (x0 + x1) / 2, (y0 + y1) / 2, CROP_SCALE * max(x1 - x0, y1 - y0)
    side = int(min(side, w, h))
    left = int(min(max(cx - side / 2, 0), w - side)); top = int(min(max(cy - side / 2, 0), h - side))
    return left, top, side


def paste(base, detail, left, top, side):
    """detail(side×side)을 위상상관으로 맞춰 타원 가중치로 base에 붙인다. 반환 (그림, dict)."""
    import cv2
    from scipy.ndimage import gaussian_filter
    region = base[top:top + side, left:left + side]
    g = lambda a: cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    (dx, dy), response = cv2.phaseCorrelate(g(detail), g(region), cv2.createHanningWindow((side, side), cv2.CV_32F))
    if max(abs(dx), abs(dy)) > MAX_SHIFT:
        return base, dict(pasted=False, shift=[round(dx, 1), round(dy, 1)], response=round(response, 3))
    moved = cv2.warpAffine(detail, np.float32([[1, 0, dx], [0, 1, dy]]), (side, side), borderMode=cv2.BORDER_REPLICATE)
    yy, xx = np.mgrid[0:side, 0:side]
    r = np.sqrt(((xx - side / 2) / (side / 2)) ** 2 + ((yy - side / 2) / (side / 2)) ** 2)
    weight = gaussian_filter(np.clip((1.0 - r) / FEATHER, 0, 1).astype(np.float32), side * 0.02)[..., None]
    out = base.copy()
    out[top:top + side, left:left + side] = np.clip(moved * weight + region * (1 - weight), 0, 255).astype(np.uint8)
    return out, dict(pasted=True, shift=[round(dx, 1), round(dy, 1)], response=round(response, 3))


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    cases = {k: Path(v) for k, v in (a.split("=", 1) for a in sys.argv[2:])}
    import torch
    from genai_lab.studio_generation import StudioRuntime
    from genai_lab.style_redraw_flux import GUIDANCE, MAX_SEQUENCE_LENGTH, STEPS, STYLE_LORA_TRIGGER, STYLE_LORA_WEIGHT
    from style_redraw_worker import MEMORY_CAP_GIB
    runtime = StudioRuntime.from_environment()
    first = json.loads((next(iter(cases.values())) / "request.json").read_text(encoding="utf-8"))
    items = []
    for name, R in cases.items():
        for f in sorted(R.glob("result-seed*.png")):
            rgb = np.asarray(Image.open(f).convert("RGB"))
            found = head_box(rgb, runtime.head_cache)
            items.append(dict(name=name, R=R, file=f, rgb=rgb, box=found[0] if found else None, score=found[1] if found else None))
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    from diffusers import Flux2KleinPipeline
    from diffusers.quantizers import PipelineQuantizationConfig
    quant = PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit", quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_use_double_quant": True}, components_to_quantize=["transformer", "text_encoder"])
    pipe = Flux2KleinPipeline.from_pretrained(first["flux_model"], torch_dtype=torch.bfloat16, quantization_config=quant, low_cpu_mem_usage=True,
                                              local_files_only=True)
    lora = Path(first["flux_style_lora"]); pipe.load_lora_weights(str(lora.parent), weight_name=lora.name, adapter_name="style")
    pipe.set_adapters(["style"], adapter_weights=[STYLE_LORA_WEIGHT]); pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling()
    records = []
    for it in items:
        rec = dict(name=it["name"], file=str(it["file"]), box=it["box"], score=it["score"])
        stem = f"{it['name']}-{it['file'].stem}"
        result = it["rgb"]
        if it["box"]:
            h, w = result.shape[:2]
            left, top, side = square(it["box"], w, h)
            crop = Image.fromarray(result[top:top + side, left:left + side]).resize((FACE_SIZE, FACE_SIZE), Image.LANCZOS)
            torch.cuda.reset_peak_memory_stats(); t0 = time.time()
            from genai_lab.flux_img2img import redraw
            detail = redraw(pipe, crop, [crop, Image.open(it["R"] / "character.png").convert("RGB")], f"{STYLE_LORA_TRIGGER} {FACE_PROMPT}",
                            STRENGTH, 7, steps=STEPS, guidance=GUIDANCE, max_sequence_length=MAX_SEQUENCE_LENGTH)
            rec.update(seconds=round(time.time() - t0, 1), peak_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2), crop=[left, top, side])
            detail.save(out / f"{stem}-face-detail.png")
            result, info = paste(result, np.asarray(detail.resize((side, side), Image.LANCZOS)), left, top, side)
            rec.update(info)
        Image.fromarray(result).save(out / f"{stem}-face.png")
        records.append(rec); print(rec, flush=True)
    pipe = None; gc.collect(); torch.cuda.empty_cache()
    import huggingface_hub.constants as hub
    hub.HF_HUB_CACHE = str(first["model_cache"])
    from imgutils.upscale import upscale_with_cdc
    for rec in records if os.environ.get("S41_UPSCALE") == "1" else []:      # CPU에서 한 장 약 5분 — 따로 켤 때만
        stem = f"{rec['name']}-{Path(rec['file']).stem}"
        t0 = time.time()
        big = upscale_with_cdc(Image.open(out / f"{stem}-face.png").convert("RGB"), model="HGSR-MHR-anime-aug_X4_320", silent=True)
        w, h = Image.open(out / f"{stem}-face.png").size
        big.resize((w * 2, h * 2), Image.LANCZOS).save(out / f"{stem}-final-2x.png")
        rec["upscale_seconds"] = round(time.time() - t0, 1)
    # 판정판: 얼굴 확대 전/후 + 2배 최종
    W = 360
    sheet = Image.new("RGB", (W * 3, (W + 22) * len(records)), "white"); dr = ImageDraw.Draw(sheet)
    for i, rec in enumerate(records):
        stem = f"{rec['name']}-{Path(rec['file']).stem}"; y = i * (W + 22)
        if not rec.get("crop"):
            continue
        left, top, side = rec["crop"]
        before = Image.open(rec["file"]).convert("RGB").crop((left, top, left + side, top + side)).resize((W, W))
        after = Image.open(out / f"{stem}-face.png").convert("RGB").crop((left, top, left + side, top + side)).resize((W, W))
        final_path = out / f"{stem}-final-2x.png"
        final = (Image.open(final_path).convert("RGB").crop((left * 2, top * 2, (left + side) * 2, (top + side) * 2)).resize((W, W))
                 if final_path.exists() else Image.new("RGB", (W, W), "white"))
        for j, (im, label) in enumerate(((before, "before"), (after, "face detail"), (final, "face detail + 2x upscale"))):
            sheet.paste(im, (j * W, y + 22)); dr.text((j * W + 4, y + 5), f"{rec['name']} {Path(rec['file']).stem} {label}", fill="black")
    sheet.save(out / "face-detail-sheet.png")
    (out / "face-detail.json").write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
