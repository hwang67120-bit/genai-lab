"""보존형 의상 교체 시험 2단계(CatVTON 전용 환경 D:/genai-cache/catvton-venv에서 실행, genai_lab을 import하지 않음).

구조: 원본에서 보존 영역 검출(1단계 protect.png) → 바꿀 영역만 생성(CatVTON) → 영역 밖은 원본 픽셀 그대로 합성 → 보존·경계 검사.
mask: CatVTON 자동 의상 마스크에서 보존 영역(5px 확장)을 빼고 미리보기를 만든 뒤 잠근다. 생성은 하지 않는다.
run : 잠금 SHA를 확인하고 CatVTON으로 생성한 뒤, 확정 마스크 안에서만 원본과 섞는다. 마스크 밖 픽셀이 하나라도 바뀌면 실패다."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CATVTON = Path(os.environ.get("CATVTON_ROOT", "D:/genai-cache/tools/CatVTON"))
SIZE = (768, 1024)
PROTECT_MARGIN = 5
FEATHER = 2


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


def catvton_imports():
    sys.path.insert(0, str(CATVTON))
    from model.cloth_masker import AutoMasker
    from model.pipeline import CatVTONPipeline
    from utils import init_weight_dtype, resize_and_padding
    return AutoMasker, CatVTONPipeline, init_weight_dtype, resize_and_padding


def attn_root(cache):
    snapshots = sorted((Path(cache) / "models--zhengchong--CatVTON" / "snapshots").iterdir())
    require(snapshots, "CatVTON 가중치가 로컬에 없습니다(다운로드하지 않음).")
    return snapshots[-1]


def final_mask(auto, protect):
    keep_out = cv2.dilate(protect.astype(np.uint8), np.ones((2 * PROTECT_MARGIN + 1,) * 2, np.uint8)) > 0
    return (auto > 127) & ~keep_out


def overlay(person, change, protect):
    out = person.astype(np.float32).copy()
    out[change] = out[change] * 0.45 + np.array([255, 40, 40]) * 0.55
    out[protect] = out[protect] * 0.45 + np.array([40, 200, 80]) * 0.55
    return out.astype(np.uint8)


def stage_mask(args):
    AutoMasker, _, _, _ = catvton_imports()
    prepare = json.loads((args.prepared / "prepare.json").read_text(encoding="utf-8"))
    require(prepare["gpu_executed"], "1단계 보존 영역(See-through)이 없습니다.")
    root = attn_root(args.cache)
    masker = AutoMasker(densepose_ckpt=str(root / "DensePose"), schp_ckpt=str(root / "SCHP"), device=args.device)
    rows, cases = [], []
    for case in prepare["cases"]:
        folder = args.prepared / case["name"]
        require(sha(folder / "person.png") == case["person_sha256"] and sha(folder / "protect.png") == case["protect_sha256"],
                f"{case['name']}: 1단계 파일이 바뀌었습니다.")
        person = Image.open(folder / "person.png").convert("RGB")
        auto = np.asarray(masker(person, args.cloth_type)["mask"].convert("L"))
        protect = load(folder / "protect.png", "L") > 127
        change = final_mask(auto, protect)
        Image.fromarray(auto).save(folder / "auto-mask.png")
        Image.fromarray(change.astype(np.uint8) * 255).save(folder / "change-mask.png")
        preview = overlay(np.asarray(person), change, protect)
        Image.fromarray(preview).save(folder / "mask-preview.png")
        rows.append(np.concatenate([np.asarray(person), preview], 1))
        cases.append(dict(name=case["name"], change_px=int(change.sum()), auto_px=int((auto > 127).sum()),
                          auto_protect_overlap_px=int(((auto > 127) & protect).sum()),
                          change_mask_sha256=sha(folder / "change-mask.png")))
        print(json.dumps(cases[-1], ensure_ascii=False), flush=True)
    sheet = np.concatenate([np.asarray(Image.fromarray(r).resize((768, 512))) for r in rows], 0)
    Image.fromarray(sheet).save(args.prepared / "mask-review.png")
    garment = Path(args.garment)
    lock = dict(cloth_type=args.cloth_type, garment=str(garment), garment_sha256=sha(garment), protect_margin=PROTECT_MARGIN,
                cases=cases, prepare_sha256=sha(args.prepared / "prepare.json"),
                code_sha256=sha(Path(__file__)), attn_root=str(root))
    write_json(args.prepared / "mask-lock.json", lock)
    digest = sha(args.prepared / "mask-lock.json")
    (args.prepared / "mask-lock.sha256").write_text(digest + "\n")
    print(json.dumps(dict(status="needs_user_review", review_sha256=digest), ensure_ascii=False))


def stage_run(args):
    import torch
    from diffusers.image_processor import VaeImageProcessor
    _, CatVTONPipeline, init_weight_dtype, resize_and_padding = catvton_imports()
    lock_file = args.prepared / "mask-lock.json"
    require(args.review_sha and sha(lock_file) == args.review_sha, "사용자가 확인한 마스크 잠금 SHA가 필요합니다.")
    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    require(lock["code_sha256"] == sha(Path(__file__)) and sha(lock["garment"]) == lock["garment_sha256"], "코드 또는 의상 이미지가 바뀌었습니다.")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    pipeline = CatVTONPipeline(base_ckpt=args.base_model, attn_ckpt=lock["attn_root"], attn_ckpt_version="mix",
                               weight_dtype=init_weight_dtype("bf16"), use_tf32=True, device="cuda")
    blur = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
    garment = resize_and_padding(Image.open(lock["garment"]).convert("RGB"), SIZE)
    rows, results = [], []
    for case in lock["cases"]:
        folder = args.prepared / case["name"]
        require(sha(folder / "change-mask.png") == case["change_mask_sha256"], f"{case['name']}: 확정 마스크가 바뀌었습니다.")
        person_img = Image.open(folder / "person.png").convert("RGB")
        person = np.asarray(person_img)
        change = load(folder / "change-mask.png", "L") > 127
        protect = load(folder / "protect.png", "L") > 127
        model_mask = blur.blur(Image.fromarray(change.astype(np.uint8) * 255), blur_factor=9)
        for seed in args.seeds:
            t0 = time.perf_counter()
            raw = pipeline(image=person_img, condition_image=garment, mask=model_mask, num_inference_steps=args.steps,
                           guidance_scale=args.guidance, generator=torch.Generator(device="cuda").manual_seed(seed))[0]
            seconds = time.perf_counter() - t0
            raw = np.asarray(raw.convert("RGB").resize(SIZE, Image.LANCZOS))
            # 확정 마스크 안에서만 섞는다. 가장자리 2px만 부드럽게 하고, 마스크 밖 가중치는 0으로 고정한다.
            weight = cv2.GaussianBlur(change.astype(np.float32), (0, 0), FEATHER)
            weight[~change] = 0
            composite = np.clip(np.rint(person * (1 - weight[..., None]) + raw * weight[..., None]), 0, 255).astype(np.uint8)
            outside_equal = bool(np.array_equal(composite[~change], person[~change]))
            protect_equal = bool(np.array_equal(composite[protect], person[protect]))
            name = f"{case['name']}_{seed}"
            Image.fromarray(raw).save(out / f"{name}_raw.png")
            Image.fromarray(composite).save(out / f"{name}.png")
            results.append(dict(case=case["name"], seed=seed, seconds=round(seconds, 1), outside_equal=outside_equal,
                                protect_equal=protect_equal, changed_px=int(np.any(composite != person, axis=2).sum()),
                                result_sha256=sha(out / f"{name}.png")))
            print(json.dumps(results[-1], ensure_ascii=False), flush=True)
            require(outside_equal and protect_equal, f"{name}: 마스크 밖 또는 보존 영역 픽셀이 바뀌었습니다.")
            rows.append(np.concatenate([person, composite], 1))
            write_json(out / "results.json", dict(lock_sha256=args.review_sha, results=results))
    Image.fromarray(np.concatenate([np.asarray(Image.fromarray(r).resize((768, 512))) for r in rows], 0)).save(out / "before_after.png")
    write_json(out / "results.json", dict(lock_sha256=args.review_sha, results=results, seconds=time.perf_counter() - started,
                                         max_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30,
                                         steps=args.steps, guidance=args.guidance, base_model=args.base_model))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("mask", "run"))
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--garment", default="C:/Users/user/Downloads/참조 의상/23e2a7bcf47406e46eccd3e249621e26.jpg")
    parser.add_argument("--cloth-type", default="overall", choices=("upper", "lower", "overall"))
    parser.add_argument("--cache", default="D:/genai-cache/huggingface")
    parser.add_argument("--base-model", default="booksforcharlie/stable-diffusion-inpainting")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--review-sha")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if args.stage == "mask":
        stage_mask(args)
    else:
        require(args.output is not None, "run에는 --output이 필요합니다.")
        stage_run(args)


if __name__ == "__main__":
    main()
