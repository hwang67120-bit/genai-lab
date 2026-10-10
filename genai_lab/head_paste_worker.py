"""분할 모델과 생성 모델을 순서대로 사용한다. 실패·취소는 완료 후보를 만들지 않는다."""
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DIFFUSERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
from PIL import Image
from genai_lab.proportion_inputs import sha, require, checked_image, json_value, ProportionOptions, validate_models
from genai_lab.qwen_record_io import write_json
from genai_lab.studio_head_paste import read, verify_manifest
from genai_lab.studio_proportion import restore_head
from genai_lab.studio_optional_finishing import restore_prompt
from genai_lab.head_paste_rules import head_crop, padding_mask, hair_mask, crop_head, paste_canvas, blend_result, blend_white_product
from genai_lab.head_paste_semantics import HeadSegmenter, save_parts, load_parts, CHECKPOINT_SHA
from genai_lab.head_paste_redraw import redraw_prepared, head_prompt
from genai_lab.onepass_generation_settings import OnePassGenerationSettings
from genai_lab.head_lines import extract_lines
from genai_lab.proportion_foreground import AnimeForeground


def save_image(directory, name, array):
    path = directory / (name + ".png")
    Image.fromarray(array).save(path)
    return str(path), sha(path)


def original_crop(request):
    head = request["head"]
    rgb = checked_image(head["normalized_file"], head["normalized_sha256"], (736, 1232), "RGB")
    mask = checked_image(head["mask_file"], head["mask_sha256"], (736, 1232), "L")
    return head_crop(rgb, mask)


def preview_overlay(original, mask, color):
    result = original.copy()
    result[mask] = (.45 * result[mask] + .55 * np.asarray(color)).astype(np.uint8)
    return result


def prepare_preview(request, directory, segmenter_factory=HeadSegmenter):
    """원본 한 번 분할 → 고정 규칙 보정 → 확인 그림·분할 고정 자료 저장."""
    verify_manifest(request["manifest"])
    started = time.perf_counter()
    original, mask, box = original_crop(request)
    segmenter = segmenter_factory(request["semantic_repo"], request["semantic_checkpoint"])
    try:
        parts = segmenter.parse(original)
        semantic = segmenter.metrics()
    finally:
        segmenter.close()
    H, counts = hair_mask(original, mask, parts, padding_mask(original))
    original_file, original_sha = save_image(directory, "original", original)
    mask_file, mask_sha = save_image(directory, "head-mask", mask.astype(np.uint8) * 255)
    h_file, h_sha = save_image(directory, "H", H.astype(np.uint8) * 255)
    corrected, _ = save_image(directory, "corrected-preview", preview_overlay(original, H, (0, 220, 120)))
    automatic, _ = save_image(directory, "automatic-preview", preview_overlay(original, parts["hair"], (0, 200, 255)))
    save_parts(directory / "original-parts.npz", parts)
    manifest = {str(directory / name): sha(directory / name) for name in
                ("original.png", "head-mask.png", "H.png", "corrected-preview.png", "automatic-preview.png", "original-parts.npz")}
    manifest[str(request["semantic_checkpoint"])] = CHECKPOINT_SHA
    verify_manifest(request["manifest"])
    result = dict(directory=str(directory), identity=request["identity"], status="needs_user_review", box=box,
        original_file=original_file, original_sha256=original_sha, H_file=h_file, H_sha256=h_sha,
        head_mask_file=mask_file, head_mask_sha256=mask_sha, automatic_preview=automatic,
        corrected_preview=corrected, manifest=manifest, measurements=counts, semantic=semantic, seconds=time.perf_counter()-started)
    write_json(directory / "result.json", json_value(result))
    return result


def prepare_paste(request, directory, segmenter_factory=HeadSegmenter, *, segmenter=None):
    """확인 H와 생성 분할로 붙이기 재료를 만든 뒤 분할 모델을 먼저 내린다."""
    preview = request["preview"]
    original, mask, box = original_crop(request)
    require(list(box) == preview["box"], "확인한 머리 자르기와 다릅니다.")
    H = checked_image(preview["H_file"], preview["H_sha256"], (1024, 1024), "L") > 127
    parts = load_parts(Path(preview["directory"]) / "original-parts.npz")
    raw = checked_image(request["raw_file"], request["raw_sha256"], (736, 1232), "RGB")
    gen = crop_head(raw, box)
    owns_segmenter = segmenter is None
    if owns_segmenter:
        segmenter = segmenter_factory(request["semantic_repo"], request["semantic_checkpoint"])
    try:
        generated_parts = segmenter.parse(gen)
        semantic = segmenter.metrics()
    finally:
        if owns_segmenter:
            segmenter.close()
    save_parts(directory / "generated-parts.npz", generated_parts)
    canvas, M, features, measurements = paste_canvas(original, mask, parts, H, gen, generated_parts)
    seed = request["seed"]
    init_file, init_sha = save_image(directory, "init", canvas)
    mask_file, mask_sha = save_image(directory, "mask", M.astype(np.uint8) * 255)
    features_file, features_sha = save_image(directory, "features", features.astype(np.uint8) * 255)
    sketch = extract_lines(canvas, np.full((1024, 1024), 255, np.uint8))
    _, sketch_sha = save_image(directory, f"sketch_{seed}", sketch)
    pasted, checks = blend_result(raw, Image.fromarray(canvas), M.astype(np.uint8) * 255, features.astype(np.uint8) * 255, box)
    paste_file, paste_sha = save_image(directory, "paste", pasted)
    entry = dict(raw=request["raw_file"], raw_sha256=request["raw_sha256"], init=init_file, init_sha256=init_sha,
                 mask=mask_file, mask_sha256=mask_sha, features=features_file, features_sha256=features_sha,
                 sketch_sha256=sketch_sha, paste=paste_file, paste_sha256=paste_sha, checks=checks, **measurements)
    return dict(box=box, seeds=[seed], H_sha256=preview["H_sha256"], semantic=semantic, per_seed={str(seed): entry})


def restore_models(request):
    fields = dict(request["settings"])
    for name in ("model_root", "adapter_root", "ip_root"):
        fields[name] = Path(fields[name])
    settings = OnePassGenerationSettings(**fields)
    options = dict(request["options"])
    options["head"] = restore_head({"head": options["head"]})
    for name in ("sketch_root", "foreground_model", "shoulder_record_file", "head_lines_file", "width_record_file"):
        if name in options and options[name] is not None:
            options[name] = Path(options[name])
    options = ProportionOptions(**options)
    models = validate_models(settings, options)
    return settings, options, models


def prepare_white_product(before, raw_redraw_file, mask, features, box, options, directory,
                          foreground_factory=AnimeForeground):
    """기존 CPU 배경 분리 모델로 알파를 재계산하고, 합성 결과와 모델 기록을 반환한다."""
    require(sha(options.foreground_model) == options.foreground_sha256, 'isnet-anime가 변경됐습니다.')
    started = time.perf_counter()
    foreground = foreground_factory(options)
    try:
        with Image.open(raw_redraw_file) as image:
            rgb = image.convert('RGB')
            alpha = foreground.alpha(rgb)
            product, checks = blend_white_product(before, np.asarray(rgb), alpha, mask, features, box)
    finally:
        foreground.close()
    alpha_file, alpha_sha = save_image(directory, 'product-alpha', alpha)
    background = dict(model=dict(id='skytnt/anime-seg', file=str(options.foreground_model),
                                sha256=options.foreground_sha256, provider='CPUExecutionProvider'),
        source_file=str(raw_redraw_file), source_sha256=sha(raw_redraw_file),
        alpha_file=alpha_file, alpha_sha256=alpha_sha, seconds=time.perf_counter()-started,
        scope='expanded_support_without_face_features')
    return product, checks, background


def redraw_models(request):
    """같은 일괄 작업의 모델 파일을 한 번 잠근다. 모델 추론은 하지 않는다."""
    settings, options, models = restore_models(request)
    model_files = {}
    for root in (settings.model_root, settings.ip_root / settings.image_encoder_subfolder, options.sketch_root):
        for path in Path(root).rglob("*"):
            if path.is_file() and path.suffix in (".safetensors", ".json", ".txt"):
                model_files[str(path)] = sha(path)
    ip_weight = settings.ip_root / settings.ip_subfolder / settings.ip_weight_name
    model_files[str(ip_weight)] = sha(ip_weight)
    model_files[str(options.foreground_model)] = options.foreground_sha256
    return settings, options, model_files


def prepare_redraw(request, directory, segmenter=None, model_context=None):
    """확인한 원본과 생성 분할로 다시 그리기 입력을 준비한다."""
    verify_manifest(request["manifest"])
    settings, options, model_files = model_context or redraw_models(request)
    manifest = {**request["manifest"], **model_files}
    rec = prepare_paste(request, directory, segmenter=segmenter)
    entry = rec["per_seed"][str(request["seed"])]
    for key in ("init", "mask", "features", "paste"):
        manifest[entry[key]] = entry[key + "_sha256"]
    manifest[str(directory / f"sketch_{request['seed']}.png")] = entry["sketch_sha256"]
    manifest[str(directory / "generated-parts.npz")] = sha(directory / "generated-parts.npz")
    inputs = SimpleNamespace(**{**request["inputs"], "prompt": restore_prompt(request["inputs"]["prompt"])})
    from genai_lab.onepass_prompt_tokenizers import load_onepass_tokenizers
    prompt = head_prompt(inputs, request["analysis"], load_onepass_tokenizers())
    prepared = dict(run=directory, rec=rec, inputs=inputs, settings=settings, options=options,
                    prompt=prompt, manifest=manifest, paste_dir=directory)
    write_json(directory / "paste-inputs.json", json_value(rec))
    verify_manifest(manifest)
    return prepared


def complete_product(request, prepared, output):
    """보호 검사와 파일 잠금을 확인한 장만 완성 결과로 기록한다."""
    directory, rec = prepared["run"], prepared["rec"]
    options, manifest = prepared["options"], prepared["manifest"]
    entry = rec["per_seed"][str(request["seed"])]
    generated = output / directory.name[:8] / f"seed-{request['seed']}"
    run = read(generated / "run.json")
    require(run["status"] == "review_pending", "머리 다시 그리기가 완료되지 않았습니다.")
    before = checked_image(request["before_file"], request["before_sha256"], (736, 1232), "RGB")
    M = checked_image(entry["mask"], entry["mask_sha256"], (1024, 1024), "L")
    features = checked_image(entry["features"], entry["features_sha256"], (1024, 1024), "L")
    product, checks, background = prepare_white_product(before, generated / "raw_redraw.png",
        M, features, rec["box"], options, directory)
    manifest[background["alpha_file"]] = background["alpha_sha256"]
    product_file, product_sha = save_image(directory, "product", product)
    for path in (generated / "head_1024.png", generated / "raw_redraw.png", generated / "run.json", directory / "paste-inputs.json"):
        manifest[str(path)] = sha(path)
    verify_manifest(manifest)
    info = dict(status="awaiting_user_review", directory=str(directory), raw_sha256=request["raw_sha256"],
                product_file=product_file, product_sha256=product_sha, before_file=request["before_file"],
                before_sha256=request["before_sha256"], checks=checks, product_background=background, redraw_record=str(generated / "run.json"),
                H_sha256=request["preview"]["H_sha256"], semantic=rec["semantic"], manifest=manifest, seed=request["seed"], rule="G3+H2",
                known_limits="모자·후드·옆모습 미검증; 원본 외곽선·얼룩·귀 조각이 남을 수 있음")
    write_json(directory / "result.json", info)
    return info



def complete_apply(request, directory):
    """기존 한 장 처리 계약을 유지한다. 일괄 처리도 같은 준비·합성 함수를 쓴다."""
    prepared = prepare_redraw(request, directory)
    state = dict(status="generating", completed=[], start_index=24, memory_mode="block")
    output = directory / "redraw"
    output.mkdir()
    redraw_prepared([prepared], output, state)
    require(not state.get("final_cleanup_errors"), "머리 모델 정리 실패")
    return complete_product(request, prepared, output)


def complete_batch(request, directory):
    """생성 분할 모두 → 분할 모델 해제 → 다시 그리기 한 번 로드 → 장별 보호 합성."""
    from genai_lab.generation_cleanup import detach_error_frames, release_cuda_cache
    import torch
    started = time.perf_counter()
    verify_manifest(request["manifest"])
    memory = release_cuda_cache(torch)
    free, total = torch.cuda.mem_get_info()
    report = dict(status="processing", items=[], timings={}, memory_before_models={**memory,
        "device_free_bytes": free, "device_total_bytes": total},
        gui_memory_before_worker=request["gui_memory_before_worker"])
    jobs = request["jobs"]
    first = jobs[0]
    require(len({job["seed"] for job in jobs}) == len(jobs), "seed가 중복됐습니다.")
    require(all(job["settings"] == first["settings"] and job["options"] == first["options"] for job in jobs), "일괄 생성 계약이 다릅니다.")
    model_context = redraw_models(first)
    segmenter = HeadSegmenter(first["semantic_repo"], first["semantic_checkpoint"])
    prepared = []
    try:
        for index, job in enumerate(jobs):
            folder = directory / f"candidate-{index+1}"
            folder.mkdir()
            item = dict(seed=job["seed"], directory=str(folder), status="failed", reason="")
            report["items"].append(item)
            try:
                require((job["semantic_repo"], job["semantic_checkpoint"]) ==
                        (first["semantic_repo"], first["semantic_checkpoint"]), "분할 모델이 다릅니다.")
                prepared.append(prepare_redraw(job, folder, segmenter, model_context))
                item["status"] = "prepared"
            except Exception as error:
                # 입력 변경은 실패한 장만 원래 결과로 표시한다. CUDA 한도는 전체 모델 작업을 중단한다.
                item["reason"] = str(error)
                detach_error_frames(error)
                if "메모리 한도" in str(error):
                    raise
    finally:
        segmenter.close()
        del segmenter
    report["timings"]["generated_segmentation_seconds"] = time.perf_counter()-started
    output = directory / "redraw"
    output.mkdir()
    state = dict(status="generating", completed=[], start_index=24, memory_mode="block", isolate_failures=True)
    redraw_started = time.perf_counter()
    if prepared:
        try:
            redraw_prepared(prepared, output, state)
        except Exception as error:
            # 전역 로드·메모리 실패는 재시도하지 않고 처리되지 않은 장에 같은 사유를 남긴다.
            report["global_error"] = str(error)
            detach_error_frames(error)
    report["timings"]["redraw_seconds"] = time.perf_counter()-redraw_started
    for index, (job, item) in enumerate(zip(jobs, report["items"])):
        if item["status"] != "prepared":
            continue
        p = next(p for p in prepared if str(p["run"]) == item["directory"])
        key = [p["run"].name[:8], job["seed"]]
        if key not in state["completed"] or state.get("final_cleanup_errors"):
            item.update(status="failed", reason=state.get("candidate_errors", {}).get(str(job["seed"]))
                or report.get("global_error") or "머리 모델 실행·정리 실패")
            continue
        try:
            info = complete_product(job, p, output)
            item.update(status="completed", product_file=info["product_file"], product_sha256=info["product_sha256"])
        except Exception as error:
            item.update(status="failed", reason=str(error))
            detach_error_frames(error)
    report.update(status="awaiting_user_review", seconds=time.perf_counter()-started,
                  segmentation_model_loads=1, redraw=state)
    verify_manifest(request["manifest"])
    write_json(directory / "result.json", report)
    return report

def main():
    mode, filename = sys.argv[1:]
    directory = Path(filename).parent
    request = read(filename)
    try:
        if mode == "hair":
            prepare_preview(request, directory)
        elif mode == "apply":
            complete_apply(request, directory)
        elif mode == "batch":
            complete_batch(request, directory)
        else:
            raise ValueError("알 수 없는 머리 처리 단계입니다.")
    except BaseException as error:
        write_json(directory / "worker-error.json", {"status": "failed", "stage": mode, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
