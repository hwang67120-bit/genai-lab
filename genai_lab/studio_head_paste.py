"""선택 후보 → 머리 영역 확인 → 붙이기 → 별도 후보. 생성 원본은 변경하지 않는다."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import os
import time
import uuid
from genai_lab.proportion_inputs import sha, require, json_value, checked_image
from genai_lab.qwen_record_io import write_json
from genai_lab.studio_proportion import restore_head
from genai_lab.studio_optional_finishing import restore_prompt
from genai_lab.onepass_generation import OnePassCancelled
from genai_lab.onepass_generation_settings import OnePassGenerationSettings


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_cancel(cancelled):
    if cancelled():
        raise OnePassCancelled("머리 붙이기를 취소했습니다. 원래 후보를 유지합니다.")


def verify_manifest(manifest):
    for path, digest in manifest.items():
        require(sha(path) == digest, "머리 붙이기 입력이 변경됐습니다: " + str(path))


def source_request(analysis, head, runtime):
    """생성 전에 원본과 확인한 윤곽만 잠근다. 생성 후보는 이 계약에 필요하지 않다."""
    require(head.confirmed, "원본 머리 윤곽이 확인되지 않았습니다.")
    confirmation = Path(head.contour_file).parent / "user-confirmation.json"
    choice = read(confirmation)
    require(choice["reviewer"] == "user" and choice["automatic_approval"] is False
            and choice["head"] == json_value(asdict(head)), "원본 머리 확인 기록이 다릅니다.")
    reference = analysis["references"]["character"]
    identity = dict(rule="G3+H2", source_sha256=reference["sha256"],
                    normalized_sha256=head.normalized_sha256, head_mask_sha256=head.mask_sha256)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest = {str(reference["source"]): reference["sha256"], str(confirmation): sha(confirmation),
                str(head.normalized_file): head.normalized_sha256, str(head.mask_file): head.mask_sha256,
                str(head.contour_file): head.contour_sha256}
    verify_manifest(manifest)
    return json_value(dict(identity=identity, cache_key=key, head=asdict(head), manifest=manifest,
        semantic_repo=runtime.head_semantic_repo, semantic_checkpoint=runtime.head_semantic_checkpoint))


def automatic_skip_reason(appearance, garments):
    """확인된 태그만 검사한다. 후드 상의 자체를 머리 덮개로 추측하지 않는다."""
    coverings = {"hat", "cap", "hood", "hood up", "helmet", "beret", "beanie", "headwear"}
    if coverings & {tag.strip().replace("_", " ") for tag in garments}:
        return "머리를 덮는 의상이 있어 머리 붙이기를 건너뛰었습니다."
    return ""


def preview_skip_reason(info, appearance=("short_hair",)):
    """캐시도 같은 픽셀로 다시 검사한다. 불명확한 배경이나 눈은 자동 승인하지 않는다.
    짧은 머리는 검증된 흰 배경만, 긴 머리는 단색 배경이면 색이 있어도 진행한다."""
    import numpy as np
    from genai_lab.head_paste_semantics import load_parts
    from genai_lab.head_paste_rules import padding_mask, dil, split_features, paste_profile, background_measure, BG_PLAIN_MIN
    from PIL import Image
    verify_preview(info, info["identity"])
    with Image.open(info["original_file"]) as image:
        original = np.asarray(image.convert("RGB"))
    with Image.open(info["head_mask_file"]) as image:
        head = np.asarray(image.convert("L")) > 127
    parts = load_parts(Path(info["directory"]) / "original-parts.npz")
    background = ~dil(np.logical_or.reduce(list(parts.values())) | head, 6) & ~padding_mask(original)
    if not background.any():
        return "원본 배경색을 확인하지 못해 머리 붙이기를 건너뛰었습니다.", None
    median = np.median(original[background], axis=0)
    distance = float(np.linalg.norm(median - 255))
    measurement = dict(median_rgb=median.tolist(), white_distance=distance, pixels=int(background.sum()),
                       distance_space="RGB", foreground_expansion_px=6)
    profile = paste_profile(appearance)
    if profile:
        measurement.update(rule=profile["rule"], plain_ratio=background_measure(original, head, parts)["plain_ratio"],
                           plain_minimum=BG_PLAIN_MIN)
        if measurement["plain_ratio"] < BG_PLAIN_MIN:
            return "원본 배경 무늬가 강해서 머리 붙이기를 건너뛰었습니다.", measurement
    elif distance > 10:
        return "원본 배경이 흰색이 아니어서 머리 붙이기를 건너뛰었습니다.", measurement
    try:
        split_features(parts["eyes"] | parts["nose"] | parts["mouth"])
    except ValueError as error:
        return str(error), measurement
    return "", measurement


def selected_request(results, runtime):
    """완료된 2단계 원본과 승인된 머리만 받아 입력 계약을 잠근다."""
    require(not results.head_paste_unavailable_reason, results.head_paste_unavailable_reason)
    results.verify_current()
    candidate = results.candidate
    preflight = results.batch.directory / "preflight.json"
    lock = read(preflight)
    require(sha(preflight) == preflight.with_suffix(".sha256").read_text().strip(), "비율 생성 사전 기록 잠금이 다릅니다.")
    head = restore_head({"head": lock["options"]["head"]})
    require(head.confirmed, "원본 머리 윤곽이 확인되지 않았습니다.")
    checked_image(head.normalized_file, head.normalized_sha256, (736, 1232), "RGB")
    checked_image(head.mask_file, head.mask_sha256, (736, 1232), "L")
    checked_image(head.contour_file, head.contour_sha256, (736, 1232), "RGB")
    confirmation_file = Path(head.contour_file).parent / "user-confirmation.json"
    confirmation = read(confirmation_file)
    require(confirmation["reviewer"] == "user" and confirmation["automatic_approval"] is False
            and confirmation["head"] == json_value(asdict(head)), "원본 머리 확인 기록이 다릅니다.")
    require(candidate.seed in lock["seeds"] and lock["settings"] == candidate.raw.record["settings"], "선택한 후보와 생성 계약이 다릅니다.")
    inputs = lock["inputs"]
    settings = lock["settings"]
    require((settings["width"], settings["height"], settings["steps"], settings["guidance_scale"], settings["max_reserved_gib"]) == (736, 1232, 28, 5.5, 6.5), "시험과 다른 생성 설정입니다.")
    source = read(results.batch.directory.parent / "inputs/analysis.json")
    references_file = results.batch.directory.parent / "inputs/references.json"
    reference = read(references_file)["character"]
    source_sha = reference["sha256"]
    manifest = {str(reference["source"]): source_sha, str(preflight): sha(preflight), str(confirmation_file): sha(confirmation_file), str(references_file): sha(references_file), str(head.normalized_file): head.normalized_sha256,
                str(head.mask_file): head.mask_sha256, str(head.contour_file): head.contour_sha256,
                str(candidate.raw.path): results.verify_original(),
                str(results.current_path): results.verify_current(), str(candidate.record_path): sha(candidate.record_path),
                str(inputs["face_file"]): inputs["face_sha256"],
                str(results.batch.directory.parent / "inputs/analysis.json"): sha(results.batch.directory.parent / "inputs/analysis.json")}
    identity = dict(rule="G3+H2", source_sha256=source_sha, normalized_sha256=head.normalized_sha256,
                    head_mask_sha256=head.mask_sha256)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    verify_manifest(manifest)
    return json_value(dict(identity=identity, cache_key=key, head=asdict(head), inputs=inputs, settings=settings,
        sketch_root=lock["options"]["sketch_root"], options=lock["options"], analysis=source,
        raw_file=candidate.raw.path, before_file=results.current_path, before_sha256=results.verify_current(),
        raw_sha256=results.verify_original(), seed=candidate.seed, manifest=manifest,
        semantic_repo=runtime.head_semantic_repo, semantic_checkpoint=runtime.head_semantic_checkpoint))


def run_worker(mode, request, directory, runtime, cancelled, progress):
    """단계마다 별도 프로세스를 끝내 모델이 서로 겹쳐 올라가지 않게 한다."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "request.json", json_value(request))
    worker = Path(__file__).with_name("head_paste_worker.py")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DIFFUSERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    start = time.perf_counter()
    status = {"status": "started", "stage": mode}
    process = None
    try:
        check_cancel(cancelled)
        progress("상태: 머리 영역 분석 중" if mode == "hair" else "상태: 원본 머리 붙이기·경계 다듬기 중 · 원본 보존")
        with (directory / "worker.log").open("wb") as log:
            process = subprocess.Popen([str(runtime.analysis_python), "-B", str(worker), mode, str(directory / "request.json")],
                stdout=log, stderr=subprocess.STDOUT, env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            while process.poll() is None:
                check_cancel(cancelled)
                require(time.perf_counter() - start <= runtime.head_paste_timeout, "머리 붙이기 시간이 초과됐습니다.")
                time.sleep(.1)
            if process.returncode:
                error_file = directory / "worker-error.json"
                reason = read(error_file).get("error", "머리 처리 실패") if error_file.is_file() else "머리 처리 실패"
                raise ValueError(reason + "\n원래 후보를 유지합니다. 기록: " + str(directory / "worker.log"))
        check_cancel(cancelled)
        result = read(directory / "result.json")
        verify_manifest(request["manifest"])
        status["status"] = "awaiting_user_review"
        return result
    except BaseException as error:
        status.update(status="cancelled" if isinstance(error, OnePassCancelled) else "failed", error=str(error))
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        status["seconds"] = time.perf_counter() - start
        write_json(directory / "operation.json", status)


def verify_preview(info, identity):
    require(info["identity"] == identity, "다른 원본의 머리 영역입니다.")
    verify_manifest(info["manifest"])
    return info


def prepare_hair(request, cache, runtime, *, cancelled=lambda: False, progress=lambda _: None):
    cache = Path(cache) / request["cache_key"]
    for path in sorted(cache.glob("*/confirmation.json")):
        choice = read(path)
        if choice.get("confirmed") is True:
            info = read(path.parent / "result.json")
            try:
                verify_preview(info, request["identity"])
                require(choice["preview_sha256"] == sha(path.parent / "result.json"), "머리 확인 기록이 변경됐습니다.")
                return {**info, "remembered": True}
            except (ValueError, OSError):
                continue  # 바뀐 자료를 승인된 영역으로 재사용하지 않는다.
    return run_worker("hair", request, cache / uuid.uuid4().hex, runtime, cancelled, progress)


def confirm_hair(info, accepted):
    """화면에서 고른 답만 저장한다. 닫거나 거부하면 생성하지 않는다."""
    verify_preview(info, info["identity"])
    directory = Path(info["directory"])
    write_json(directory / "confirmation.json", dict(confirmed=bool(accepted), reviewer="user",
        preview_sha256=sha(directory / "result.json"), identity=info["identity"],
        reason=None if accepted else "사용자가 머리 영역을 승인하지 않음"))


def apply_head(request, preview, directory, runtime, *, cancelled=lambda: False, progress=lambda _: None):
    verify_preview(preview, request["identity"])
    confirmed = read(Path(preview["directory"]) / "confirmation.json")
    require(confirmed["confirmed"] and confirmed["identity"] == request["identity"]
            and confirmed["preview_sha256"] == sha(Path(preview["directory"]) / "result.json"), "머리 영역을 먼저 확인해 주세요.")
    manifest = {**request["manifest"], **preview["manifest"],
                str(Path(preview["directory"]) / "confirmation.json"): sha(Path(preview["directory"]) / "confirmation.json")}
    job = {**request, "preview": preview, "manifest": manifest}
    return run_worker("apply", job, directory, runtime, cancelled, progress)


def head_product_info(directory, original_sha):
    """채택·선택·저장 전에 파일과 실행 검사를 다시 확인한다."""
    directory = Path(directory)
    info = read(directory / "result.json")
    require(info["status"] == "awaiting_user_review" and info["raw_sha256"] == original_sha, "머리 결과의 기준 원본이 다릅니다.")
    verify_manifest(info["manifest"])
    checks = info["checks"]
    require(checks["outside_equal"] and checks["face_equal"], "머리 밖·얼굴 보호 검사가 실패했습니다.")
    require(sha(info["product_file"]) == info["product_sha256"], "머리 붙이기 결과가 변경됐습니다.")
    record = read(info["redraw_record"])
    require(record["status"] == "review_pending" and len(record["unet_calls"]) == 28
            and len(record["callback"]) == 28 and record["adapter_calls"] == 1
            and record["max_reserved_gib"] <= 6.5
            and all(row.get("batch") == 2 and row.get("adapter") is True for row in record["unet_calls"])
            and all(row.get("index") == index and row.get("mode") == ("overwrite" if index < 24 else "blend")
                    for index, row in enumerate(record["callback"])), "머리 다시 그리기 실행 검사가 실패했습니다.")
    return info


def apply_head_batch(batch, preview, runtime, *, cancelled=lambda: False, progress=lambda _: None):
    """4장 계약을 잠그고 별도 프로세스 하나로 처리한다. 원래 후보는 덮어쓰지 않는다."""
    from genai_lab.studio_generation import StudioResults
    from genai_lab.generation_cleanup import release_cuda_cache, GenerationCleanupError
    memory = release_cuda_cache(sys.modules.get("torch"))
    if memory["allocated_bytes"] or memory["reserved_bytes"]:
        raise GenerationCleanupError("머리 붙이기 전에 생성 GPU 자원이 남아 있습니다.")
    results = StudioResults(batch)
    jobs = []
    for index in range(len(batch.candidates)):
        check_cancel(cancelled)
        results.select(index)
        request = selected_request(results, runtime)
        verify_preview(preview, request["identity"])
        confirmation = Path(preview["directory"]) / "confirmation.json"
        accepted = read(confirmation)
        require(accepted["confirmed"] and accepted["identity"] == request["identity"]
                and accepted["preview_sha256"] == sha(Path(preview["directory"]) / "result.json"),
                "머리 영역을 먼저 확인해 주세요.")
        jobs.append({**request, "preview": preview, "manifest": {**request["manifest"], **preview["manifest"],
                     str(confirmation): sha(confirmation)}})
    manifest = {}
    for job in jobs:
        manifest.update(job["manifest"])
    directory = batch.directory / "auto-head-pastes" / uuid.uuid4().hex
    request = dict(jobs=jobs, manifest=manifest, gui_memory_before_worker=memory)
    report = run_worker("batch", request, directory, runtime, cancelled, progress)
    require(len(report["items"]) == len(jobs), "머리 결과 수가 다릅니다.")
    for item, job in zip(report["items"], jobs):
        require(item["seed"] == job["seed"], "머리 결과 순서가 다릅니다.")
        if item["status"] == "completed":
            head_product_info(item["directory"], job["raw_sha256"])
    return report
