"""Optional scene-lineart integration into the existing reference-only candidate flow."""
from pathlib import Path
from time import perf_counter
import gc
import json
import math
import tempfile
import numpy as np
from PIL import Image

from genai_lab.scene_reference import (
    AnalysisRejected, SceneCondition, analyze_with_retry, binary_mask,
    warp_garment, compose_scene, LineartExtractor,
)


def scene_settings(config):
    settings = config.get("scene_lineart", {})
    if not settings.get("enabled", False):
        return None
    if not config.get("clothing_reference_generation", {}).get("enabled", False):
        raise ValueError("장면 선화는 참조 생성 모드에서만 사용할 수 있습니다.")
    if config.get("model", {}).get("family") != "sdxl":
        raise ValueError("장면 선화에는 SDXL 모델이 필요합니다.")
    if not config.get("style", {}).get("enabled", False):
        raise ValueError("장면 선화에는 RGB IP-Adapter 참조도 필요합니다.")
    if settings.get("model_id") != "TheMistoAI/MistoLine":
        raise ValueError("검증 대상 SDXL 선화 모델만 허용합니다.")
    if not isinstance(settings.get("extractor_output_white_lines"), bool):
        raise ValueError("선화 추출기 극성을 설정하고 확인해야 합니다.")
    for key, default in (("conditioning_scale", .5), ("guidance_end", .8)):
        value = float(settings.get(key, default))
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"scene_lineart.{key}: 0~1 범위가 필요합니다.")
    if not 1 <= int(settings.get("analysis_attempts", 2)) <= 3:
        raise ValueError("장면 분석 시도는 1~3회로 제한됩니다.")
    timeout = float(settings.get("timeout_seconds", 1800))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("장면 생성 제한 시간이 올바르지 않습니다.")
    return settings


def validate_scene_condition(condition, size):
    if not isinstance(condition, SceneCondition):
        raise ValueError("검증된 SceneCondition이 필요합니다. 임의 RGB 초기 이미지는 허용하지 않습니다.")
    hint = condition.hint
    if hint.mode != "RGB" or hint.size != size:
        raise ValueError("선화 조건의 형식/출력 좌표 크기가 다릅니다.")
    pixels = np.asarray(hint)
    if not (np.array_equal(pixels[..., 0], pixels[..., 1])
            and np.array_equal(pixels[..., 1], pixels[..., 2])):
        raise ValueError("선화 조건에 RGB 색상 이미지가 들어왔습니다.")
    if pixels.min() == pixels.max():
        raise ValueError("선화 조건이 단색으로 비어 있습니다.")
    if not 2 <= len(condition.references) <= 9:
        raise ValueError("부위 참조 개수는 얼굴·의상 포함 2~9개여야 합니다.")
    occupied = np.zeros((size[1], size[0]), dtype=bool)
    names = set()
    for ref in condition.references:
        region = binary_mask(ref.region, size, ref.name)
        if (not region.any() or np.any(occupied & region)
                or ref.rgb.mode != "RGB" or min(ref.rgb.size) <= 0):
            raise ValueError("부위 RGB 참조 또는 영향 영역이 잘못되었습니다.")
        if ref.name in names or not np.isfinite(ref.scale) or not 0 <= ref.scale <= 1:
            raise ValueError("부위 이름/강도 오류")
        occupied |= region
        names.add(ref.name)
    if not {"identity", "garment"} <= names:
        raise ValueError("얼굴·헤어와 의상 참조가 모두 필요합니다.")


def prepare_scene_inputs(source, garment, config, run_log=None,
                         cancelled=lambda: False, status=lambda message: None):
    from genai_lab.visual_reference import VisualInputs
    settings = scene_settings(config)
    analyzer = settings.get("analyzer")
    if analyzer is None:
        raise ValueError(
            "장면 선화 자동 분석기가 아직 연결되지 않았습니다. "
            "의상 기준점·특수 부위·가려짐 분석 구현을 연결해야 합니다. "
            "임의 좌표나 초기 이미지로 대체하지 않습니다.")
    directory = Path(tempfile.mkdtemp(prefix="genai-scene-"))
    timings = {}
    scene = None
    extractor = None
    condition = None
    outcome = "failed"
    failure = None
    started = perf_counter()
    try:
        try:
            scene = analyze_with_retry(analyzer, source, garment,
                attempts=int(settings.get("analysis_attempts", 2)),
                cancelled=cancelled, status=status)
        finally:
            analyzer.close()
            gc.collect()
        timings["analysis_seconds"] = perf_counter() - started
        started = perf_counter()
        warped = warp_garment(scene.garment_rgba, scene.src_points, scene.dst_points,
                              source.size, cancelled=cancelled)
        try:
            if cancelled():
                raise InterruptedError("scene preprocessing cancelled")
            extractor = LineartExtractor(config["model"]["cache_dir"],
                output_white_lines=settings["extractor_output_white_lines"])
            condition = compose_scene(scene, warped, extractor, directory,
                garment_scale=float(config["clothing_reference_generation"].get(
                    "garment_reference_scale", .45)))
        finally:
            warped.close()
        timings["alignment_lineart_seconds"] = perf_counter() - started
        validate_scene_condition(condition, source.size)
        identity = next(p for p in condition.references if p.name == "identity")
        garment_ref = next(p for p in condition.references if p.name == "garment")
        inputs = VisualInputs(source.copy(), identity.rgb.copy(), garment_ref.rgb.copy(),
                              identity.region.copy(), garment_ref.region.copy(),
                              scene_condition=condition)
        condition = None  # Ownership transferred to VisualInputs.
        outcome = "prepared"
        return inputs
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if scene is not None:
            scene.close()
        if condition is not None:
            condition.close()
        if extractor is not None:
            extractor.close()
        gc.collect()
        (directory / "timings.json").write_text(json.dumps(
            {"status": outcome, "error": failure, "timings": timings},
            ensure_ascii=False, indent=2), encoding="utf-8")
        if run_log is not None:
            run_log.write_stage("장면 선화 전처리", f"진단={directory}, 시간={timings}")


def reference_scales(inputs, identity_scale, garment_scale):
    if inputs.scene_condition is None:
        return [identity_scale, garment_scale, *[
            ref.scale for ref in getattr(inputs, 'extra_references', ())]]
    return [identity_scale if ref.name == "identity" else
            garment_scale if ref.name == "garment" else ref.scale
            for ref in inputs.scene_condition.references]


def scene_arguments(pipeline, config, inputs, size, log=None):
    settings = scene_settings(config)
    condition = getattr(inputs, "scene_condition", None)
    if settings is None:
        if condition is not None:
            raise ValueError("선화 조건이 있지만 선화 설정이 꺼져 있습니다.")
        return {}
    validate_scene_condition(condition, size)
    if getattr(pipeline, "_genai_lab_scene_lineart_model_id", None) != settings["model_id"]:
        raise ValueError("현재 모델은 장면 선화용이 아닙니다. 모델을 다시 준비하세요.")
    started = perf_counter()
    timeout = float(settings.get("timeout_seconds", 1800))
    cancelled = settings.get("cancelled", lambda: False)

    def on_step_end(pipe, step, timestep, callback_kwargs):
        elapsed = perf_counter() - started
        if log is not None:
            import torch
            memory = ""
            if torch.cuda.is_initialized():
                memory = (f", VRAM allocated={torch.cuda.memory_allocated() / 2**20:.1f}MiB"
                          f", reserved={torch.cuda.memory_reserved() / 2**20:.1f}MiB")
            log.write_stage("장면 선화 추론", f"단계={step + 1}, 경과={elapsed:.1f}초{memory}")
        if cancelled():
            raise InterruptedError("장면 선화 생성을 취소했습니다.")
        if elapsed > timeout:
            raise TimeoutError("장면 선화 생성 제한 시간을 초과했습니다.")
        return callback_kwargs

    if cancelled():
        raise InterruptedError("장면 선화 생성을 취소했습니다.")
    return {
        # Img2Img의 시작 RGB는 generator가 별도로 전달한다.
        "control_image": condition.hint,
        "controlnet_conditioning_scale": float(settings.get("conditioning_scale", .5)),
        "control_guidance_end": float(settings.get("guidance_end", .8)),
        "callback_on_step_end": on_step_end,
    }
