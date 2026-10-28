"""R6 머리 윤곽 비교 실행기다. 기본은 CPU 사전 검사이며 GPU를 사용하지 않는다. main → run_trial → run_case 순서로 읽는다. 운영
모듈은 변경하지 않고 잠긴 BASE 조건은 기존 단일 어댑터 실행기에 맡긴다.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path, PurePath
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.head_contour_trial_inputs import (
    DESIGN, BASE_SHAS, CONTOUR_SHA, canonical, preflight, read, require, sha, verify_download,
)
from genai_lab.onepass_generation import (
    DiffusersOnePassBackend, OnePassGenerationError, check_pipeline_support,
    encode_prompt_plan, generate_onepass_image, pipeline_callback_kwargs,
)
from genai_lab.onepass_generation_settings import scheduler_config

OFFLOAD = "text_encoder->text_encoder_2->image_encoder->adapter->unet->vae"
STRENGTHS = [1.2, .9]
NEW_SEEDS = (209212002, 209212004, 209212005, 209212006)
MAP_SEEDS = (209212003, 209212001, *NEW_SEEDS)


def encode_record_path(value):
    """경로만 JSON 문자열로 바꾼다. 알 수 없는 객체는 여전히 오류다."""
    if isinstance(value, PurePath):
        return str(value)
    raise TypeError(f"Unsupported trial record value: {type(value).__name__}")


def write_trial_record(path, record):
    """파일을 열기 전에 중첩 경로를 직렬화해 인코딩 실패 시 기존 파일을 보존한다."""
    text = json.dumps(record, ensure_ascii=False, indent=2, default=encode_record_path)
    Path(path).write_text(text + chr(10), encoding="utf-8")


def load_adapter_pair(adapter_type, multi_type, settings, sketch_root, dtype):
    """검증한 로컬 모델 두 개만 fp16 종류를 명시해 로드한다."""
    pose = adapter_type.from_pretrained(str(settings.adapter_root), torch_dtype=dtype,
                                        use_safetensors=True, local_files_only=True)
    sketch = adapter_type.from_pretrained(str(sketch_root), torch_dtype=dtype, variant="fp16",
                                          use_safetensors=True, local_files_only=True)
    return multi_type([pose, sketch])


def call_with_contour(pipe, embeds, images, contour, settings, seed_generator, callbacks, sketch_strength=.9):
    """얼굴 입력은 따로 두고 제어 입력 순서는 항상 자세 다음 머리 스케치다."""
    return pipe(**embeds, image=[images[0], contour], ip_adapter_image=images[1],
                width=settings.width, height=settings.height,
                num_inference_steps=settings.steps, guidance_scale=settings.guidance_scale,
                adapter_conditioning_scale=trial_strengths(sketch_strength),
                adapter_conditioning_factor=settings.adapter_factor,
                generator=seed_generator, **callbacks)


class HeadContourBackend(DiffusersOnePassBackend):
    """시험 분기만 두 번째 어댑터를 로드한다. 해제는 제품 코드를 재사용한다."""

    def __init__(self, settings, sketch_root, sketch_strength=.9, sketch_maps=None):
        self.sketch_maps = sketch_maps
        self.sketch_strength = trial_strengths(sketch_strength)[1]
        import torch
        import diffusers
        from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, MultiAdapter
        from diffusers import EulerAncestralDiscreteScheduler
        from transformers import CLIPVisionModelWithProjection
        self.torch, self.settings = torch, settings
        self.pipe = None
        self.scheduler_type = EulerAncestralDiscreteScheduler
        self.versions = {"torch": torch.__version__, "diffusers": diffusers.__version__}
        self.callback_mode = check_pipeline_support(StableDiffusionXLAdapterPipeline)
        require(torch.cuda.is_available(), "CUDA is unavailable")
        try:
            adapter = load_adapter_pair(T2IAdapter, MultiAdapter, settings, sketch_root, torch.float16)
            encoder = CLIPVisionModelWithProjection.from_pretrained(
                str(settings.ip_root / settings.image_encoder_subfolder), torch_dtype=torch.float16,
                use_safetensors=True, local_files_only=True)
            self.pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
                str(settings.model_root), adapter=adapter, image_encoder=encoder,
                torch_dtype=torch.float16, use_safetensors=True, local_files_only=True)
            self.pipe.load_ip_adapter(str(settings.ip_root), subfolder=settings.ip_subfolder,
                                      weight_name=settings.ip_weight_name, image_encoder_folder=None,
                                      local_files_only=True)
            self.pipe.set_ip_adapter_scale(0.)
            self.pipe.model_cpu_offload_seq = OFFLOAD
            self.pipe.enable_model_cpu_offload()
        except BaseException:
            self.close()
            raise

    def generate(self, inputs, images, seed, observation, cancelled):
        torch, pipe, settings = self.torch, self.pipe, self.settings
        contour = load_sketch_image(seed, self.sketch_maps)
        pipe.scheduler = self.scheduler_type.from_config(scheduler_config())
        pipe.set_ip_adapter_scale(inputs.ip_early)
        handle = pipe.unet.register_forward_hook(observation, with_kwargs=True)
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            embeds = encode_prompt_plan(inputs.prompt, (pipe.text_encoder, pipe.text_encoder_2),
                                         pipe._execution_device, pipe.text_encoder_2.dtype)
            result = call_with_contour(pipe, embeds, images, contour, settings,
                torch.Generator(device="cuda").manual_seed(seed),
                pipeline_callback_kwargs(pipe, settings, cancelled, self.callback_mode), self.sketch_strength)
            torch.cuda.synchronize()
            require(len(result.images) == 1, "Expected exactly one output")
            return result.images[0], {
                "generation_seconds": time.perf_counter() - started,
                "timing_scope": "prompt_encoding_and_pipeline",
                "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
                "versions": self.versions, "scheduler_actual_config": dict(pipe.scheduler.config),
            }
        finally:
            handle.remove()
            contour.close()


class StepGuard:
    """제품 호출을 먼저 관찰하며 강도 일정이나 메모리 표본이 처음 어긋날 때 멈춘다."""

    def __init__(self, observation, settings, ip_early, cuda):
        self.observation, self.settings = observation, settings
        self.ip_early, self.cuda = ip_early, cuda

    def __call__(self, module, args, kwargs, output):
        self.observation(module, args, kwargs, output)
        index = len(self.observation.calls) - 1
        call = self.observation.calls[-1]
        expected_ip = self.ip_early if index < self.settings.ip_start else self.settings.ip_scale
        if (index >= self.settings.steps or call["ip_scales"] != [expected_ip]
                or call["adapter_applied"] != (index < self.settings.adapter_steps)):
            raise OnePassGenerationError(f"Trial schedule mismatch at UNet {index}")
        check_memory(self.cuda, self.settings)


def memory_record(cuda):
    return {"max_memory_allocated_bytes": cuda.max_memory_allocated(),
            "max_memory_reserved_bytes": cuda.max_memory_reserved(),
            "max_allocated_gib": cuda.max_memory_allocated() / 2**30,
            "max_reserved_gib": cuda.max_memory_reserved() / 2**30}


def check_memory(cuda, settings):
    if cuda.max_memory_reserved() / 2**30 > settings.max_reserved_gib:
        raise OnePassGenerationError("reserved exceeded fixed 6.5 GiB trial limit")


class MonitoredBackend:
    """두 실행기의 입력을 바꾸지 않고 읽기 전용 검사를 붙인다."""

    def __init__(self, backend, settings, contour):
        self.backend, self.settings, self.contour = backend, settings, contour
        self.pipe, self.callback_mode = backend.pipe, backend.callback_mode
        self.forward_calls = [0, 0] if contour else [0]
        self.metrics = {}
        self.handles = []
        adapters = list(self.pipe.adapter.adapters) if contour else [self.pipe.adapter]
        for index, adapter in enumerate(adapters):
            def count(_module, _args, _output, index=index):
                self.forward_calls[index] += 1
            self.handles.append(adapter.register_forward_hook(count))

    def generate(self, inputs, images, seed, observation, cancelled):
        cuda = self.backend.torch.cuda
        self.forward_calls[:] = [0] * len(self.forward_calls)
        try:
            guard = StepGuard(observation, self.settings, inputs.ip_early, cuda)
            result, timing = self.backend.generate(inputs, images, seed, guard, cancelled)
            # 제품은 최종 메모리 검사 전에 완성 원본을 저장한다.
            # VAE 완료 후 여기서 예외를 던져 완성 이미지를 잃지 않게 한다.
            return result, timing
        finally:
            self.metrics = memory_record(cuda)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.backend.close()


def verify_runtime_models(report, contour):
    selected = report["models"] if contour else report["models"][:1]
    for model in selected:
        for item in model["files"].values():
            require(sha(item["path"]) == item["sha256"], "Model file changed before loading")


def create_backend(settings, contour, report):
    """모델을 로드하는 유일한 분기 직전에 스케치 바이트를 다시 검증한다."""
    verify_runtime_models(report, contour)
    import torch
    torch.cuda.reset_peak_memory_stats()
    if contour:
        sketch_root, _ = verify_download(DESIGN / "download.json")
        require(str(sketch_root) == report["sketch_root"], "Sketch path changed")
        backend = HeadContourBackend(settings, sketch_root, options_for(report)["sketch_strength"],
                                     **({"sketch_maps": report["sketch_maps"]} if report.get("sketch_maps") else {}))
    else:
        backend = DiffusersOnePassBackend(settings)
    try:
        check_memory(torch.cuda, settings)
        monitored = MonitoredBackend(backend, settings, contour)
        monitored.loading_memory = memory_record(torch.cuda)
        return monitored
    except BaseException:
        backend.close()
        raise


def enrich_record(path, report, contour, backend=None, error=None, seed=None):
    """원본과 제품 검증을 유지하면서 실제 사용 모델 두 개를 명시한다."""
    record = read(path) if path.exists() else {"completed": False, "valid": False}
    selected = report["models"] if contour else report["models"][:1]
    record.setdefault("models", {})["adapter"] = selected if contour else selected[0]
    record["trial"] = {
        "name": "head_contour_4px", "contour_enabled": contour,
        "adapters": selected, "strengths": trial_strengths(options_for(report)["sketch_strength"]) if contour else [1.2],
        "options": options_for(report), "seeds": options_for(report)["seeds"],
        "reused_baselines": report.get("reused_baselines", []),
        "control_image_sha256": report["control_shas"] if contour else report["control_shas"][:1],
        "face_sha256": report["face_sha256"], "expected_applied_steps": list(range(11)),
        "actual_shared_residual_steps": record.get("adapter_applied_calls", []),
        "individual_adapter_forward_calls": backend.forward_calls if backend else [],
        "observation_note": "UNet receives a weighted sum, not individually distinguishable adapter residuals",
        "spatial_input": "locked contour; no coordinate transformation",
    }
    if report.get("sketch_maps") and contour:
        selected_map = report["sketch_maps"]["items"][str(seed)]
        record["trial"]["sketch_input"] = selected_map
        record["trial"]["sketch_mode"] = "seed_specific"
        record["trial"]["control_image_sha256"] = [report["control_shas"][0], selected_map["sha256"]]
        record["trial"]["spatial_input"] = "locked seed-specific sketch; no coordinate transformation"
    if backend is not None:
        record.update(backend.metrics)
        record["trial"]["loading_memory"] = getattr(backend, "loading_memory", None)
    if error is not None:
        record.update(valid=False, error_type=type(error).__name__, error=str(error))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_trial_record(path, record)


def run_case(case, destination, inputs, settings, report, backend_factory=create_backend):
    """한 번만 시도한다. 생성 전 모델 로드에 실패해도 실패 기록을 남긴다."""
    folder = destination / case["id"]
    require(not folder.exists(), "Refusing to overwrite case " + case["id"])
    backend = None
    try:
        if report.get("sketch_maps"):
            require(sha(report["sketch_maps"]["manifest_path"]) == report["sketch_maps"]["manifest_sha256"],
                    "Sketch manifest changed before loading")
            with load_sketch_image(case["seed"], report["sketch_maps"]):
                pass
        backend = backend_factory(settings, case["contour"], report)
        candidate = generate_onepass_image(backend, inputs, case["seed"], folder, settings=settings,
                                           expected_sha256=case["expected_sha256"])
        require(backend.forward_calls == ([1, 1] if case["contour"] else [1]),
                "Adapter forward count mismatch")
        enrich_record(folder / "run.json", report, case["contour"], backend, seed=case["seed"])
        return candidate
    except BaseException as error:
        enrich_record(folder / "run.json", report, case["contour"], backend, error, seed=case["seed"])
        raise
    finally:
        if backend is not None:
            try:
                backend.close()
            except BaseException as error:
                enrich_record(folder / "run.json", report, case["contour"], backend, error, seed=case["seed"])
                raise
        if (folder / "run.json").exists():
            record = read(folder / "run.json")
            record.setdefault("seed", case["seed"])
            record["trial"]["case"] = case
            write_trial_record(folder / "run.json", record)


def validate_new_seeds(seeds, contour_only):
    """명시적 seed 선택은 이 시험에만 적용하며 중복을 자동 제거하지 않는다."""
    if seeds is None:
        return
    require(bool(seeds) and all(type(seed) is int and seed in NEW_SEEDS for seed in seeds),
            "Explicit seeds must come from the four approved new seeds")
    require(len(seeds) == len(set(seeds)), "Duplicate seeds are not allowed")
    require(not contour_only, "New seeds require BASE generation; contour-only is not allowed")


def cases(contour_only=False, seeds=None, seed_maps=False):
    selected = list(MAP_SEEDS if seed_maps else BASE_SHAS) if seeds is None else list(seeds)
    if seed_maps:
        validate_map_options(selected, contour_only, .5)
    elif selected != list(BASE_SHAS):
        validate_new_seeds(selected, contour_only)
    return [{"id": f"{mode}_{seed}", "seed": seed, "contour": mode == "CONTOUR",
             "expected_sha256": BASE_SHAS.get(seed) if mode == "BASE" else None,
             "sha_comparison": ("reference_required" if seed in BASE_SHAS else "no_reference_available")
                               if mode == "BASE" else "not_applicable_contour"}
            for seed in selected for mode in ("BASE", "CONTOUR")
            if not contour_only or mode == "CONTOUR"]


def run_trial(destination, inputs, settings, report, backend_factory=create_backend):
    """선택 사례를 순서대로 실행한다. 이어 실행·재시도·덮어쓰기는 없다."""
    options = options_for(report)
    strengths = trial_strengths(options["sketch_strength"])
    planned = cases(options["contour_only"], options["seeds"], bool(options["sketch_map_dir"]))
    if options["output"] is not None:
        require(Path(options["output"]) == destination, "Output differs from locked options")
    if options["sketch_map_dir"]:
        require(canonical(verify_seed_maps(report)) == canonical(report.get("sketch_maps")),
                "Seed sketches changed since preflight")
    elif options["contour_only"]:
        require(canonical(verify_reused_baselines(report)) == canonical(report.get("reused_baselines")),
                "Reused BASE changed since preflight")
    destination.mkdir(parents=True, exist_ok=False)
    state = {"status": "started", "planned": planned, "completed": [],
             "strengths": strengths, "options": options, "seeds": options["seeds"],
             "sketch_mode": options["sketch_mode"], "sketch_maps": report.get("sketch_maps"),
             "reused_baselines": report.get("reused_baselines", []),
             "automatic_retry": False, "started_unix": time.time()}
    try:
        write_trial_record(destination / "preflight.json", report)
        for case in planned:
            state.update(status="running", current=case["id"])
            write_trial_record(destination / "status.json", state)
            print("START", case["id"], flush=True)
            run_case(case, destination, inputs, settings, report, backend_factory)
            state["completed"].append(case["id"])
            write_trial_record(destination / "status.json", state)
            print("DONE", case["id"], flush=True)
        state.update(status="completed_pending_visual_review", current=None)
    except BaseException as error:
        state.update(status="stopped", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        state["ended_unix"] = time.time()
        write_trial_record(destination / "status.json", state)


def trial_strengths(sketch_strength):
    require(sketch_strength in (.9, .5), "Sketch strength must be 0.9 or 0.5")
    return [STRENGTHS[0], sketch_strength]


def options_for(report):
    return {"sketch_strength": .9, "contour_only": False, "output": None,
            "seeds": list(BASE_SHAS), "sketch_mode": "head_only", "sketch_map_dir": None,
            **report.get("options", {})}


def configure_trial(report, sketch_strength, contour_only, output, seeds=None, sketch_map_dir=None):
    """선택 옵션과 검증한 제어 이미지를 CPU 잠금에 연결한다."""
    if sketch_map_dir is not None:
        seeds = list(MAP_SEEDS) if seeds is None else list(seeds)
        validate_map_options(seeds, contour_only, sketch_strength)
    else:
        validate_new_seeds(seeds, contour_only)
    configured = dict(report, strengths=trial_strengths(sketch_strength),
                      options={"sketch_strength": sketch_strength, "contour_only": contour_only,
                               "output": str(output), "seeds": list(BASE_SHAS) if seeds is None else list(seeds),
                               "sketch_mode": "seed_specific" if sketch_map_dir is not None else "head_only",
                               "sketch_map_dir": str(sketch_map_dir) if sketch_map_dir is not None else None})
    if sketch_map_dir is not None:
        configured["sketch_maps"] = verify_seed_maps(configured)
        configured["reused_baselines"] = [item["base"] for item in configured["sketch_maps"]["items"].values()]
    else:
        configured["reused_baselines"] = verify_reused_baselines(configured) if contour_only else []
    return configured


def verify_reused_baselines(report):
    """잠긴 입력이 같고 원본 해시가 유효한 gpu-02 BASE 이미지만 재사용한다."""
    source = DESIGN / "gpu-02"
    previous = read(source / "preflight.json")
    for key, value in report.items():
        if key not in ("protected", "strengths", "options", "reused_baselines", "sketch_maps"):
            require(canonical(previous.get(key)) == canonical(value), "Reused BASE input changed: " + key)
    # 이 옵션으로 시험 실행기 자체는 바뀌지만 나머지 보호 대상 코드는 고정한다.
    runner_path = str(Path(__file__))
    old_code = {k: v for k, v in previous["protected"].items() if k != runner_path}
    new_code = {k: v for k, v in report["protected"].items() if k != runner_path}
    require(old_code == new_code, "Reused BASE protected code changed")
    status = read(source / "status.json")
    evidence = []
    for seed, expected in BASE_SHAS.items():
        name = f"BASE_{seed}"
        folder = source / name
        record = read(folder / "run.json")
        require(name in status["completed"] and record["completed"] and record["valid"],
                "Reused BASE is not complete and valid: " + name)
        require(record["seed"] == seed and record["raw_sha256"] == sha(folder / "raw.png") == expected,
                "Reused BASE raw SHA mismatch: " + name)
        require(canonical(record["settings"]) == canonical(report["settings"]), "Reused BASE settings changed")
        require(canonical(record["prompt"]) == canonical(report["base_inputs"]["prompt"]),
                "Reused BASE prompt changed")
        require(not record["validation_errors"] and record["max_reserved_gib"] <= report["settings"]["max_reserved_gib"],
                "Reused BASE validation failed")
        require(record["adapter_applied_calls"] == list(range(11)) and record["trial"]["strengths"] == [1.2],
                "Reused BASE adapter condition changed")
        evidence.append({"seed": seed, "raw_file": folder / "raw.png", "raw_sha256": expected,
                         "run_file": folder / "run.json", "run_sha256": sha(folder / "run.json")})
    return {"source": source, "preflight_sha256": sha(source / "preflight.json"),
            "status_sha256": sha(source / "status.json"), "cases": evidence}


def validate_map_options(seeds, contour_only, strength):
    require(contour_only and strength == .5, "Seed sketches require contour-only and sketch strength 0.5")
    require(bool(seeds) and all(type(seed) is int and seed in MAP_SEEDS for seed in seeds),
            "Seed sketches support only the six approved seeds")
    require(len(seeds) == len(set(seeds)), "Duplicate seeds are not allowed")


def load_sketch_image(seed, sketch_maps=None):
    """사용 전에 선택한 바이트를 검증한다. 크기 변경·재작성·대체 입력은 없다."""
    from PIL import Image
    if sketch_maps is None:
        path = DESIGN / "head_draft/contour_4px.png"
        require(sha(path) == CONTOUR_SHA, "Contour changed before generation")
        with Image.open(path) as im:
            return im.copy()
    import numpy as np
    item = sketch_maps["items"][str(seed)]
    path = Path(item["path"])
    require(sha(path) == item["sha256"], "Seed sketch SHA mismatch: " + str(seed))
    with Image.open(path) as im:
        require(im.format == "PNG" and im.size == (736, 1232) and im.mode in ("L", "RGB"),
                "Seed sketch must be a 736x1232 L/RGB PNG")
        pixels = np.asarray(im)
        require(bool(np.all((pixels == 0) | (pixels == 255))), "Seed sketch contains non-binary pixels")
        if im.mode == "RGB":
            require(bool(np.all(pixels == pixels[:, :, :1])), "Seed sketch is not black and white")
        return im.copy()


def verify_map_base(report, seed, entry):
    """몸 외곽을 제공한 기존 BASE를 검사한다. 재생성하지 않는다."""
    source = DESIGN / ("gpu-02" if seed in BASE_SHAS else "gpu-04")
    folder = source / f"BASE_{seed}"
    raw = folder / "raw.png"
    require(Path(entry["base_raw"]).resolve() == raw.resolve(), "Unexpected sketch BASE source")
    expected = BASE_SHAS.get(seed, entry["base_raw_sha256"])
    require(sha(raw) == entry["base_raw_sha256"] == expected, "Sketch BASE SHA mismatch")
    previous = read(source / "preflight.json")
    for key in ("settings", "base_inputs", "models", "versions", "control_shas", "face_sha256", "applied_steps"):
        require(canonical(previous[key]) == canonical(report[key]), "Sketch BASE inputs changed: " + key)
    runner_path = str(Path(__file__))
    require({k: v for k, v in previous["protected"].items() if k != runner_path}
            == {k: v for k, v in report["protected"].items() if k != runner_path},
            "Sketch BASE protected code changed")
    record = read(folder / "run.json")
    status = read(source / "status.json")
    require(folder.name in status["completed"] and record["completed"] and record["valid"]
            and record["seed"] == seed and record["raw_sha256"] == expected,
            "Sketch BASE is not complete and valid")
    require(not record["validation_errors"] and record["max_reserved_gib"] <= report["settings"]["max_reserved_gib"]
            and record["adapter_applied_calls"] == list(range(11)) and record["trial"]["strengths"] == [1.2],
            "Sketch BASE validation failed")
    require(canonical(record["settings"]) == canonical(report["settings"])
            and canonical(record["prompt"]) == canonical(report["base_inputs"]["prompt"]),
            "Sketch BASE request mismatch")
    return {"seed": seed, "raw_file": raw, "raw_sha256": expected,
            "run_file": folder / "run.json", "run_sha256": sha(folder / "run.json"),
            "preflight_sha256": sha(source / "preflight.json"), "status_sha256": sha(source / "status.json")}


def verify_seed_maps(report):
    options = options_for(report)
    validate_map_options(options["seeds"], options["contour_only"], options["sketch_strength"])
    directory = Path(options["sketch_map_dir"])
    manifest_path = directory.parent / "maps.json"
    manifest = read(manifest_path)
    result = {"mode": "seed_specific", "manifest_path": manifest_path,
              "manifest_sha256": sha(manifest_path), "items": {}}
    for seed in options["seeds"]:
        entry = manifest[str(seed)]
        result["items"][str(seed)] = {"seed": seed, "mode": "seed_specific",
            "path": directory / f"sketch_{seed}.png", "sha256": entry["sketch_sha256"]}
        with load_sketch_image(seed, result):
            pass
        result["items"][str(seed)]["base"] = verify_map_base(report, seed, entry)
    return result


def preflight_record_path(report):
    """GPU 폴더를 만들지 않고 각 출력 폴더 옆에 CPU 잠금을 저장한다."""
    output = options_for(report)["output"]
    if output is None:
        return DESIGN / "runner_preflight.json"
    output = Path(output)
    return output.with_name(output.name + "_preflight.json")


def save_preflight(report):
    path = preflight_record_path(report)
    write_trial_record(path, report)
    path.with_suffix(".sha256").write_text(sha(path) + chr(10), encoding="utf-8")
    return path


def require_saved_preflight(report):
    path = preflight_record_path(report)
    require(sha(path) == path.with_suffix(".sha256").read_text().strip(), "CPU report lock changed")
    require(canonical(report) == canonical(read(path)), "CPU inputs/code/models changed; rerun preflight")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run"), nargs="?", default="preflight")
    parser.add_argument("--output", type=Path, default=DESIGN / "gpu-01")
    parser.add_argument("--sketch-strength", type=float, choices=(.9, .5), default=.9)
    parser.add_argument("--contour-only", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", choices=MAP_SEEDS, default=None)
    parser.add_argument("--sketch-map-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.sketch_map_dir is not None:
            validate_map_options(args.seeds if args.seeds is not None else MAP_SEEDS,
                                 args.contour_only, args.sketch_strength)
        else:
            validate_new_seeds(args.seeds, args.contour_only)
    except ValueError as error:
        parser.error(str(error))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    if args.action == "preflight":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    inputs, settings, report = preflight()
    report = configure_trial(report, args.sketch_strength, args.contour_only, args.output, args.seeds, args.sketch_map_dir)
    if args.action == "preflight":
        print("CPU PASS; GPU NOT RUN:", save_preflight(report), flush=True)
    else:
        require_saved_preflight(report)
        run_trial(args.output, inputs, settings, report)


if __name__ == "__main__":
    main()
