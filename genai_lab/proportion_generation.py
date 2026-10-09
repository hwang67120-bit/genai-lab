"""Opt-in two-pass generation. Read generate_proportion_batch for the complete flow."""
from dataclasses import asdict, dataclass
from pathlib import Path
import time
import numpy as np
from PIL import Image

from genai_lab.onepass_generation import (
    OnePassCandidate, OnePassCancelled, OnePassGenerationError,
    DiffusersOnePassBackend, generate_onepass_image, validate_seed)
from genai_lab.onepass_generation_settings import OnePassGenerationSettings
from genai_lab.proportion_inputs import (
    ProportionOptions, validate_proportion_request, validate_models, make_sketch, white_background,
    sha, require, json_value, checked_image)
from genai_lab.proportion_foreground import AnimeForeground
from genai_lab.proportion_backend import ProportionBackend, GuardedBase
from genai_lab.qwen_record_io import write_json
from genai_lab.head_lines import compose_head_lines


@dataclass(frozen=True)
class ProportionCandidate:
    seed: int
    raw: OnePassCandidate
    path: Path
    record_path: Path
    record: dict


@dataclass(frozen=True)
class ProportionBatch:
    directory: Path
    candidates: tuple[ProportionCandidate, ...]
    review_stage: str = "proportion_unreviewed"
    final_return_eligible: bool = False


def check_cancel(cancelled):
    if cancelled():
        raise OnePassCancelled("비율 생성을 취소했습니다. 완료된 중간 이미지는 보존합니다.")


def save_state(directory, state):
    write_json(directory / "status.json", json_value(state))


def generate_stage(inputs, seeds, directory, settings, factory, cancelled, state, stage, expected):
    """Own one GPU pipeline; always release it before the next processing phase."""
    candidates, backend = [], None
    try:
        check_cancel(cancelled)
        backend = factory()
        for seed in seeds:
            check_cancel(cancelled)
            state["current_seed"] = seed
            save_state(directory, state)
            candidate = generate_onepass_image(backend, inputs, seed, directory / f"{stage}_{seed}",
                                               settings=settings, cancelled=cancelled,
                                               expected_sha256=expected.get(seed))
            if stage == "CONTOUR" and candidate.record.get("adapter_forward_counts") != [1, 1]:
                candidate.record.update(valid=False, validation_errors=["두 어댑터 실제 호출 수 불일치"])
                write_json(candidate.record_path, candidate.record)
                raise OnePassGenerationError("두 어댑터가 각각 1회 실행되지 않았습니다.")
            candidates.append(candidate)
            state[stage].append(seed)
            save_state(directory, state)
        return candidates
    finally:
        if backend is not None:
            backend.close()


def build_maps(bases, head_mask, head_contour, directory, foreground_factory, options, cancelled, expected):
    """Derive each sketch from its own first-pass raw; never borrow another seed's body."""
    maps, foreground = {}, None
    try:
        foreground = foreground_factory(options)
        for base in bases:
            check_cancel(cancelled)
            require(sha(base.path) == base.record["raw_sha256"], "1단계 raw가 변경됐습니다.")
            with Image.open(base.path) as image:
                alpha = foreground.alpha(image)
            check_cancel(cancelled)
            sketch, head_lines = compose_head_lines(make_sketch(alpha, head_mask, head_contour),options)
            path = directory / f"sketch_{base.seed}.png"
            Image.fromarray(alpha).save(directory / f"base_alpha_{base.seed}.png")
            Image.fromarray(sketch).save(path)
            digest = sha(path)
            require(base.seed not in expected or digest == expected[base.seed], "재현 스케치 SHA 불일치")
            maps[base.seed] = {"path": str(path), "sha256": digest,
                               "base_raw": str(base.path), "base_raw_sha256": base.record["raw_sha256"],
                               "mode": "body_outline_plus_confirmed_head_4px", "outline_source": "base",
                               "original_sha256": options.head.normalized_sha256}
            if head_lines is not None:
                maps[base.seed]["head_lines"] = head_lines
            write_json(directory / "maps.json", {str(key): value for key, value in maps.items()})
        return maps
    finally:
        if foreground is not None:
            foreground.close()



def build_original_maps(seeds, head_mask, head_contour, directory, foreground_factory, options, cancelled, expected):
    """One normalized source, one CPU inference, one shared map; no generated body."""
    check_cancel(cancelled)
    head = options.head
    pixels = checked_image(head.normalized_file, head.normalized_sha256,
                           (head_mask.shape[1], head_mask.shape[0]), "RGB")
    foreground = None
    try:
        foreground = foreground_factory(options)
        with Image.fromarray(pixels) as image:
            alpha = foreground.alpha(image)
        check_cancel(cancelled)
        sketch = make_sketch(alpha, head_mask, head_contour)
        path = directory / "sketch_original.png"
        Image.fromarray(alpha).save(directory / "original_alpha.png")
        Image.fromarray(sketch).save(path)
        digest = sha(path)
        for seed in seeds:
            require(seed not in expected or digest == expected[seed], "원본 스케치 SHA 불일치")
        entry = {"path": str(path), "sha256": digest, "outline_source": "original",
                 "original_file": str(head.normalized_file), "original_sha256": head.normalized_sha256,
                 "mode": "body_outline_plus_confirmed_head_4px"}
        maps = {seed: dict(entry) for seed in seeds}
        write_json(directory / "maps.json", {str(key): value for key, value in maps.items()})
        return maps
    finally:
        if foreground is not None:
            foreground.close()


def finish_candidates(raws, directory, foreground_factory, options, cancelled, state, on_image):
    """Keep raw untouched; publish white-background products only after all checks pass."""
    products, foreground = [], None
    try:
        foreground = foreground_factory(options)
        for raw in raws:
            check_cancel(cancelled)
            require(sha(raw.path) == raw.record["raw_sha256"], "2단계 raw가 변경됐습니다.")
            with Image.open(raw.path) as image:
                alpha = foreground.alpha(image)
                product = white_background(np.asarray(image), alpha)
            check_cancel(cancelled)
            folder = directory / f"seed-{raw.seed}"
            folder.mkdir()
            path = folder / "product.png"
            Image.fromarray(alpha).save(folder / "alpha.png")
            Image.fromarray(product).save(path)
            record = {**raw.record, "proportion_mode": "two_pass", "raw_file": str(raw.path),
                      "product_sha256": sha(path), "product_file": str(path),
                      "alpha_sha256": sha(folder / "alpha.png"), "review_stage": "proportion_unreviewed",
                      "final_return_eligible": False}
            record_path = folder / "run.json"
            write_json(record_path, record)
            candidate = ProportionCandidate(raw.seed, raw, path, record_path, record)
            products.append(candidate)
            state["products"].append(raw.seed)
            save_state(directory, state)
            on_image(candidate)
        return products
    finally:
        if foreground is not None:
            foreground.close()


def generate_proportion_batch(inputs, seeds, directory, *, settings=OnePassGenerationSettings(),
                              options, cancelled=lambda: False, on_image=lambda _image: None,
                              base_factory=DiffusersOnePassBackend, contour_factory=ProportionBackend,
                              foreground_factory=AnimeForeground,
                              expected_base=None, expected_sketch=None, expected_raw=None):
    """Existing generation -> CPU sketch -> two adapters -> CPU white background, no retries.

    Expected hashes are optional reproduction assertions, never alternate inputs.
    Six-seed replay and the four-seed product entry point use this same function.
    """
    seeds = tuple(seeds)
    require(seeds and len(set(seeds)) == len(seeds), "중복 없는 seed가 필요합니다.")
    for seed in seeds:
        validate_seed(seed)
    expected_base, expected_sketch, expected_raw = expected_base or {}, expected_sketch or {}, expected_raw or {}
    for expected in (expected_base, expected_sketch, expected_raw):
        require(set(expected) <= set(seeds), "재현 SHA에 요청 밖 seed가 있습니다.")
    require(options.outline_source == "base" or not expected_base, "원본 윤곽 모드에서는 1단계 SHA를 검사할 수 없습니다.")
    check_cancel(cancelled)
    # Validation before output creation/loading: rejected inputs cannot consume GPU work.
    mask, contour, models = validate_proportion_request(inputs, settings, options)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    lock = json_value({"inputs": asdict(inputs), "settings": asdict(settings), "options": asdict(options),
                       "seeds": seeds, "models": models, "adapter_strengths": [1.2, .5],
                       "shared_adapter_steps": list(range(11)), "expected_base": expected_base,
                       "expected_sketch": expected_sketch, "expected_raw": expected_raw,
                       "outline_source": options.outline_source, "original_sha256": options.head.normalized_sha256,
                       "base_stage": "generated" if options.outline_source == "base" else "skipped_unused_outline"})
    lock["code_sha256"] = {name: sha(Path(__file__).parent / name) for name in (
        "proportion_generation.py", "proportion_inputs.py", "proportion_foreground.py", "proportion_backend.py",
        "onepass_generation.py", "onepass_generation_settings.py")}
    write_json(directory / "preflight.json", lock)
    (directory / "preflight.sha256").write_text(sha(directory / "preflight.json") + "\n")
    state = {"status": "started", "phase": "BASE" if options.outline_source == "base" else "sketch", "BASE": [], "CONTOUR": [], "products": [],
             "seeds": list(seeds), "adapter_strengths": [1.2, .5], "proportion_mode": "two_pass",
             "outline_source": options.outline_source, "base_stage": lock["base_stage"]}
    started = time.monotonic()
    save_state(directory, state)
    try:
        def first_backend():
            require(validate_models(settings, options) == models, "로딩 전 모델 파일 변경")
            return GuardedBase(base_factory(settings), settings)
        if options.outline_source == "base":
            bases = generate_stage(inputs, seeds, directory, settings, first_backend, cancelled, state, "BASE", expected_base)
            state["phase"] = "sketch"
            save_state(directory, state)
            maps = build_maps(bases, mask, contour, directory, foreground_factory, options, cancelled, expected_sketch)
        else:
            state["phase"] = "sketch"
            save_state(directory, state)
            maps = build_original_maps(seeds, mask, contour, directory, foreground_factory, options, cancelled, expected_sketch)
        state["sketch_sha256"] = {str(seed): entry["sha256"] for seed, entry in maps.items()}
        state["phase"] = "CONTOUR"
        save_state(directory, state)
        def second_backend():
            require(validate_models(settings, options) == models, "로딩 전 모델 파일 변경")
            return contour_factory(settings, options, maps)
        raws = generate_stage(inputs, seeds, directory, settings, second_backend, cancelled, state, "CONTOUR", expected_raw)
        state["phase"] = "white_background"
        save_state(directory, state)
        products = finish_candidates(raws, directory, foreground_factory, options, cancelled, state, on_image)
        check_cancel(cancelled)
        state.update(status="review_pending", phase="done")
        return ProportionBatch(directory, tuple(products))
    except BaseException as error:
        state.update(status="cancelled" if isinstance(error, OnePassCancelled) else "failed",
                     error_type=type(error).__name__, error=str(error))
        raise
    finally:
        state["elapsed_seconds"] = time.monotonic() - started
        save_state(directory, state)
        write_json(directory / "run.json", {**lock, **state, "final_return_eligible": False})
