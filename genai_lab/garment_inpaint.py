"""승인된 Human-Agnostic 이미지와 의상 참조를 2D Inpaint로 연결한다."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
import subprocess
from threading import Event, Thread
from time import perf_counter
from typing import Callable

import numpy as np
from PIL import Image
from genai_lab.inpaint_quality import NeutralResidualDiagnostic, inspect_neutral_residual

from genai_lab.garment_reference_board import (
    GarmentReferenceBoard,
    GarmentReferenceBoardError,
    GarmentReferenceBoardSettings,
    create_garment_reference_board,
    validate_garment_reference_board_settings,
)

GARMENT_INPAINT_PROGRESS_PHASES = frozenset({
    "runner_started",
    "pipeline_loading",
    "ip_adapter_loading",
    "diffusion_running",
    "output_saving",
    "completed",
})
PROGRESS_POLL_INTERVAL_SECONDS = 1.0
PROGRESS_HEARTBEAT_SECONDS = 5.0


@dataclass(frozen=True)
class GarmentInpaintSettings:
    python_executable: Path
    runner_path: Path
    temporary_root: Path
    cache_dir: Path
    benchmark_root: Path | None = None
    base_model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    model_variant: str = "fp16"
    adapter_repository: str = "h94/IP-Adapter"
    adapter_subfolder: str = "sdxl_models"
    adapter_weight: str = "ip-adapter-plus_sdxl_vit-h.safetensors"
    adapter_image_encoder_subfolder: str = "models/image_encoder"
    strength: float = 0.90
    inference_steps: int = 28
    guidance_scale: float = 5.5
    ip_adapter_scale: float = 0.80
    padding_mask_crop: int = 64
    mask_threshold: int = 128
    garment_board_size: int = 1024
    garment_board_outer_padding: int = 32
    garment_board_cell_padding: int = 16
    garment_board_minimum_component_pixels: int = 16
    garment_board_maximum_components: int = 8
    timeout_seconds: int = 1800
    dtype: str = "float16"
    step5_vram_before_allocated_mib: float | None = None
    step5_vram_after_allocated_mib: float | None = None
    step5_vram_after_reserved_mib: float | None = None
    neutral_rgb: tuple[int, int, int] = (127, 127, 127)
    neutral_residual_tolerance: int = 8


@dataclass(frozen=True)
class GarmentInpaintProgress:
    phase: str
    phase_elapsed_seconds: float
    total_elapsed_seconds: float
    current_step: int | None = None
    configured_steps: int | None = None
    message: str = ""


@dataclass(frozen=True)
class GarmentInpaintExecutionMetrics:
    pipeline_load_seconds: float
    ip_adapter_load_seconds: float
    diffusion_seconds: float
    output_save_seconds: float
    runner_total_seconds: float
    parent_total_seconds: float
    progress_event_count: int
    invalid_progress_event_count: int
    heartbeat_count: int


@dataclass(frozen=True)
class GarmentInpaintReviewCandidate:
    base_character_preview: Image.Image
    human_agnostic_preview: Image.Image
    approved_mask_preview: Image.Image
    garment_reference_preview: Image.Image
    garment_reference_board_preview: Image.Image
    raw_inpaint_output: Image.Image
    protected_output: Image.Image
    difference_preview: Image.Image
    canvas_size: tuple[int, int]
    inpaint_mask_pixels: int
    inpaint_soft_mask_pixels: int
    raw_changed_inside_mask_pixels: int
    raw_changed_outside_mask_pixels: int
    raw_changed_from_initial_inside_mask_pixels: int
    protected_changed_inside_mask_pixels: int
    protected_changed_outside_mask_pixels: int
    mean_rgb_l1_inside_mask: float
    garment_source_component_count: int
    garment_retained_component_count: int
    garment_board_occupied_pixel_count: int
    benchmark_directory: Path
    benchmark_file_count: int
    elapsed_seconds: float
    execution_metrics: GarmentInpaintExecutionMetrics
    settings: GarmentInpaintSettings
    automatic_save_count: int = 0
    neutral_residual: NeutralResidualDiagnostic | None = None

    def close(self) -> None:
        for image in (
            self.base_character_preview,
            self.human_agnostic_preview,
            self.approved_mask_preview,
            self.garment_reference_preview,
            self.garment_reference_board_preview,
            self.raw_inpaint_output,
            self.protected_output,
            self.difference_preview,
        ):
            image.close()
        if self.neutral_residual is not None:
            self.neutral_residual.close()


@dataclass(frozen=True)
class GarmentInpaintApprovedInput:
    image: Image.Image
    canvas_size: tuple[int, int]
    changed_inside_mask_pixels: int
    changed_outside_mask_pixels: int
    mean_rgb_l1_inside_mask: float

    def close(self) -> None:
        self.image.close()


class GarmentInpaintError(RuntimeError):
    pass


class GarmentGenerationEngine(ABC):
    """GUI와 구체 생성 엔진을 분리하는 의상 생성 계약."""

    @abstractmethod
    def generate_inpaint(
        self,
        base_character_image: Image.Image,
        approved_human_agnostic_image: Image.Image,
        approved_change_mask: Image.Image,
        garment_reference_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        seed: int,
        settings: GarmentInpaintSettings,
        progress_callback: Callable[[GarmentInpaintProgress], None] | None = None,
    ) -> GarmentInpaintReviewCandidate:
        """사용자 승인 전 검토 후보를 만들고 자동 저장하지 않는다."""


class SubprocessSDXLGarmentGenerationEngine(GarmentGenerationEngine):
    """모델 생명주기를 별도 프로세스 한 번으로 제한하는 SDXL 엔진."""

    def generate_inpaint(
        self,
        base_character_image: Image.Image,
        approved_human_agnostic_image: Image.Image,
        approved_change_mask: Image.Image,
        garment_reference_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        seed: int,
        settings: GarmentInpaintSettings,
        progress_callback: Callable[[GarmentInpaintProgress], None] | None = None,
    ) -> GarmentInpaintReviewCandidate:
        return _execute_garment_inpaint(
            base_character_image,
            approved_human_agnostic_image,
            approved_change_mask,
            garment_reference_image,
            prompt,
            negative_prompt,
            seed,
            settings,
            progress_callback,
        )


def execute_garment_inpaint(
    base_character_image: Image.Image,
    approved_human_agnostic_image: Image.Image,
    approved_change_mask: Image.Image,
    garment_reference_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    seed: int,
    settings: GarmentInpaintSettings,
    progress_callback: Callable[[GarmentInpaintProgress], None] | None = None,
) -> GarmentInpaintReviewCandidate:
    """기본 별도 프로세스 엔진으로 사용자 승인 전 후보를 만든다."""
    return SubprocessSDXLGarmentGenerationEngine().generate_inpaint(
        base_character_image,
        approved_human_agnostic_image,
        approved_change_mask,
        garment_reference_image,
        prompt,
        negative_prompt,
        seed,
        settings,
        progress_callback,
    )


def _execute_garment_inpaint(
    base_character_image: Image.Image,
    approved_human_agnostic_image: Image.Image,
    approved_change_mask: Image.Image,
    garment_reference_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    seed: int,
    settings: GarmentInpaintSettings,
    progress_callback: Callable[[GarmentInpaintProgress], None] | None = None,
) -> GarmentInpaintReviewCandidate:
    """별도 GPU 프로세스를 실행하고 마스크로 보호한 후보를 만든다."""
    started_at = perf_counter()
    _validate_settings(settings)
    _validate_inputs(
        base_character_image,
        approved_human_agnostic_image,
        approved_change_mask,
        prompt,
        seed,
        settings.mask_threshold,
    )
    base_rgb = _convert_to_rgb_on_white(base_character_image)
    human_agnostic_rgb = _convert_to_rgb_on_white(
        approved_human_agnostic_image
    )
    mask_l = approved_change_mask.convert("L")
    garment_rgb = _convert_to_rgb_on_white(garment_reference_image)
    try:
        garment_board = create_garment_reference_board(
            garment_reference_image,
            GarmentReferenceBoardSettings(
                board_size=settings.garment_board_size,
                outer_padding=settings.garment_board_outer_padding,
                cell_padding=settings.garment_board_cell_padding,
                alpha_threshold=settings.mask_threshold,
                minimum_component_pixels=(
                    settings.garment_board_minimum_component_pixels
                ),
                maximum_components=settings.garment_board_maximum_components,
            ),
        )
    except GarmentReferenceBoardError as error:
        for image in (base_rgb, human_agnostic_rgb, mask_l, garment_rgb):
            image.close()
        raise GarmentInpaintError(
            f"IP-Adapter 의상 참조 보드를 만들 수 없습니다: {error}"
        ) from error
    initial = human_agnostic_rgb.copy()
    raw_output = None
    review_candidate = None
    progress_events: list[GarmentInpaintProgress] = []
    progress_counters = {"invalid": 0, "heartbeat": 0}
    benchmark_directory = _create_benchmark_directory(settings, seed)
    paths = {
        "initial": benchmark_directory / "initial.png",
        "mask": benchmark_directory / "mask.png",
        "garment_source": benchmark_directory / "garment_source.png",
        "garment_board": benchmark_directory / "garment_board.png",
        "output": benchmark_directory / "raw_output_A.png",
        "protected": benchmark_directory / "protected_output.png",
        "progress": benchmark_directory / "progress.jsonl",
        "prompt_record": benchmark_directory / "prompt_execution.json",
        "stdout": benchmark_directory / "stdout.log",
        "stderr": benchmark_directory / "stderr.log",
        "metadata": benchmark_directory / "metadata.json",
    }
    for image, name in (
        (initial, "initial"), (mask_l, "mask"),
        (garment_rgb, "garment_source"),
        (garment_board.image, "garment_board"),
    ):
        image.save(paths[name])
    metadata = _create_benchmark_metadata(
        settings, paths, prompt, negative_prompt, seed, base_rgb.size,
        garment_board,
    )
    _write_benchmark_metadata(paths["metadata"], metadata)
    try:
        command = _build_runner_command(
            settings, paths, prompt, negative_prompt, seed, base_rgb.size,
        )
        monitor_stop = Event()
        monitor_thread = Thread(
            target=_monitor_progress_file,
            args=(
                paths["progress"],
                monitor_stop,
                progress_callback,
                progress_events,
                progress_counters,
                started_at,
            ),
            name="garment-inpaint-progress-monitor",
            daemon=True,
        )
        monitor_thread.start()
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=settings.timeout_seconds, check=False,
            )
        finally:
            monitor_stop.set()
            monitor_thread.join(timeout=PROGRESS_POLL_INTERVAL_SECONDS * 2.0)
        stdout_text = _decode_process_output(completed.stdout)
        stderr_text = _decode_process_output(completed.stderr)
        paths["stdout"].write_text(stdout_text, encoding="utf-8")
        paths["stderr"].write_text(stderr_text, encoding="utf-8")
        prompt_execution = _read_optional_json(paths["prompt_record"])
        if prompt_execution is not None:
            metadata["prompt_execution"] = prompt_execution
        if completed.returncode != 0:
            raise GarmentInpaintError(
                "2D 의상 Inpaint 별도 프로세스가 실패했습니다.\n"
                f"표준 출력: {stdout_text[-2000:]}\n"
                f"표준 오류: {stderr_text[-4000:]}"
            )
        if not paths["output"].is_file():
            raise GarmentInpaintError("Inpaint 결과 이미지를 만들지 않았습니다.")
        with Image.open(paths["output"]) as opened:
            raw_output = opened.convert("RGB").copy()
        if raw_output.size != base_rgb.size:
            raise GarmentInpaintError(
                f"Inpaint 결과 크기 {raw_output.size} != 기준 {base_rgb.size}"
            )
        elapsed_seconds = perf_counter() - started_at
        execution_metrics = _create_execution_metrics(
            progress_events,
            parent_total_seconds=elapsed_seconds,
            invalid_progress_event_count=progress_counters["invalid"],
            heartbeat_count=progress_counters["heartbeat"],
        )
        review_candidate = _create_review_candidate(
            base_rgb, initial, mask_l, garment_rgb, garment_board.image,
            raw_output,
            settings, elapsed_seconds, execution_metrics,
            benchmark_directory, garment_board,
        )
        review_candidate.protected_output.save(paths["protected"])
        residual = review_candidate.neutral_residual
        metadata.update({
            "neutral_residual": {
                "severity": "warning_only",
                "neutral_rgb": list(settings.neutral_rgb),
                "tolerance": settings.neutral_residual_tolerance,
                "evaluated_pixel_count": residual.evaluated_pixel_count,
                "suspected_pixel_count": residual.suspected_pixel_count,
                "suspected_percent": residual.suspected_percent,
            },
            "status": "completed",
            "finished_at": datetime.now().astimezone().isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "execution_metrics": {
                key: value
                for key, value in execution_metrics.__dict__.items()
            },
            "output_sha256": {
                "raw_output_A": _sha256_file(paths["output"]),
                "protected_output": _sha256_file(paths["protected"]),
            },
        })
        _write_benchmark_metadata(paths["metadata"], metadata)
        review_candidate = replace(
            review_candidate,
            benchmark_file_count=len(tuple(benchmark_directory.iterdir())),
        )
        return review_candidate
    except subprocess.TimeoutExpired as error:
        paths["stdout"].write_text(
            _decode_process_output(error.stdout), encoding="utf-8"
        )
        paths["stderr"].write_text(
            _decode_process_output(error.stderr), encoding="utf-8"
        )
        metadata.update({
            "status": "timeout",
            "finished_at": datetime.now().astimezone().isoformat(),
            "elapsed_seconds": perf_counter() - started_at,
            "last_progress": _progress_to_dict(
                progress_events[-1] if progress_events else None
            ),
            "error_type": type(error).__name__,
            "error_message": str(error),
        })
        _write_benchmark_metadata(paths["metadata"], metadata)
        raise GarmentInpaintError(
            f"2D 의상 Inpaint 제한 시간 {settings.timeout_seconds}초 초과, "
            f"벤치마크 보존={benchmark_directory}"
        ) from error
    except Exception as error:
        if metadata.get("status") == "running":
            metadata.update({
                "status": "failed",
                "finished_at": datetime.now().astimezone().isoformat(),
                "elapsed_seconds": perf_counter() - started_at,
                "last_progress": _progress_to_dict(
                    progress_events[-1] if progress_events else None
                ),
                "error_type": type(error).__name__,
                "error_message": str(error),
            })
            _write_benchmark_metadata(paths["metadata"], metadata)
        raise
    finally:
        for image in (
            base_rgb, human_agnostic_rgb, mask_l, garment_rgb, initial,
        ):
            image.close()
        garment_board.close()
        if raw_output is not None:
            raw_output.close()


def approve_garment_inpaint_review(
    candidate: GarmentInpaintReviewCandidate,
) -> GarmentInpaintApprovedInput:
    if candidate.protected_changed_outside_mask_pixels != 0:
        raise GarmentInpaintError("승인 마스크 밖 최종 변경이 0px가 아닙니다.")
    if candidate.protected_changed_inside_mask_pixels == 0:
        raise GarmentInpaintError("승인 마스크 안 최종 변경이 0px입니다.")
    if candidate.raw_changed_from_initial_inside_mask_pixels == 0:
        raise GarmentInpaintError(
            "Inpaint가 Human-Agnostic 시작 이미지를 변경하지 않았습니다."
        )
    if candidate.automatic_save_count != 0:
        raise GarmentInpaintError("자동 저장 수가 0개가 아닙니다.")
    return GarmentInpaintApprovedInput(
        image=candidate.protected_output.copy(),
        canvas_size=candidate.canvas_size,
        changed_inside_mask_pixels=candidate.protected_changed_inside_mask_pixels,
        changed_outside_mask_pixels=candidate.protected_changed_outside_mask_pixels,
        mean_rgb_l1_inside_mask=candidate.mean_rgb_l1_inside_mask,
    )


def _create_review_candidate(
    base_rgb: Image.Image,
    initial: Image.Image,
    mask_l: Image.Image,
    garment_rgb: Image.Image,
    garment_board_rgb: Image.Image,
    raw_output: Image.Image,
    settings: GarmentInpaintSettings,
    elapsed_seconds: float,
    execution_metrics: GarmentInpaintExecutionMetrics,
    benchmark_directory: Path,
    garment_board: GarmentReferenceBoard,
) -> GarmentInpaintReviewCandidate:
    base_array = np.asarray(base_rgb, dtype=np.uint8)
    initial_array = np.asarray(initial, dtype=np.uint8)
    raw_array = np.asarray(raw_output, dtype=np.uint8)
    mask_array = np.asarray(mask_l, dtype=np.uint8)
    allowed = mask_array > 0
    hard_allowed = mask_array >= settings.mask_threshold
    protected = Image.composite(raw_output, base_rgb, mask_l)
    protected_array = np.asarray(protected, dtype=np.uint8)
    raw_changed = np.any(raw_array != base_array, axis=2)
    raw_changed_from_initial = np.any(raw_array != initial_array, axis=2)
    protected_changed = np.any(protected_array != base_array, axis=2)
    difference = np.abs(
        protected_array.astype(np.int16) - base_array.astype(np.int16)
    )
    difference_preview = np.clip(difference * 4, 0, 255).astype(np.uint8)
    mean_l1 = float(np.sum(difference, axis=2)[allowed].mean())
    return GarmentInpaintReviewCandidate(
        base_character_preview=base_rgb.copy(),
        human_agnostic_preview=initial.copy(),
        approved_mask_preview=mask_l.copy(),
        garment_reference_preview=garment_rgb.copy(),
        garment_reference_board_preview=garment_board_rgb.copy(),
        raw_inpaint_output=raw_output.copy(),
        protected_output=protected,
        difference_preview=Image.fromarray(difference_preview),
        canvas_size=base_rgb.size,
        inpaint_mask_pixels=int(np.count_nonzero(hard_allowed)),
        inpaint_soft_mask_pixels=int(np.count_nonzero(
            (mask_array > 0) & (mask_array < settings.mask_threshold)
        )),
        raw_changed_inside_mask_pixels=int(np.count_nonzero(
            raw_changed & allowed
        )),
        raw_changed_outside_mask_pixels=int(np.count_nonzero(
            raw_changed & ~allowed
        )),
        raw_changed_from_initial_inside_mask_pixels=int(np.count_nonzero(
            raw_changed_from_initial & allowed
        )),
        protected_changed_inside_mask_pixels=int(np.count_nonzero(
            protected_changed & allowed
        )),
        protected_changed_outside_mask_pixels=int(np.count_nonzero(
            protected_changed & ~allowed
        )),
        mean_rgb_l1_inside_mask=mean_l1,
        garment_source_component_count=garment_board.source_component_count,
        garment_retained_component_count=garment_board.retained_component_count,
        garment_board_occupied_pixel_count=(
            garment_board.board_occupied_pixel_count
        ),
        benchmark_directory=benchmark_directory,
        benchmark_file_count=0,
        elapsed_seconds=elapsed_seconds,
        execution_metrics=execution_metrics,
        settings=settings,
        neutral_residual=inspect_neutral_residual(
            initial, protected, mask_l,
            settings.neutral_rgb, settings.neutral_residual_tolerance,
        ),
    )


def _validate_inputs(
    base: Image.Image,
    human_agnostic: Image.Image,
    approved_mask: Image.Image,
    prompt: str,
    seed: int,
    threshold: int,
) -> None:
    expected = base.size
    for name, actual in {
        "Human-Agnostic 이미지": human_agnostic.size,
        "승인 변경 마스크": approved_mask.size,
    }.items():
        if actual != expected:
            raise GarmentInpaintError(
                f"{name} 크기가 기준 캐릭터와 다릅니다: {actual} != {expected}"
            )
    if not prompt.strip():
        raise GarmentInpaintError("의상 Inpaint 프롬프트가 비어 있습니다.")
    if not 0 <= seed <= 2**63 - 1:
        raise GarmentInpaintError("시드는 0~2^63-1 범위여야 합니다.")
    base_image = base.convert("RGB")
    agnostic_image = human_agnostic.convert("RGB")
    mask_image = approved_mask.convert("L")
    try:
        base_array = np.asarray(base_image, dtype=np.uint8)
        agnostic_array = np.asarray(agnostic_image, dtype=np.uint8)
        mask = np.asarray(mask_image, dtype=np.uint8)
    finally:
        for image in (base_image, agnostic_image, mask_image):
            image.close()
    if int(np.count_nonzero(mask >= threshold)) == 0:
        raise GarmentInpaintError("승인 변경 마스크 픽셀이 0개입니다.")
    changed = np.any(base_array != agnostic_array, axis=2)
    outside_change = int(np.count_nonzero(changed & (mask == 0)))
    if outside_change != 0:
        raise GarmentInpaintError(
            "Human-Agnostic 이미지가 승인 마스크 밖을 변경했습니다: "
            f"{outside_change}px"
        )
    if int(np.count_nonzero(changed & (mask > 0))) == 0:
        raise GarmentInpaintError(
            "Human-Agnostic 이미지의 승인 마스크 안 변경이 0px입니다."
        )


def _validate_settings(settings: GarmentInpaintSettings) -> None:
    if (len(settings.neutral_rgb) != 3
            or any(type(v) is not int or not 0 <= v <= 255 for v in settings.neutral_rgb)
            or type(settings.neutral_residual_tolerance) is not int
            or not 0 <= settings.neutral_residual_tolerance <= 255):
        raise GarmentInpaintError("중립색 RGB와 진단 허용 오차는 0~255 정수여야 합니다.")
    if not settings.python_executable.is_file():
        raise GarmentInpaintError(
            f"Inpaint Python 실행 파일이 없습니다: {settings.python_executable}"
        )
    if not settings.runner_path.is_file():
        raise GarmentInpaintError(
            f"Inpaint 실행기 파일이 없습니다: {settings.runner_path}"
        )
    for name, value in (
        ("strength", settings.strength),
        ("ip_adapter_scale", settings.ip_adapter_scale),
    ):
        if not 0.0 <= value <= 1.0:
            raise GarmentInpaintError(f"{name}은 0.0~1.0이어야 합니다.")
    if settings.inference_steps < 1 or settings.guidance_scale <= 0:
        raise GarmentInpaintError("steps는 1 이상, guidance는 0 초과여야 합니다.")
    if settings.padding_mask_crop < 0 or settings.timeout_seconds < 1:
        raise GarmentInpaintError("padding은 0 이상, 제한은 1초 이상이어야 합니다.")
    if not 1 <= settings.mask_threshold <= 255:
        raise GarmentInpaintError("마스크 임계값은 1~255여야 합니다.")
    if settings.dtype not in {"float16", "bfloat16"}:
        raise GarmentInpaintError("dtype은 float16 또는 bfloat16이어야 합니다.")
    for name, value in (
        ("base_model_id", settings.base_model_id),
        ("model_variant", settings.model_variant),
        ("adapter_repository", settings.adapter_repository),
        ("adapter_subfolder", settings.adapter_subfolder),
        ("adapter_weight", settings.adapter_weight),
        (
            "adapter_image_encoder_subfolder",
            settings.adapter_image_encoder_subfolder,
        ),
    ):
        if not value.strip():
            raise GarmentInpaintError(f"{name}은 비어 있을 수 없습니다.")
    try:
        validate_garment_reference_board_settings(
            GarmentReferenceBoardSettings(
            board_size=settings.garment_board_size,
            outer_padding=settings.garment_board_outer_padding,
            cell_padding=settings.garment_board_cell_padding,
            alpha_threshold=settings.mask_threshold,
            minimum_component_pixels=(
                settings.garment_board_minimum_component_pixels
            ),
            maximum_components=settings.garment_board_maximum_components,
            )
        )
    except ValueError as error:
        raise GarmentInpaintError(
            "의상 참조 보드 설정값이 유효하지 않습니다."
        ) from error


def _convert_to_rgb_on_white(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    try:
        return Image.alpha_composite(white, rgba).convert("RGB")
    finally:
        white.close()
        rgba.close()


def _create_benchmark_directory(
    settings: GarmentInpaintSettings,
    seed: int,
) -> Path:
    root = (
        settings.benchmark_root
        if settings.benchmark_root is not None
        else settings.temporary_root / "debug_benchmark"
    )
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    directory = root / f"{timestamp}_seed{seed}"
    directory.mkdir(parents=False, exist_ok=False)
    return directory


def _create_benchmark_metadata(
    settings: GarmentInpaintSettings,
    paths: dict[str, Path],
    prompt: str,
    negative_prompt: str,
    seed: int,
    canvas_size: tuple[int, int],
    garment_board: GarmentReferenceBoard,
) -> dict[str, object]:
    return {
        "schema_version": 4,
        "status": "running",
        "initial_image_source": "approved_human_agnostic",
        "tps_rgb_composite_enabled": False,
        "garment_controlnet_enabled": False,
        "started_at": datetime.now().astimezone().isoformat(),
        "seed": seed,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": canvas_size[0],
        "height": canvas_size[1],
        "dtype": settings.dtype,
        "inference_steps": settings.inference_steps,
        "guidance_scale": settings.guidance_scale,
        "strength": settings.strength,
        "ip_adapter_scale": settings.ip_adapter_scale,
        "padding_mask_crop": settings.padding_mask_crop,
        "mask_threshold": settings.mask_threshold,
        "timeout_seconds": settings.timeout_seconds,
        "base_model_id": settings.base_model_id,
        "model_variant": settings.model_variant,
        "adapter_repository": settings.adapter_repository,
        "adapter_subfolder": settings.adapter_subfolder,
        "adapter_weight": settings.adapter_weight,
        "adapter_image_encoder_subfolder": (
            settings.adapter_image_encoder_subfolder
        ),
        "garment_reference_board": {
            "layout_method": garment_board.layout_method,
            "board_size": list(garment_board.image.size),
            "source_component_count": garment_board.source_component_count,
            "retained_component_count": garment_board.retained_component_count,
            "discarded_component_count": garment_board.discarded_component_count,
            "source_foreground_pixel_count": (
                garment_board.source_foreground_pixel_count
            ),
            "retained_foreground_pixel_count": (
                garment_board.retained_foreground_pixel_count
            ),
            "board_occupied_pixel_count": (
                garment_board.board_occupied_pixel_count
            ),
            "components": [
                {
                    "component_index": component.component_index,
                    "source_bbox_xywh": list(component.source_bbox_xywh),
                    "board_bbox_xywh": list(component.board_bbox_xywh),
                    "foreground_pixel_count": component.foreground_pixel_count,
                }
                for component in garment_board.components
            ],
        },
        "python_version": platform.python_version(),
        "torch_version": _package_version("torch"),
        "diffusers_version": _package_version("diffusers"),
        "accelerate_version": _package_version("accelerate"),
        "step5_memory_release_mib": {
            "before_allocated": settings.step5_vram_before_allocated_mib,
            "after_allocated": settings.step5_vram_after_allocated_mib,
            "after_reserved": settings.step5_vram_after_reserved_mib,
        },
        "input_files": {
            name: {
                "name": paths[name].name,
                "sha256": _sha256_file(paths[name]),
            }
            for name in ("initial", "mask", "garment_board")
        },
        "audit_files": {
            "garment_source": {
                "name": paths["garment_source"].name,
                "sha256": _sha256_file(paths["garment_source"]),
            },
        },
    }


def _write_benchmark_metadata(
    metadata_path: Path,
    metadata: dict[str, object],
) -> None:
    temporary_path = metadata_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(metadata_path)


def _read_optional_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _decode_process_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _progress_to_dict(
    progress: GarmentInpaintProgress | None,
) -> dict[str, object] | None:
    if progress is None:
        return None
    return {
        "phase": progress.phase,
        "phase_elapsed_seconds": progress.phase_elapsed_seconds,
        "total_elapsed_seconds": progress.total_elapsed_seconds,
        "current_step": progress.current_step,
        "configured_steps": progress.configured_steps,
        "message": progress.message,
    }


def _build_runner_command(
    settings: GarmentInpaintSettings,
    paths: dict[str, Path],
    prompt: str,
    negative_prompt: str,
    seed: int,
    canvas_size: tuple[int, int],
) -> list[str]:
    return [
        str(settings.python_executable), str(settings.runner_path),
        "--base-model-id", settings.base_model_id,
        "--model-variant", settings.model_variant,
        "--adapter-repository", settings.adapter_repository,
        "--adapter-subfolder", settings.adapter_subfolder,
        "--adapter-weight", settings.adapter_weight,
        "--adapter-image-encoder-subfolder",
        settings.adapter_image_encoder_subfolder,
        "--cache-dir", str(settings.cache_dir),
        "--initial-image", str(paths["initial"]),
        "--mask-image", str(paths["mask"]),
        "--garment-image", str(paths["garment_board"]),
        "--output-image", str(paths["output"]),
        "--prompt", prompt, "--negative-prompt", negative_prompt,
        "--prompt-record-file", str(paths["prompt_record"]),
        "--width", str(canvas_size[0]), "--height", str(canvas_size[1]),
        "--strength", str(settings.strength),
        "--inference-steps", str(settings.inference_steps),
        "--guidance-scale", str(settings.guidance_scale),
        "--ip-adapter-scale", str(settings.ip_adapter_scale),
        "--padding-mask-crop", str(settings.padding_mask_crop),
        "--seed", str(seed), "--dtype", settings.dtype,
        "--progress-file", str(paths["progress"]),
    ]


def _monitor_progress_file(
    progress_path: Path,
    stop_event: Event,
    progress_callback: Callable[[GarmentInpaintProgress], None] | None,
    progress_events: list[GarmentInpaintProgress],
    counters: dict[str, int],
    parent_started_at: float,
) -> None:
    read_offset = 0
    last_progress: GarmentInpaintProgress | None = None
    last_callback_at = perf_counter()
    while not stop_event.wait(PROGRESS_POLL_INTERVAL_SECONDS):
        events, read_offset, invalid_count = _read_progress_events(
            progress_path,
            read_offset,
        )
        counters["invalid"] += invalid_count
        for progress in events:
            progress_events.append(progress)
            last_progress = progress
            _emit_progress_callback(progress_callback, progress)
            last_callback_at = perf_counter()
        now = perf_counter()
        if now - last_callback_at >= PROGRESS_HEARTBEAT_SECONDS:
            heartbeat = GarmentInpaintProgress(
                phase=(
                    last_progress.phase
                    if last_progress is not None
                    else "runner_started"
                ),
                phase_elapsed_seconds=(
                    last_progress.phase_elapsed_seconds
                    if last_progress is not None
                    else 0.0
                ),
                total_elapsed_seconds=now - parent_started_at,
                current_step=(
                    last_progress.current_step
                    if last_progress is not None
                    else None
                ),
                configured_steps=(
                    last_progress.configured_steps
                    if last_progress is not None
                    else None
                ),
                message="heartbeat",
            )
            _emit_progress_callback(progress_callback, heartbeat)
            counters["heartbeat"] += 1
            last_callback_at = now
    events, _, invalid_count = _read_progress_events(progress_path, read_offset)
    counters["invalid"] += invalid_count
    for progress in events:
        progress_events.append(progress)
        _emit_progress_callback(progress_callback, progress)


def _read_progress_events(
    progress_path: Path,
    read_offset: int,
) -> tuple[list[GarmentInpaintProgress], int, int]:
    if not progress_path.is_file():
        return [], read_offset, 0
    events: list[GarmentInpaintProgress] = []
    invalid_count = 0
    with progress_path.open("r", encoding="utf-8", errors="replace") as stream:
        stream.seek(read_offset)
        for line in stream:
            try:
                payload = json.loads(line)
                events.append(_parse_progress_payload(payload))
            except (GarmentInpaintError, json.JSONDecodeError, TypeError, ValueError):
                invalid_count += 1
        new_offset = stream.tell()
    return events, new_offset, invalid_count


def _parse_progress_payload(payload: object) -> GarmentInpaintProgress:
    if not isinstance(payload, dict):
        raise GarmentInpaintError("Inpaint 진행 이벤트는 객체여야 합니다.")
    phase = payload.get("phase")
    if phase not in GARMENT_INPAINT_PROGRESS_PHASES:
        raise GarmentInpaintError(f"알 수 없는 Inpaint 진행 단계입니다: {phase}")
    phase_elapsed = _parse_nonnegative_float(
        payload.get("phase_elapsed_seconds"),
        "phase_elapsed_seconds",
    )
    total_elapsed = _parse_nonnegative_float(
        payload.get("total_elapsed_seconds"),
        "total_elapsed_seconds",
    )
    current_step = _parse_optional_integer(
        payload.get("current_step"),
        "current_step",
        minimum=0,
    )
    configured_steps = _parse_optional_integer(
        payload.get("configured_steps"),
        "configured_steps",
        minimum=1,
    )
    if (
        current_step is not None
        and configured_steps is not None
        and current_step > configured_steps
    ):
        raise GarmentInpaintError("현재 콜백 횟수가 설정 단계보다 큽니다.")
    message = payload.get("message", "")
    if not isinstance(message, str):
        raise GarmentInpaintError("Inpaint 진행 메시지는 문자열이어야 합니다.")
    return GarmentInpaintProgress(
        phase=str(phase),
        phase_elapsed_seconds=phase_elapsed,
        total_elapsed_seconds=total_elapsed,
        current_step=current_step,
        configured_steps=configured_steps,
        message=message,
    )


def _parse_nonnegative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GarmentInpaintError(f"{name}은 숫자여야 합니다.")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise GarmentInpaintError(f"{name}은 0 이상의 유한값이어야 합니다.")
    return converted


def _parse_optional_integer(
    value: object,
    name: str,
    *,
    minimum: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GarmentInpaintError(f"{name}은 {minimum} 이상의 정수여야 합니다.")
    return value


def _emit_progress_callback(
    progress_callback: Callable[[GarmentInpaintProgress], None] | None,
    progress: GarmentInpaintProgress,
) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(progress)
    except Exception:
        # 화면 상태 전달 실패가 GPU 생성과 임시 파일 정리를 중단하면 안 된다.
        return


def _create_execution_metrics(
    progress_events: list[GarmentInpaintProgress],
    *,
    parent_total_seconds: float,
    invalid_progress_event_count: int,
    heartbeat_count: int,
) -> GarmentInpaintExecutionMetrics:
    def phase_seconds(phase: str) -> float:
        values = [
            event.phase_elapsed_seconds
            for event in progress_events
            if event.phase == phase
        ]
        return max(values, default=0.0)

    runner_totals = [
        event.total_elapsed_seconds
        for event in progress_events
        if event.phase == "completed"
    ]
    if not runner_totals:
        runner_totals = [
            event.total_elapsed_seconds for event in progress_events
        ]
    return GarmentInpaintExecutionMetrics(
        pipeline_load_seconds=phase_seconds("pipeline_loading"),
        ip_adapter_load_seconds=phase_seconds("ip_adapter_loading"),
        diffusion_seconds=phase_seconds("diffusion_running"),
        output_save_seconds=phase_seconds("output_saving"),
        runner_total_seconds=max(runner_totals, default=0.0),
        parent_total_seconds=parent_total_seconds,
        progress_event_count=len(progress_events),
        invalid_progress_event_count=invalid_progress_event_count,
        heartbeat_count=heartbeat_count,
    )
