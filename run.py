"""GenAI Lab의 단일 실행점.

처음 배우는 사람이 입력 -> 검사 -> 모델 준비 -> 생성 -> 저장 순서를
한 파일에서 따라갈 수 있도록 기준 이미지가 성공하기 전까지 분리하지 않는다.

공식 참고 문서:
- Stable Diffusion 추론: https://huggingface.co/docs/diffusers/using-diffusers/conditional_image_generation
- IP-Adapter: https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter
- 시드 재현: https://huggingface.co/docs/diffusers/main/using-diffusers/reusing_seeds
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from genai_lab.generator import generate_images
from genai_lab.model import prepare_pipeline
from genai_lab.result import (
    create_new_run_directory,
    initial_result,
    load_existing_result,
    resolve_resume_directory,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parent


class AppError(Exception):
    """사용자가 직접 고칠 수 있는 입력 또는 환경 오류."""


def configure_console_encoding() -> None:
    """Windows에서도 한글 안내가 UTF-8로 기록되도록 출력 인코딩을 고정한다."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def configure_system_certificates() -> None:
    """Windows 인증서 저장소를 모델 다운로드 연결에 사용한다.

    truststore의 inject_into_ssl은 라이브러리가 아닌 응용 프로그램과
    스크립트에서 사용하도록 공식 안내되어 있다.
    https://pypi.org/project/truststore/
    """
    try:
        import truststore
    except ImportError as error:
        raise AppError(
            "Windows 인증서를 연결하는 truststore가 없습니다. "
            "'python -m pip install -r requirements.txt'를 실행하세요."
        ) from error
    truststore.inject_into_ssl()


@dataclass(frozen=True)
class PromptItem:
    request_id: str
    description_ko: str
    prompt: str
    negative_prompt: str
    seed: int

    @property
    def filename(self) -> str:
        return f"{self.request_id}_{self.seed}.png"


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="참조 그림의 특징을 반영해 이미지를 한 장씩 생성합니다."
    )
    parser.add_argument("--config", required=True, help="설정 YAML 파일 경로")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="모델을 받지 않고 GPU, 설정과 입력만 확인",
    )
    parser.add_argument(
        "--resume",
        help="중단된 결과 폴더를 지정하여 빠진 이미지부터 다시 생성",
    )
    return parser.parse_args(argv)


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise AppError(
            "설정 파일을 읽는 PyYAML이 없습니다. "
            "먼저 'python -m pip install -r requirements.txt'를 실행하세요."
        ) from error

    if not path.is_file():
        raise AppError(f"설정 파일이 없습니다: {path}")

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise AppError(f"설정 파일을 읽을 수 없습니다: {error}") from error

    if not isinstance(data, dict):
        raise AppError("설정 파일의 최상위 값은 항목 묶음이어야 합니다.")
    return data


def require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise AppError(f"설정에 '{key}' 항목 묶음이 필요합니다.")
    return value


def require_value(mapping: dict[str, Any], key: str, section: str) -> Any:
    value = mapping.get(key)
    if value is None or value == "":
        raise AppError(f"설정 '{section}.{key}' 값이 필요합니다.")
    return value


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_config(config: dict[str, Any]) -> None:
    model = require_mapping(config, "model")
    generation = require_mapping(config, "generation")
    style = require_mapping(config, "style")
    paths = require_mapping(config, "paths")

    require_value(model, "id", "model")
    require_value(model, "cache_dir", "model")
    family = require_value(model, "family", "model")
    if family not in {"sd15", "sdxl"}:
        raise AppError("설정 'model.family'은 sd15 또는 sdxl이어야 합니다.")
    if model.get("dtype") != "float16":
        raise AppError("RTX 4060 8GB 기준 설정 'model.dtype'은 float16이어야 합니다.")

    width = generation.get("width")
    height = generation.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        raise AppError(
            "설정 'generation.width'와 'generation.height'는 정수여야 합니다."
        )
    if width < 256 or height < 256:
        raise AppError(
            "이미지의 가로와 세로는 각각 256픽셀 이상이어야 합니다."
        )
    if width % 8 != 0 or height % 8 != 0:
        raise AppError(
            "이미지의 가로와 세로는 모델 처리 단위인 8의 배수여야 합니다."
        )
    if width > 1536 or height > 1536 or width * height > 1024 * 1024:
        raise AppError(
            "현재 RTX 4060 8GB 기준으로 한 변은 1536픽셀 이하이고 "
            "전체 픽셀 수는 1024×1024 이하여야 합니다. "
            f"현재 설정: {width}×{height}"
        )

    steps = generation.get("steps")
    if not isinstance(steps, int) or steps < 1:
        raise AppError("설정 'generation.steps'는 1 이상의 정수여야 합니다.")

    limit = generation.get("limit")
    if not isinstance(limit, int) or not 1 <= limit <= 3:
        raise AppError("설정 'generation.limit'은 1~3 사이의 정수여야 합니다.")

    guidance = generation.get("guidance_scale")
    if not isinstance(guidance, (int, float)) or guidance <= 0:
        raise AppError("설정 'generation.guidance_scale'은 0보다 커야 합니다.")

    generation_mode = generation.get("mode", "text_to_image")
    if generation_mode not in {"text_to_image", "image_to_image"}:
        raise AppError(
            "설정 'generation.mode'는 text_to_image 또는 "
            "image_to_image여야 합니다."
        )
    if generation_mode == "image_to_image":
        original_image_change_strength = generation.get(
            "original_image_change_strength"
        )
        if not isinstance(original_image_change_strength, (int, float)) or not (
            0.15 <= original_image_change_strength <= 0.35
        ):
            raise AppError(
                "원본 유지 모드의 'generation.original_image_change_strength'는 "
                "0.15부터 0.35 사이여야 합니다."
            )

    pose_control = config.get("pose_control")
    if pose_control is not None:
        if not isinstance(pose_control, dict):
            raise AppError("설정 'pose_control'은 항목 묶음이어야 합니다.")
        if not isinstance(pose_control.get("enabled"), bool):
            raise AppError("설정 'pose_control.enabled'는 true 또는 false여야 합니다.")
        if pose_control.get("enabled"):
            if family != "sdxl" or generation_mode != "image_to_image":
                raise AppError(
                    "자세 제어는 SDXL 이미지 수정 방식에서만 사용할 수 있습니다."
                )
            require_value(pose_control, "model_id", "pose_control")
            conditioning_scale = pose_control.get("conditioning_scale")
            if not isinstance(conditioning_scale, (int, float)) or not (
                0.0 < conditioning_scale <= 2.0
            ):
                raise AppError(
                    "설정 'pose_control.conditioning_scale'은 0 초과 2.0 이하여야 합니다."
                )
            guidance_start = pose_control.get("guidance_start")
            guidance_end = pose_control.get("guidance_end")
            if not isinstance(guidance_start, (int, float)) or not (
                0.0 <= guidance_start <= 1.0
            ):
                raise AppError(
                    "설정 'pose_control.guidance_start'는 0.0~1.0이어야 합니다."
                )
            if not isinstance(guidance_end, (int, float)) or not (
                0.0 <= guidance_end <= 1.0
            ):
                raise AppError(
                    "설정 'pose_control.guidance_end'는 0.0~1.0이어야 합니다."
                )
            if guidance_start >= guidance_end:
                raise AppError(
                    "자세 제어 시작 비율은 종료 비율보다 작아야 합니다."
                )
            pose_image_strength = pose_control.get(
                "original_image_change_strength"
            )
            if not isinstance(pose_image_strength, (int, float)) or not (
                0.15 <= pose_image_strength <= 0.60
            ):
                raise AppError(
                    "설정 'pose_control.original_image_change_strength'는 "
                    "0.15~0.60이어야 합니다."
                )

    pose_result_policy = config.get("pose_result_policy")
    if pose_result_policy is not None:
        validate_pose_result_policy_config(pose_result_policy)

    reference_quality = config.get("reference_quality")
    if reference_quality is not None:
        if not isinstance(reference_quality, dict):
            raise AppError("설정 'reference_quality'는 항목 묶음이어야 합니다.")
        validate_reference_quality_config(reference_quality)
    detail_correction = config.get("detail_correction")
    if detail_correction is not None:
        if not isinstance(detail_correction, dict):
            raise AppError("설정 'detail_correction'은 항목 묶음이어야 합니다.")
        validate_detail_correction_config(detail_correction)
    if not isinstance(style.get("enabled"), bool):
        raise AppError("설정 'style.enabled'는 true 또는 false여야 합니다.")
    if style["enabled"]:
        for key in (
            "reference_image",
            "adapter_repository",
            "adapter_subfolder",
            "adapter_weight",
            "scale",
        ):
            require_value(style, key, "style")
        scale = style["scale"]
        if not isinstance(scale, (int, float)) or not 0 <= scale <= 1:
            raise AppError("설정 'style.scale'은 0~1 사이 숫자여야 합니다.")
        reference_path = resolve_project_path(style["reference_image"])
        if not reference_path.is_file():
            raise AppError(
                "참조 그림 사용이 켜져 있지만 파일이 없습니다: "
                f"{reference_path}\ninputs/reference/style.png를 넣거나 "
                "style.enabled를 false로 바꾸세요."
            )

    require_value(paths, "prompts_file", "paths")
    require_value(paths, "output_dir", "paths")


def validate_pose_result_policy_config(policy: dict[str, Any]) -> None:
    """결과 우선 임시 자세 정책이 관측 전용인지 검사한다."""
    if not isinstance(policy, dict):
        raise AppError("설정 'pose_result_policy'는 항목 묶음이어야 합니다.")
    if policy.get("mode") != "observe_only":
        raise AppError(
            "임시 정책 'pose_result_policy.mode'는 observe_only여야 합니다."
        )
    if policy.get("target_sample_count") != 3:
        raise AppError(
            "임시 정책 'pose_result_policy.target_sample_count'는 3이어야 합니다."
        )
    for key in (
        "block_on_pose_mismatch",
        "switch_to_text_to_image",
        "use_identity_crop",
    ):
        if policy.get(key) is not False:
            raise AppError(
                f"임시 관측 정책 'pose_result_policy.{key}'는 false여야 합니다."
            )


def validate_reference_quality_config(
    reference_quality: dict[str, Any],
) -> None:
    """참조 이미지 화질 검사와 확대 복원 설정을 검사한다."""
    if not isinstance(reference_quality.get("enabled"), bool):
        raise AppError("설정 'reference_quality.enabled'는 true 또는 false여야 합니다.")
    if not reference_quality["enabled"]:
        return

    for key in ("minimum_short_side", "tile_size", "tile_overlap", "maximum_long_side"):
        if not isinstance(reference_quality.get(key), int):
            raise AppError(f"설정 'reference_quality.{key}'는 정수여야 합니다.")
    if reference_quality["minimum_short_side"] < 256:
        raise AppError("'reference_quality.minimum_short_side'는 256 이상이어야 합니다.")
    if reference_quality["maximum_long_side"] < reference_quality["minimum_short_side"]:
        raise AppError(
            "'reference_quality.maximum_long_side'는 minimum_short_side 이상이어야 합니다."
        )
    if reference_quality["tile_size"] < 64:
        raise AppError("'reference_quality.tile_size'는 64 이상이어야 합니다.")
    if not 0 <= reference_quality["tile_overlap"] < reference_quality["tile_size"]:
        raise AppError(
            "'reference_quality.tile_overlap'은 0 이상이고 tile_size보다 작아야 합니다."
        )
    sharpness = reference_quality.get("minimum_sharpness_score")
    if not isinstance(sharpness, (int, float)) or sharpness <= 0:
        raise AppError(
            "'reference_quality.minimum_sharpness_score'는 0보다 커야 합니다."
        )
    require_value(reference_quality, "model_path", "reference_quality")


def validate_detail_correction_config(
    detail_correction: dict[str, Any],
) -> None:
    """얼굴·손 탐지, 마스크 제한과 Inpaint 설정을 검사한다."""
    if not isinstance(detail_correction.get("enabled"), bool):
        raise AppError("설정 'detail_correction.enabled'는 true 또는 false여야 합니다.")
    if not detail_correction["enabled"]:
        return

    for key in ("detector_repository", "face_model", "hand_model"):
        require_value(detail_correction, key, "detail_correction")
    confidence = detail_correction.get("minimum_confidence")
    if not isinstance(confidence, (int, float)) or not 0 < confidence <= 1:
        raise AppError("'detail_correction.minimum_confidence'는 0 초과 1 이하여야 합니다.")
    for key in (
        "maximum_face_regions",
        "maximum_hand_regions",
        "inpaint_steps",
        "padding_mask_crop",
    ):
        if not isinstance(detail_correction.get(key), int) or detail_correction[key] < 1:
            raise AppError(f"설정 'detail_correction.{key}'는 1 이상의 정수여야 합니다.")

    minimum_area = detail_correction.get("minimum_mask_area_ratio")
    maximum_area = detail_correction.get("maximum_mask_area_ratio")
    if not isinstance(minimum_area, (int, float)) or not isinstance(
        maximum_area, (int, float)
    ):
        raise AppError("부분 보정 마스크 면적 비율은 숫자여야 합니다.")
    if not 0 < minimum_area < maximum_area < 1:
        raise AppError(
            "부분 보정 마스크 비율은 0 < 최소 < 최대 < 1 순서여야 합니다."
        )
    for key in ("mask_padding_ratio", "inpaint_strength", "guidance_scale"):
        if not isinstance(detail_correction.get(key), (int, float)):
            raise AppError(f"설정 'detail_correction.{key}'는 숫자여야 합니다.")
    if not 0 < detail_correction["inpaint_strength"] <= 1:
        raise AppError("'detail_correction.inpaint_strength'는 0 초과 1 이하여야 합니다.")


def read_prompts(path: Path, limit: int) -> list[PromptItem]:
    if not path.is_file():
        raise AppError(f"생성 요청 파일이 없습니다: {path}")

    items: list[PromptItem] = []
    seen_ids: set[str] = set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"id", "prompt", "negative_prompt", "seed"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise AppError(
                    "prompts.csv에는 id, prompt, negative_prompt, seed 열이 필요합니다."
                )
            for row_number, row in enumerate(reader, start=2):
                request_id = (row.get("id") or "").strip()
                prompt = (row.get("prompt") or "").strip()
                if not request_id or not prompt:
                    raise AppError(
                        f"prompts.csv {row_number}번째 줄의 id와 prompt가 필요합니다."
                    )
                if request_id in seen_ids:
                    raise AppError(f"prompts.csv에 중복 요청 번호가 있습니다: {request_id}")
                try:
                    seed = int((row.get("seed") or "").strip())
                except ValueError as error:
                    raise AppError(
                        f"prompts.csv {row_number}번째 줄의 seed는 정수여야 합니다."
                    ) from error
                if seed < 0:
                    raise AppError(
                        f"prompts.csv {row_number}번째 줄의 seed는 0 이상이어야 합니다."
                    )
                seen_ids.add(request_id)
                items.append(
                    PromptItem(
                        request_id=request_id,
                        description_ko=(row.get("description_ko") or "").strip(),
                        prompt=prompt,
                        negative_prompt=(row.get("negative_prompt") or "").strip(),
                        seed=seed,
                    )
                )
    except UnicodeDecodeError as error:
        raise AppError("prompts.csv를 UTF-8 형식으로 저장하세요.") from error
    except OSError as error:
        raise AppError(f"prompts.csv를 읽을 수 없습니다: {error}") from error

    if len(items) < limit:
        raise AppError(
            f"생성 요청은 {limit}개가 필요하지만 prompts.csv에는 {len(items)}개만 있습니다."
        )
    return items[:limit]


def config_fingerprint(config: dict[str, Any], prompts: list[PromptItem]) -> str:
    data = {
        "config": config,
        "prompts": [asdict(item) for item in prompts],
    }
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_environment() -> dict[str, Any]:
    if sys.version_info[:2] != (3, 10):
        raise AppError(
            f"Python 3.10이 필요하지만 현재 버전은 {platform.python_version()}입니다."
        )
    try:
        import torch
    except ImportError as error:
        raise AppError(
            "PyTorch가 설치되지 않았습니다. README의 CUDA 12.8 설치 명령을 실행하세요."
        ) from error

    if not torch.cuda.is_available():
        raise AppError(
            "PyTorch에서 NVIDIA GPU를 찾지 못했습니다. "
            "NVIDIA 드라이버와 CUDA용 PyTorch 설치를 확인하세요."
        )

    device_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory
    if "RTX 4060" not in device_name.upper():
        print(f"주의: 계획한 RTX 4060과 다른 GPU가 감지되었습니다: {device_name}")

    missing = [
        name
        for name in (
            "diffusers",
            "transformers",
            "accelerate",
            "safetensors",
            "yaml",
            "truststore",
        )
        if package_version("PyYAML" if name == "yaml" else name) is None
    ]
    if missing:
        raise AppError(
            "필요한 도구가 설치되지 않았습니다: "
            + ", ".join(missing)
            + "\n'python -m pip install -r requirements.txt'를 실행하세요."
        )

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": device_name,
        "vram_bytes": total_vram,
        "packages": {
            name: package_version(name)
            for name in (
                "diffusers",
                "transformers",
                "accelerate",
                "safetensors",
                "PyYAML",
                "Pillow",
                "truststore",
            )
        },
    }


def execute(args: argparse.Namespace) -> Path | None:
    config_path = resolve_project_path(args.config)
    config = load_yaml(config_path)
    validate_config(config)

    prompts_path = resolve_project_path(config["paths"]["prompts_file"])
    prompts = read_prompts(prompts_path, config["generation"]["limit"])
    environment = check_environment()

    print(f"GPU 확인: {environment['gpu']}")
    print(f"GPU 메모리: {environment['vram_bytes'] / 1024**3:.1f}GB")
    print(f"생성 요청: {len(prompts)}개, 한 장씩 순서대로 처리")
    if args.check_only:
        print("환경, 설정과 입력 검사를 모두 통과했습니다.")
        return None

    output_root = resolve_project_path(config["paths"]["output_dir"])
    fingerprint = config_fingerprint(config, prompts)
    if args.resume:
        run_directory = resolve_resume_directory(
            resolve_project_path(args.resume), output_root
        )
        result = load_existing_result(run_directory, fingerprint)
        result["status"] = "running"
    else:
        run_directory = create_new_run_directory(output_root)
        result = initial_result(config, prompts, environment, fingerprint)
        write_json(run_directory / "result.json", result)

    print(f"결과 폴더: {run_directory}")
    pipeline = prepare_pipeline(config)
    generate_images(pipeline, config, prompts, run_directory, result, PROJECT_ROOT)
    print("모든 이미지 생성을 완료했습니다.")
    return run_directory


def main(argv: list[str] | None = None) -> int:
    configure_console_encoding()
    configure_system_certificates()
    args = parse_arguments(argv)
    try:
        execute(args)
        return 0
    except (AppError, ValueError, RuntimeError) as error:
        print(f"오류: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n사용자가 실행을 중단했습니다. 생성된 파일은 그대로 남아 있습니다.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
