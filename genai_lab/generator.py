"""생성 요청을 한 장씩 처리한다."""

import time
from datetime import datetime
from pathlib import Path
from typing import Any

from genai_lab.result import refresh_summary, update_request_result, write_json
from genai_lab.style import load_reference_image


def generate_images(
    pipeline,
    config: dict[str, Any],
    prompts: list[Any],
    run_directory: Path,
    result: dict[str, Any],
    project_root: Path,
) -> None:
    import torch

    generation = config["generation"]
    reference_image = load_reference_image(config, project_root)
    default_negative = generation.get("default_negative_prompt", "")
    result_path = run_directory / "result.json"

    for index, item in enumerate(prompts, start=1):
        output_path = run_directory / "images" / item.filename
        if output_path.is_file():
            print(f"[{index}/{len(prompts)}] 이미 완료되어 건너뜀: {item.filename}")
            update_request_result(result, item.request_id, status="skipped")
            refresh_summary(result)
            write_json(result_path, result)
            continue

        print(f"[{index}/{len(prompts)}] 생성 중: {item.description_ko or item.request_id}")
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        random_start = torch.Generator(device="cpu").manual_seed(item.seed)
        arguments: dict[str, Any] = {
            "prompt": item.prompt,
            "negative_prompt": item.negative_prompt or default_negative,
            "width": generation["width"],
            "height": generation["height"],
            "num_inference_steps": generation["steps"],
            "guidance_scale": generation["guidance_scale"],
            "generator": random_start,
        }
        if reference_image is not None:
            arguments["ip_adapter_image"] = [reference_image] if not isinstance(reference_image, list) else reference_image

        try:
            image = pipeline(**arguments).images[0]
            image.save(output_path)
            elapsed = round(time.perf_counter() - started, 3)
            peak_vram = torch.cuda.max_memory_allocated()
            update_request_result(
                result,
                item.request_id,
                status="completed",
                output=str(output_path.relative_to(project_root)),
                elapsed_seconds=elapsed,
                peak_vram_bytes=peak_vram,
            )
            refresh_summary(result)
            write_json(result_path, result)
            print(f"저장 완료: {output_path.name} ({elapsed}초)")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            is_memory_error = isinstance(error, torch.cuda.OutOfMemoryError) or (
                "out of memory" in str(error).lower()
            )
            torch.cuda.empty_cache()
            update_request_result(
                result,
                item.request_id,
                status="failed",
                error="GPU 메모리 부족" if is_memory_error else str(error),
            )
            result["status"] = "failed"
            refresh_summary(result)
            write_json(result_path, result)
            if is_memory_error:
                raise RuntimeError(
                    "GPU 메모리가 부족합니다. 해상도와 설정은 자동으로 바꾸지 않았습니다.\n"
                    "다른 GPU 사용 프로그램을 닫고 같은 결과 폴더로 이어서 실행하세요.\n"
                    f"결과 폴더: {run_directory}"
                ) from error
            raise

    result["status"] = "completed"
    result["finished_at"] = datetime.now().astimezone().isoformat()
    refresh_summary(result)
    write_json(result_path, result)

