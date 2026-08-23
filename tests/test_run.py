from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import run
from genai_lab.generator import generate_images
from genai_lab.result import initial_result


def valid_config(tmp_path: Path) -> dict:
    reference = tmp_path / "style.png"
    reference.write_bytes(b"not-used-by-validation")
    return {
        "model": {
            "family": "sd15",
            "id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "cache_dir": str(tmp_path / "cache"),
            "dtype": "float16",
        },
        "generation": {
            "width": 512,
            "height": 512,
            "steps": 20,
            "guidance_scale": 7.5,
            "limit": 1,
            "default_negative_prompt": "low quality",
        },
        "style": {
            "enabled": False,
            "reference_image": str(reference),
            "adapter_repository": "h94/IP-Adapter",
            "adapter_subfolder": "models",
            "adapter_weight": "ip-adapter_sd15.bin",
            "scale": 0.55,
        },
        "paths": {
            "prompts_file": str(tmp_path / "prompts.csv"),
            "output_dir": str(tmp_path / "outputs"),
        },
    }


def write_prompts(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["id", "description_ko", "prompt", "negative_prompt", "seed"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_config_accepts_fixed_first_version(tmp_path: Path) -> None:
    run.validate_config(valid_config(tmp_path))


def test_config_accepts_portrait_resolution(tmp_path: Path) -> None:
    config = valid_config(tmp_path)
    config["generation"]["width"] = 576
    config["generation"]["height"] = 896
    run.validate_config(config)


def test_config_rejects_resolution_not_divisible_by_eight(
    tmp_path: Path,
) -> None:
    config = valid_config(tmp_path)
    config["generation"]["width"] = 577
    with pytest.raises(run.AppError, match="8의 배수"):
        run.validate_config(config)


def test_config_rejects_resolution_over_gpu_pixel_budget(
    tmp_path: Path,
) -> None:
    config = valid_config(tmp_path)
    config["generation"]["width"] = 1536
    config["generation"]["height"] = 1024
    with pytest.raises(run.AppError, match="1024×1024"):
        run.validate_config(config)


def test_config_accepts_animagine_trial_resolution(tmp_path: Path) -> None:
    config = valid_config(tmp_path)
    config["model"]["family"] = "sdxl"
    config["model"]["id"] = "cagliostrolab/animagine-xl-3.1"
    config["generation"]["width"] = 576
    config["generation"]["height"] = 896
    run.validate_config(config)


def test_style_requires_reference_image(tmp_path: Path) -> None:
    config = valid_config(tmp_path)
    config["style"]["enabled"] = True
    config["style"]["reference_image"] = str(tmp_path / "missing.png")
    with pytest.raises(run.AppError, match="참조 그림"):
        run.validate_config(config)


def test_reads_prompts_and_builds_expected_filename(tmp_path: Path) -> None:
    path = tmp_path / "prompts.csv"
    write_prompts(
        path,
        [
            {
                "id": "001",
                "description_ko": "숲",
                "prompt": "a forest",
                "negative_prompt": "",
                "seed": "101",
            }
        ],
    )
    items = run.read_prompts(path, 1)
    assert items[0].filename == "001_101.png"
    assert items[0].description_ko == "숲"


def test_duplicate_request_id_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "prompts.csv"
    row = {
        "id": "001",
        "description_ko": "숲",
        "prompt": "a forest",
        "negative_prompt": "",
        "seed": "101",
    }
    write_prompts(path, [row, row])
    with pytest.raises(run.AppError, match="중복 요청 번호"):
        run.read_prompts(path, 2)


def test_resume_rejects_changed_configuration(tmp_path: Path) -> None:
    config = valid_config(tmp_path)
    prompt = run.PromptItem("001", "숲", "a forest", "", 101)
    fingerprint = run.config_fingerprint(config, [prompt])
    run_directory = tmp_path / "outputs" / "run-1"
    (run_directory / "images").mkdir(parents=True)
    (run_directory / "result.json").write_text(
        json.dumps({"config_fingerprint": fingerprint}), encoding="utf-8"
    )
    loaded = run.load_existing_result(run_directory, fingerprint)
    assert loaded["config_fingerprint"] == fingerprint
    with pytest.raises(ValueError, match="설정 또는 prompts.csv가 다릅니다"):
        run.load_existing_result(run_directory, "different")


def test_resume_directory_must_stay_under_outputs(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    outside = tmp_path / "outside"
    (outside / "images").mkdir(parents=True)
    (outside / "result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="outputs 안"):
        run.resolve_resume_directory(outside, output_root)


class FakeOutOfMemoryError(RuntimeError):
    pass


class FakeCuda:
    OutOfMemoryError = FakeOutOfMemoryError

    @staticmethod
    def reset_peak_memory_stats() -> None:
        return None

    @staticmethod
    def max_memory_allocated() -> int:
        return 1234

    @staticmethod
    def empty_cache() -> None:
        return None


class FakeGenerator:
    def __init__(self, device: str):
        self.device = device
        self.seed = None

    def manual_seed(self, seed: int):
        self.seed = seed
        return self


class FakeImage:
    def save(self, path: Path) -> None:
        Path(path).write_bytes(b"fake-png")


class FakePipeline:
    def __init__(self, fail_at: int | None = None):
        self.calls: list[dict] = []
        self.fail_at = fail_at

    def __call__(self, **arguments):
        self.calls.append(arguments)
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise FakeOutOfMemoryError("CUDA out of memory")
        return SimpleNamespace(images=[FakeImage()])


def fake_torch_module() -> SimpleNamespace:
    return SimpleNamespace(cuda=FakeCuda(), Generator=FakeGenerator)


def generation_config(tmp_path: Path, limit: int) -> dict:
    config = valid_config(tmp_path)
    config["generation"]["limit"] = limit
    return config


def prompt_items(count: int) -> list[run.PromptItem]:
    return [
        run.PromptItem(f"{index:03d}", f"요청 {index}", f"prompt {index}", "", index)
        for index in range(1, count + 1)
    ]


def test_three_requests_are_generated_sequentially_and_resume_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "torch", fake_torch_module())
    monkeypatch.setattr("genai_lab.generator.load_reference_image", lambda *_: None)
    config = generation_config(tmp_path, 3)
    prompts = prompt_items(3)
    run_directory = tmp_path / "outputs" / "run-3"
    (run_directory / "images").mkdir(parents=True)
    result = initial_result(config, prompts, {"gpu": "fake"}, "fingerprint")
    pipeline = FakePipeline()

    generate_images(pipeline, config, prompts, run_directory, result, tmp_path)

    assert len(pipeline.calls) == 3
    assert [call["generator"].seed for call in pipeline.calls] == [1, 2, 3]
    assert len(list((run_directory / "images").glob("*.png"))) == 3
    assert result["summary"] == {"total": 3, "completed": 3, "failed": 0}

    generate_images(pipeline, config, prompts, run_directory, result, tmp_path)
    assert len(pipeline.calls) == 3


def test_memory_error_keeps_completed_image_and_korean_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "torch", fake_torch_module())
    monkeypatch.setattr("genai_lab.generator.load_reference_image", lambda *_: None)
    config = generation_config(tmp_path, 2)
    prompts = prompt_items(2)
    run_directory = tmp_path / "outputs" / "memory-error"
    (run_directory / "images").mkdir(parents=True)
    result = initial_result(config, prompts, {"gpu": "fake"}, "fingerprint")

    with pytest.raises(RuntimeError, match="GPU 메모리가 부족합니다"):
        generate_images(
            FakePipeline(fail_at=2), config, prompts, run_directory, result, tmp_path
        )

    assert (run_directory / "images" / "001_1.png").is_file()
    saved = json.loads((run_directory / "result.json").read_text(encoding="utf-8"))
    assert saved["status"] == "failed"
    assert saved["requests"][1]["error"] == "GPU 메모리 부족"
