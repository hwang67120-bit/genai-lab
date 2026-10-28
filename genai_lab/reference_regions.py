"""합성·중립화와 분리된 참조 영역 분석 경로."""
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import json
import math
import subprocess
import tempfile
import numpy as np
from PIL import Image


ANALYSIS_SCOPE_CONTRACTS = {
    "legacy": {
        "required_regions": ("identity", "garment"),
        "use_target_garment_tags": True,
    },
    "character_input": {
        "required_regions": ("identity",),
        "use_target_garment_tags": False,
    },
    "base_output": {
        "required_regions": ("identity",),
        "use_target_garment_tags": False,
    },
    "garment_edit_input": {
        "required_regions": ("identity", "garment"),
        "use_target_garment_tags": False,
    },
    "final_output": {
        "required_regions": ("identity", "garment"),
        "use_target_garment_tags": True,
    },
}


def analysis_scope_contract(scope):
    name = str(scope or "legacy")
    try:
        return name, ANALYSIS_SCOPE_CONTRACTS[name]
    except KeyError as error:
        raise ValueError(f"unknown reference-region analysis scope: {name}") from error


@dataclass
class ReferenceRegions:
    masks: dict[str, Image.Image]
    record: dict

    def close(self):
        for mask in self.masks.values():
            mask.close()


def read_binary_mask(mask, size):
    if mask.mode != "L" or mask.size != size:
        raise ValueError("참조 마스크 형식 또는 좌표 크기가 다릅니다.")
    values = np.asarray(mask)
    if not np.isin(values, (0, 255)).all():
        raise ValueError("참조 마스크는 0/255여야 합니다.")
    return values == 255


def run_reference_process(command, cwd, timeout, cancelled):
    """소유한 분석 프로세스만 취소·시간 초과 시 종료한다."""
    if cancelled():
        raise InterruptedError("참조 영역 분석을 취소했습니다.")
    deadline = perf_counter() + timeout
    with subprocess.Popen(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    ) as process:
        try:
            while True:
                if cancelled():
                    raise InterruptedError("참조 영역 분석을 취소했습니다.")
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    raise TimeoutError("참조 영역 분석 시간 제한을 초과했습니다.")
                try:
                    stdout, stderr = process.communicate(timeout=min(.25, remaining))
                    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            process.kill()
            process.communicate()
            raise


def load_reference_regions(
        directory, size, *, required_regions=None, analysis_scope="legacy"):
    scope, scope_contract = analysis_scope_contract(analysis_scope)
    required = tuple(
        scope_contract["required_regions"]
        if required_regions is None else required_regions
    )
    allowed_required = {"identity", "hair", "face", "garment", "foreground"}
    if not required or set(required) - allowed_required:
        raise ValueError("invalid required reference regions")
    record = json.loads((directory / "regions.json").read_text(encoding="utf-8"))
    if record.get("mode") != "reference_regions_v1" or record.get("size") != list(size):
        raise ValueError("참조 영역 분석 출력 계약 또는 좌표가 다릅니다.")
    masks = {}
    try:
        for name in ("identity", "hair", "face", "garment", "foreground"):
            with Image.open(directory / f"{name}.png") as image:
                masks[name] = image.copy()
            selected = read_binary_mask(masks[name], size)
            if int(selected.sum()) != record["pixel_counts"][name]:
                raise ValueError(f"{name}: 기록과 실제 마스크 픽셀 수가 다릅니다.")
            if name in required and not selected.any():
                raise ValueError(f"{name}: 참조 영역이 비었습니다.")
        identity = read_binary_mask(masks["identity"], size)
        garment = read_binary_mask(masks["garment"], size)
        hair = read_binary_mask(masks["hair"], size)
        face = read_binary_mask(masks["face"], size)
        foreground = read_binary_mask(masks["foreground"], size)
        if (np.any(identity & garment) or np.any(hair & ~identity)
                or np.any(face & ~identity) or np.any(garment & ~foreground)):
            raise ValueError("참조 영역 간 포함·충돌 관계가 잘못됐습니다.")
        record["analysis_scope"] = scope
        record["required_regions"] = list(required)
        record["target_garment_tags_applied"] = bool(
            scope_contract["use_target_garment_tags"])
        return ReferenceRegions(masks, record)
    except BaseException:
        for mask in masks.values():
            mask.close()
        raise


def analyze_reference_regions(
        source, config, root, *, cancelled=lambda: False, run_log=None,
        allow_parser_foreground_fallback=False, analysis_scope="legacy"):
    scope, scope_contract = analysis_scope_contract(analysis_scope)
    # 기존 모델 경로만 재사용한다. 의상 제거·팽창 설정은 읽지 않는다.
    cfg = config["character_body_comparison"]
    paths = {name: Path(cfg[name]) for name in (
        "python_executable", "repository_path", "runner_path", "temporary_root", "cache_dir")}
    if not paths["runner_path"].is_absolute():
        paths["runner_path"] = root / paths["runner_path"]
    for name in ("python_executable", "repository_path", "runner_path"):
        if not paths[name].exists():
            raise ValueError(f"참조 분석 경로가 없습니다: {name}={paths[name]}")
    width, height = int(cfg["width"]), int(cfg["height"])
    timeout = float(cfg["timeout_seconds"])
    if min(width, height) < 256 or width % 8 or height % 8:
        raise ValueError("참조 분석 크기는 256 이상인 8의 배수여야 합니다.")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("참조 분석 시간 제한이 잘못됐습니다.")
    if cancelled():
        raise InterruptedError("참조 영역 분석을 취소했습니다.")
    paths["temporary_root"].mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    with tempfile.TemporaryDirectory(prefix="genai-reference-regions-", dir=paths["temporary_root"]) as temp:
        directory = Path(temp)
        with source.convert("RGB") as rgb:
            rgb.save(directory / "source.png")
        command = [
            str(paths["python_executable"]), str(paths["runner_path"]),
            "--repository-path", str(paths["repository_path"]),
            "--person-image", str(directory / "source.png"),
            "--reference-output-dir", str(directory),
            "--clothing-type", "overall", "--layered-target-masks",
            "--cache-dir", str(paths["cache_dir"]),
            "--width", str(width), "--height", str(height),
            "--foreground-model-id", cfg.get("foreground_model_id", "isnet-anime"),
        ]
        if allow_parser_foreground_fallback:
            command.append("--allow-parser-foreground-fallback")
        approved_garment_tags = (
            config.get("clothing_reference_generation", {}).get(
                "approved_tags", ())
            if scope_contract["use_target_garment_tags"] else ()
        )
        if approved_garment_tags:
            if (not isinstance(approved_garment_tags, (tuple, list))
                    or not all(isinstance(tag, str) for tag in approved_garment_tags)):
                raise ValueError("승인 의상 태그는 문자열 목록이어야 합니다.")
            command.extend([
                "--approved-garment-tags-json",
                json.dumps(list(approved_garment_tags), ensure_ascii=False),
            ])
        # 공용 실행기의 구형 CLI 인자. 참조 모드에서는 생성·조회하지 않는다.
        for flag in ("raw-mask", "protection-mask", "foreground-mask", "densepose", "metadata-json"):
            command.extend([f"--output-{flag}", str(directory / f"unused-{flag}")])
        result = run_reference_process(command, paths["repository_path"], timeout, cancelled)
        if run_log is not None:
            run_log.write_stage("참조 영역 분석", f"종료={result.returncode}, 소요={perf_counter()-started:.2f}초, 중립화/제거 검증=미실행")
        if result.returncode:
            raise RuntimeError("참조 영역 분석 실패: " + (result.stderr or result.stdout)[-3000:])
        if cancelled():
            raise InterruptedError("참조 영역 분석을 취소했습니다.")
        regions = load_reference_regions(
            directory, source.size, analysis_scope=scope)
        regions.record["elapsed_seconds"] = perf_counter() - started
        if run_log is not None:
            run_log.write_stage(
                "승인 의상 기반 마스크",
                str(regions.record.get("garment_mask_policy", {
                    "mode": "legacy_output_without_policy",
                })),
            )
        return regions
