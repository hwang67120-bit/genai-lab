"""Benchmark cumulative Base similarity for one approved GUI replay bundle."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import gc
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from PIL import Image, ImageDraw
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from genai_lab.approved_reference_run import approve_reference_run
from genai_lab.generation_orchestrator import GenerationOrchestrator
from genai_lab.generation_replay import load_generation_replay_bundle
from genai_lab.model import prepare_pipeline
from genai_lab.body_proportion_presets import active_body_proportion_preset_id
from genai_lab.run_log import create_generation_run_log
from run import (
    check_environment,
    configure_console_encoding,
    configure_system_certificates,
    validate_config,
)

COMPONENTS = ("character", "hair", "garment", "vibe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure cumulative Base similarity for one seed sequence."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument(
        "--prefixes", type=int, nargs="+", default=(1, 2, 4, 6, 8)
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def configure_benchmark(config: dict[str, Any], count: int) -> None:
    if not 2 <= count <= 100:
        raise ValueError("candidate-count must be between 2 and 100")
    generation = dict(config["clothing_reference_generation"])
    generation["candidate_count"] = count
    config["clothing_reference_generation"] = generation
    pipeline = dict(config.get("candidate_pipeline", {}))
    pipeline.update({
        "enabled": True,
        "target_valid_candidates": count,
        "maximum_attempts": count,
        "gender_retry_attempts": 0,
        "quality_retry_attempts": 0,
        "repair_best_candidate_only": True,
        "return_policy": "hard_safety_only",
    })
    config["candidate_pipeline"] = pipeline
    config["candidate_similarity_benchmark"] = {
        "continue_after_scoring_error": True,
    }


def load_records(directory: Path) -> list[dict[str, Any]]:
    paths = sorted(
        directory.glob("candidate_*.json"),
        key=lambda item: int(item.stem.split("_")[-1]),
    )
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def candidate_rows(
    records: list[dict[str, Any]], retained_numbers: set[int]
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        number = int(record.get("candidate_number", 0))
        similarity = record.get("similarity", {})
        semantic = record.get("candidate_semantic_gate", {})
        structure = record.get("candidate_structure_gate", {})
        integrity = record.get("generated_image_integrity", {})
        decision = record.get("candidate_decision", {})
        values = {name: numeric(similarity.get(name)) for name in COMPONENTS}
        base = [values["character"], values["hair"]]
        complete = all(value is not None for value in base)
        rows.append({
            "candidate_number": number,
            "seed": record.get("seed"),
            "hard_safe": number in retained_numbers,
            **values,
            "base_minimum": min(base) if complete else None,
            "base_mean": sum(base) / 2 if complete else None,
            "integrity_status": integrity.get("status"),
            "semantic_status": semantic.get("status"),
            "semantic_violations": semantic.get("violations", []),
            "structure_status": structure.get("status"),
            "structure_violations": structure.get("violations", []),
            "decision": decision.get("action"),
            "decision_reasons": decision.get("reasons", []),
        })
    return rows


def prefix_summary(rows: list[dict[str, Any]], prefix: int) -> dict[str, Any]:
    observed = [row for row in rows if row["candidate_number"] <= prefix]
    safe = [row for row in observed if row["hard_safe"]]
    scorable = [
        row for row in safe
        if row["base_minimum"] is not None and row["base_mean"] is not None
    ]
    ranked = sorted(
        scorable,
        key=lambda row: (
            row["base_minimum"],
            row["base_mean"],
            row["character"],
            row["hair"],
        ),
        reverse=True,
    )
    selected = ranked[0] if ranked else None

    def average(name: str) -> float | None:
        values = [row[name] for row in scorable if row[name] is not None]
        return sum(values) / len(values) if values else None

    return {
        "prefix": prefix,
        "attempts_observed": len(observed),
        "hard_safe_count": len(safe),
        "scorable_count": len(scorable),
        "selected_candidate": selected["candidate_number"] if selected else None,
        "selected_seed": selected["seed"] if selected else None,
        "best_character": max(
            (row["character"] for row in scorable), default=None
        ),
        "best_hair": max((row["hair"] for row in scorable), default=None),
        "best_base_minimum": selected["base_minimum"] if selected else None,
        "best_base_mean": selected["base_mean"] if selected else None,
        "mean_character": average("character"),
        "mean_hair": average("hair"),
        "target_65_reached": bool(
            selected and selected["base_minimum"] >= 0.65
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "candidate_number", "seed", "hard_safe", "character", "hair",
        "garment", "vibe", "base_minimum", "base_mean",
        "integrity_status", "semantic_status", "structure_status", "decision",
    )
    with path.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(
            target, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def write_contact_sheet(
    source: Path, rows: list[dict[str, Any]], output: Path
) -> None:
    cell_width, cell_height, columns = 260, 455, 4
    row_count = max(1, (len(rows) + columns - 1) // columns)
    canvas = Image.new(
        "RGB", (cell_width * columns, cell_height * row_count), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for offset, row in enumerate(rows):
        number = row["candidate_number"]
        image_path = source / f"candidate_{number}.png"
        x = (offset % columns) * cell_width
        y = (offset // columns) * cell_height
        if image_path.is_file():
            with Image.open(image_path) as opened:
                preview = opened.convert("RGB")
                preview.thumbnail((240, 360))
                canvas.paste(
                    preview,
                    (x + (cell_width - preview.width) // 2, y + 5),
                )
        lines = (
            f"#{number} seed={row['seed']} safe={row['hard_safe']}",
            f"character={percent(row['character'])}",
            f"hair={percent(row['hair'])}",
            f"base min={percent(row['base_minimum'])}",
            f"semantic={row['semantic_status']} structure={row['structure_status']}",
        )
        text_y = y + 365
        for line in lines:
            draw.text((x + 8, text_y), line, fill="black")
            text_y += 18
    canvas.save(output)


def main() -> int:
    configure_console_encoding()
    args = parse_args()
    prefixes = sorted({
        value for value in args.prefixes
        if 1 <= value <= args.candidate_count
    })
    if not prefixes:
        raise ValueError("No valid prefixes")
    if prefixes[-1] != args.candidate_count:
        prefixes.append(args.candidate_count)

    output = args.output_dir or (
        PROJECT_ROOT / "outputs" / "candidate-count-benchmarks" /
        datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output.mkdir(parents=True, exist_ok=False)

    inputs = request = batch = pipeline_object = None
    run_log = create_generation_run_log(PROJECT_ROOT)
    started = time.perf_counter()
    try:
        inputs, config, request, source_approval = (
            load_generation_replay_bundle(args.bundle)
        )
        configure_benchmark(config, args.candidate_count)
        validate_config(config)
        approve_reference_run(inputs, config, request)
        benchmark_approval = inputs.approved_generation

        configure_system_certificates()
        environment = check_environment()
        body_preset_id = active_body_proportion_preset_id(request)
        pipeline_object = (
            prepare_pipeline(
                config,
                body_proportion_preset_id=body_preset_id,
            )
            if body_preset_id is not None
            else prepare_pipeline(config)
        )
        orchestrator = GenerationOrchestrator(
            config,
            request,
            PROJECT_ROOT,
            run_log,
            cancelled=lambda: False,
            status_callback=lambda message: print(message, flush=True),
        )
        orchestrator.restore_approved_inputs(inputs)
        batch = orchestrator.generate_base_candidates(
            pipeline_object, inputs
        )
        generated_directory = Path(batch.directory)
        retained_numbers = {
            int(candidate.design_reference_record["candidate_number"])
            for candidate in batch.candidates
        }
        elapsed = time.perf_counter() - started

        records = load_records(generated_directory)
        rows = candidate_rows(records, retained_numbers)
        recorded_numbers = {row["candidate_number"] for row in rows}
        for quarantined in batch.quarantined:
            number = int(quarantined.get("candidate_number", 0))
            if number in recorded_numbers:
                continue
            failure_report = quarantined.get("report", {})
            rows.append({
                "candidate_number": number,
                "seed": quarantined.get("seed"),
                "hard_safe": False,
                "character": None, "hair": None,
                "garment": None, "vibe": None,
                "base_minimum": None, "base_mean": None,
                "integrity_status": failure_report.get("status"),
                "semantic_status": None,
                "semantic_violations": [],
                "structure_status": None,
                "structure_violations": [],
                "decision": "quarantined",
                "decision_reasons": [
                    quarantined.get("failure_stage", "unknown")
                ],
                "failure_stage": quarantined.get("failure_stage"),
            })
        rows.sort(key=lambda row: row["candidate_number"])
        summaries = [prefix_summary(rows, prefix) for prefix in prefixes]

        artifacts = output / "candidates"
        artifacts.mkdir()
        for path in generated_directory.glob("candidate_*.*"):
            if path.suffix.lower() in {".png", ".json"}:
                shutil.copy2(path, artifacts / path.name)
        corrupted = generated_directory / "corrupted"
        if corrupted.is_dir():
            shutil.copytree(
                corrupted, artifacts / "corrupted", dirs_exist_ok=True
            )
        write_csv(output / "candidate_scores.csv", rows)
        write_contact_sheet(
            generated_directory, rows, output / "contact_sheet.png"
        )
        report = {
            "version": "candidate_count_similarity_benchmark_v1",
            "source_bundle": str(args.bundle.resolve()),
            "source_approval_fingerprint": source_approval.fingerprint,
            "benchmark_approval_fingerprint": benchmark_approval.fingerprint,
            "base_seed": request.seed,
            "candidate_count": args.candidate_count,
            "seed_rule": "(base_seed + candidate_number - 1) modulo 2**32",
            "model_id": request.model_id,
            "elapsed_seconds": elapsed,
            "gpu": environment,
            "generated_directory": str(generated_directory),
            "candidate_rows": rows,
            "prefix_summaries": summaries,
            "interpretation": {
                "scores_are_probabilities": False,
                "single_sequence_estimates_success_probability": False,
                "ranking": "maximin(character,hair), then mean(character,hair)",
                "hard_safety_only": True,
                "native_refinement_executed": False,
            },
        }
        (output / "benchmark.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    finally:
        if batch is not None:
            batch.close()
        if inputs is not None:
            inputs.close()
        if request is not None:
            request.reference_image.close()
        if (
            pipeline_object is not None
            and hasattr(pipeline_object, "maybe_free_model_hooks")
        ):
            pipeline_object.maybe_free_model_hooks()
        pipeline_object = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        run_log.close()


if __name__ == "__main__":
    raise SystemExit(main())

