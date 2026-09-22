"""Rebuild current reference analysis and run an Animagine Base GPU regression."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

from PIL import Image, ImageOps
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from genai_lab.body_proportion_presets import active_body_proportion_preset_id
from genai_lab.generation_orchestrator import GenerationOrchestrator
from genai_lab.generation_replay import load_generation_replay_bundle
from genai_lab.model import prepare_pipeline
from genai_lab.native_pipeline_contract import prepare_character_only_base_request
from genai_lab.reference_experiment_approval import approve_reference_experiment
from genai_lab.run_log import create_generation_run_log
from run import configure_console_encoding, configure_system_certificates, validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--character-image", type=Path, required=True)
    parser.add_argument("--garment-image", type=Path, required=True)
    parser.add_argument("--garment-root", type=Path, required=True)
    parser.add_argument("--character-tag", action="append", default=[])
    parser.add_argument("--garment-tag", action="append", default=[])
    parser.add_argument("--garment-detail-tag", action="append", default=[])
    parser.add_argument(
        "--character-gender",
        choices=("male", "female", "unspecified"),
        default="unspecified",
    )
    parser.add_argument("--body-proportion-preset-id")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_input_roles(
    character_image: Path,
    garment_image: Path,
    garment_root: Path,
) -> dict[str, str]:
    character = character_image.resolve(strict=True)
    garment = garment_image.resolve(strict=True)
    root = garment_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"garment root is not a directory: {root}")
    if character.is_relative_to(root):
        raise ValueError("character reference must not come from garment root")
    if not garment.is_relative_to(root):
        raise ValueError("garment reference must come from garment root")
    character_sha256 = _sha256(character)
    garment_sha256 = _sha256(garment)
    if character_sha256 == garment_sha256:
        raise ValueError("the same image cannot serve as character and garment")
    return {
        "character_role": "character_reference_only",
        "character_path": str(character),
        "character_sha256": character_sha256,
        "garment_role": "garment_reference_only",
        "garment_path": str(garment),
        "garment_sha256": garment_sha256,
        "garment_root": str(root),
    }


def normalize_character_canvas(image: Image.Image, multiple: int = 8):
    if type(multiple) is not int or multiple < 1:
        raise ValueError("canvas multiple must be a positive integer")
    target_width = ((image.width + multiple - 1) // multiple) * multiple
    target_height = ((image.height + multiple - 1) // multiple) * multiple
    left = (target_width - image.width) // 2
    top = (target_height - image.height) // 2
    right = target_width - image.width - left
    bottom = target_height - image.height - top
    fill = image.getpixel((0, 0))
    normalized = ImageOps.expand(
        image, border=(left, top, right, bottom), fill=fill)
    return normalized, {
        "version": "character_canvas_multiple_v1",
        "original_size": [image.width, image.height],
        "normalized_size": [target_width, target_height],
        "padding_ltrb": [left, top, right, bottom],
        "multiple": multiple,
        "content_resized": False,
        "fill_source": "top_left_pixel",
    }
CURRENT_ANALYSIS_POLICY_KEYS = (
    "reference_face_observation",
    "reference_hair_observation",
    "reference_base_mask_validation",
)


def refresh_current_analysis_policy(config: dict) -> dict:
    # Refresh only analysis gates changed after the replay was approved.
    current_path = PROJECT_ROOT / "configs" / "animagine.yaml"
    current = yaml.safe_load(current_path.read_text(encoding="utf-8"))
    refreshed = {}
    for key in CURRENT_ANALYSIS_POLICY_KEYS:
        if key not in current:
            raise ValueError(f"current analysis policy is missing: {key}")
        config[key] = deepcopy(current[key])
        refreshed[key] = deepcopy(current[key])
    current_parts = current.get("reference_analysis", {}).get(
        "part_detection", {}
    )
    if "maximum_foreground_area_ratios" not in current_parts:
        raise ValueError(
            "current analysis policy is missing part area upper bounds"
        )
    config.setdefault("reference_analysis", {}).setdefault(
        "part_detection", {}
    )["maximum_foreground_area_ratios"] = deepcopy(
        current_parts["maximum_foreground_area_ratios"]
    )
    refreshed["reference_analysis.part_detection."
              "maximum_foreground_area_ratios"] = deepcopy(
        current_parts["maximum_foreground_area_ratios"]
    )
    return {
        "version": "current_analysis_policy_refresh_v1",
        "source": str(current_path),
        "refreshed": refreshed,
        "model_seed_and_input_contract_changed": False,
    }


INPUT_BOUND_APPROVAL_KEYS = (
    "approved_tags",
    "approved_detail_tags",
    "approved_garment_topology",
    "garment_detail_report",
    "eye_color_report",
    "hair_detail_report",
    "hair_prompt_delivery",
    "part_color_descriptions",
    "approved_part_color_descriptions",
)


def reset_input_bound_approval_state(
    section: dict,
    *,
    character_tags: tuple[str, ...],
    character_gender: str,
    garment_name: str,
    garment_tags: tuple[str, ...] = (),
    garment_detail_tags: tuple[str, ...] = (),
) -> dict:
    """Replace replay-bound evidence with approval for the current inputs.

    Garment tags remain available to the post-Base refinement contract. The
    character-only Base profile still excludes them from both its prompt and
    image-adapter inputs.
    """
    for key in INPUT_BOUND_APPROVAL_KEYS:
        section[key] = (
            ()
            if key in {
                "approved_tags",
                "approved_detail_tags",
                "part_color_descriptions",
                "approved_part_color_descriptions",
            }
            else None
        )
    section["approved_tags"] = garment_tags
    section["approved_detail_tags"] = garment_detail_tags
    section["approved_character_tags"] = character_tags
    section["character_gender"] = character_gender
    section["source_name"] = garment_name
    section["character_feature_routing"] = {
        "selected": character_tags,
        "excluded": (),
        "unresolved": (),
        "eye_status": "not_evaluated",
        "eye_reasons": (),
        "policy": "CLI explicit current-input approval; inherited semantic state removed",
    }
    return {
        "version": "input_bound_approval_reset_v1",
        "cleared_keys": list(INPUT_BOUND_APPROVAL_KEYS),
        "character_tags": list(character_tags),
        "character_gender": character_gender,
        "garment_name": garment_name,
        "garment_tags": list(garment_tags),
        "garment_detail_tags": list(garment_detail_tags),
    }


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return str(value)


def main() -> int:
    configure_console_encoding()
    configure_system_certificates()
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    old_inputs = inputs = batch = pipeline = request = None
    old_request = None
    source = garment = None
    log = create_generation_run_log(PROJECT_ROOT)
    started = perf_counter()
    report_path = output / "integration_report.json"
    try:
        role_validation = validate_input_roles(
            args.character_image,
            args.garment_image,
            args.garment_root,
        )
        # This bundle supplies only immutable model/config/request defaults.
        # The current character and garment are re-analysed and receive a new
        # approval below, so an older semantic approval must not be inherited.
        old_inputs, config, old_request, _ = load_generation_replay_bundle(
            args.source_bundle.resolve(), verify_approved_run=False
        )
        policy_refresh = refresh_current_analysis_policy(config)
        config["paths"] = dict(config["paths"], output_dir=str(output))
        section = config["clothing_reference_generation"]
        fresh_character_tags = tuple(args.character_tag)
        fresh_garment_tags = tuple(args.garment_tag)
        fresh_garment_detail_tags = tuple(args.garment_detail_tag)
        input_bound_reset = reset_input_bound_approval_state(
            section,
            character_tags=fresh_character_tags,
            character_gender=args.character_gender,
            garment_name=args.garment_image.name,
            garment_tags=fresh_garment_tags,
            garment_detail_tags=fresh_garment_detail_tags,
        )
        validate_config(config)

        with Image.open(args.character_image.resolve()) as opened:
            loaded_source = opened.convert("RGB")
        try:
            source, canvas_normalization = normalize_character_canvas(
                loaded_source)
        finally:
            loaded_source.close()
        role_validation["character_canvas"] = canvas_normalization
        with Image.open(args.garment_image.resolve()) as opened:
            garment = opened.convert("RGB")

        request_template = replace(
            old_request,
            reference_image=source.copy(),
            reference_image_name=args.character_image.name,
            width=source.width,
            height=source.height,
            body_proportion_preset_id=args.body_proportion_preset_id,
        )
        preset_id = active_body_proportion_preset_id(request_template)
        pipeline = (
            prepare_pipeline(config, body_proportion_preset_id=preset_id)
            if preset_id is not None
            else prepare_pipeline(config)
        )
        request, prompt_record = prepare_character_only_base_request(
            request_template,
            (pipeline.tokenizer, pipeline.tokenizer_2),
            character_tags=fresh_character_tags,
            character_gender=args.character_gender,
        )
        section["approved_prompt_pair"] = (
            request.prompt,
            request.negative_prompt,
        )
        section["approved_prompt_record"] = prompt_record

        orchestrator = GenerationOrchestrator(
            config,
            request,
            PROJECT_ROOT,
            log,
            cancelled=lambda: False,
            status_callback=lambda message: print(message, flush=True),
        )
        inputs = orchestrator.prepare_visual_inputs(source, garment)
        section['part_color_descriptions'] = inputs.part_color_descriptions
        section['approved_part_color_descriptions'] = ()
        approve_reference_experiment(inputs)
        approval_fingerprint = orchestrator.approve_visual_inputs(inputs)
        batch = orchestrator.generate_base_candidates(pipeline, inputs)

        preserved = output / "base_candidates"
        if preserved.exists():
            raise FileExistsError(preserved)
        shutil.copytree(batch.directory, preserved)
        candidate_paths = [preserved / Path(path).name for path in batch.paths]
        candidate_records = []
        for path in candidate_paths:
            record_path = path.with_suffix(".json")
            candidate_records.append(
                json.loads(record_path.read_text(encoding="utf-8"))
                if record_path.is_file()
                else {"missing_record": str(record_path)}
            )

        analysis = inputs.analysis_record or {}
        output_region_records = []
        for path in sorted(preserved.glob("output_regions/*/regions.json")):
            output_region_records.append(
                json.loads(path.read_text(encoding="utf-8"))
            )
        report = {
            "version": "contract_scope_gpu_regression_v1",
            "status": "completed_base_only",
            "elapsed_seconds": round(perf_counter() - started, 3),
            "source_bundle": str(args.source_bundle.resolve()),
            "source_bundle_usage": "template_only_new_inputs_reapproved",
            "runtime_analysis_policy_refresh": policy_refresh,
            "input_bound_approval_reset": input_bound_reset,
            "character_image": str(args.character_image.resolve()),
            "garment_image": str(args.garment_image.resolve()),
            "input_role_validation": role_validation,
            "character_tags": list(fresh_character_tags),
            "character_gender": args.character_gender,
            "garment_tags": list(fresh_garment_tags),
            "garment_detail_tags": list(fresh_garment_detail_tags),
            "seed": request.seed,
            "body_proportion_preset_id": request.body_proportion_preset_id,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "prompt_record": prompt_record,
            "prompt_contains_literal_none": any(
                part.strip().lower() == "none"
                for part in request.prompt.split(",")
            ),
            "approval_fingerprint": approval_fingerprint,
            "replay_bundle": str(orchestrator.replay_bundle),
            "visual_input_analysis": analysis,
            "candidate_count": len(candidate_paths),
            "candidate_paths": [str(path) for path in candidate_paths],
            "candidate_records": candidate_records,
            "output_region_records": output_region_records,
            "warning": batch.warning,
            "review_stage": batch.review_stage,
            "orchestrator_trace": orchestrator.trace,
        }
        report_path.write_text(
            json.dumps(_json_safe(report), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "status": report["status"],
            "report": str(report_path),
            "candidate_paths": report["candidate_paths"],
            "prompt_contains_literal_none": report[
                "prompt_contains_literal_none"
            ],
        }, ensure_ascii=False, indent=2), flush=True)
        return 0
    except BaseException as error:
        failure = {
            "version": "contract_scope_gpu_regression_v1",
            "status": "failed",
            "elapsed_seconds": round(perf_counter() - started, 3),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        report_path.write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        if batch is not None:
            batch.close()
        if inputs is not None:
            inputs.close()
        if old_inputs is not None:
            old_inputs.close()
        if source is not None:
            source.close()
        if garment is not None:
            garment.close()
        if request is not None and request is not old_request:
            request.reference_image.close()
        if old_request is not None:
            old_request.reference_image.close()
        if pipeline is not None and hasattr(pipeline, "maybe_free_model_hooks"):
            pipeline.maybe_free_model_hooks()
        pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())


