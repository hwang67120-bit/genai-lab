"""Run one FLUX.2 Klein native full-image refinement candidate.

No image merge, hard paste, regional paste, or candidate transfer is used.
The approved character input and isolated garment board are the only images.

Official model page:
https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from typing import Any

import numpy as np
from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from genai_lab.native_refinement_policy import (
    ENGINE_NAME,
    decide_refinement,
    gate_native_candidate,
    validate_approved_base,
    validate_approved_source,
)

ENGINE_FILES = {
    "flux2_klein": (
        PROJECT_ROOT / "scripts" / "flux2_klein_smoke.py",
        "flux2-klein-smoke.json",
        "flux2-klein-smoke.png",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-image", type=Path, required=True)
    parser.add_argument(
        "--input-mode",
        choices=("approved_base", "source_character_direct"),
        default="approved_base",
    )
    parser.add_argument("--candidate-record", type=Path, required=True)
    parser.add_argument("--approved-run", type=Path, required=True)
    parser.add_argument("--garment-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "animagine.yaml",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("D:/genai-cache/venv/Scripts/python.exe"),
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "huggingface",
    )
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--flux-steps", type=int, default=4)
    parser.add_argument("--minimum-similarity", type=float, default=0.65)
    parser.add_argument("--minimum-color-similarity", type=float, default=0.55)
    parser.add_argument("--instruction")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def common_instruction(
    candidate_record: dict[str, Any],
    approved_run: dict[str, Any],
    override: str | None,
) -> str:
    del candidate_record
    if override:
        return override.strip()
    from genai_lab.native_pipeline_contract import (
        build_native_refinement_instruction,
    )
    instruction, _ = build_native_refinement_instruction(approved_run)
    return instruction

def engine_prompt(
    engine: str,
    instruction: str,
    input_mode: str = "approved_base",
) -> str:
    first_authority = (
        "approved original character reference"
        if input_mode == "source_character_direct"
        else "approved Animagine Base"
    )
    return (
        f"Image 1 is the {first_authority}. "
        "Image 2 is the isolated garment-only board. "
        f"{instruction}"
    )


def decision_for_input_mode(
    decision: dict[str, Any],
    input_mode: str,
) -> dict[str, Any]:
    if (
        input_mode == "source_character_direct"
        and decision.get("action") == "keep_approved_base_and_seed"
    ):
        return {
            **decision,
            "status": "SOURCE_LOCKED",
            "action": "require_explicit_animagine_fallback",
        }
    return decision


def engine_garment_reference(
    engine: str,
    args: argparse.Namespace,
) -> Path:
    if engine != ENGINE_NAME:
        raise ValueError(f"unsupported native engine: {engine}")
    return args.garment_reference


def run_engine(
    engine: str,
    *,
    args: argparse.Namespace,
    contract: dict[str, Any],
    prompt: str,
) -> dict[str, Any]:
    script, report_name, image_name = ENGINE_FILES[engine]
    directory = args.output_dir / engine
    directory.mkdir(parents=True, exist_ok=True)
    if engine != ENGINE_NAME:
        raise ValueError(f"unsupported native engine: {engine}")
    steps = args.flux_steps
    garment_reference = engine_garment_reference(engine, args)
    command = [
        str(args.python),
        "-u",
        str(script),
        "--character",
        str(args.base_image),
        "--garment",
        str(garment_reference),
        "--output-dir",
        str(directory),
        "--cache-dir",
        str(args.model_cache_dir),
        "--prompt",
        engine_prompt(engine, prompt, args.input_mode),
        "--seed",
        str(contract["seed"]),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--steps",
        str(steps),
    ]
    if not args.allow_download:
        command.append("--local-files-only")
    environment = os.environ.copy()
    if not args.allow_download:
        environment.update({
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DIFFUSERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        })
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    (directory / "stdout.log").write_text(
        completed.stdout or "", encoding="utf-8"
    )
    (directory / "stderr.log").write_text(
        completed.stderr or "", encoding="utf-8"
    )
    report_path = directory / report_name
    smoke = read_json(report_path) if report_path.is_file() else {
        "status": "failed",
        "error": "engine report was not written",
    }
    output = directory / image_name
    return {
        "engine": engine,
        "process_return_code": completed.returncode,
        "smoke_report": smoke,
        "output": str(output.resolve()) if output.is_file() else None,
        "native_full_image": True,
        "received_primary_character": str(args.base_image.resolve()),
        "received_garment_reference": str(garment_reference.resolve()),
        "received_garment_layout": "component_grid",
        "received_input_mode": args.input_mode,
        "received_other_engine_output": False,
        "pixel_merge_used": False,
    }


class VisionEncoder:
    def __init__(self, cache_dir: Path, local_files_only: bool):
        import torch
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = "h94/IP-Adapter:models/image_encoder"
        self.processor = CLIPImageProcessor()
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            cache_dir=str(cache_dir),
            torch_dtype=(
                torch.float16 if self.device.type == "cuda" else torch.float32
            ),
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()

    def feature(self, image: Image.Image) -> np.ndarray:
        values = self.processor(
            images=image.convert("RGB"), return_tensors="pt"
        ).pixel_values.to(
            self.device, dtype=next(self.model.parameters()).dtype
        )
        with self.torch.inference_mode():
            feature = (
                self.model(values).image_embeds.detach().float().cpu().numpy()
                .reshape(-1)
            )
        norm = np.linalg.norm(feature)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("invalid CLIP image feature")
        return feature / norm

    def close(self) -> None:
        model = self.model
        self.model = None
        del model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


def color_histogram(image: Image.Image) -> np.ndarray:
    import cv2

    rgb = np.asarray(image.convert("RGB"))
    keep = np.any(rgb < 245, axis=-1)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    values = hsv[keep]
    if not len(values):
        values = hsv.reshape(-1, 3)
    histogram = cv2.calcHist(
        [values.reshape(-1, 1, 3)],
        [0, 1],
        None,
        [36, 32],
        [0, 180, 0, 256],
    )
    cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
    return histogram.reshape(-1)


def color_similarity(left: np.ndarray, right: np.ndarray) -> float:
    import cv2

    distance = cv2.compareHist(
        left.astype(np.float32),
        right.astype(np.float32),
        cv2.HISTCMP_BHATTACHARYYA,
    )
    return float(np.clip(1.0 - distance, 0.0, 1.0))


def evaluate_candidate(
    engine: str,
    image_path: Path,
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    candidate_record: dict[str, Any],
    approved_run: dict[str, Any],
) -> dict[str, Any]:
    from genai_lab.candidate_semantic_gate import (
        CandidateSemanticAnalyzer,
        compare_lower_body_transition,
        resolve_candidate_semantic_gate,
    )
    from genai_lab.candidate_structure_gate import (
        evaluate_candidate_structure,
        resolve_candidate_structure_gate,
    )
    from genai_lab.generated_image_integrity import (
        analyze_generated_image_integrity,
        resolve_generated_image_integrity,
    )
    from genai_lab.visual_reference import (
        detect_output_similarity_regions,
        masked_crop,
    )

    semantic_analyzer = CandidateSemanticAnalyzer(config)
    encoder_cache = Path(
        config.get("garment_inpaint", {}).get(
            "cache_dir", args.model_cache_dir
        )
    )
    encoder = VisionEncoder(encoder_cache, not args.allow_download)
    base_regions = None
    output_regions = None
    with Image.open(args.base_image) as source:
        base = source.convert("RGB").copy()
    with Image.open(args.garment_reference) as source:
        garment_reference = source.convert("RGB").copy()
    with Image.open(image_path) as source:
        candidate = source.convert("RGB").copy()
    try:
        semantic_settings = resolve_candidate_semantic_gate(config)
        base_semantic = semantic_analyzer.analyze(
            base,
            approved_run,
            semantic_settings,
            stage="final",
        )
        semantic = semantic_analyzer.analyze(
            candidate,
            approved_run,
            semantic_settings,
            stage="final",
        )
        lower_body_transition = compare_lower_body_transition(
            base_semantic, semantic
        )
        structure = evaluate_candidate_structure(
            candidate, resolve_candidate_structure_gate(config)
        )
        integrity = analyze_generated_image_integrity(
            candidate, resolve_generated_image_integrity(config)
        )

        base_regions, base_directory = detect_output_similarity_regions(
            base,
            config,
            PROJECT_ROOT,
            args.output_dir,
            int(candidate_record["candidate_number"]),
            suffix="native_approved_base",
        )
        output_regions, output_directory = detect_output_similarity_regions(
            candidate,
            config,
            PROJECT_ROOT,
            args.output_dir,
            int(candidate_record["candidate_number"]),
            suffix=engine,
        )

        reference_features: dict[str, np.ndarray] = {}
        for name in ("identity", "hair", "foreground"):
            mask = base_regions.masks.get(name)
            if mask is None:
                raise ValueError(f"approved Base region unavailable: {name}")
            with masked_crop(base, mask) as crop:
                reference_features[name] = encoder.feature(crop)
        reference_features["garment"] = encoder.feature(garment_reference)

        scores: dict[str, float | None] = {}
        for score_name, mask_name, reference_name in (
            ("character", "identity", "identity"),
            ("hair", "hair", "hair"),
            ("garment", "garment", "garment"),
            ("vibe", "foreground", "foreground"),
        ):
            mask = output_regions.masks.get(mask_name)
            if mask is None:
                scores[score_name] = None
                continue
            with masked_crop(candidate, mask) as crop:
                scores[score_name] = cosine(
                    reference_features[reference_name], encoder.feature(crop)
                )

        base_garment_mask = base_regions.masks.get("garment")
        output_garment_mask = output_regions.masks.get("garment")
        if base_garment_mask is None or output_garment_mask is None:
            raise ValueError("garment region unavailable for final gate")
        with masked_crop(base, base_garment_mask) as crop:
            garment_baseline = cosine(
                reference_features["garment"], encoder.feature(crop)
            )
            color_baseline = color_similarity(
                color_histogram(garment_reference), color_histogram(crop)
            )
        with masked_crop(candidate, output_garment_mask) as crop:
            candidate_color = color_similarity(
                color_histogram(garment_reference), color_histogram(crop)
            )

        gate = gate_native_candidate(
            semantic_report=semantic,
            structure_report=structure,
            integrity_report=integrity,
            similarity=scores,
            minimum_similarity=args.minimum_similarity,
            color_similarity=candidate_color,
            minimum_color_similarity=args.minimum_color_similarity,
            garment_baseline=garment_baseline,
            color_baseline=color_baseline,
            return_policy=str(
                config.get("candidate_pipeline", {}).get(
                    "return_policy", "strict_all"
                )
            ),
        )
        return {
            "semantic": semantic,
            "base_semantic": base_semantic,
            "lower_body_transition": lower_body_transition,
            "structure": structure,
            "integrity": integrity,
            "similarity": scores,
            "similarity_encoder": encoder.model_id,
            "color": {
                "baseline": color_baseline,
                "result": candidate_color,
                "minimum": args.minimum_color_similarity,
            },
            "base_regions_directory": str(base_directory),
            "output_regions_directory": str(output_directory),
            "gate": gate,
            "native_full_image": True,
            "pixel_merge_used": False,
        }
    finally:
        if base_regions is not None:
            base_regions.close()
        if output_regions is not None:
            output_regions.close()
        base.close()
        garment_reference.close()
        candidate.close()
        semantic_analyzer.close()
        encoder.close()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.allow_download:
        os.environ.update({
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DIFFUSERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        })
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    candidate_record = read_json(args.candidate_record)
    approved_run = read_json(args.approved_run)
    from genai_lab.native_pipeline_contract import (
        build_native_refinement_instruction,
        require_isolated_garment_board,
    )
    args.garment_reference = require_isolated_garment_board(
        args.garment_reference, approved_run
    )
    return_policy = str(
        config.get("candidate_pipeline", {}).get(
            "return_policy", "strict_all"
        )
    )
    if args.input_mode == "source_character_direct":
        contract = validate_approved_source(
            args.base_image,
            candidate_record,
            approved_run,
            return_policy=return_policy,
        )
    else:
        contract = validate_approved_base(
            args.base_image,
            candidate_record,
            approved_run,
            minimum_similarity=args.minimum_similarity,
            return_policy=return_policy,
        )
    prompt = common_instruction(candidate_record, approved_run, args.instruction)
    if args.instruction:
        prompt_contract = {
            "version": "native_pipeline_contract_v2",
            "override": True,
            "animagine_prompt_reused": False,
        }
    else:
        _, prompt_contract = build_native_refinement_instruction(approved_run)
    report: dict[str, Any] = {
        "version": "native_refinement_run_v1",
        "status": "running",
        "input_mode": args.input_mode,
        "animagine_base_used": args.input_mode == "approved_base",
        "started_at": utc_now(),
        "contract": contract,
        "pipeline_contract": prompt_contract,
        "garment_references": {
            "flux2_klein": {
                "path": str(args.garment_reference.resolve()),
                "kind": "component_grid",
            },
        },
        "strategy": "flux2_klein_only",
        "return_policy": return_policy,
        "image_merge_used": False,
        "attempts": {},
    }
    write_json(args.output_dir / "run.json", report)

    try:
        for engine in (ENGINE_NAME,):
            decision = decide_refinement(report["attempts"])
            if decision["action"] != f"run_{engine}":
                break
            attempt = run_engine(
                engine,
                args=args,
                contract=contract,
                prompt=prompt,
            )
            output = attempt.get("output")
            if (
                attempt.get("smoke_report", {}).get("status") == "completed"
                and output
            ):
                try:
                    attempt.update(evaluate_candidate(
                        engine,
                        Path(output),
                        args=args,
                        config=config,
                        candidate_record=candidate_record,
                        approved_run=approved_run,
                    ))
                except Exception as error:
                    attempt["gate"] = {
                        "status": "FAIL",
                        "reason": "evaluation_error",
                    }
                    attempt["evaluation_error"] = {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    }
            else:
                attempt["gate"] = {
                    "status": "FAIL",
                    "reason": "generation_failed",
                }
            report["attempts"][engine] = attempt
            write_json(args.output_dir / engine / "native-gate.json", attempt)
            write_json(args.output_dir / "run.json", report)
            next_decision = decide_refinement(report["attempts"])
            if next_decision["action"] == "select_existing_candidate":
                break

        decision = decision_for_input_mode(
            decide_refinement(report["attempts"]),
            args.input_mode,
        )
        report["decision"] = decision
        report["finished_at"] = utc_now()
        if decision["action"] == "select_existing_candidate":
            engine = decision["selected_engine"]
            report["status"] = "PASS"
            report["selected_output"] = report["attempts"][engine]["output"]
            write_json(
                args.output_dir / "selected.json",
                {
                    "engine": engine,
                    "path": report["selected_output"],
                    "image_merge_used": False,
                    "refinement_required": decision.get(
                        "refinement_required", False),
                    "overall_similarity_percentage": decision.get(
                        "overall_similarity_percentage"),
                    "refinement_targets": decision.get(
                        "refinement_targets", []),
                },
            )
        else:
            direct = args.input_mode == "source_character_direct"
            report["status"] = "SOURCE_LOCKED" if direct else "BASE_LOCKED"
            lock_name = (
                "approved-source-lock.json"
                if direct else "approved-base-lock.json"
            )
            write_json(
                args.output_dir / lock_name,
                {
                    "input_mode": args.input_mode,
                    "base_image": contract["base_image"],
                    "base_sha256": contract["base_sha256"],
                    "seed": contract["seed"],
                    "approval_fingerprint": contract["approval_fingerprint"],
                    "reason": decision["reason"],
                    "failed_candidate_returned": False,
                    "requires_explicit_animagine_fallback": direct,
                },
            )
        write_json(args.output_dir / "run.json", report)
        print(args.output_dir / "run.json", flush=True)
        return 0 if report["status"] == "PASS" else 2
    except BaseException as error:
        report["status"] = "ERROR"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        report["finished_at"] = utc_now()
        write_json(args.output_dir / "run.json", report)
        print(args.output_dir / "run.json", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

