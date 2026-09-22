"""승인된 GUI 재생 번들로 동일한 제품 생성 파이프라인을 실행한다."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import traceback

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from genai_lab.generation_orchestrator import GenerationOrchestrator
from genai_lab.generation_replay import load_generation_replay_bundle
from genai_lab.model import prepare_pipeline
from genai_lab.body_proportion_presets import active_body_proportion_preset_id
from genai_lab.native_pipeline_contract import native_direct_enabled
from genai_lab.run_log import create_generation_run_log
from run import (
    check_environment,
    configure_console_encoding,
    configure_system_certificates,
    validate_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GUI가 승인한 번들을 공통 GenerationOrchestrator로 재실행합니다."
        )
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--selected-candidate",
        type=int,
        help="GUI와 같은 1부터 시작하는 Base 후보 번호",
    )
    parser.add_argument(
        "--native-output-dir",
        type=Path,
        help="Native 결과 폴더. 생략하면 프로젝트 outputs 아래에 생성합니다.",
    )
    return parser.parse_args()


def main() -> int:
    configure_console_encoding()
    args = parse_args()
    inputs = request = batch = pipeline = None
    run_log = create_generation_run_log(PROJECT_ROOT)
    try:
        inputs, config, request, approval = load_generation_replay_bundle(
            args.bundle)
        validate_config(config)
        run_log.write_stage(
            "Codex 제품 재생",
            f"번들={args.bundle}, 승인={approval.fingerprint}",
        )
        configure_system_certificates()
        environment = check_environment()
        run_log.write_stage(
            "환경 검사",
            f"GPU={environment['gpu']}, "
            f"GPU 메모리={environment['vram_bytes'] / 1024**3:.1f}GB",
        )
        direct_route = native_direct_enabled(config)
        if not direct_route:
            body_preset_id = active_body_proportion_preset_id(request)
            pipeline = (
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

        if direct_route:
            selection = orchestrator.select_direct_source(inputs)
            result = orchestrator.finalize_selected_candidate(
                selection,
                output_directory=args.native_output_dir,
                status_callback=lambda message: print(message, flush=True),
            )
            summary = {
                "execution_scope": "product_pipeline",
                "runtime_smoke": False,
                "approval_fingerprint": approval.fingerprint,
                "route": "source_character_direct",
                "input_directory": str(selection.batch_directory),
                "candidate_count": len((getattr(result, "report", {}) or {}).get("attempts", {})),
                "selected_candidate": result.selected_engine,
                "status": "FINAL_GATE_PASS",
                "native_status": result.status,
                "selected_engine": result.selected_engine,
                "selected_image": str(result.selected_image_path),
                "native_report": str(result.report_path),
                "final_return_eligible": True,
                "user_approval_status": "pending",
                "final_review_evidence": (getattr(result, "report", {}) or {}).get(
                    "final_review", {}
                ).get("evidence_path"),
                "image_merge_used": False,
            }
            summary_directory = selection.batch_directory
        else:
            batch = orchestrator.generate_base_candidates(pipeline, inputs)
            summary = {
                "execution_scope": "product_pipeline",
                "runtime_smoke": False,
                "approval_fingerprint": approval.fingerprint,
                "route": "approved_animagine_base",
                "base_directory": str(batch.directory),
                "base_review_stage": batch.review_stage,
                "candidate_count": len(batch.candidates),
                "selected_candidate": args.selected_candidate,
                "status": "BASE_READY_FOR_SELECTION",
                "final_return_eligible": False,
            }
            if args.selected_candidate is not None:
                index = args.selected_candidate - 1
                selection = orchestrator.select_base_candidate(batch, index)
                result = orchestrator.finalize_selected_candidate(
                    selection,
                    output_directory=args.native_output_dir,
                    status_callback=lambda message: print(message, flush=True),
                )
                summary.update({
                    "status": (
                        "FINAL_GATE_PASS"
                        if result.status == "PASS"
                        else "FINAL_BASE_LOCKED"
                    ),
                    "native_status": result.status,
                    "selected_engine": result.selected_engine,
                    "selected_image": str(result.selected_image_path),
                    "native_report": str(result.report_path),
                    "final_return_eligible": True,
                    "user_approval_status": "pending",
                    "final_review_evidence": (getattr(result, "report", {}) or {}).get(
                        "final_review", {}
                    ).get("evidence_path"),
                })
            summary_directory = Path(batch.directory)
        summary_path = Path(summary_directory) / "codex_product_result.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        run_log.write_failure(
            "Codex 제품 재생",
            error,
            "재생 번들 무결성, 모델 경로, GPU 상태와 전체 추적 정보를 확인하세요.",
        )
        print(traceback.format_exc(), file=sys.stderr)
        return 2
    finally:
        if batch is not None:
            batch.close()
        if inputs is not None:
            inputs.close()
        if request is not None:
            request.reference_image.close()
        if pipeline is not None and hasattr(pipeline, "maybe_free_model_hooks"):
            pipeline.maybe_free_model_hooks()
        pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        run_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
