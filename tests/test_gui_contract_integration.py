"""GPU 없이 GUI 전체 계약 연결을 검증한다.

무거운 분석기와 이미지 생성기만 즉시 완료 대역으로 바꾼다. GUI의 실제
상태 전이, Base 선택, Native 인계, 최종 후보 표시는 제품 코드를 그대로
통과한다.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtWidgets import QApplication, QDialog
import pytest

import gui_main
from genai_lab.external_candidate import load_external_character_candidate
from genai_lab.generation_orchestrator import GenerationPhase
from genai_lab.native_pipeline_contract import (
    native_refinement_enabled,
    validate_native_pipeline_config,
)
from genai_lab.native_refinement_execution import NativeRefinementExecutionResult
from genai_lab.visual_reference import CandidateBatch
from genai_lab.workflow import GenerationWorkflowStage
from gui_main import GenAILabWindow
from run import load_yaml, validate_config


@pytest.mark.parametrize(
    ("native_status", "selected_engine"),
    (("PASS", "flux2_klein"), ("BASE_LOCKED", None)),
)
def test_gui_full_contract_routes_approved_inputs_to_final_review(
    monkeypatch,
    tmp_path: Path,
    native_status: str,
    selected_engine: str | None,
) -> None:
    """입력 등록부터 Native 성공·Base 고정 표시까지 GUI 전체를 검증한다."""
    application = QApplication.instance() or QApplication([])
    project_root = Path(gui_main.__file__).resolve().parent
    config = load_yaml(project_root / "configs" / "animagine.yaml")
    config["generation_run_context"]["root"] = str(tmp_path / "runs")
    validate_config(config)
    validate_native_pipeline_config(config)
    assert native_refinement_enabled(config)

    character_path = tmp_path / "character.png"
    outfit_path = tmp_path / "outfit.png"
    base_path = tmp_path / "candidate_1.png"
    refined_path = tmp_path / "flux2_klein.png"
    garment_board_path = tmp_path / "input_garment.png"
    candidate_record_path = base_path.with_suffix(".json")
    approved_run_path = tmp_path / "approved_generation.json"
    report_path = tmp_path / "native_refinement_report.json"
    for path, color in (
        (character_path, "lightblue"),
        (outfit_path, "navy"),
        (base_path, "slateblue"),
        (refined_path, "royalblue"),
        (garment_board_path, "navy"),
    ):
        Image.new("RGB", (48, 80), color).save(path)
    candidate_record_path.write_text("{}", encoding="utf-8")
    approved_run_path.write_text("{}", encoding="utf-8")
    report_path.write_text("{}", encoding="utf-8")

    window = GenAILabWindow()
    window.body_proportion_combo.setCurrentIndex(
        window.body_proportion_combo.findData("standard_7_5h_shoulder")
    )
    window.config = config
    window.style_path = str(character_path)
    window.selected_outfit_path = outfit_path
    contract_before_gui_routing = deepcopy({
        "native_pipeline_v2": config["native_pipeline_v2"],
        "native_refinement": config["native_refinement"],
    })
    trace: list[str] = []
    routed_paths: dict[str, Path] = {}

    def complete_reference(path: Path) -> None:
        assert path == character_path
        trace.append("reference_approved")
        window.approved_reference_image = SimpleNamespace(
            image=Image.new("RGB", (48, 80), "lightblue"),
            enhancement_applied=False,
            enhancement_model_id=None,
            quality_status=SimpleNamespace(value="good"),
        )
        window.advance_generation_workflow()

    def complete_outfit_mask(path: Path) -> None:
        assert path == outfit_path
        trace.append("garment_mask_approved")
        window.outfit_path = str(path)
        window.pending_clothing_extraction = SimpleNamespace(
            extracted_image=Image.new("RGB", (48, 80), "navy"),
            clothing_mask=Image.new("L", (48, 80), 255),
        )
        window.advance_generation_workflow()

    def complete_outfit_analysis() -> None:
        trace.append("garment_analysis_approved")
        window.confirmed_clothing_design = SimpleNamespace(
            design_tags=("navy jacket", "mini skirt"),
        )
        window.advance_generation_workflow()

    def complete_base_generation(thread=None) -> None:
        trace.append("base_generated")
        assert window.worker is not None
        prepared = window.worker.config["clothing_reference_generation"]
        assert prepared["garment_reference_scale"] == 0.0
        assert prepared["native_base_profile"] == "character_only"
        loaded = load_external_character_candidate(
            base_path,
            reference_image_name=character_path.name,
            clothing_reference_name=outfit_path.name,
            framing_type="full_body",
        )
        thumbnail = loaded.image.copy()
        thumbnail.thumbnail((24, 40))
        loaded.image.close()
        candidate = replace(
            loaded,
            image=thumbnail,
            seed=123456,
            candidate_number=1,
            model_id="Animagine XL 4.0",
            detail_correction_status="native_base_gate_passed",
        )
        batch = CandidateBatch(
            candidates=[candidate],
            paths=[base_path],
            directory=tmp_path,
            review_stage="native_base",
        )
        window.worker.orchestrator.phase = GenerationPhase.BASE_COMPLETED
        window.worker.orchestrator._base_batch = batch
        window.worker.completed.emit(batch, None)

    def complete_native_refinement(
            base_candidate, *, orchestrator, selection) -> None:
        trace.append("native_handoff")
        assert orchestrator is window.worker.orchestrator
        routed_paths.update({
            "base_image": selection.base_image,
            "candidate_record": selection.candidate_record,
            "approved_run": selection.approved_run,
            "garment_reference": selection.garment_reference,
        })
        # GUI 라우팅이 백엔드 설정·게이트 임계값을 다시 쓰지 않았는지 검사한다.
        assert {
            "native_pipeline_v2": window.config["native_pipeline_v2"],
            "native_refinement": window.config[
                "native_refinement"
            ],
        } == contract_before_gui_routing
        window.pending_native_base_candidate = base_candidate
        window.native_refinement_completed(
            NativeRefinementExecutionResult(
                status=native_status,
                selected_engine=selected_engine,
                selected_image_path=(
                    refined_path if native_status == "PASS" else base_path
                ),
                report_path=report_path,
                output_directory=tmp_path,
                report={
                    "status": native_status,
                    "reason": (
                        "final_gate_passed"
                        if native_status == "PASS"
                        else "all_native_candidates_rejected"
                    ),
                },
            )
        )
        trace.append("final_review")

    selection = SimpleNamespace(currentData=lambda: 0)
    monkeypatch.setattr(
        gui_main,
        "VisualCandidateReview",
        lambda *args, **kwargs: SimpleNamespace(selection=selection),
    )
    def approve_dialog(dialog) -> int:
        if hasattr(dialog, "character_gender_input"):
            dialog.character_gender_input.setCurrentIndex(
                dialog.character_gender_input.findData("male")
            )
            dialog.candidate_count_input.setValue(2)
            dialog.identity_scale_input.setValue(0.85)
            dialog.garment_scale_input.setValue(0.30)
        return int(QDialog.DialogCode.Accepted)

    monkeypatch.setattr(window, "execute_approval_dialog", approve_dialog)
    monkeypatch.setattr(window, "start_reference_preparation", complete_reference)
    monkeypatch.setattr(
        window,
        "start_outfit_region_preparation",
        complete_outfit_mask,
    )
    monkeypatch.setattr(
        window,
        "start_clothing_design_analysis",
        complete_outfit_analysis,
    )
    run_log = SimpleNamespace(
        write_stage=lambda *args: None,
        write_failure=lambda *args: None,
        close=lambda: None,
        file_path=tmp_path / "gui-contract.log",
    )
    monkeypatch.setattr(gui_main, "create_generation_run_log", lambda *args: run_log)
    monkeypatch.setattr(gui_main, "load_yaml", lambda *args: deepcopy(config))
    monkeypatch.setattr(gui_main, "load_character_gender", lambda path: None)
    monkeypatch.setattr(gui_main, "save_character_gender", lambda *args: None)
    monkeypatch.setattr(gui_main.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(gui_main.QThread, "start", complete_base_generation)
    monkeypatch.setattr(
        window,
        "start_native_refinement",
        complete_native_refinement,
    )

    try:
        window.start_generation()

        assert trace == [
            "reference_approved",
            "garment_mask_approved",
            "garment_analysis_approved",
            "base_generated",
            "native_handoff",
            "final_review",
        ]
        assert window.workflow_context is not None
        assert window.workflow_context.character_image_path == character_path
        assert window.workflow_context.clothing_image_path == outfit_path
        assert window.workflow_context.pose_image_path is None
        assert (
            window.workflow_context.current_stage
            is GenerationWorkflowStage.FINAL_REVIEW
        )
        assert window.workflow_context.failed_stage is None
        assert routed_paths["base_image"].name == "approved-base.png"
        assert routed_paths["candidate_record"].name == "approved-base.json"
        assert routed_paths["approved_run"].name == "approved-generation.json"
        assert routed_paths["garment_reference"].name == "input_garment.png"
        assert routed_paths["base_image"].read_bytes() == base_path.read_bytes()
        assert (
            routed_paths["garment_reference"].read_bytes()
            == garment_board_path.read_bytes()
        )
        assert window.pending_character_candidate is not None
        assert window.pending_character_candidate.seed == 123456
        if native_status == "PASS":
            assert window.pending_character_candidate.model_id == "flux2_klein"
            assert (
                window.pending_character_candidate.detail_correction_status
                == "native_refinement_flux2_klein_final_gate_passed"
            )
            assert "최종 검토" in window.status_label.text()
        else:
            assert (
                window.pending_character_candidate.detail_correction_status
                == "native_refinement_base_locked"
            )
            assert "승인 Base·Seed 고정" in window.status_label.text()
        assert (
            window.pending_character_candidate.design_reference_record[
                "native_refinement"
            ]["image_merge_used"]
            is False
        )
        assert {
            "native_pipeline_v2": window.config["native_pipeline_v2"],
            "native_refinement": window.config[
                "native_refinement"
            ],
        } == contract_before_gui_routing
    finally:
        if window.worker is not None:
            window.worker.generation_request.reference_image.close()
        if window.config is not None:
            garment = window.config.get(
                "clothing_reference_generation", {}
            ).pop("garment_image", None)
            if garment is not None:
                garment.close()
        window.close()
        application.processEvents()
