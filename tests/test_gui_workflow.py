import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image, ImageDraw
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog

from genai_lab.pose_estimation import (
    PoseEstimationReviewCandidate,
    PoseJointCoordinateCandidate,
)
from genai_lab.workflow import (
    GenerationWorkflowContext,
    GenerationWorkflowStage,
)
from gui_main import GenAILabWindow, build_garment_inpaint_prompts


def create_window() -> tuple[QApplication, GenAILabWindow]:
    application = QApplication.instance() or QApplication([])
    return application, GenAILabWindow()


def close_window(window: GenAILabWindow) -> None:
    window.approved_reference_image = None
    window.workflow_context = None
    window.close()


def create_pose_review_candidate(
    detected_joint_count: int,
) -> PoseEstimationReviewCandidate:
    joint_names = (
        "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
        "left_shoulder", "left_elbow", "left_wrist", "right_hip",
        "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
        "right_eye", "left_eye", "right_ear", "left_ear",
    )
    coordinates = tuple(
        PoseJointCoordinateCandidate(
            joint_name=name,
            x=float(index),
            y=float(index + 1),
            confidence_score=0.90 if index < detected_joint_count else 0.10,
            detected=index < detected_joint_count,
        )
        for index, name in enumerate(joint_names)
    )
    control_map = Image.new("RGB", (64, 128), "black")
    ImageDraw.Draw(control_map).line((32, 10, 32, 118), fill="white", width=3)
    return PoseEstimationReviewCandidate(
        source_image=Image.new("RGB", (64, 128), "white"),
        overlay_image=Image.new("RGB", (64, 128), "green"),
        control_map_image=control_map,
        joint_coordinates=coordinates,
        detected_joint_count=detected_joint_count,
        missing_joint_count=18 - detected_joint_count,
        minimum_pose_confidence=0.30,
        model_ids=("test-dwpose",),
        elapsed_seconds=1.0,
    )


def test_automatic_workflow_starts_base_generation_without_optional_inputs(
    monkeypatch,
) -> None:
    application, window = create_window()
    started_stages: list[str] = []
    window.approved_reference_image = object()
    window.workflow_context = GenerationWorkflowContext(
        character_image_path=Path("character.png"),
        clothing_image_path=None,
        pose_image_path=None,
    )
    monkeypatch.setattr(
        window,
        "_start_model_generation",
        lambda: started_stages.append("base"),
    )

    window.advance_generation_workflow()

    assert started_stages == ["base"]
    assert (
        window.workflow_context.current_stage
        is GenerationWorkflowStage.BASE_GENERATING
    )
    close_window(window)
    application.processEvents()


def test_automatic_workflow_runs_clothing_before_pose(monkeypatch) -> None:
    application, window = create_window()
    started_paths: list[Path] = []
    clothing_path = Path("clothing.png")
    window.approved_reference_image = object()
    window.workflow_context = GenerationWorkflowContext(
        character_image_path=Path("character.png"),
        clothing_image_path=clothing_path,
        pose_image_path=Path("pose.png"),
    )
    monkeypatch.setattr(
        window,
        "start_outfit_region_preparation",
        lambda image_path: started_paths.append(image_path),
    )

    window.advance_generation_workflow()

    assert started_paths == [clothing_path]
    assert (
        window.workflow_context.current_stage
        is GenerationWorkflowStage.CLOTHING_MASKING
    )
    close_window(window)
    application.processEvents()


def test_automatic_workflow_runs_pose_after_clothing_approval(monkeypatch) -> None:
    application, window = create_window()
    reviewed_paths: list[Path] = []
    pose_path = Path("pose.png")
    window.approved_reference_image = object()
    window.confirmed_clothing_design = object()
    window.workflow_context = GenerationWorkflowContext(
        character_image_path=Path("character.png"),
        clothing_image_path=Path("clothing.png"),
        pose_image_path=pose_path,
    )
    monkeypatch.setattr(
        window,
        "review_pose_reference",
        lambda image_path: reviewed_paths.append(image_path),
    )

    window.advance_generation_workflow()

    assert reviewed_paths == [pose_path]
    assert (
        window.workflow_context.current_stage
        is GenerationWorkflowStage.POSE_ESTIMATING
    )
    window.confirmed_clothing_design = None
    close_window(window)
    application.processEvents()


def test_pose_quality_failure_uses_saved_pose_fallback(monkeypatch) -> None:
    application, window = create_window()
    review_candidate = create_pose_review_candidate(7)
    fallback_reasons: list[str] = []
    monkeypatch.setattr(
        window,
        "offer_saved_pose_fallback",
        lambda failure_reason, failed_source_image: (
            fallback_reasons.append(failure_reason) or ("approved", "")
        ),
    )

    window.pose_reference_estimation_completed(review_candidate)

    assert len(fallback_reasons) == 1
    assert "탐지 관절=7/18개, 최소=8/18개" in fallback_reasons[0]
    close_window(window)
    application.processEvents()


def test_pose_approval_saves_last_approved_pose(monkeypatch) -> None:
    application, window = create_window()
    review_candidate = create_pose_review_candidate(18)
    saved_counts: list[int] = []

    def approve_dialog(dialog: QDialog) -> int:
        dialog.is_approved = True
        return int(QDialog.DialogCode.Accepted)

    monkeypatch.setattr(window, "execute_approval_dialog", approve_dialog)
    monkeypatch.setattr(
        window,
        "save_last_approved_pose",
        lambda approved_pose, source_preview_image: (
            saved_counts.append(approved_pose.detected_joint_count)
            or (True, "a" * 64)
        ),
    )

    window.pose_reference_estimation_completed(review_candidate)

    assert saved_counts == [18]
    assert window.approved_pose_estimation is not None
    assert window.approved_pose_estimation.detected_joint_count == 18
    close_window(window)
    application.processEvents()


def test_image_selection_registers_paths_without_starting_ai(monkeypatch) -> None:
    application, window = create_window()
    started_ai_stages: list[str] = []
    selected_paths = iter(
        (
            "C:/input/character.png",
            "C:/input/clothing.jpg",
            "C:/input/pose.png",
        )
    )
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (next(selected_paths), "이미지 파일"),
    )
    monkeypatch.setattr(
        window,
        "start_reference_preparation",
        lambda path: started_ai_stages.append("reference"),
    )
    monkeypatch.setattr(
        window,
        "start_outfit_region_preparation",
        lambda path: started_ai_stages.append("clothing"),
    )
    monkeypatch.setattr(
        window,
        "review_pose_reference",
        lambda path: started_ai_stages.append("pose"),
    )

    window.select_image("style")
    window.select_image("outfit")
    window.select_image("pose")

    assert window.style_path == "C:/input/character.png"
    assert window.selected_outfit_path == Path("C:/input/clothing.jpg")
    assert window.selected_pose_path == Path("C:/input/pose.png")
    assert started_ai_stages == []
    assert window.generate_button.isEnabled() is True
    close_window(window)
    application.processEvents()


def test_approval_dialog_blocks_duplicate_resume_until_closed(monkeypatch) -> None:
    application, window = create_window()
    resumed_stages: list[str] = []
    window.workflow_context = GenerationWorkflowContext(
        character_image_path=Path("character.png"),
        clothing_image_path=None,
        pose_image_path=None,
    )
    monkeypatch.setattr(
        window,
        "advance_generation_workflow",
        lambda: resumed_stages.append("resumed"),
    )
    approval_dialog = QDialog(window)

    def attempt_resume_while_dialog_is_open() -> None:
        assert window.approval_dialog_open is True
        window.resume_generation_workflow()
        assert resumed_stages == []
        approval_dialog.accept()

    QTimer.singleShot(0, attempt_resume_while_dialog_is_open)
    dialog_result = window.execute_approval_dialog(approval_dialog)

    assert dialog_result == int(QDialog.DialogCode.Accepted)
    assert resumed_stages == []
    application.processEvents()
    assert resumed_stages == ["resumed"]
    close_window(window)
    application.processEvents()


def test_clothing_workflow_starts_inpaint_after_body_approval(
    monkeypatch,
) -> None:
    application, window = create_window()
    started: list[str] = []
    window.approved_reference_image = object()
    window.confirmed_clothing_design = object()
    window.confirmed_character_body_comparison = object()
    window.pending_clothing_base_candidate = object()
    window.workflow_context = GenerationWorkflowContext(
        character_image_path=Path("character.png"),
        clothing_image_path=Path("clothing.png"),
        pose_image_path=None,
    )
    monkeypatch.setattr(
        window,
        "start_garment_inpaint",
        lambda: started.append("inpaint"),
    )

    window.advance_generation_workflow()

    assert started == ["inpaint"]
    assert (
        window.workflow_context.current_stage
        is GenerationWorkflowStage.CLOTHING_COMPOSITING
    )
    window.confirmed_character_body_comparison = None
    window.pending_clothing_base_candidate = None
    window.confirmed_clothing_design = None
    close_window(window)
    application.processEvents()


def test_garment_prompt_removes_only_app_outfit_preservation_conflicts() -> None:
    prompt, negative_prompt = build_garment_inpaint_prompts(
        "same character as reference image, matching outfit and colors, smile",
        "different character, different outfit, mismatched colors, blurry",
        ("blue jacket", "gold buttons"),
    )

    assert "matching outfit and colors" not in prompt
    assert "blue jacket" in prompt
    assert "gold buttons" in prompt
    assert "different outfit" not in negative_prompt
    assert "mismatched colors" not in negative_prompt
    assert "different character" in negative_prompt
    assert "blurry" in negative_prompt
    assert prompt.startswith(
        "blue jacket, gold buttons, reference garment, "
        "preserve garment color pattern seams accessories"
    )


def test_garment_inpaint_settings_use_vit_h_image_encoder() -> None:
    application, window = create_window()
    try:
        settings = window.create_garment_inpaint_settings()
        assert settings.adapter_weight == (
            "ip-adapter-plus_sdxl_vit-h.safetensors"
        )
        assert settings.adapter_image_encoder_subfolder == (
            "models/image_encoder"
        )
    finally:
        close_window(window)
        application.processEvents()


def test_step5_pipeline_release_clears_gui_and_worker_references(
    monkeypatch,
) -> None:
    application, window = create_window()
    calls: list[str] = []

    class Pipeline:
        def maybe_free_model_hooks(self) -> None:
            calls.append("maybe_free_model_hooks")

        def remove_all_hooks(self) -> None:
            calls.append("remove_all_hooks")

    class Worker:
        def __init__(self, pipeline) -> None:
            self.pipeline = pipeline

    pipeline = Pipeline()
    worker = Worker(pipeline)
    window.pipeline = pipeline
    window.worker = worker
    monkeypatch.setattr("gui_main.torch.cuda.is_available", lambda: True)
    memory_values = iter((2**20, 0))
    monkeypatch.setattr(
        "gui_main.torch.cuda.memory_allocated",
        lambda: next(memory_values),
    )
    monkeypatch.setattr("gui_main.torch.cuda.memory_reserved", lambda: 0)
    monkeypatch.setattr(
        "gui_main.torch.cuda.synchronize",
        lambda: calls.append("synchronize"),
    )
    monkeypatch.setattr(
        "gui_main.torch.cuda.empty_cache",
        lambda: calls.append("empty_cache"),
    )

    metrics = window.release_step5_pipeline()

    assert window.pipeline is None
    assert worker.pipeline is None
    assert calls == [
        "maybe_free_model_hooks",
        "remove_all_hooks",
        "synchronize",
        "empty_cache",
    ]
    assert metrics == {
        "before_allocated_mib": 1.0,
        "after_allocated_mib": 0.0,
        "after_reserved_mib": 0.0,
    }
    window.worker = None
    close_window(window)
    application.processEvents()
