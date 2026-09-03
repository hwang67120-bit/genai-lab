from pathlib import Path

from genai_lab.workflow import (
    GenerationWorkflowContext,
    GenerationWorkflowStage,
)


def test_workflow_reports_fixed_eight_stage_progress() -> None:
    context = GenerationWorkflowContext(
        character_image_path=Path("character.png"),
        clothing_image_path=Path("clothing.png"),
        pose_image_path=Path("pose.png"),
    )

    context.move_to(GenerationWorkflowStage.POSE_ESTIMATING)

    assert context.progress == (4, 8)
    assert context.active is True


def test_workflow_retry_returns_to_failed_stage() -> None:
    context = GenerationWorkflowContext(
        character_image_path=Path("character.png"),
        clothing_image_path=Path("clothing.png"),
        pose_image_path=None,
    )
    context.fail(GenerationWorkflowStage.CLOTHING_MASKING)

    assert context.progress == (2, 8)
    assert context.active is False

    context.retry()

    assert context.current_stage is GenerationWorkflowStage.CLOTHING_MASKING
    assert context.failed_stage is None
    assert context.retry_count == 1
    assert context.active is True


def test_workflow_completed_state_stops_automatic_progress() -> None:
    context = GenerationWorkflowContext(
        character_image_path=Path("character.png"),
        clothing_image_path=None,
        pose_image_path=None,
    )

    context.move_to(GenerationWorkflowStage.COMPLETED)

    assert context.progress == (8, 8)
    assert context.active is False
