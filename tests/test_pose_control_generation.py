from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

from genai_lab.generator import generate_character_candidate
from genai_lab.pose_estimation import (
    PoseEstimationApprovedInput,
    PoseJointCoordinateCandidate,
)
from genai_lab.request import (
    CharacterFramingType,
    CharacterGenerationRequest,
)


class RecordingPipeline:
    """모델 실행 없이 전달 인수만 기록하는 시험 파이프라인."""

    def __init__(self) -> None:
        self.arguments = None

    def __call__(self, **arguments):
        self.arguments = arguments
        return SimpleNamespace(
            images=[Image.new("RGB", (768, 1344), "white")]
        )


class RecordingRunLog:
    def __init__(self) -> None:
        self.stages: list[tuple[str, str]] = []

    def write_stage(self, stage: str, detail: str) -> None:
        self.stages.append((stage, detail))


def test_generation_passes_approved_pose_to_controlnet(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    reference_image = Image.new("RGB", (256, 512), "white")
    control_map = Image.new("RGB", (100, 200), "black")
    ImageDraw.Draw(control_map).line((50, 10, 50, 190), fill="white", width=3)
    joint_coordinates = tuple(
        PoseJointCoordinateCandidate(
            joint_name=f"joint_{index}",
            x=float(index),
            y=float(index),
            confidence_score=0.9,
            detected=True,
        )
        for index in range(18)
    )
    approved_pose = PoseEstimationApprovedInput(
        control_map_image=control_map,
        joint_coordinates=joint_coordinates,
        detected_joint_count=18,
        missing_joint_count=0,
        minimum_pose_confidence=0.30,
        model_ids=("test-dwpose",),
    )
    generation_request = CharacterGenerationRequest(
        reference_image=reference_image,
        reference_image_name="reference.png",
        reference_enhancement_applied=False,
        reference_enhancement_model_id=None,
        reference_quality_status="passed",
        framing_type=CharacterFramingType.FULL_BODY,
        width=768,
        height=1344,
        prompt="test prompt",
        negative_prompt="test negative",
        seed=42,
        candidate_number=1,
        inference_steps=28,
        guidance_scale=5.5,
        original_image_change_strength=0.25,
        reference_image_strength=0.80,
        model_id="test-sdxl",
        reference_adapter_id="test-ip-adapter",
    )
    config = {
        "model": {"cache_dir": "D:/genai-cache/huggingface"},
        "generation": {"mode": "image_to_image"},
        "pose_control": {
            "enabled": True,
            "model_id": "test-controlnet",
            "conditioning_scale": 0.65,
            "guidance_start": 0.00,
            "guidance_end": 0.80,
            "original_image_change_strength": 0.35,
        },
        "pose_result_policy": {
            "mode": "observe_only",
            "target_sample_count": 3,
            "block_on_pose_mismatch": False,
            "switch_to_text_to_image": False,
            "use_identity_crop": False,
        },
        "detail_correction": {"enabled": False},
    }
    pipeline = RecordingPipeline()
    run_log = RecordingRunLog()
    candidate = generate_character_candidate(
        pipeline=pipeline,
        config=config,
        generation_request=generation_request,
        project_root=Path("."),
        run_log=run_log,
        approved_pose_estimation=approved_pose,
    )
    try:
        assert pipeline.arguments is not None
        assert pipeline.arguments["control_image"].size == (768, 1344)
        assert pipeline.arguments["controlnet_conditioning_scale"] == 0.65
        assert pipeline.arguments["control_guidance_start"] == 0.00
        assert pipeline.arguments["control_guidance_end"] == 0.80
        assert pipeline.arguments["strength"] == 0.35
        assert candidate.pose_control_status == "applied"
        assert candidate.pose_control_model_id == "test-controlnet"
        policy_logs = [
            detail
            for stage, detail in run_log.stages
            if stage == "임시 자세 결과 정책"
        ]
        assert len(policy_logs) == 1
        assert "모드=observe_only" in policy_logs[0]
        assert "목표 표본=3건" in policy_logs[0]
        assert "자세 불일치 차단=0회" in policy_logs[0]
        assert "Text2Img 전환=미사용" in policy_logs[0]
        assert "IP-Adapter 크롭=미사용" in policy_logs[0]
    finally:
        candidate.image.close()
        approved_pose.close()
        reference_image.close()
