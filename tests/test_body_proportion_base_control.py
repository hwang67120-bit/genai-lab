from types import SimpleNamespace
import sys
import pytest

from PIL import Image
import yaml

from genai_lab.body_proportion_presets import (
    body_proportion_contract,
    prepare_body_proportion_control,
    resolve_body_proportion_control,
)


def project_root():
    from pathlib import Path
    return Path(__file__).resolve().parents[1]


def load_config():
    root = project_root()
    return yaml.safe_load(
        (root / "configs" / "animagine.yaml").read_text(encoding="utf-8")
    )


def request(preset_id="standard_7_5h_shoulder", framing="full_body"):
    return SimpleNamespace(
        body_proportion_preset_id=preset_id,
        framing_type=SimpleNamespace(value=framing),
        width=768,
        height=1344,
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "body_proportion_presets.enabled 가 false 로 꺼져 있어 "
        "프리셋 경로가 실행되지 않는다. 테스트는 활성 상태를 기대한다. "
        "기능을 다시 켜면 이 테스트는 XPASS 가 되고 strict=True 이므로 "
        "실패한다. 그때 마커를 제거하거나 테스트를 갱신할 것. "
        "이전 A/B 기록: 프리셋 적용 여부와 무관하게 출력 해시가 전부 동일했다. "
        "근거 설정: configs/animagine.yaml:270"
    ),
)
def test_full_body_preset_becomes_animagine_control():
    config = load_config()
    image, record = prepare_body_proportion_control(
        config, request(), project_root()
    )
    try:
        assert record["status"] == "active"
        assert record["preset_id"] == "standard_7_5h_shoulder"
        assert record["runtime_size"] == [768, 1344]
        assert record["coordinate_source"] == "project_preset_output_canvas"
        assert record["settings"]["openpose_used"] is False
        assert image.mode == "RGB"
        assert image.size == (768, 1344)
        assert image.getbbox() is not None
    finally:
        image.close()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "body_proportion_presets.enabled 가 false 로 꺼져 있어 "
        "프리셋 경로가 실행되지 않는다. 테스트는 활성 상태를 기대한다. "
        "기능을 다시 켜면 이 테스트는 XPASS 가 되고 strict=True 이므로 "
        "실패한다. 그때 마커를 제거하거나 테스트를 갱신할 것. "
        "이전 A/B 기록: 프리셋 적용 여부와 무관하게 출력 해시가 전부 동일했다. "
        "근거 설정: configs/animagine.yaml:270"
    ),
)
def test_non_full_body_keeps_selection_but_skips_control():
    config = load_config()
    image, record = prepare_body_proportion_control(
        config, request(framing="upper_body"), project_root()
    )
    assert image is None
    assert record["status"] == "skipped_non_full_body"
    assert record["preset_id"] == "standard_7_5h_shoulder"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "body_proportion_presets.enabled 가 false 로 꺼져 있어 "
        "프리셋 경로가 실행되지 않는다. 테스트는 활성 상태를 기대한다. "
        "기능을 다시 켜면 이 테스트는 XPASS 가 되고 strict=True 이므로 "
        "실패한다. 그때 마커를 제거하거나 테스트를 갱신할 것. "
        "이전 A/B 기록: 프리셋 적용 여부와 무관하게 출력 해시가 전부 동일했다. "
        "근거 설정: configs/animagine.yaml:270"
    ),
)
def test_legacy_request_without_selection_does_not_change_pipeline():
    config = load_config()
    record = body_proportion_contract(
        config, request(preset_id=None), project_root()
    )
    assert record["status"] == "not_selected"


def test_model_loader_uses_cached_canny_controlnet_only_when_selected(
    monkeypatch, tmp_path
):
    from genai_lab.model import prepare_pipeline

    calls = []
    auto_calls = []
    pipe = SimpleNamespace(enable_model_cpu_offload=lambda: calls.append("offload"))
    control_factory = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: (
            calls.append(("controlnet", args, kwargs)) or "control"
        )
    )
    pipeline_factory = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: (
            calls.append(("pipeline", args, kwargs)) or pipe
        )
    )
    base_factory = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: pipe
    )
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            AutoPipelineForImage2Image=SimpleNamespace(
                from_pipe=lambda value: auto_calls.append(value) or value
            ),
            ControlNetModel=control_factory,
            StableDiffusionPipeline=base_factory,
            StableDiffusionXLControlNetImg2ImgPipeline=pipeline_factory,
            StableDiffusionXLControlNetPipeline=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: (_ for _ in ()).throw(
                    AssertionError("text-to-image ControlNet must not be used")
                )
            ),
            StableDiffusionXLPipeline=base_factory,
        ),
    )
    config = {
        "model": {
            "family": "sdxl",
            "id": "animagine",
            "cache_dir": str(tmp_path),
        },
        "generation": {"mode": "image_to_image"},
        "style": {"enabled": False},
        "body_proportion_presets": {
            "enabled": True,
            "status": "animagine_base_connected",
            "model_id": "xinsir/controlnet-canny-sdxl-1.0",
            "local_files_only": True,
            "conditioning_scale": .42,
            "guidance_start": 0.0,
            "guidance_end": .72,
            "user_selection_required": True,
        },
    }

    result = prepare_pipeline(
        config,
        body_proportion_preset_id="standard_7_5h_shoulder",
    )

    assert result is pipe
    assert auto_calls == []
    assert pipe._genai_lab_pose_control_enabled is False
    assert (
        pipe._genai_lab_body_proportion_model_id
        == "xinsir/controlnet-canny-sdxl-1.0"
    )
    control_call = next(item for item in calls if isinstance(item, tuple)
                        and item[0] == "controlnet")
    assert control_call[2]["local_files_only"] is True
    assert resolve_body_proportion_control(config).conditioning_scale == .42