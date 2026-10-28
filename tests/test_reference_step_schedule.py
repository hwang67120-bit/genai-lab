"""단계별 참조 강도 전환의 설정/경계 검사. 이미지 품질 검사는 아니다."""
from types import SimpleNamespace

import pytest

from genai_lab.reference_step_schedule import (
    build_reference_scale_schedule,
    effective_denoising_steps,
)


def entries():
    return tuple(SimpleNamespace(name=name, scale=scale) for name, scale in (
        ("identity", .7), ("ears", .35), ("tail", .35), ("garment", .45)))


def config():
    return {"reference_analysis": {"reference_step_schedule": {
        "enabled": True,
        "phases": [
            {"end_ratio": .35, "scale_factors": {
                "identity": 1., "ears": .15, "tail": .30, "garment": 1.}},
            {"end_ratio": .70, "scale_factors": {
                "identity": .93, "ears": .71, "tail": .86, "garment": .93}},
            {"end_ratio": 1., "scale_factors": {
                "identity": .79, "ears": 1.57, "tail": 1.43, "garment": .67}},
        ],
    }}}


def test_schedule_uses_approved_base_scales_and_switches_only_at_boundaries():
    schedule, record = build_reference_scale_schedule(config(), entries(), 28)
    assert schedule.initial_scales == pytest.approx([.7, .0525, .105, .45])
    assert [phase["end_step"] for phase in record["phases"]] == [10, 20, 28]

    calls = []
    pipe = SimpleNamespace(set_ip_adapter_scale=calls.append)
    payload = {"latents": object()}
    for step in range(28):
        assert schedule(pipe, step, 0, payload) is payload
    assert calls[0] == pytest.approx((.651 + .2485 + .301 + .4185) / 4)
    assert calls[1] == pytest.approx((.553 + .5495 + .5005 + .3015) / 4)


def test_disabled_schedule_preserves_base_record_without_callback():
    schedule, record = build_reference_scale_schedule({}, entries(), 28)
    assert schedule is None
    assert record["enabled"] is False
    assert record["base_scales"] == {
        "identity": .7, "ears": .35, "tail": .35, "garment": .45}


def test_img2img_effective_steps_follow_strength_before_building_schedule():
    total_steps = effective_denoising_steps(28, "image_to_image", .25)
    assert total_steps == 7

    schedule, record = build_reference_scale_schedule(config(), entries(), total_steps)
    assert [phase["end_step"] for phase in record["phases"]] == [3, 5, 7]

    calls = []
    pipe = SimpleNamespace(set_ip_adapter_scale=calls.append)
    for step in range(total_steps):
        schedule(pipe, step, 0, {"latents": object()})
    assert len(calls) == 2


def test_text_to_image_effective_steps_use_full_inference_count():
    assert effective_denoising_steps(28, "text_to_image") == 28


@pytest.mark.parametrize("strength", [None, 0, 1.01, float("nan")])
def test_invalid_img2img_strength_is_rejected(strength):
    with pytest.raises(ValueError, match="strength"):
        effective_denoising_steps(28, "image_to_image", strength)


@pytest.mark.parametrize("change,match", [
    (("ratio", 1, .2), "비율"),
    (("missing", 0, "ears"), "누락"),
    (("factor", 2, ("ears", 3.0)), "0~1"),
])
def test_invalid_schedule_is_rejected(change, match):
    value = config()
    phases = value["reference_analysis"]["reference_step_schedule"]["phases"]
    kind, index, payload = change
    if kind == "ratio":
        phases[index]["end_ratio"] = payload
    elif kind == "missing":
        phases[index]["scale_factors"].pop(payload)
    else:
        name, factor = payload
        phases[index]["scale_factors"][name] = factor
    with pytest.raises(ValueError, match=match):
        build_reference_scale_schedule(value, entries(), 28)
