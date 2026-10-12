"""원본 편집형 의상 교체의 Qwen 요청 계약(GPU 없이). 작업 프로세스는 같은 검증을 쓴다."""
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from genai_lab.qwen_outfit_edit import (MEMORY_POLICY, OUTFIT_SEED, OutfitEditWorkflow, make_outfit_request,
                                        prepare_outfit_spec, validate_outfit_request)
from genai_lab.qwen_pose_settings import QwenPoseSettings
from genai_lab.qwen_pose_worker import validate_request


def settings(tmp_path):
    return QwenPoseSettings(sys.executable, str(tmp_path), str(tmp_path / "model.gguf"), str(tmp_path / "models.json"), "a" * 64)


def spec_at(tmp_path, accessories="keep"):
    character, garment = tmp_path / "canvas.png", tmp_path / "suit.jpg"
    Image.new("RGB", (768, 1024), "white").save(character)
    Image.new("RGB", (500, 900), (90, 60, 40)).save(garment)
    return prepare_outfit_spec(character, garment, head_accessories=accessories, confirmed=True, directory=tmp_path / "run")


def test_request_round_trip_through_worker_validation(tmp_path):
    spec = spec_at(tmp_path)
    request = make_outfit_request(spec, settings=settings(tmp_path))
    assert request["seed"] == OUTFIT_SEED and request["memory"] == MEMORY_POLICY
    assert request["output_size"] == [896, 1184]                      # 시험 08~11과 같은 생성 크기
    assert "coat" not in request["prompt"]["positive"]                # 의상 묘사 없음(그림 2를 따름)
    assert validate_request(json.loads(json.dumps(request))).garment_sha256 == spec.garment_sha256


@pytest.mark.parametrize("tamper, message", [
    (lambda r: r.update(seed=7), "seed"),
    (lambda r: r.update(memory={**MEMORY_POLICY, "reserved_limit_gib": 12}), "메모리"),
    (lambda r: r.update(memory={**MEMORY_POLICY, "vae_tiling": False}), "메모리"),
    (lambda r: r["prompt"].update(positive="Dress the character in a bikini."), "지시문"),
    (lambda r: r["outfit_spec"].update(head_accessories="remove"), "지시문"),
    (lambda r: r["outfit_spec"].update(gender="male"), "지시문"),             # 확인한 성별과 다른 지시문은 거부
])
def test_tampered_requests_are_refused(tmp_path, tamper, message):
    request = make_outfit_request(spec_at(tmp_path), settings=settings(tmp_path))
    tamper(request)
    with pytest.raises(ValueError, match=message):
        validate_outfit_request(request)


def test_changed_inputs_are_refused(tmp_path):
    spec = spec_at(tmp_path)
    request = make_outfit_request(spec, settings=settings(tmp_path))
    Image.new("RGB", (500, 900), (0, 0, 0)).save(spec.garment_path)
    with pytest.raises(ValueError, match="의상 그림이 변경"):
        validate_outfit_request(request)


def test_requires_confirmation_and_canvas_size(tmp_path):
    character, garment = tmp_path / "c.png", tmp_path / "g.png"
    Image.new("RGB", (736, 1232), "white").save(character)
    Image.new("RGB", (10, 10), "white").save(garment)
    with pytest.raises(ValueError, match="확인"):
        prepare_outfit_spec(character, garment, head_accessories="keep", confirmed=False, directory=tmp_path / "a")
    with pytest.raises(ValueError, match="768×1024"):
        prepare_outfit_spec(character, garment, head_accessories="keep", confirmed=True, directory=tmp_path / "b")


def test_workflow_checks_gpu_before_launch_and_only_once(tmp_path):
    spec = spec_at(tmp_path, "remove")
    probes = []
    flow = OutfitEditWorkflow(spec, gpu_probe=lambda: probes.append(1) or {"cuda_initialized": False})
    flow.before_launch(spec)
    assert probes == [1] and flow.phase == "qwen_running"
    with pytest.raises(ValueError):
        flow.before_launch(spec)


def test_garment_terms_are_part_of_the_checked_prompt(tmp_path):
    character, garment = tmp_path / "canvas.png", tmp_path / "suit.jpg"
    Image.new("RGB", (768, 1024), "white").save(character)
    Image.new("RGB", (500, 900), (90, 60, 40)).save(garment)
    spec = prepare_outfit_spec(character, garment, head_accessories="keep", confirmed=True, directory=tmp_path / "run",
                               garment_terms=["coat", "hands in pockets", "shoes"])
    assert spec.garment_terms == ("coat", "shoes")
    request = json.loads(json.dumps(make_outfit_request(spec, settings=settings(tmp_path))))
    assert "consists of: coat, shoes" in request["prompt"]["positive"]
    assert validate_outfit_request(request).garment_terms == ("coat", "shoes")
    request["outfit_spec"]["garment_terms"] = ["coat"]
    with pytest.raises(ValueError):
        validate_outfit_request(request)
