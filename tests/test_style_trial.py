"""그림체 LoRA 맛보기(D-159, 시험 전용): 기본 꺼짐, 명세·지시문·작업 프로세스 요청·기록, 오염 방지 표시."""
import json
from pathlib import Path

import pytest

from genai_lab import free_generation_pipeline as pipeline
from genai_lab import style_loras
from test_free_generation import fake_models, run  # noqa: F401


def test_registry_rejects_unknown_names_bad_strength_and_unrecorded_sha(tmp_path, monkeypatch):
    assert style_loras.style_lora("") is None
    with pytest.raises(ValueError):
        style_loras.style_lora("pretty")
    for bad in (0, -0.2, 1.6, "0.6", True):
        with pytest.raises(ValueError):
            style_loras.check_strength(bad)
    assert style_loras.check_strength(0.6) == 0.6
    assert len(style_loras.style_lora("anime_borrowed")["sha256"]) == 64
    with pytest.raises(ValueError, match="파일이 없거나"):
        style_loras.check_file("anime_borrowed", tmp_path / "Qwen-Image-Edit-2511")
    monkeypatch.setitem(style_loras.STYLE_LORAS, "anime_borrowed", dict(style_loras.STYLE_LORAS["anime_borrowed"], sha256=None))
    with pytest.raises(ValueError, match="SHA"):
        style_loras.check_file("anime_borrowed", tmp_path / "Qwen-Image-Edit-2511")        # SHA 기록 없이는 쓸 수 없다


def test_default_request_has_no_style_and_no_trigger(tmp_path, fake_models):
    base, calls = fake_models
    run(tmp_path, base)
    request = next(c[1] for c in calls if c[0] == "qwen")
    assert request["free_spec"]["style_lora"] == "" and request["free_spec"]["style_strength"] == 0.0
    assert not request["prompt"]["positive"].startswith("Transform into anime.")


def test_style_trial_reaches_the_worker_request_and_is_marked_as_trial(tmp_path, fake_models, monkeypatch):
    base, calls = fake_models
    monkeypatch.setattr(style_loras, "check_file", lambda name, root: Path(root) / "style.safetensors")
    result = run(tmp_path, base, style_trial=dict(name="anime_borrowed", strength=0.6), experiment=dict(single_attempt=True))
    request = next(c[1] for c in calls if c[0] == "qwen")
    assert request["free_spec"]["style_lora"] == "anime_borrowed" and request["free_spec"]["style_strength"] == 0.6
    assert request["prompt"]["positive"].startswith("Transform into anime. ")
    trace = json.loads(Path(result["trace"]).read_text(encoding="utf-8"))
    assert trace["style_trial"]["trial"] == "style_lora_borrowed" and "운영 집계" in trace["style_trial"]["note"]
    assert "style_loras" in trace["execution_contract"]["code_sha256"]


def test_style_trial_shape_and_missing_file_are_rejected(tmp_path, fake_models):
    from genai_lab.free_generation import FreeGenerationRejected
    base, _ = fake_models
    for bad in (dict(name="anime_borrowed"), dict(name="anime_borrowed", strength=2.0), "anime_borrowed"):
        with pytest.raises(FreeGenerationRejected) as error:
            run(tmp_path / str(len(str(bad))), base, style_trial=bad)
        assert error.value.code == "invalid_input"


def test_trial_runner_makes_none_and_two_strengths_per_case_in_its_own_folder(tmp_path, monkeypatch):
    import sys
    from PIL import Image
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import style_trial_run
    base = tmp_path / "base.png"
    Image.new("RGB", (40, 80), "white").save(base)
    calls = []
    def fake_run(path, features, request, **kw):
        calls.append(dict(kw, request=request))
        Path(kw["directory"]).mkdir(parents=True)
        raw = Path(kw["directory"]) / "raw.png"
        Image.new("RGB", (40, 80), "gray").save(raw)
        return dict(status="awaiting_user_review", result=str(raw))
    monkeypatch.setattr(pipeline, "run_free_generation", fake_run)
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([dict(case="X", base=str(base), gender="male", body_parts=[], features={}, scene_seed=7)]), encoding="utf-8")
    status = style_trial_run.main([str(cases), str(tmp_path / "out")])
    assert [c["style_trial"] for c in calls] == [None, dict(name="anime_borrowed", strength=0.6), dict(name="anime_borrowed", strength=1.0)]
    assert all(c["experiment"] == dict(lightning8=True, single_attempt=True) for c in calls)
    assert len({c["request"] for c in calls}) == 1 and status["trial"] == "style_lora_borrowed"
    assert (tmp_path / "out" / "sheet.png").is_file() and status["rows"][0]["state_1.0"] == "awaiting_user_review"


def test_worker_loads_the_style_adapter_after_the_accelerator():
    source = (Path(__file__).resolve().parents[1] / "genai_lab" / "qwen_pose_worker.py").read_text(encoding="utf-8")
    assert source.index('adapter_name=accel_name') < source.index('adapter_name="style"') < source.index("enable_group_offload")
    assert 'pipe.set_adapters(names + ["style"], weights + [strength])' in source
    assert "그림체 LoRA SHA 불일치" in source
