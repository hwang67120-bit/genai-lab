"""그림체 후처리 한 번(D-161 시험 A, 시험 전용): 트리거 문장 하나·입력 1장·같은 크기, LoRA 없는 대조, 실행기, 적용 모듈 기록."""
import hashlib
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from genai_lab import style_pass
from genai_lab.qwen_pose_settings import QwenPoseSettings

ROOT = Path(__file__).resolve().parents[1]


def settings(tmp):
    return QwenPoseSettings(sys.executable, str(tmp), str(tmp / "m.gguf"), str(tmp / "m.json"), "a" * 64)


def picture(path):
    Image.new("RGB", (896, 1184), "gray").save(path)
    return path


def test_request_is_the_trigger_alone_on_one_image_at_the_same_size(tmp_path):
    spec = style_pass.prepare_style_spec(picture(tmp_path / "a.png"), directory=tmp_path / "run", style_lora="anime_borrowed")
    request = style_pass.make_style_request(spec, settings=settings(tmp_path))
    assert request["task_kind"] == "style_pass" and request["images"] == 1 and request["seed"] == 42
    assert request["prompt"]["positive"] == "Transform into anime." and request["prompt"]["negative"] == " "
    assert request["basis_size"] == request["output_size"] == [896, 1184]
    assert spec.lora_name == "anime_borrowed" and spec.style_strength == 1.0
    assert style_pass.validate_style_request(request) == spec
    from genai_lab.qwen_pose_worker import validate_request
    assert validate_request(request) == spec                 # 작업 프로세스도 같은 검사로 받는다


@pytest.mark.parametrize("change", [
    lambda r: r["prompt"].update(positive="Transform into anime. Keep the same character."),
    lambda r: r["prompt"].update(negative="blurry"),
    lambda r: r.update(seed=43),
    lambda r: r.update(images=2),
    lambda r: r["memory"].update(reserved_limit_gib=8),
    lambda r: r.update(output_size=[1024, 1024]),
])
def test_other_instructions_seeds_or_relaxed_memory_are_rejected(tmp_path, change):
    spec = style_pass.prepare_style_spec(picture(tmp_path / "a.png"), directory=tmp_path / "run", style_lora="anime_borrowed")
    request = style_pass.make_style_request(spec, settings=settings(tmp_path))
    change(request)
    with pytest.raises(ValueError):
        style_pass.validate_style_request(request)


def test_control_sends_the_same_sentence_without_the_lora_and_bad_specs_fail(tmp_path):
    spec = style_pass.prepare_style_spec(picture(tmp_path / "a.png"), directory=tmp_path / "run", style_lora="anime_borrowed",
                                         apply_lora=False)
    assert spec.lora_name == "" and spec.style_strength == 0.0 and spec.prompt() == "Transform into anime."
    with pytest.raises(ValueError):
        style_pass.StylePassSpec(spec.image_path, spec.image_sha256, "anime_borrowed", apply_lora=False, style_strength=0.6).verify_image()
    with pytest.raises(ValueError):
        style_pass.StylePassSpec(spec.image_path, "0" * 64, "anime_borrowed").verify_image()
    with pytest.raises(ValueError):
        style_pass.prepare_style_spec(tmp_path / "a.png", directory=tmp_path / "r2", style_lora="anime_borrowed", strength=2.0)


def test_worker_takes_one_image_under_the_memory_policy_and_records_applied_modules():
    source = (ROOT / "genai_lab" / "qwen_pose_worker.py").read_text(encoding="utf-8")
    assert '("outfit_edit", "free_generation", "style_pass")' in source
    assert 'sources = [(spec.image_path, spec.image_sha256)] + ([reference] if reference else [])' in source
    assert 'else spec.lora_name if request.get("task_kind") == "style_pass"' in source
    assert source.index('pipe.set_adapters(names + ["style"]') < source.index('"style_modules.json"') < source.index("enable_group_offload")
    assert "applied_modules=len(applied)" in source


def test_runner_converts_each_case_once_and_reports_conversion_and_total_time(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import style_pass_run
    image = picture(tmp_path / "judged.png")
    calls = []
    def fake_run(request, directory, workflow, **kw):
        calls.append(request)
        directory.mkdir(parents=True)
        Image.new("RGB", (896, 1184), "white").save(directory / "raw.png")
        (directory / "run.final.json").write_text(json.dumps(dict(seconds=900.0, generation_memory=dict(reserved_bytes=6 * 2**30),
                                                                  call_arguments=dict(num_inference_steps=40, true_cfg_scale=4.0))), encoding="utf-8")
        return directory / "product.png"
    monkeypatch.setattr(style_pass, "run_style_pass_edit", fake_run)
    monkeypatch.setattr(style_pass.OutfitEditWorkflow, "before_launch", lambda self, spec: None)
    sha = hashlib.sha256(image.read_bytes()).hexdigest()
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(dict(style_lora="anime_borrowed", strength=1.0, variants=["style", "none"], cases=[
        dict(case="A1", image=str(image), sha256=sha, original_seconds=4324),
        dict(case="X", image=str(image), sha256="0" * 64, original_seconds=10)])), encoding="utf-8")
    status = style_pass_run.main([str(plan), str(tmp_path / "out")])
    assert [r["task_kind"] for r in calls] == ["style_pass", "style_pass"]
    assert [r["style_spec"]["apply_lora"] for r in calls] == [True, False]
    a1, bad = status["rows"]
    assert a1["state_style"] == a1["state_none"] == "completed" and a1["worker_seconds_style"] == 900.0
    assert a1["total_seconds_style"] == 4324 + a1["seconds_style"]
    assert bad["state_style"] == "failed" and "SHA" in bad["error_style"]            # 판정한 그림과 다르면 돌리지 않는다
    assert (tmp_path / "out" / "sheet.png").is_file() and status["trial"] == "style_pass_borrowed"
