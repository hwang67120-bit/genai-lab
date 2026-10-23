"""필수 모델 공통 검사(D-172): 준비 화면과 실제 실행이 같은 검사, 누락은 성공이 아니다, 안내는 모델·위치·할 일."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from genai_lab import model_check

ROOT = Path(__file__).resolve().parents[1]


def tiny_home(tmp_path):
    """작은 가짜 모델 폴더 + 그에 맞는 목록(실제 배치와 같은 상대 위치)."""
    files = {"wd_tagger": "hf/models--SmilingWolf--wd-vit-tagger-v3/snapshots/x/model.onnx",
             "anime_seg": "hf/models--skytnt--anime-seg/snapshots/x/isnetis.onnx",
             "dwpose": "dwpose/checkpoints/yolox_l.onnx",
             "head_detection": "head-detection/hub/models--deepghs--anime_head_detection/snapshots/x/model.onnx",
             "head_segmentation": "hf/models--24yearsold--l2d_sam_iter2/snapshots/x/checkpoint-18000.pt",
             "see_through_code": "tools/see-through/modules/semanticsam.py"}
    folders = {"wd_tagger": "hf/models--SmilingWolf--wd-vit-tagger-v3", "anime_seg": "hf/models--skytnt--anime-seg",
               "dwpose": "dwpose/checkpoints", "head_detection": "head-detection/hub/models--deepghs--anime_head_detection",
               "head_segmentation": "hf/models--24yearsold--l2d_sam_iter2", "see_through_code": "tools/see-through"}
    home = tmp_path / "home"
    components = []
    for key, rel in files.items():
        path = home / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(key.encode() * 10)
        components.append(dict(key=key, folder=folders[key], missing_message="x",
                               files=[dict(path=rel, size=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())]))
    manifest = tmp_path / "model_home_files.json"
    manifest.write_text(json.dumps(dict(components=components)), encoding="utf-8")
    runtime = SimpleNamespace(model_cache=home / "hf", pose_models=home / "dwpose/checkpoints", head_cache=home / "head-detection/hub",
                              head_semantic_repo=home / "tools/see-through", head_semantic_checkpoint=home / files["head_segmentation"])
    qwen_root = home / "qwen/Qwen-Image-Edit-2511"
    (qwen_root / "vae").mkdir(parents=True)
    (qwen_root / "vae/config.json").write_text("{}", encoding="utf-8")
    qwen_manifest = tmp_path / "qwen_model_files.json"
    qwen_manifest.write_text(json.dumps(dict(files={"vae/config.json": "0" * 64})), encoding="utf-8")
    gguf = home / "qwen/model.gguf"
    gguf.write_bytes(b"g")
    settings = SimpleNamespace(model_root=str(qwen_root), gguf_file=str(gguf), artifact_manifest=str(qwen_manifest))
    return home, manifest, runtime, settings, files


def test_complete_folder_passes_for_both_flows(tmp_path):
    home, manifest, runtime, settings, _ = tiny_home(tmp_path)
    for flow in ("outfit", "staging"):
        result = model_check.check_models(runtime, settings, flow, manifest_path=manifest)
        assert result["ok"] and not result["problems"]


def test_missing_wd_tagger_stops_with_model_location_and_action(tmp_path):
    home, manifest, runtime, settings, files = tiny_home(tmp_path)
    (home / files["wd_tagger"]).unlink()
    result = model_check.check_models(runtime, settings, "staging", manifest_path=manifest)
    assert not result["ok"] and result["problems"][0]["component"] == "wd_tagger" and result["problems"][0]["problem"] == "없음"
    text = model_check.message(result)
    assert "WD 태거" in text and str(home / files["wd_tagger"]) in text and "확인한 위치" in text and "할 일" in text


def test_size_and_hash_differences_are_reported_and_deep_only_hashes(tmp_path, monkeypatch):
    import genai_lab.qwen_pose_settings as qps
    monkeypatch.setattr(qps, "validate_model_files", lambda settings: None)          # Qwen SHA 대조는 기존 함수 몫(여기선 분석 모델만)
    home, manifest, runtime, settings, files = tiny_home(tmp_path)
    (home / files["dwpose"]).write_bytes(b"short")
    assert model_check.check_models(runtime, settings, "staging", manifest_path=manifest)["problems"][0]["problem"] == "크기 다름"
    home2, manifest2, runtime2, settings2, files2 = tiny_home(tmp_path / "b")
    data = (home2 / files2["see_through_code"]).read_bytes()
    (home2 / files2["see_through_code"]).write_bytes(bytes(reversed(data)))         # 크기는 같고 내용만 다름
    assert model_check.check_models(runtime2, settings2, "staging", manifest_path=manifest2)["ok"]       # 빠른 검사는 크기까지
    deep = model_check.check_models(runtime2, settings2, "staging", deep=True, manifest_path=manifest2)
    assert any(p["problem"] == "SHA 다름" and p["component"] == "see_through_code" for p in deep["problems"])


def test_missing_qwen_files_and_gguf_are_reported(tmp_path):
    home, manifest, runtime, settings, _ = tiny_home(tmp_path)
    Path(settings.gguf_file).unlink()
    (Path(settings.model_root) / "vae/config.json").unlink()
    result = model_check.check_models(runtime, settings, "outfit", manifest_path=manifest)
    assert {p["path"] for p in result["problems"]} == {settings.gguf_file, str(Path(settings.model_root) / "vae/config.json")}
    with pytest.raises(model_check.MissingModelFiles, match="이미지 생성 모델"):
        if not result["ok"]:
            raise model_check.MissingModelFiles(result)


def test_outfit_flow_needs_more_models_than_staging():
    assert set(model_check.FLOWS["staging"]) < set(model_check.FLOWS["outfit"])
    assert {"anime_seg", "head_detection"} <= set(model_check.FLOWS["outfit"]) - set(model_check.FLOWS["staging"])


@pytest.mark.skipif(not Path("D:/genai-cache/huggingface").is_dir(), reason="개발 PC 모델 캐시가 있을 때만")
def test_this_development_pc_passes_the_quick_check_with_its_existing_paths():
    from genai_lab.qwen_pose_settings import QwenPoseSettings
    from genai_lab.studio_generation import StudioRuntime
    runtime = StudioRuntime()
    settings = QwenPoseSettings.load(runtime.qwen_settings)
    for flow in ("outfit", "staging"):
        result = model_check.check_models(runtime, settings, flow)
        assert result["ok"], model_check.message(result)


def test_tagger_failure_is_not_turned_into_empty_tags(tmp_path, monkeypatch):
    from PIL import Image
    from genai_lab import studio_free_generation as free_ui
    picture = tmp_path / "c.png"
    Image.new("RGB", (8, 8), "white").save(picture)
    runtime = SimpleNamespace(model_cache=tmp_path / "hf")
    monkeypatch.setattr(free_ui, "measure_result_scores", lambda image, rt: None)          # 태거를 불러오지 못함
    with pytest.raises(RuntimeError, match="WD 태거"):
        free_ui.analyze_base(picture, runtime)
    monkeypatch.setattr(free_ui, "measure_result_scores", lambda image, rt: {})            # 태거는 돌았고 태그가 없음 = 정상
    assert free_ui.analyze_base(picture, runtime)["character_scores"] == {}


def test_generate_stops_before_any_analysis_when_models_are_missing(monkeypatch, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.delenv("GENAI_STUDIO_DEVELOPER", raising=False)
    from PySide6.QtWidgets import QApplication
    from genai_lab import qwen_outfit_edit, studio_free_generation as free_ui, studio_generation
    QApplication.instance() or QApplication([])
    from gui_main import GenAILabWindow
    window = GenAILabWindow()
    window.studio.output_root = tmp_path / "runs"
    picture = tmp_path / "c.png"
    from PIL import Image
    Image.new("RGB", (8, 8), "white").save(picture)
    window.style_path = str(picture)
    monkeypatch.setattr(qwen_outfit_edit, "load_qwen_settings", lambda path: object())
    missing = dict(ok=False, problems=[dict(component="wd_tagger", label="x", path="P", problem="없음")], locations={"wd_tagger": "L"})
    monkeypatch.setattr(model_check, "require_product_models", lambda *a, **k: (_ for _ in ()).throw(model_check.MissingModelFiles(missing)))
    monkeypatch.setattr(free_ui, "analyze_base", lambda *a, **k: pytest.fail("모델이 없으면 분석을 시작하지 않는다"))
    monkeypatch.setattr(studio_generation, "analyze_inputs", lambda *a, **k: pytest.fail("모델이 없으면 분석을 시작하지 않는다"))
    failures = []
    monkeypatch.setattr(window.studio, "fail", lambda error, details="": failures.append(error))
    window.studio.product.start()
    assert len(failures) == 1 and isinstance(failures[0], model_check.MissingModelFiles) and "할 일" in str(failures[0])
    assert window.studio.task is None
    window.close()
