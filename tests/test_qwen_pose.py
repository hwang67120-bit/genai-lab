from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from genai_lab.qwen_preservation import (PreservationSpec, PreservationItem, confirm_item,
    draft_from_reports, file_sha, json_sha)
from genai_lab.qwen_pose_prompt import PoseEditInstructions, assemble_pose_edit_prompt
from genai_lab.qwen_pose_settings import QwenPoseSettings, EXPECTED_VERSIONS, check_runtime_versions, output_dimensions
from genai_lab.qwen_pose_edit import PoseEditWorkflow, make_request, product_preview, run_pose_edit, write_json, assert_parent_gpu_released
from genai_lab.qwen_pose_worker import validate_request, UntruncatedProcessor
from genai_lab.qwen_pose_review import save_user_review, GLOBAL_FIELDS
from genai_lab.qwen_preservation_analysis import report_candidates

FIXTURE = json.loads((Path(__file__).parent / "fixtures/qwen_preservation_b.json").read_text(encoding="utf-8"))
HANDS = "Both arms hang down at the sides of the body, and each hand rests lightly beside the hip, next to the side seam of the shorts."


@pytest.fixture
def image(tmp_path):
    p = tmp_path / "basis.png"
    Image.new("RGB", (736, 1232), "white").save(p)
    return p


def approved(image):
    item = PreservationItem("a", "outfit", "흰 상의", "The top is white.", "user", file_sha(image),
                            "상의", "confirmed", "사용자 확인")
    return PreservationSpec(str(image), file_sha(image), "이미지 승인", (item,))


def settings(tmp_path):
    return QwenPoseSettings(str(Path(__import__("sys").executable)), str(tmp_path), str(tmp_path / "model.gguf"),
                            str(tmp_path / "models.json"), "a"*64)


@pytest.mark.parametrize("character", ("raccoon", "ordinary-female"))
def test_exact_recorded_b_prompt_utf8_and_order(image, character):
    data = FIXTURE["characters"][character]
    items = tuple(PreservationItem(row["id"], "style" if "그림체" in row["kind"] else row["kind"],
        row["fact"], row["english"], row["source"], file_sha(image), row["check_at"],
        "confirmed", row["approval"], approved_gender=True) for row in data["items"])
    spec = replace(approved(image), items=items)
    result = assemble_pose_edit_prompt(spec, PoseEditInstructions("front-facing", HANDS, "시험 승인"))
    assert result["positive"].encode("utf-8") == data["expected"].encode("utf-8")
    assert result["positive_sha256"] == hashlib.sha256(data["expected"].encode()).hexdigest()
    assert result["negative"] == " "


def test_optional_pose_not_injected(image):
    result = assemble_pose_edit_prompt(approved(image))["positive"]
    assert "front-facing" not in result and HANDS not in result
    assert "in the body pose of Picture 2" in result
    with pytest.raises(ValueError): PoseEditInstructions("front-facing")


def test_unconfirmed_excluded_and_revisions(image):
    spec = approved(image)
    other = replace(spec.items[0], id="b", status="draft", confirmation="")
    spec = replace(spec, items=(*spec.items, other, replace(other, id="c", status="excluded")))
    result = assemble_pose_edit_prompt(spec)
    assert result["included"] == ["a"]
    changed = spec.revise_item("a", english="The top is black.")
    assert changed.items[0].status == "draft"
    with pytest.raises(ValueError): assemble_pose_edit_prompt(changed)
    changed = confirm_item(changed, "a", confirmation="수정 승인")
    assert "black" in assemble_pose_edit_prompt(changed)["positive"]
    assert changed.sha256 != spec.sha256
    assert all(x.status == "draft" for x in changed.rebind(image, "새 승인").items)


def test_basis_hash_change_blocks(image):
    spec = approved(image)
    Image.new("RGB", (736, 1232), "red").save(image)
    with pytest.raises(ValueError, match="변경"): assemble_pose_edit_prompt(spec)


@pytest.mark.parametrize("english", ("A raccoon tail.", "A fox tail.", "hands in pockets", "small breasts", "petite", "skinny", "1boy"))
def test_disallowed_preservation_instructions(image, english):
    with pytest.raises(ValueError): replace(approved(image).items[0], english=english)


def test_conflicting_evidence_requires_resolution(image):
    reports = [{"source": "eye", "kind": "eye", "image_sha256": "b"*64,
                "candidates": [{"fact_ko":"파란 눈", "english":"Blue eyes."}]}]
    spec = draft_from_reports(image, "승인", reports, ("red eyes",))
    assert spec.items[0].status == "draft"
    assert "different_evidence_image" in spec.items[0].issues
    assert "compare_with_generation_tags" in spec.items[0].issues
    with pytest.raises(ValueError): confirm_item(spec, "1", confirmation="확인")
    result = confirm_item(spec, "1", confirmation="확인", resolution="완성 이미지에서 파란색 직접 확인")
    assert result.items[0].status == "confirmed"


@pytest.mark.parametrize("source", ("eye_color_analysis", "hair_detail_analysis", "garment_detail_analysis",
                                    "extra_parts_analysis", "accessory_analysis"))
def test_analysis_failure_never_invents_facts(image, source):
    report = report_candidates(source, file_sha(image), {"status": "failed"})
    assert all(not x["english"] and x["issues"] for x in report["candidates"])


def test_local_wd_analysis_cannot_download(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError
    from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, download_wd14_model_files, ClothingDesignAnalysisError
    calls = []
    def missing(**kwargs):
        calls.append(kwargs)
        raise LocalEntryNotFoundError("not cached")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing)
    with pytest.raises(ClothingDesignAnalysisError):
        download_wd14_model_files(ClothingDesignAnalysisSettings(cache_dir=tmp_path, local_files_only=True))
    assert len(calls) == 1 and calls[0]["local_files_only"] is True


@pytest.mark.parametrize("package", tuple(EXPECTED_VERSIONS))
def test_runtime_version_mismatch_stops_before_load(package):
    versions = dict(EXPECTED_VERSIONS, **{package: "0.0.0"})
    with pytest.raises(RuntimeError, match=package): check_runtime_versions(versions.__getitem__)


def test_runtime_exact_versions_and_missing():
    assert check_runtime_versions(EXPECTED_VERSIONS.__getitem__) == EXPECTED_VERSIONS
    def missing(_): raise importlib.metadata.PackageNotFoundError()
    with pytest.raises(RuntimeError): check_runtime_versions(missing)


def test_unknown_offload_rejected(tmp_path):
    with pytest.raises(ValueError, match="오프로드"): replace(settings(tmp_path), offload="unknown")


def test_dimensions_and_product_preview_preserve_raw(tmp_path):
    assert output_dimensions(736, 1232) == (800, 1312)
    raw = tmp_path / "raw.png"; product = tmp_path / "product.png"
    Image.new("RGB", (800, 1312), "red").save(raw)
    before = file_sha(raw)
    record = product_preview(raw, product)
    assert record["size"] == [736, 1207] and record["offset"] == [0, 12]
    assert file_sha(raw) == before
    with Image.open(product) as p:
        assert p.size == (736,1232)
        assert p.getpixel((0,11)) == (255,255,255)
        assert p.getpixel((0,12)) == (255,0,0)
        assert p.getpixel((0,1219)) == (255,255,255)


def ready_workflow(image):
    class Analyzer:
        def analyze(self, *args): return []
        def close(self): pass
    flow = PoseEditWorkflow(gpu_probe=lambda: {"allocated": 0})
    flow.release_generation(); flow.analyze(image, Analyzer); flow.confirm(approved(image))
    return flow


def test_gpu_order_release_analysis_confirm_launch(image):
    events = []
    class Analyzer:
        def __init__(self): events.append("analyze_load")
        def analyze(self, *args): return []
        def close(self): events.append("analyze_close")
    flow = PoseEditWorkflow((lambda: events.append("generation_close"),), gpu_probe=lambda: events.append("probe"))
    with pytest.raises(RuntimeError): flow.analyze(image, Analyzer)
    flow.release_generation(); flow.analyze(image, Analyzer)
    assert events == ["generation_close", "probe", "analyze_load", "analyze_close", "probe"]
    with pytest.raises(RuntimeError): flow.before_launch(approved(image))
    flow.confirm(approved(image)); flow.before_launch(approved(image))
    assert flow.phase == "qwen_running"


def test_analysis_error_releases_and_allows_manual(image):
    events = []
    class Analyzer:
        def analyze(self, *args): raise ValueError("분석 실패")
        def close(self): events.append("closed")
    flow = PoseEditWorkflow(gpu_probe=lambda: {})
    flow.release_generation()
    reports, failure = flow.analyze(image, Analyzer)
    assert reports == [] and failure["message"] == "분석 실패" and events == ["closed"]
    assert flow.phase == "awaiting_confirmation"


def test_release_failure_and_gpu_residue_block(image):
    def residue(): raise RuntimeError("잔여 GPU")
    flow = PoseEditWorkflow(gpu_probe=residue)
    with pytest.raises(RuntimeError): flow.release_generation()
    assert flow.phase == "blocked"
    with pytest.raises(RuntimeError): flow.confirm(approved(image))


def test_parent_gpu_probe_blocks_allocated(monkeypatch):
    import sys
    cuda = SimpleNamespace(is_initialized=lambda: True, synchronize=lambda: None,
        empty_cache=lambda: None, memory_allocated=lambda: 123, memory_reserved=lambda: 256)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    with pytest.raises(RuntimeError, match="해제되지"): assert_parent_gpu_released()


def test_changed_spec_requires_confirmation(image):
    flow = ready_workflow(image)
    changed = replace(approved(image), revision=2)
    with pytest.raises(RuntimeError): flow.before_launch(changed)


def test_request_hash_and_prompt_validation(image, tmp_path):
    req = make_request(approved(image), image, file_sha(image), settings=settings(tmp_path))
    validate_request(req)
    req["prompt"]["positive"] += " corrupted"
    with pytest.raises(ValueError, match="프롬프트"): validate_request(req)


def test_worker_versions_fail_before_inference_import(image, tmp_path, monkeypatch):
    import genai_lab.qwen_pose_worker as worker
    req = make_request(approved(image), image, file_sha(image), settings=settings(tmp_path))
    write_json(tmp_path / "request.json", req)
    def mismatch(): raise RuntimeError("version mismatch")
    monkeypatch.setattr(worker, "check_runtime_versions", mismatch)
    monkeypatch.setattr(worker, "validate_model_files", lambda _: pytest.fail("must not load"))
    assert worker.main([str(tmp_path / "request.json")]) == 1
    result = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed" and "version mismatch" in result["error"]
    with pytest.raises(SystemExit): worker.main([str(tmp_path / "request.json")])


def test_processor_records_without_truncation():
    calls = []
    def processor(**kw): calls.append(kw); return {"input_ids": SimpleNamespace(shape=(1, 960))}
    rec = {}; wrapped = UntruncatedProcessor(processor, rec)
    wrapped(text=["complete"], images=[], padding=True, return_tensors="pt")
    assert rec["processor_inputs"][0]["sequence_length"] == 960 and calls[0]["text"] == ["complete"]
    with pytest.raises(ValueError): wrapped(text=["x"], truncation=True)


def test_subprocess_request_and_user_review(image, tmp_path):
    req = make_request(approved(image), image, file_sha(image), settings=settings(tmp_path))
    flow = ready_workflow(image)
    out = tmp_path / "edit"
    seen = []
    class Process:
        pid=123; returncode=0
        def __init__(self, command, **kwargs):
            seen.append((command,kwargs))
            Image.new("RGB", (800,1312), "white").save(out / "raw.png")
            write_json(out / "run.json", {"status":"completed", "raw_sha256":file_sha(out/"raw.png"),
                                          "request_sha256":json_sha(req)})
        def poll(self): return self.returncode
    result = run_pose_edit(req, out, flow, popen=Process)
    assert result.exists() and seen[0][0][1:3] == ["-m", "genai_lab.qwen_pose_worker"]
    assert seen[0][1]["shell"] is False
    assert "gguf" not in seen[0][0]
    with pytest.raises(ValueError): save_user_review(out, approved(image), {}, {}, adopted=True, reviewer="user")
    record = save_user_review(out, approved(image), {"a":"변형"}, dict.fromkeys(GLOBAL_FIELDS,"확인 불가"),
                             adopted=False, reviewer="user")
    assert record["adopted"] is False and record["automatic_verdict"] is False


def test_subprocess_failure_not_auto_retried(image, tmp_path):
    req = make_request(approved(image), image, file_sha(image), settings=settings(tmp_path))
    calls = []
    class Process:
        pid=123; returncode=1
        def __init__(self, *a, **k): calls.append(1)
        def poll(self): return 1
    with pytest.raises(RuntimeError, match="프로세스 실패"):
        run_pose_edit(req, tmp_path/"failed", ready_workflow(image), popen=Process)
    assert calls == [1]
    assert json.loads((tmp_path/"failed/launcher.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_cancel_terminates_process_without_retry(image,tmp_path):
    req=make_request(approved(image),image,file_sha(image),settings=settings(tmp_path))
    events=[]
    class Process:
        pid=1; returncode=None
        def __init__(self,*a,**k): events.append("spawn")
        def poll(self): return self.returncode
        def terminate(self): self.returncode=-1; events.append("terminate")
        def wait(self,timeout=None): return -1
    stop=lambda: bool(events)
    with pytest.raises(RuntimeError,match="취소"):
        run_pose_edit(req,tmp_path/"cancel",ready_workflow(image),popen=Process,cancelled=stop)
    assert events==["spawn","terminate"]


def test_existing_output_never_overwritten(image, tmp_path):
    req=make_request(approved(image),image,file_sha(image),settings=settings(tmp_path))
    out=tmp_path/'existing'; out.mkdir(); (out/'launcher.json').write_text('original',encoding='utf-8')
    with pytest.raises(FileExistsError):
        run_pose_edit(req,out,ready_workflow(image),popen=lambda *a,**k: pytest.fail('must not spawn'))
    assert (out/'launcher.json').read_text(encoding='utf-8') == 'original'


def test_model_file_manifest_checks_all_files(tmp_path):
    from genai_lab.qwen_pose_settings import validate_model_files
    root=tmp_path/'model'; root.mkdir()
    names=['model_index.json','scheduler/scheduler_config.json','text_encoder/config.json',
           'text_encoder/model.safetensors','vae/config.json','vae/model.safetensors',
           'transformer/config.json','tokenizer/tokenizer.json','processor/preprocessor_config.json']
    entries={}
    for name in names:
        path=root/name; path.parent.mkdir(exist_ok=True,parents=True); path.write_bytes(b'local')
        entries[name]=file_sha(path)
    manifest=tmp_path/'models.json'; write_json(manifest,{'files':entries})
    gguf=tmp_path/'test.gguf'; gguf.write_bytes(b'gguf')
    cfg=replace(settings(tmp_path),model_root=str(root),gguf_file=str(gguf),gguf_sha256=file_sha(gguf),
                artifact_manifest=str(manifest),artifact_manifest_sha256=file_sha(manifest))
    assert validate_model_files(cfg)['files_checked']==len(names)
    (root/'vae/model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA'): validate_model_files(cfg)


def test_gui_defaults_unconfirmed_and_optional_pose_empty(image,tmp_path):
    from PySide6.QtWidgets import QApplication
    from genai_lab.qwen_pose_gui import QwenPoseDialog
    app=QApplication.instance() or QApplication([])
    dialog=QwenPoseDialog(image,image,settings(tmp_path),PoseEditWorkflow(gpu_probe=lambda:{}),tmp_path)
    dialog.analysis_done(([],None))
    try:
        assert not dialog.approve_image.isChecked()
        assert dialog.viewpoint.text()==dialog.hands.text()==''
        assert dialog.table.cellWidget(0,5).currentIndex()==0
        dialog.add_manual()
        assert dialog.table.rowCount()==2
        assert all(item.status=='draft' for item in dialog.confirmed_spec().items)
    finally:
        dialog.close()


def test_changed_image_after_analysis_cannot_be_reapproved_silently(image):
    flow = ready_workflow(image)
    flow.phase = "awaiting_confirmation"
    Image.new("RGB", (736, 1232), "red").save(image)
    with pytest.raises(ValueError, match="분석 기준"):
        flow.confirm(approved(image))


def test_analyzer_gets_hash_of_exact_opened_bytes(image):
    original = file_sha(image)
    seen = []
    class Analyzer:
        def analyze(self, rgb, sha):
            Image.new("RGB", (736, 1232), "red").save(image)
            seen.append((sha, rgb.getpixel((0, 0))))
            return []
        def close(self): pass
    flow = PoseEditWorkflow(gpu_probe=lambda: {})
    flow.release_generation(); flow.analyze(image, Analyzer)
    assert seen == [(original, (255, 255, 255))]
    with pytest.raises(ValueError): flow.confirm(approved(image))


def test_replay_preparation_cpu_only(image, tmp_path):
    from scripts.qwen_pose_replay import prepare_replay
    root = tmp_path/'repo'; trial = root/'outputs/qwen-pose-identity-20261003'
    trial.mkdir(parents=True)
    source = root/'outputs/source.png'; source.write_bytes(image.read_bytes())
    data = dict(FIXTURE['characters']['raccoon'])
    data.update(start='Z:/repo/outputs/source.png', start_sha256=file_sha(source))
    write_json(trial/'preservation.json', {'characters': {'raccoon': data}})
    write_json(trial/'prompts.json', {'prompts': {'raccoon': {'B': data['expected']}}})
    write_json(trial/'manifest.json', {'skeleton': {'path':'Z:/repo/outputs/source.png', 'sha256':file_sha(source)}})
    spec, request = prepare_replay(root, 'raccoon', settings(tmp_path))
    assert request['prompt']['positive'] == data['expected']
    assert request['images'] == 2 and request['seed'] == 209212001
