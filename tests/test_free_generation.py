"""자유 생성 계약·실패·재시도·출력 좌표 검증. 모든 모델 실행은 모의 처리한다."""
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
from genai_lab import free_generation_pipeline as pipeline
from genai_lab.character_features import TAG_ITEMS
from genai_lab.free_generation import check_request, free_conditions, part_counts
from genai_lab.free_generation_results import FreeGenerationResults
from genai_lab.outfit_edit import sha, write_json
from genai_lab.qwen_free_generation import prepare_free_spec, make_free_request
from genai_lab.qwen_pose_settings import QwenPoseSettings
from genai_lab.qwen_pose_worker import validate_request


def features():
    return {k: ["long_hair"] if k == "hair_length" else [] for k in TAG_ITEMS}


def settings(tmp):
    return QwenPoseSettings(sys.executable, str(tmp), str(tmp/"m.gguf"), str(tmp/"m.json"), "a"*64)


@pytest.mark.parametrize("text", ["", "눈 내리는 거리", "눈을 감은 표정", "머리를 기울여 서기", "close the eyes in a snowy street", "walking at sunset", "귀를 기울이는 자세"])
def test_staging_words_do_not_imply_appearance_change(text):
    assert check_request(text)["ok"]
    assert check_request(text)["semantic_status"] == "not_reviewed"


@pytest.mark.parametrize("text", ["머리색을 바꿔", "remove the tail", "change eye color to blue", "wear a new dress", "누드", "make the character childlike"])
def test_explicit_appearance_and_safety_conflicts_are_rejected(text):
    assert not check_request(text)["ok"]


def test_background_is_a_condition_and_white_is_excluded_only_with_scene():
    scene = free_conditions("", "female", (), "눈 내리는 거리", 42)
    assert next(c for c in scene["conditions"] if c["key"] == "background")["source"] == "user"
    face = free_conditions("", "female", (), "눈을 감은 표정", 42)
    assert "white background" in next(c for c in face["conditions"] if c["key"] == "background")["text"]


def test_components_are_not_anatomical_count():
    mask = np.zeros((50,50), bool)
    mask[2:20,2:20] = True
    mask[25:45,25:45] = True
    report = part_counts({"tail":mask})
    assert report["tail"]["mask_components"] == 2
    assert report["tail"]["count"] is None and report["tail"]["status"] == "unmeasurable"


def spec_at(tmp):
    path = tmp/"c.png"
    Image.new("RGB", (768,1024), "white").save(path)
    return prepare_free_spec(path,path,directory=tmp/"spec",request_text="",body_parts=(),base_kind="original")


def test_worker_validates_same_direction_for_both_seeds(tmp_path):
    spec = spec_at(tmp_path)
    first, second = (make_free_request(spec, settings=settings(tmp_path), seed=n) for n in (42,43))
    assert first["prompt"] == second["prompt"]
    assert first["seed"] != second["seed"]
    assert validate_request(json.loads(json.dumps(second))).selection_seed == 42


@pytest.mark.parametrize("mutation", [lambda r:r.update(seed=5), lambda r:r["free_spec"].update(confirmed=False),
    lambda r:r["free_spec"].update(body_parts=None), lambda r:r["free_spec"].update(gender="bad"),
    lambda r:r["free_spec"].update(base_kind="bad"), lambda r:r["prompt"].update(positive_sha256="b"*64),
    lambda r:r["memory"].update(reserved_limit_gib=8), lambda r:r["prompt"].update(positive="other")])
def test_tampering_is_rejected(tmp_path, mutation):
    request = make_free_request(spec_at(tmp_path),settings=settings(tmp_path))
    mutation(request)
    with pytest.raises(ValueError):
        validate_request(request)


@pytest.fixture
def fake_models(tmp_path, monkeypatch):
    base = tmp_path/"base.png"
    Image.new("RGB",(768,1024),"white").save(base)
    calls = []
    def segment(prepared, directory, runtime, **kwargs):
        directory.mkdir()
        mask = np.zeros((1024,768),bool)
        # 출력 마스크 위치가 기준과 다르다. 기준 마스크 재사용이면 검사에서 걸린다.
        mask[300:450,400:550] = True if "result-analysis" in str(directory) else False
        if "result-analysis" not in str(directory):
            mask[40:100,30:100] = True
        np.savez(directory/"parts.npz", hair=mask, face=mask, eyes=mask)
        calls.append(("segmentation",str(directory),prepared["canvas_sha256"]))
        return dict(parts=str(directory/"parts.npz"))
    def generate(request, directory, flow, **kwargs):
        directory.mkdir()
        write_json(directory/"request.json",request)
        Image.new("RGB",(896,1184),"white").save(directory/"raw.png")
        calls.append(("qwen",request))
        return directory/"raw.png"
    monkeypatch.setattr(pipeline,"segment_canvas",segment)
    monkeypatch.setattr(pipeline,"run_free_generation_edit",generate)
    monkeypatch.setattr(pipeline,"measure_result_scores",lambda *a:{"long_hair":.9,"1girl":.9})
    monkeypatch.setattr(pipeline,"detect_joints",lambda *a,**kw:[])
    return base,calls


def run(tmp, base, **kwargs):
    return pipeline.run_free_generation(base,features(),base_kind="original",gender="female",body_parts=(),
        confirmed=True,adult_confirmed=True,settings=settings(tmp),runtime=SimpleNamespace(),directory=tmp/"run",**kwargs)


def test_completed_output_is_pending_and_segmented_in_its_own_coordinates(tmp_path,fake_models):
    base,calls = fake_models
    result = run(tmp_path,base)
    trace = json.loads(Path(result["trace"]).read_text(encoding="utf-8"))
    assert result["status"] == "awaiting_user_review" and result["quality_pass"] is None
    assert not result["composited"] and result["result"].endswith("raw.png")
    assert [c[0] for c in calls] == ["segmentation","qwen","segmentation"]
    assert "result-analysis" in calls[-1][1] and calls[-1][1] != calls[0][1]
    obs = trace["attempts"][0]["observation"]
    assert obs["coordinate_basis"] == "generated_output" and obs["limbs"]["status"] == "unmeasurable"
    assert next(j for j in result["condition_judgements"] if j["key"] == "keep_character")["status"] == "not_reviewed"
    assert trace["input_facts"]["base"]["kind"] == "original" and trace["input_facts"]["base"]["proves_outfit_edit"] is False
    results = FreeGenerationResults(result)
    with pytest.raises(ValueError):
        results.export(tmp_path/"saved.png")
    results.approve(dict(character=True,garment=True,request=True,exposure=True))
    results.export(tmp_path/"saved.png")
    human = json.loads(Path(result["trace"]).read_text(encoding="utf-8"))["human_judgements"][-1]
    assert human["judge"] == "user" and human["target"]["sha256"] == sha(tmp_path/"saved.png")
    assert json.loads((tmp_path/"saved.review.json").read_text(encoding="utf-8"))["head_composited"] is False


def test_automatic_retry_keeps_direction_and_pending_quality(tmp_path,fake_models,monkeypatch):
    base,calls = fake_models
    monkeypatch.setattr(pipeline,"verify_features",lambda *a:dict(items={"hair_length":dict(status="changed")},summary=dict(changed=["hair_length"])))
    result = run(tmp_path,base)
    requests = [c[1] for c in calls if c[0] == "qwen"]
    assert [r["seed"] for r in requests] == [42,43]
    assert requests[0]["prompt"] == requests[1]["prompt"]
    assert result["status"] == "awaiting_user_review" and result["quality_pass"] is None


def test_failure_preserves_attempt_request_and_never_returns_source_as_success(tmp_path,fake_models,monkeypatch):
    base,_ = fake_models
    monkeypatch.setattr(pipeline,"run_free_generation_edit",lambda *a,**k:(_ for _ in ()).throw(RuntimeError("memory stopped")))
    with pytest.raises(RuntimeError,match="memory stopped"):
        run(tmp_path,base)
    trace = json.loads((tmp_path/"run/trace.json").read_text(encoding="utf-8"))
    assert trace["status"] == "failed" and trace["attempts"][0]["request"]["seed"] == 42
    assert trace["judgement"]["quality_pass"] is None and not (tmp_path/"run/outcome.json").exists()


def test_cancel_before_generation_records_cancelled(tmp_path,fake_models):
    base,calls = fake_models
    with pytest.raises(RuntimeError):
        run(tmp_path,base,cancelled=lambda:True)
    assert json.loads((tmp_path/"run/trace.json").read_text(encoding="utf-8"))["status"] == "cancelled" and not calls


def test_edited_base_requires_matching_human_approval(tmp_path,fake_models):
    base,_ = fake_models
    args = (base,"edited",features(),"female",(),"",True,True)
    with pytest.raises(FileNotFoundError):
        pipeline.validate_inputs(*args)
    sidecar = base.with_suffix(".review.json")
    write_json(sidecar,dict(mode="outfit_edit",reviewer="user",sha256=sha(base),checks=dict(character=True,garment=True,exposure=True,original_outfit_gone=True)))
    _,record,_ = pipeline.validate_inputs(*args)
    assert record["approval"]["sha256"] == sha(sidecar)
    Image.new("RGB",(768,1024),"red").save(base)
    with pytest.raises(ValueError,match="승인"):
        pipeline.validate_inputs(*args)


def test_missing_analysis_is_unmeasurable_and_no_source_masks_are_used(tmp_path,fake_models,monkeypatch):
    base,_ = fake_models
    original = pipeline.segment_canvas
    def segment(prepared,directory,*args,**kwargs):
        if "result-analysis" in str(directory):
            raise ValueError("segmentation unavailable")
        return original(prepared,directory,*args,**kwargs)
    monkeypatch.setattr(pipeline,"segment_canvas",segment)
    result = run(tmp_path,base)
    obs = result["attempts"][0]["observation"]
    assert obs["errors"]["segmentation"] == "segmentation unavailable"
    assert obs["body_part_counts"]["tail"]["count"] is None


def test_unsafe_result_cannot_be_approved(tmp_path,fake_models,monkeypatch):
    base,_ = fake_models
    readings = iter([{"long_hair":.9}, {"nude":.9}, {"nude":.9}])
    monkeypatch.setattr(pipeline,"measure_result_scores",lambda *a:next(readings))
    result = run(tmp_path,base)
    assert result["status"] == "blocked" and len(result["attempts"]) == 2
    with pytest.raises(ValueError,match="차단"):
        FreeGenerationResults(result)


def test_derived_reference_must_be_crop_of_same_basis(tmp_path):
    spec = spec_at(tmp_path)
    Image.new("RGB",(768,1024),"red").save(spec.reference_path)
    # SHA도 바꾸더라도 별도 참조를 주입할 수 없어야 한다.
    with pytest.raises(ValueError,match="같은 기준"):
        replace(spec,reference_sha256=sha(spec.reference_path)).verify_image()


def test_source_snapshot_survives_changes_to_original(tmp_path,fake_models):
    base,_=fake_models
    result=run(tmp_path,base)
    trace=json.loads(Path(result["trace"]).read_text(encoding="utf-8"))
    fact=trace["input_facts"]["base"]
    Image.new("RGB",(768,1024),"blue").save(base)
    assert sha(fact["snapshot"])==fact["sha256"] and sha(base)!=fact["sha256"]
    results=FreeGenerationResults(result)
    Image.new("RGB",(896,1184),"red").save(results.current_path)
    with pytest.raises(ValueError,match="변경"):
        results.approve(dict(character=True,garment=True,request=True,exposure=True))


def test_confirmed_adult_still_rejects_forbidden_input_observation(tmp_path,fake_models,monkeypatch):
    base,calls=fake_models
    monkeypatch.setattr(pipeline,"measure_result_scores",lambda *a:{"nude":.9})
    with pytest.raises(ValueError,match="금지"):
        run(tmp_path,base)
    assert not calls


def test_result_measurement_receives_only_result_masks(tmp_path,fake_models,monkeypatch):
    base,_=fake_models
    received=[]
    def verify(reference,scores,rgb,parts):
        assert parts["hair"][350,450] and not parts["hair"][50,50]
        received.append(True)
        return dict(items={},summary={})
    monkeypatch.setattr(pipeline,"verify_features",verify)
    run(tmp_path,base)
    assert received==[True]


def test_selection_seed_is_independent_and_explicitly_reproducible(tmp_path,fake_models):
    base,calls=fake_models
    result=run(tmp_path,base,selection_seed=91)
    req=next(c[1] for c in calls if c[0]=="qwen")
    assert req["seed"]==42 and req["free_spec"]["selection_seed"]==91
    assert validate_request(json.loads(json.dumps(req))).selection_seed==91
    trace=json.loads(Path(result["trace"]).read_text(encoding="utf-8"))
    assert next(c for c in trace["conditions"] if c["key"]=="request")["basis"]["seed"]==91


def test_requested_composition_is_not_overridden_with_full_body():
    conditions=free_conditions("","female",(),"close-up portrait, eyes closed",42)
    assert not any(c["key"]=="framing" for c in conditions["conditions"])
    assert any(c["key"]=="framing" for c in conditions["excluded"])
