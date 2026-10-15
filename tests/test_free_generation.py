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


@pytest.mark.parametrize("text", ["", "눈 내리는 거리", "눈을 감은 표정", "머리를 기울여 서기", "close the eyes in a snowy street", "walking at sunset", "귀를 기울이는 자세",
                                  # 검토 3번 오거부였던 문장, SD·4등신 그림체(사용자 결정 2026-10-14)
                                  "change the pose so her hair flows in the wind", "노을이 길게 드리운 거리에서 꼬리를 흔들며 걷기",
                                  "change the background to a beach while her hair blows", "holding her hat in one hand", "chibi style, sitting"])
def test_staging_words_do_not_imply_appearance_change(text):
    assert check_request(text)["ok"]
    assert check_request(text)["semantic_status"] == "not_reviewed"


@pytest.mark.parametrize("text", ["머리색을 바꿔", "remove the tail", "change eye color to blue", "wear a new dress", "누드", "make the character childlike",
                                  # 검토 3번: 머리 장식 추가·길이 변경도 외형 변경이다
                                  "give her a hat", "put glasses on her", "add a crown", "머리를 짧게 잘라", "꼬리를 없애고"])
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
    def verify(reference,scores,rgb,parts,skip):
        assert parts["hair"][350,450] and not parts["hair"][50,50] and skip == {}
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


def test_safety_observation_blocks_exposure_but_only_records_age_style_tags():
    # 검토 2번: 태거 어휘에 있는 노출 태그로 차단한다. child·chibi는 SD·4등신 그림체가 있어 차단하지 않고 근거로만 남긴다
    from genai_lab.free_generation import safety_observation
    assert safety_observation({"completely_nude": .9, "topless": .6})["blocked"] == ["completely_nude", "topless"]
    observed = safety_observation({"child": .8, "chibi": .7, "1girl": .9})
    assert observed["blocked"] == [] and observed["review"] == {"child": .8, "chibi": .7}
    assert safety_observation({"loli": .6})["blocked"] == ["loli"]


def test_result_with_completely_nude_is_blocked(tmp_path,fake_models,monkeypatch):
    base,_ = fake_models
    readings = iter([{"long_hair":.9}, {"completely_nude":.9}, {"completely_nude":.9}])
    monkeypatch.setattr(pipeline,"measure_result_scores",lambda *a:next(readings))
    result = run(tmp_path,base)
    assert result["status"] == "blocked"
    with pytest.raises(ValueError,match="차단"):
        FreeGenerationResults(result)


def test_when_both_attempts_violate_the_one_with_fewer_violations_is_shown(tmp_path,fake_models,monkeypatch):
    # 검토 7번: 두 시도 모두 자동 위반이면 위반이 적은 시도를 보여 준다. 두 시도 모두 trace에 남는다
    base,_ = fake_models
    reports = iter([dict(items={}, summary=dict(changed=["hair_length"], added=[])),
                    dict(items={}, summary=dict(changed=["hair_length", "eye_color"], added=["head_accessories"]))])
    monkeypatch.setattr(pipeline,"verify_features",lambda *a:next(reports))
    result = run(tmp_path,base)
    assert result["selected_attempt"] == 1 and result["result"].endswith(str(Path("qwen") / "raw.png"))
    assert [a["violation_score"] for a in result["attempts"]] == [1, 3]
    trace = json.loads(Path(result["trace"]).read_text(encoding="utf-8"))
    assert trace["selected_attempt"] == 1 and len(trace["attempts"]) == 2


def test_free_judgement_uses_the_common_format_and_is_counted_by_the_summary(tmp_path,fake_models):
    # 검토 4번: 자유 생성 판정도 D-114 형식(recorded_at·항목 검증)으로 남고 운영 집계(D-118)에 따로 잡힌다
    import subprocess
    base,_ = fake_models
    root = tmp_path/"studio"
    result = pipeline.run_free_generation(base,features(),base_kind="original",gender="female",body_parts=(),
        confirmed=True,adult_confirmed=True,settings=settings(tmp_path),runtime=SimpleNamespace(),directory=root/"r1"/"free-generation")
    results = FreeGenerationResults(result)
    results.approve(dict(character=True,garment=True,request=True,exposure=True))
    entry = json.loads((root/"r1"/"free-generation"/"judgement.json").read_text(encoding="utf-8"))[-1]
    assert entry["recorded_at"] and entry["request_followed"] == "yes" and entry["mode"] == "free_generation"
    assert entry["quality_pass"] is True and entry["target"]["attempt"] == 1
    script = Path(__file__).resolve().parents[1]/"scripts"/"studio_judgement_summary.py"
    out = subprocess.run([sys.executable,"-B",str(script),"--root",str(root),"--json"],capture_output=True,text=True,check=True,encoding="utf-8")
    summary = json.loads(out.stdout)
    assert summary["total"] == 0 and summary["free_generation"]["judged"] == 1 and summary["free_generation"]["quality_pass"] == 1


def test_unknown_features_are_kept_apart_from_confirmed_none(tmp_path,fake_models):
    # 검토 5번: 모름(None)은 unknown, 없음([])은 사용자가 확정한 값으로 남는다
    base,_ = fake_models
    confirmed = dict(features(), skin=None)
    result = pipeline.run_free_generation(base,confirmed,base_kind="original",gender="female",body_parts=(),
        confirmed=True,adult_confirmed=True,settings=settings(tmp_path),runtime=SimpleNamespace(),directory=tmp_path/"run")
    sheet = json.loads((Path(result["directory"])/"character-features"/"features.json").read_text(encoding="utf-8"))
    assert sheet["items"]["skin"]["state"] == "unknown" and "skin" in sheet["unknown"]
    assert sheet["items"]["head_accessories"]["confirmed"] is True and sheet["items"]["head_accessories"]["values"] == []


def sheet(**items):
    base = {k: dict(state="unknown") for k in TAG_ITEMS}
    base.update(items)
    return dict(items=base)


def test_placement_free_hair_tags_leave_the_prompt_and_english_notes_enter_it():
    from genai_lab.free_generation import HAIR_PLACEMENT, features_sentence
    text = features_sentence(sheet(hair_color=dict(state="measured", tags=["grey_hair", "multicolored_hair", "streaked_hair"]),
                                   distinctive=dict(state="measured", values=["heart-shaped tail tip", "꼬리 끝 하트"])))
    assert "grey hair" in text and "multicolored" not in text and "streaked" not in text and HAIR_PLACEMENT in text
    assert "Keep these distinctive details exactly as in image 1: heart-shaped tail tip. " in text and "꼬리" not in text


@pytest.mark.parametrize("note", ["change the tail to a heart", "nude", "꼬리를 없애고"])
def test_distinctive_notes_reject_change_and_unsafe_phrases(note):
    from genai_lab.free_generation import distinctive_notes
    with pytest.raises(ValueError):
        distinctive_notes([note])


def test_request_conflicts_and_garment_tag_changes_are_observations():
    from genai_lab.free_generation import garment_tag_changes, request_conflicts
    assert request_conflicts("standing with eyes closed, head tilted slightly") == {"eye_color": "요청과 충돌: 눈 감기"}
    assert set(request_conflicts("뒷모습으로 걷기")) == {"eye_color", "skin"} and request_conflicts("눈 내리는 거리") == {}
    changes = garment_tag_changes(dict(shorts=0.8, crop_top=0.7, long_hair=0.9), dict(pants=0.8, crop_top=0.6))
    assert changes["lost"] == ["shorts"] and changes["added"] == ["pants"] and changes["status"] == "observed"


def test_reference_crop_follows_the_face_and_wide_crop_changes_the_wording(tmp_path):
    canvas = Image.new("RGB", (768, 1024), "white")
    canvas.save(tmp_path / "canvas.png")
    parts = {k: np.zeros((1024, 768), bool) for k in ("hair", "face", "eyes")}
    parts["face"][100:160, 350:410] = True
    parts["hair"][60:900, 300:460] = True               # 긴 양갈래가 몸까지 내려간다(N1)
    path, record = pipeline.prepare_reference(canvas, parts, tmp_path)
    assert record["status"] == "derived_head_crop" and record["scope"] == "head" and record["box"][3] < 330
    no_face = {k: v for k, v in parts.items() if k != "face"}
    no_face["eyes"] = np.zeros((1024, 768), bool)
    _, wide = pipeline.prepare_reference(canvas, no_face, tmp_path)
    assert wide["status"] == "hair_box_fallback" and wide["scope"] == "wide"
    text = "".join(c["text"] for c in free_conditions("", "female", (), "", 42, reference_scope="wide")["conditions"])
    assert "cropped reference in image 2" in text and "head close-up" not in text


def test_only_confident_violations_retry(tmp_path,fake_models,monkeypatch):
    # D-126 ⑤: 경계값·엇갈림처럼 retry에 없는 바뀜은 재시도하지 않는다
    base,calls = fake_models
    monkeypatch.setattr(pipeline,"verify_features",lambda *a:dict(items={},summary=dict(changed=["eye_color"]),retry=[]))
    result = run(tmp_path,base)
    assert len([c for c in calls if c[0] == "qwen"]) == 1 and result["attempts"][0]["retry_reason"] is None


def test_both_attempts_are_candidates_and_the_user_choice_is_recorded(tmp_path,fake_models,monkeypatch):
    base,_ = fake_models
    reports = iter([dict(items={}, summary={}, retry=["hair_color", "eye_color"]), dict(items={}, summary={}, retry=["hair_color"])])
    monkeypatch.setattr(pipeline,"verify_features",lambda *a:next(reports))
    result = run(tmp_path,base)
    assert [c["attempt"] for c in result["candidates"]] == [1, 2] and result["selected_attempt"] == 2
    results = FreeGenerationResults(result)
    assert results.candidate["attempt"] == 2 and results.current_path.parent.name == "qwen-2"
    results.select(0)
    assert results.current_path.parent.name == "qwen"
    with pytest.raises(ValueError):
        results.select(2)
    results.approve(dict(character=True,garment=True,request=True,exposure=True))
    entry = json.loads((Path(result["directory"])/"judgement.json").read_text(encoding="utf-8"))[-1]
    assert entry["target"]["attempt"] == 1 and entry["target"]["sha256"] == sha(Path(result["directory"])/"qwen"/"raw.png")


def test_distinctive_notes_must_be_confirmed_at_approval(tmp_path,fake_models):
    base,_ = fake_models
    confirmed = dict(features(), distinctive=["heart-shaped tail tip"])
    result = pipeline.run_free_generation(base,confirmed,base_kind="original",gender="female",body_parts=(),
        confirmed=True,adult_confirmed=True,settings=settings(tmp_path),runtime=SimpleNamespace(),directory=tmp_path/"run")
    assert result["distinctive_review"] == ["heart-shaped tail tip"]
    request = json.loads((Path(result["directory"])/"qwen"/"request.json").read_text(encoding="utf-8"))
    assert "heart-shaped tail tip" in request["prompt"]["positive"]
    results = FreeGenerationResults(result)
    with pytest.raises(ValueError):
        results.approve(dict(character=True,garment=True,request=True,exposure=True))
    results.approve(dict(character=True,garment=True,request=True,exposure=True,distinctive=True))
