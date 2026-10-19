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
    results.approve(dict(character=True,garment=True,request=True,exposure=True,finish=True))
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
        results.approve(dict(character=True,garment=True,request=True,exposure=True,finish=True))


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
        assert parts["hair"][350,450] and not parts["hair"][50,50] and set(skip) <= {"hands"}   # 무작위 고정 자세 중 뒷짐(D-128)
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
    results.approve(dict(character=True,garment=True,request=True,exposure=True,finish=True))
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


def test_skin_is_kept_by_tone_not_by_tag_words():
    from genai_lab.free_generation import SKIN_TONE, features_sentence
    text = features_sentence(sheet(skin=dict(state="measured", tags=["dark_skin"])))
    assert SKIN_TONE in text and "dark skin" not in text and "dark_skin" not in text
    assert SKIN_TONE not in features_sentence(sheet(skin=dict(state="measured", tags=["dark_skin"])), hidden={"skin": "x"})
    assert SKIN_TONE not in features_sentence(sheet(skin=dict(state="measured", tags=[])))


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
    results.approve(dict(character=True,garment=True,request=True,exposure=True,finish=True))
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
        results.approve(dict(character=True,garment=True,request=True,exposure=True,finish=True))
    results.approve(dict(character=True,garment=True,request=True,exposure=True,finish=True,distinctive=True))


def test_manual_free_judgement_does_not_use_the_head_paste_rule(tmp_path,fake_models):
    # 편집 단계 기준(머리 합성 안 됨 = 실패)이 자유 생성 수동 기록에 쓰이지 않는다(2026-10-15 K-1 기록에서 발견)
    from genai_lab.outfit_judgement import record_judgement
    base,_ = fake_models
    result = run(tmp_path,base)
    directory = Path(result["directory"])
    record_judgement(directory, dict(judge="user", note="눈 색만 봄"))
    assert json.loads((directory/"judgement.json").read_text(encoding="utf-8"))[-1]["quality_pass"] is None
    record_judgement(directory, dict(judge="user", same_character="yes", outfit="same", request_followed="yes", severe_defect="none",
                                     finish_quality="good"))                    # 완성도까지 판정해야 통과(D-133)
    trace = json.loads((directory/"trace.json").read_text(encoding="utf-8"))
    assert trace["human_judgements"][-1]["quality_pass"] is True and trace["judgement"]["quality"]["quality_pass"] is True
    record_judgement(directory, dict(judge="user", same_character="no"))
    assert json.loads((directory/"judgement.json").read_text(encoding="utf-8"))[-1]["quality_pass"] is False


def test_request_hidden_features_leave_the_prompt_but_stay_for_review(tmp_path,fake_models,monkeypatch):
    # D-127 (요청 21 K-2): 눈 색 설명·눈 메모 + eyes closed → 눈을 뜸. 눈 감기 요청이면 지시문에서 눈 색을 빼고 확인 체크로만 남긴다
    from genai_lab.free_generation import features_sentence
    sheet_items = sheet(eye_color=dict(state="measured", tags=["heterochromia", "blue_eyes"]),
                        distinctive=dict(state="measured", values=["blue eye on the viewer's left, red eye on the right", "heart-shaped tail tip"]))
    hidden = features_sentence(sheet_items, hidden={"eye_color": "요청과 충돌: 눈 감기"})
    assert "heterochromia" not in hidden and "viewer's left" not in hidden and "heart-shaped tail tip" in hidden
    assert "heterochromia" in features_sentence(sheet_items)
    base,calls = fake_models
    monkeypatch.setattr(pipeline,"measure_result_scores",lambda *a:{"long_hair":.9,"1girl":.9,"closed_eyes":.01})
    confirmed = dict(features(), eye_color=["blue_eyes"], distinctive=["blue eye on the left", "heart-shaped tail tip"])
    result = pipeline.run_free_generation(base,confirmed,"standing with eyes closed",base_kind="original",gender="female",body_parts=(),
        confirmed=True,adult_confirmed=True,settings=settings(tmp_path),runtime=SimpleNamespace(),directory=tmp_path/"run")
    prompt = next(c[1] for c in calls if c[0] == "qwen")["prompt"]["positive"]
    assert "blue eyes" not in prompt and "blue eye on the left" not in prompt and "heart-shaped tail tip" in prompt
    assert result["distinctive_review"] == ["blue eye on the left", "heart-shaped tail tip"]
    trace = json.loads(Path(result["trace"]).read_text(encoding="utf-8"))
    assert any(e["key"] == "distinctive" and e["notes"] == ["blue eye on the left"] for e in trace["excluded"])
    row = next(j for j in result["condition_judgements"] if j["key"] == "request.eyes_closed")
    assert row["status"] == "needs_review" and row["evidence"]["status"] == "not_observed"
    assert len([c for c in calls if c[0] == "qwen"]) == 1                          # 요청 관찰은 재시도 근거가 아니다


def test_automatic_kept_is_not_fulfilled():
    # D-127: 자동 '유지'는 충족이 아니다 — 위반 관찰 없음, 여유가 적으면 확인 필요. 충족은 사람만
    observation = dict(verification=dict(items=dict(hair_color=dict(status="kept", review=["색 차이 12.3 / 문턱 15"]),
                                                    skin=dict(status="kept", review=[]))),
                       safety=dict(status="not_reviewed"), request_tags=[])
    rows = {r["key"]: r["status"] for r in pipeline.condition_judgements([], observation, 1, Path(__file__))}
    assert rows == {"feature.hair_color": "needs_review", "feature.skin": "not_violated", "safety": "not_reviewed"}


@pytest.mark.parametrize("text,hidden", [("standing with her hands behind her back", {"hands"}), ("뒷짐을 지고 서기", {"hands"}),
                                         ("hands in his pockets, walking", {"hands"}), ("sitting so that her tail is hidden", {"tail", "body_parts"}),
                                         ("꼬리를 숨기고 앉기", {"tail", "body_parts"}), ("waving with one hand", set()), ("tail swaying", set())])
def test_hands_and_tail_hiding_requests_are_conflicts(text, hidden):
    # D-128: 손·꼬리를 가리는 연출도 눈 감기와 같은 원칙(사용자 결정 2026-10-15)
    from genai_lab.free_generation import request_conflicts
    assert check_request(text)["ok"] and set(request_conflicts(text)) == hidden


def test_hidden_hands_and_tail_are_not_demanded_visible_in_the_prompt():
    from genai_lab.free_generation import features_sentence, staging_text, POSES
    items = sheet(distinctive=dict(state="measured", values=["heart-shaped tip at the end of the tail", "black claw-like fingernails", "two horns"]))
    hands = features_sentence(items, hidden={"hands": "x"})
    assert "fingernails" not in hands and "heart-shaped" in hands and "two horns" in hands
    tail = features_sentence(items, hidden={"tail": "x", "body_parts": "x"})
    assert "heart-shaped" not in tail and "fingernails" in tail
    text = "".join(c["text"] for c in free_conditions("", "female", ("tail", "wings"), "sitting so that her tail is hidden", 42)["conditions"])
    assert "the tail may be hidden by the requested pose, but do not remove it or add extra tails" in text
    plain = "".join(c["text"] for c in free_conditions("", "female", ("tail",), "sitting on a chair", 42)["conditions"])
    assert "keep them exactly as in image 1" in plain and "may be hidden" not in plain
    seed = next(i for i, p in enumerate(POSES) if "behind the back" in p)      # 빈 요청의 고정 자세도 가림 기준
    assert staging_text("", seed) == POSES[seed] and "hands" in __import__("genai_lab.free_generation", fromlist=["x"]).request_conflicts(staging_text("", seed))


def test_hands_hidden_request_is_observed_with_tags():
    from genai_lab.free_generation import request_observations
    seen = request_observations("standing with her hands behind her back", {"arms_behind_back": 0.8})
    assert seen == [dict(key="hands_hidden", status="observed", tags=["arms_behind_back", "hands_in_pockets", "hand_in_pocket"], score=0.8,
                         note="태그 관찰. 요청 반영 판정은 사람이 한다")]
    assert request_observations("hands in pockets", {})[0]["status"] == "not_observed"


def test_pose_and_head_accessory_tags_are_not_garment_changes():
    # 요청 23 N2-hands: 뒷짐이 hands_in_pockets(의상 분류)로 잡혀 '의상 생김'에 섞였다. 머리 장식은 특징표가 따로 본다
    from genai_lab.free_generation import garment_tag_changes
    changes = garment_tag_changes(dict(shorts=0.8, hair_bow=0.9), dict(shorts=0.8, hands_in_pockets=0.7, holding_hat=0.6, hat=0.8, pants=0.6))
    assert changes["lost"] == [] and changes["added"] == ["pants"]


def test_scene_requests_ask_to_keep_own_colours_but_white_background_does_not():
    # D-129: 요청 21 N2-2 눈길에서 분홍 머리가 회색으로(흰 배경 요청 23 N2-hands는 유지) → 장면 요청일 때만 고유 색 유지 문장
    scene = {c["key"]: c for c in free_conditions("", "male", ("ears", "tail"), "walking in a snowy street", 42)["conditions"]}
    assert scene["scene_colors"]["text"].startswith("Keep the character's own hair, eye, skin and outfit colors")
    white = {c["key"] for c in free_conditions("", "male", ("ears", "tail"), "", 42)["conditions"]}
    assert "scene_colors" not in white
    closed = {c["key"]: c for c in free_conditions("", "female", (), "standing with eyes closed in a snowy street", 42)["conditions"]}
    assert "eye" not in closed["scene_colors"]["basis"]["items"]          # 가린 특징은 보이게 요구하지 않는다(D-127)


def test_hair_restore_moves_only_hair_colour_toward_the_reference():
    # D-130: 눈길에서 회색이 된 분홍 머리(요청 21·24 N2-2) — 머리 색감만 기준 분포로, 명암·머리 밖은 그대로
    from genai_lab.character_features import to_lab
    from genai_lab.hair_color_restore import hair_ab_quantiles, restore_hair_color
    reference = np.full((60, 60, 3), (230, 170, 195), np.uint8)
    reference[:, 30:] = (200, 190, 196)                                       # 분홍 반, 회분홍 반
    quantiles = hair_ab_quantiles(reference, np.ones((60, 60), bool))
    result = np.full((80, 80, 3), (190, 205, 225), np.uint8)                 # 푸른 배경
    result[10:50, 10:50] = (185, 185, 190)                                    # 회색 머리
    result[10:50, 10:30] = (150, 150, 155)                                    # 그늘
    mask = np.zeros((80, 80), bool)
    mask[10:50, 10:50] = True
    restored, report = restore_hair_color(result, mask, quantiles)
    assert report["status"] == "restored" and report["chroma_after"] > report["chroma_before"] + 5
    assert report["lightness_change"] < 3
    assert (restored[70:, 70:] == result[70:, 70:]).all()                    # 머리 밖은 그대로
    lab = to_lab(restored)
    assert lab[30, 20, 0] < lab[30, 40, 0]                                    # 그늘은 그늘대로(명암 유지)
    assert restore_hair_color(result, np.zeros((80, 80), bool), quantiles)[0] is None


def test_hair_restored_candidate_is_offered_but_not_selected(tmp_path,fake_models,monkeypatch):
    base,_ = fake_models
    monkeypatch.setattr(pipeline,"verify_features",lambda *a:dict(items=dict(hair_color=dict(status="kept",review=["색 있는 머리 비중 감소"])),
                                                                  summary=dict(review=["hair_color"]),retry=[]))
    result = run(tmp_path,base)
    variants = [(c["attempt"], c["variant"]) for c in result["candidates"]]
    assert variants == [(1, "raw"), (1, "hair_restored")]
    restored = result["candidates"][1]
    assert restored["raw"].endswith("hair-restored.png") and restored["hair_restore"]["status"] == "restored"
    results = FreeGenerationResults(result)
    assert results.candidate["variant"] == "raw"                              # 처음엔 원 결과
    results.select(1)
    assert "머리색 보정본" in results.head_notice
    results.approve(dict(character=True,garment=True,request=True,exposure=True,finish=True))
    entry = json.loads((Path(result["directory"])/"judgement.json").read_text(encoding="utf-8"))[-1]
    assert entry["target"]["variant"] == "hair_restored" and entry["target"]["sha256"] == restored["sha256"]
    saved = results.export(tmp_path/"out.png")
    assert json.loads(saved.with_suffix(".review.json").read_text(encoding="utf-8"))["variant"] == "hair_restored"


@pytest.mark.parametrize("text,code", [("nude", "unsafe"), ("two characters standing together", "multiple_characters"),
                                       ("wear a new dress", "outfit_change"), ("머리색을 바꿔", "appearance_change"),
                                       ("give her a hat", "appearance_change"), ("make the character childlike", "unsafe")])
def test_rejections_carry_a_fixed_code_for_external_callers(text, code):
    # D-133 호환성: 외부 프롬프트 작성기가 처리할 수 있게 고정 거부 코드를 준다(사람용 문장은 그대로)
    from genai_lab.free_generation import FreeGenerationRejected, REJECTION_CODES
    checked = check_request(text)
    assert not checked["ok"] and checked["code"] == code and code in REJECTION_CODES
    with pytest.raises(FreeGenerationRejected) as error:
        free_conditions("", "female", (), text, 42)
    assert error.value.code == code and isinstance(error.value, ValueError)
    with pytest.raises(FreeGenerationRejected) as too_long:
        check_request("a" * 2001)
    assert too_long.value.code == "request_too_long"


def test_headless_call_returns_versioned_outcome_and_coded_rejection(tmp_path,fake_models):
    # GUI 없이 공통 함수를 부른다(향후 API와 같은 길). 성공은 버전 있는 결과, 거부는 코드가 trace에 남는다
    from genai_lab.free_generation import FreeGenerationRejected
    base,_ = fake_models
    result = run(tmp_path,base)
    assert result["schema"] == pipeline.OUTCOME_SCHEMA and result["status"] == "awaiting_user_review"
    assert {"candidates","selected_attempt","condition_judgements","automatic_features","trace","quality_pass"} <= set(result)
    with pytest.raises(FreeGenerationRejected) as error:
        pipeline.run_free_generation(base,features(),"wear a new dress",base_kind="original",gender="female",body_parts=(),
            confirmed=True,adult_confirmed=True,settings=settings(tmp_path),runtime=SimpleNamespace(),directory=tmp_path/"rejected")
    trace = json.loads((tmp_path/"rejected"/"trace.json").read_text(encoding="utf-8"))
    assert trace["status"] == "rejected" and trace["rejection"]["code"] == error.value.code == "outfit_change"
    with pytest.raises(FreeGenerationRejected) as unconfirmed:
        pipeline.run_free_generation(base,features(),"",base_kind="original",gender="female",body_parts=(),
            confirmed=True,adult_confirmed=False,settings=settings(tmp_path),runtime=SimpleNamespace(),directory=tmp_path/"unconfirmed")
    assert unconfirmed.value.code == "confirmation_required"


def test_approval_requires_finish_quality_and_records_it(tmp_path,fake_models):
    # D-133 완료 기준: 완성도(배경·선·마무리)도 사람이 확인한다
    base,_ = fake_models
    results = FreeGenerationResults(run(tmp_path,base))
    with pytest.raises(ValueError):
        results.approve(dict(character=True,garment=True,request=True,exposure=True))
    results.approve(dict(character=True,garment=True,request=True,exposure=True,finish=True))
    entry = json.loads((results.directory/"judgement.json").read_text(encoding="utf-8"))[-1]
    assert entry["finish_quality"] == "good" and entry["quality_pass"] is True


def test_adult_male_topless_is_recorded_not_blocked():
    # 사용자 결정 2026-10-16: 성인 남성 상의 탈의는 차단하지 않는다. 다른 노출·여성·불확실한 경우는 그대로 차단
    from genai_lab.free_generation import safety_observation
    male = safety_observation({"1boy": .98, "topless_male": .89, "nipples": .74})
    assert male["blocked"] == [] and male["review"]["nipples"] == .74
    assert safety_observation({"1boy": .97, "topless_male": .04, "nipples": .86})["blocked"] == ["nipples"]
    assert safety_observation({"1girl": .9, "nipples": .8})["blocked"] == ["nipples"]
    assert safety_observation({"1boy": .98, "topless_male": .9, "penis": .6})["blocked"] == ["penis"]



@pytest.mark.parametrize("text", ["kneeling on one knee in a flower field", "standing in a garden", "walking in a meadow", "standing on a rooftop",
                                  "sitting by the window", "standing in the woods", "kneeling on one knee on a beach at sunset",
                                  "꽃밭에서 무릎 꿇기", "정원에 서 있기", "해 질 녘 바닷가", "산길을 걷기", "강가에 앉기", "밤에 걷기"])
def test_scene_words_count_as_background_requests(text):
    from genai_lab.free_generation import WHITE_BACKGROUND, free_conditions, mentions_background
    assert mentions_background(text)
    conditions = free_conditions("", "female", (), text, 42, base_kind="original")["conditions"]
    assert WHITE_BACKGROUND not in " ".join(c["text"] for c in conditions)


@pytest.mark.parametrize("text", ["close-up portrait", "sitting on a simple stool", "kneeling on one knee", "산책하듯 걷기", "다리를 꼬고 앉기",
                                  "강하게 웃기", "팔짱 끼고 서기", "standing with the hands in the pockets", "sea-green eyes looking up"])
def test_pose_only_requests_keep_the_white_background(text):
    from genai_lab.free_generation import mentions_background
    assert not mentions_background(text)


def test_staging_without_composition_words_asks_for_full_body():
    from genai_lab.free_generation import FULL_BODY, free_prompt
    assert FULL_BODY in free_prompt("", "female", (), "standing in a sunny park with trees in the background", 42)
    assert FULL_BODY in free_prompt("", "male", (), "sitting on a simple stool in a cozy cafe", 42)
    for text in ("close-up portrait in a cafe", "upper body shot on a beach", "상반신만 보이게 카페에서"):
        assert FULL_BODY not in free_prompt("", "female", (), text, 42)


def test_skin_word_follows_measured_reference_lightness():
    from genai_lab.free_generation import SKIN_TONE, features_sentence
    dark = features_sentence(sheet(skin=dict(state="measured", tags=["dark_skin"], color=dict(lab=[31.2, 11.5, 6.1]))))
    tan = features_sentence(sheet(skin=dict(state="measured", tags=["dark_skin"], color=dict(lab=[71.3, 16.4, 16.3]))))
    assert "dark brown skin" in dark and SKIN_TONE in dark
    assert "brown skin" not in tan and "dark" not in tan.replace("darker", "") and SKIN_TONE in tan


def test_auto_selected_hair_restore_is_shown_first_but_still_needs_approval(tmp_path,fake_models,monkeypatch):
    import genai_lab.hair_auto_restore as auto
    base,_ = fake_models
    monkeypatch.setattr(auto,"decide",lambda *a,**k:dict(decision="auto_selected",reason="머리만의 색 변화 + 재검사 통과",
                                                         cast=dict(status="hair_only_shift"),restored=np.full((1184,896,3),90,np.uint8)))
    result = run(tmp_path,base)
    assert result["schema"] == "free_generation_outcome/2"
    assert result["auto_selection"] == dict(attempt=1,variant="hair_auto",reason="머리만의 색 변화 + 재검사 통과")
    auto_candidate = next(c for c in result["candidates"] if c["variant"] == "hair_auto")
    assert auto_candidate["auto_selected"] and auto_candidate["raw"].endswith("hair-auto.png")
    assert any(c["variant"] == "raw" for c in result["candidates"])                 # 원 결과 보존
    results = FreeGenerationResults(result)
    assert results.candidate["variant"] == "hair_auto" and results.status == "awaiting_user_review"
    assert "자동 보정본" in results.head_notice
    results.approve(dict(character=True,garment=True,request=True,exposure=True,finish=True))
    saved = results.export(tmp_path/"out.png")
    assert json.loads(saved.with_suffix(".review.json").read_text(encoding="utf-8"))["variant"] == "hair_auto"


def test_hair_auto_decision_errors_never_fail_the_generation(tmp_path,fake_models,monkeypatch):
    import genai_lab.hair_auto_restore as auto
    base,_ = fake_models
    def broken(*a,**k):
        raise RuntimeError("boom")
    monkeypatch.setattr(auto,"decide",broken)
    result = run(tmp_path,base)
    assert result["auto_selection"] is None and all(c["variant"] != "hair_auto" for c in result["candidates"])
    assert result["attempts"][0]["hair_auto"]["decision"] == "unmeasurable"


def test_experiment_hair_color_words_off_drops_only_the_colour_words():
    from genai_lab.free_generation import HAIR_PLACEMENT, features_sentence
    items = sheet(hair_length=dict(state="measured", value="short_hair"), hair_color=dict(state="measured", tags=["blue_hair"]))
    on, off = features_sentence(items), features_sentence(items, hair_color_words=False)
    assert "blue hair" in on and "blue" not in off
    assert "short hair" in off and HAIR_PLACEMENT in off and HAIR_PLACEMENT in on


def test_experiment_moves_absent_parts_to_the_negative_prompt():
    import re
    from genai_lab.free_generation import absent_parts_negative_prompt, free_prompt
    text = "kneeling on one knee on a beach at sunset"
    default = free_prompt("", "male", (), text, 42)
    moved = free_prompt("", "male", (), text, 42, absent_parts_negative=True)
    assert "no animal ears, no tail and no wings" in default and re.search(r"\btails?\b", moved) is None
    assert moved == default.replace("The character has no animal ears, no tail and no wings; do not add any. ", "")
    assert absent_parts_negative_prompt(()) == "animal tail, animal ears, wings"
    assert absent_parts_negative_prompt(("ears", "tail")) == "wings"
    assert absent_parts_negative_prompt(None) == " "
    partial = free_prompt("", "female", ("ears",), text, 42, absent_parts_negative=True)
    assert "animal ears" in partial and "no a tail" not in partial and "do not add any" not in partial


def test_experiment_switches_reach_the_worker_request_and_trace(tmp_path,fake_models,monkeypatch):
    base,calls = fake_models
    monkeypatch.setattr(pipeline,"verify_features",lambda *a,**k:dict(items=dict(skin=dict(status="changed",confident=True)),
                                                                      summary=dict(changed=["skin"]),retry=["skin"]))
    result = run(tmp_path,base,experiment=dict(absent_parts_negative=True,hair_color_words=False,single_attempt=True))
    requests = [c[1] for c in calls if c[0] == "qwen"]
    assert len(requests) == 1                                                  # 확신 위반이어도 한 장만
    assert requests[0]["prompt"]["negative"] == "animal tail, animal ears, wings"
    assert "no tail" not in requests[0]["prompt"]["positive"]
    from genai_lab.qwen_free_generation import validate_free_request
    assert validate_free_request(json.loads(json.dumps(requests[0]))).absent_parts_negative is True
    trace = json.loads(Path(result["trace"]).read_text(encoding="utf-8"))
    assert trace["experiment"]["single_attempt"] is True and len(result["attempts"]) == 1


def test_default_request_keeps_the_blank_negative_prompt(tmp_path,fake_models):
    base,calls = fake_models
    run(tmp_path,base)
    request = next(c[1] for c in calls if c[0] == "qwen")
    assert request["prompt"]["negative"] == " " and "no animal ears, no tail and no wings" in request["prompt"]["positive"]
    tampered = json.loads(json.dumps(request))
    tampered["prompt"]["negative"] = "animal tail"
    from genai_lab.qwen_free_generation import validate_free_request
    with pytest.raises(ValueError):
        validate_free_request(tampered)


def test_unknown_experiment_switch_is_rejected(tmp_path,fake_models):
    base,_ = fake_models
    from genai_lab.free_generation import FreeGenerationRejected
    with pytest.raises(FreeGenerationRejected) as error:
        run(tmp_path,base,experiment=dict(stronger_identity=True))
    assert error.value.code == "invalid_input"
