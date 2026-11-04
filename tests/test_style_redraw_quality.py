"""자동 진단/보정의 실패 사례를 CPU로 검증한다. GPU·모델 가중치·다운로드를 사용하지 않는다."""
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from genai_lab.style_redraw_quality import (
    audit_flux_prompt, build_contract, choose_correction, correction_regressions, diagnose_candidate)
from genai_lab.style_redraw_flux import flux_prompt, UPPER_FRAMING, COWBOY_FRAMING

KINDS = {tag: dict(kind=kind, **{kind: 1.0}) for tag, kind in [
    ("blue_eyes", "body"), ("long_hair", "body"), ("furry", "body"),
    ("red_shirt", "outfit"), ("white_shirt", "outfit"), ("black_vest", "outfit"),
    ("black_pants", "outfit"), ("shoes", "outfit"), ("black_footwear", "outfit"), ("socks", "outfit")]}
ORIGINAL = dict(blue_eyes=.95, long_hair=.8, red_shirt=.9)
GARMENT = dict(white_shirt=.9, black_vest=.95, shoes=.9, black_pants=.9)


def contract(**kwargs):
    return build_contract(ORIGINAL, GARMENT, KINDS, with_garment=True, **kwargs)


def test_replacement_does_not_preserve_old_outfit():
    c = contract()
    assert "red_shirt" not in c["target_scores"]
    assert set(c["identity"]) == {"blue_eyes", "long_hair"}
    assert {"white_shirt", "black_vest"} <= set(c["garment"])


@pytest.mark.parametrize("shot,absent,present", [
    (UPPER_FRAMING, ["shoes", "black_pants"], ["white_shirt", "black_vest"]),
    (COWBOY_FRAMING, ["shoes"], ["black_pants", "white_shirt"]),
])
def test_offscreen_clothes_do_not_trigger_correction(shot, absent, present):
    c = contract(framing=shot)
    assert all(tag not in c["target_scores"] for tag in absent)
    assert all(tag in c["target_scores"] for tag in present)


def test_new_eye_loss_rejects_better_garment_score():
    c = contract()
    before = {**ORIGINAL, **GARMENT, "white_shirt": .1}
    after = {**before, "white_shirt": .9, "blue_eyes": .1}
    result = choose_correction(c, before, after, ["white_shirt"])
    assert not result["use_correction"]
    assert result["newly_missing"] == ["blue_eyes"]


def test_other_garment_regression_also_rejects():
    before = {**ORIGINAL, **GARMENT, "white_shirt": .1}
    result = choose_correction(contract(), before, {**before, "white_shirt": .9, "black_vest": .1}, ["white_shirt"])
    assert not result["use_correction"]
    assert "black_vest" in result["newly_missing"]


@pytest.mark.parametrize("tag", ["signature", "watermark", "text", "logo"])
def test_new_contamination_is_not_a_success(tag):
    before = {**ORIGINAL, **GARMENT, "white_shirt": .1}
    result = choose_correction(contract(), before, {**before, "white_shirt": .9, tag: .9}, ["white_shirt"])
    assert not result["use_correction"]
    assert result["new_contamination_suspected"] == [tag]


def test_ties_keep_original_and_real_improvement_can_be_selected():
    before = {**ORIGINAL, **GARMENT, "white_shirt": .1}
    assert not choose_correction(contract(), before, before, ["white_shirt"])["use_correction"]
    assert choose_correction(contract(), before, {**before, "white_shirt": .9}, ["white_shirt"])["use_correction"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 1.1, True, "0.9"])
def test_invalid_observation_is_failure_not_pass(value):
    with pytest.raises(ValueError):
        diagnose_candidate(contract(), {"blue_eyes": value})


def test_empty_observation_is_not_pass():
    with pytest.raises(ValueError):
        diagnose_candidate(contract(), {})


def test_unknown_tags_are_unassessed_not_correction_demands():
    c = build_contract({**ORIGINAL, "new_unknown": .9}, None, KINDS)
    assert "new_unknown" in c["unclassified"]
    assert "new_unknown" not in c["target_scores"]
    assert diagnose_candidate(c, ORIGINAL)["state"] == "inconclusive"


def test_automatic_diagnostics_never_approve_style():
    d = diagnose_candidate(contract(), {**ORIGINAL, **GARMENT}, style_weight=1.0)
    assert d["state"] == "no_tag_issue_detected"
    assert not d["user_quality_approved"]
    assert d["style"]["state"] == "user_review_required"
    assert diagnose_candidate(contract(), {**ORIGINAL, **GARMENT}, style_weight=0)["style"]["state"] == "not_applied"


class TinyTokenizer:
    def apply_chat_template(self, messages, **options):
        assert options == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return "<user> " + messages[0]["content"] + " <assistant>"
    def __call__(self, text, **options):
        assert options == dict(truncation=False)
        return {"input_ids": text.split()}
    def decode(self, ids):
        return " ".join(ids)


def test_chat_template_counts_towards_limit():
    assert audit_flux_prompt(TinyTokenizer(), "one two", 4)["state"] == "ready"
    audit = audit_flux_prompt(TinyTokenizer(), "one two three", 4)
    assert audit["truncated"] and audit["token_count"] == 5
    assert audit["omitted_text"] == "<assistant>"


def test_overflow_stops_before_cuda_or_weights(tmp_path, monkeypatch):
    from genai_lab import style_redraw_worker as worker
    (tmp_path / "prepare.json").write_text(json.dumps({"prompt": "long"}))
    monkeypatch.setattr(worker, "prompt_delivery", lambda *args: dict(truncated=True, token_count=513, limit=512))
    fake = MagicMock()
    monkeypatch.setitem(sys.modules, "torch", fake)
    with pytest.raises(ValueError, match="입력 한도"):
        worker.generate_flux(tmp_path, {})
    assert not fake.mock_calls


def test_garment_colour_source_is_consistent():
    p = flux_prompt([], with_garment=True, scene="Walking.")
    assert "clothes take their colours from image 2" in p
    assert "skin and clothes keep their colours from image 1" not in p
    q = flux_prompt([], scene="Walking.")
    assert "skin and clothes keep their colours from image 1" in q


@pytest.mark.parametrize("regress,blocked_retry,postprocess", [
    (True, False, None), (False, False, None), (False, True, None),
    (False, False, "colour"), (False, False, "face"), (False, False, "extended_condition"),
    (False, False, "anatomy_autofix"), (False, False, "anatomy_face"),
    (False, False, "anatomy_colour"), (False, True, "anatomy_initial"),
    (False, False, "arm_autofix"), (False, False, "arm_face"),
    (False, False, "arm_colour"), (False, True, "arm_initial")])
def test_worker_uses_revalidation_and_keeps_evidence(tmp_path, monkeypatch, regress, blocked_retry, postprocess):
    """실제 작업 함수의 생성/태그 모델만 대역으로 바꿔 후보 승격과 저장 결과를 검증한다."""
    from genai_lab import style_redraw_worker as worker, style_redraw_flux as flux
    from genai_lab import clothing_analysis, identity_autofix
    prepared = dict(prompt="short", with_garment=True, framing=UPPER_FRAMING,
                    body_parts=[], identity_anchors=[], style_lora_weight=1)
    (tmp_path / "prepare.json").write_text(json.dumps(prepared))
    Image.new("RGB", (8, 8), (10, 10, 10)).save(tmp_path / "character.png")
    Image.new("RGB", (8, 8), (20, 20, 20)).save(tmp_path / "garment.png")
    np.savez(tmp_path / "original-masks.npz", skin=np.ones((8,8), bool))
    calls = []
    condition_marker = object()
    class Pipe:
        tokenizer = TinyTokenizer()
        vae = SimpleNamespace(enable_tiling=lambda: None)
        def load_lora_weights(self, *a, **kw): pass
        def set_adapters(self, *a, **kw): pass
        def enable_model_cpu_offload(self): pass
        def __call__(self, **kw):
            if postprocess == "extended_condition":
                assert "prompt" not in kw and kw["prompt_embeds"] is condition_marker
                assert kw["max_sequence_length"] == 316
            calls.append(kw)
            marker = 30 if len(calls) <= 3 else 40
            return SimpleNamespace(images=[Image.new("RGB", (8,8), (marker,)*3)])
    pipe = Pipe()
    torch = MagicMock()
    torch.cuda.get_device_properties.return_value.total_memory = 8 * 2**30
    torch.cuda.max_memory_reserved.return_value = 100
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(
        Flux2KleinPipeline=SimpleNamespace(from_pretrained=lambda *a, **kw: pipe)))
    monkeypatch.setitem(sys.modules, "diffusers.quantizers", SimpleNamespace(PipelineQuantizationConfig=lambda **kw: kw))
    class Tagger:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def analyze(self, image):
            marker = np.asarray(image)[0,0,0]
            if marker == 10: scores = ORIGINAL
            elif marker == 20: scores = GARMENT
            elif marker == 30: scores = {**ORIGINAL, **GARMENT, "white_shirt": .1}
            elif marker == 50: scores = {**ORIGINAL, **GARMENT, "blue_eyes": .1}
            else: scores = {**ORIGINAL, **GARMENT, "blue_eyes": .1 if regress else .95}
            return SimpleNamespace(raw_general_scores=scores)
    monkeypatch.setattr(clothing_analysis, "WdTagSession", Tagger)
    monkeypatch.setattr(identity_autofix, "kinds_table", lambda: KINDS)
    monkeypatch.setattr(worker, "prompt_delivery", lambda d,r,p,phase,*a:
                        dict(truncated=blocked_retry and phase == "autofix"))
    if postprocess == "extended_condition":
        from genai_lab import flux_prompt_delivery
        monkeypatch.setattr(flux_prompt_delivery, "build_flux_condition",
                            lambda *args: (dict(prompt_embeds=condition_marker, max_sequence_length=316),
                                           dict(mode="prompt_embeds", encoding_length=316)))
    def detected(images, req):
        bad_marker = {"anatomy_autofix": 40, "anatomy_face": 60,
                      "anatomy_colour": 60, "anatomy_initial": 30}.get(postprocess)
        # 정상 대역은 손 상자를 화면 안에 둔다. 경계 잘림은 별도 회귀 검사한다.
        boxes = [((0.2, 4, 2, 6), "hand", 0.8), ((6, 4, 7.8, 6), "hand", 0.8), ((3, 4, 5, 6), "hand", 0.8)]
        return [boxes[:3 if np.asarray(im)[0, 0, 0] == bad_marker else 2] for im in images]
    monkeypatch.setattr(worker, "hand_detections", detected)
    def poses(directory, images, request, phase):
        bad_marker = {"arm_autofix": 40, "arm_face": 60, "arm_colour": 60, "arm_initial": 30}.get(postprocess)
        records = []
        for image in images:
            points = [[4, 0], [4, 1], [2, 1], [1.5, 3], [1, 5], [6, 1], [6.5, 3], [7, 5]]
            if np.asarray(image)[0, 0, 0] == bad_marker:
                points[3] = [2, 1.2]  # 손 수가 같아도 팔 연결 길이가 모순인 결과다.
            records.append(dict(state="observed", points=[points], scores=[[.9] * 8]))
        return records
    monkeypatch.setattr(worker, "observe_arm_poses", poses)
    monkeypatch.setattr(worker, "person_mask", lambda image, req: np.ones(image.shape[:2], bool))
    monkeypatch.setattr(worker, "person_and_parts", lambda rgb, req: ({}, {}))
    monkeypatch.setattr(flux, "colour_match", lambda rgb,*a,**kw:
                        (np.full_like(rgb, 50) if postprocess == "colour" else np.full_like(rgb, 60) if postprocess in ("anatomy_colour", "arm_colour") else rgb, {}))
    from genai_lab import face_detail
    monkeypatch.setattr(face_detail, "detail_face", lambda pipe,rgb,*a,**kw:
                        (np.full_like(rgb, 60 if postprocess in ("anatomy_face", "arm_face") else 50), dict(pasted=True)))
    req = dict(flux_model="not-loaded", flux_style_lora="not-loaded.safetensors", model_cache="unused")
    if postprocess in ("face", "anatomy_face", "anatomy_initial", "arm_face", "arm_initial"):
        req["head_cache"] = "fake"
    worker.generate_flux(tmp_path, req)
    torch.cuda.set_per_process_memory_fraction.assert_called_once_with(6.5 / 8)
    result = json.loads((tmp_path / "generation.json").read_text())
    expected = 0 if regress or blocked_retry or postprocess in ("anatomy_autofix", "arm_autofix", "anatomy_initial", "arm_initial") else 1
    assert result["autofix"]["picks"] == [expected] * 3
    assert len(calls) == (3 if blocked_retry else 6)
    assert (tmp_path / "quality-contract.json").exists()
    for candidate in result["candidates"]:
        assert not candidate["quality_diagnostic"]["user_quality_approved"]
        assert (tmp_path / f"quality-seed{candidate['seed']}.json").exists()
        if postprocess in ("colour", "face"):
            assert "blue_eyes" not in candidate["quality_diagnostic"]["missing"]["identity"]
            if postprocess == "colour":
                assert not candidate["colour_applied"]
                assert (tmp_path / f"result-seed{candidate['seed']}-colour-rejected.png").exists()
            else:
                assert not candidate["face_detail"]["pasted"]
                assert (tmp_path / f"result-seed{candidate['seed']}-face-rejected.png").exists()
                assert "blue_eyes" in candidate["face_detail"]["revalidation"]["newly_missing"]
        if postprocess == "anatomy_autofix":
            assert result["autofix"]["decisions"][0]["reason"] == "excess_hands_suspected"
            assert candidate["hands"] == 2
        if postprocess == "anatomy_face":
            assert not candidate["face_detail"]["pasted"] and candidate["hands"] == 2
            assert candidate["face_detail"]["anatomy"]["state"] == "excess_hands_suspected"
        if postprocess == "anatomy_colour":
            assert not candidate["colour_applied"] and candidate["hands"] == 2
        if postprocess == "anatomy_initial":
            assert not candidate["face_detail"]["pasted"]
            assert candidate["hands"] == 3 and not candidate["anatomy"]["eligible_for_review"]
        if postprocess == "arm_autofix":
            assert result["autofix"]["decisions"][0]["reason"] == "anatomy_review_required"
        if postprocess == "arm_face":
            assert not candidate["face_detail"]["pasted"]
            assert candidate["face_detail"]["anatomy"]["state"] == "anatomy_review_required"
        if postprocess == "arm_colour":
            assert not candidate["colour_applied"]
        if postprocess == "arm_initial":
            assert not candidate["face_detail"]["pasted"] and not candidate["colour_applied"]
            assert candidate["anatomy"]["eligible_for_review"]
            assert not candidate["anatomy"]["automatic_correction_eligible"]
        if expected:
            assert (tmp_path / f"result-raw-seed{candidate['seed']}-r0.png").exists()
        elif not blocked_retry:
            assert (tmp_path / f"result-raw-seed{candidate['seed']}-fix.png").exists()


def test_review_screen_shows_diagnostics_without_approving(tmp_path):
    from genai_lab.style_redraw_results import StyleRedrawResults
    path = tmp_path / "result-seed42.png"
    Image.new("RGB", (8,8)).save(path)
    d = diagnose_candidate(contract(), {"blue_eyes": .9, "white_shirt": .1, "signature": .9}, style_weight=1)
    (tmp_path / "generation.json").write_text(json.dumps(dict(candidates=[dict(file=path.name, seed=42, hands=2, quality_diagnostic=d)])))
    results = StyleRedrawResults(dict(state="review_pending", candidates=[str(path)]))
    assert "의상 누락 의심" in results.head_notice and "오염 의심" in results.head_notice
    assert results.status == "awaiting_user_review"


def test_compact_prompt_keeps_style_and_roles_separate():
    from genai_lab.style_redraw_prompt import compact_flux_prompt
    from genai_lab.style_redraw_flux import STYLE, STYLE_LORA_TRIGGER
    plan = compact_flux_prompt([dict(name="eyes", text="green eyes"), dict(name="clothes", text="old red shirt")],
                               True, [], "walking by a lake", {}, UPPER_FRAMING, "male",
                               ["black vest", "white shirt", "black footwear", "shoes", "black pants"])
    text = plan["prompt"]
    assert text.startswith(STYLE_LORA_TRIGGER + " " + STYLE)
    assert "green eyes" in text and "Masculine adult man" in text
    assert "black vest" in text and "white shirt" in text and "Image 2 clothes only" in text
    assert "old red shirt" not in text and "shoes" not in text and "black pants" not in text
    assert "no legs" in text and "walking by a lake" in text
    assert "No animal ears/tail/wings" in text


def test_compact_scene_preserves_specific_content_and_removes_conflicting_full_body():
    from genai_lab.scene_composer import compose_scene
    from genai_lab.style_redraw_prompt import compact_flux_prompt
    from genai_lab.style_redraw_flux import FULL_BODY_FRAMING
    scene = compose_scene(42)
    plan = compact_flux_prompt([], False, [], scene["text"], {}, UPPER_FRAMING)
    for key in ("action", "place", "time", "lighting", "palette", "weather"):
        value = scene["parts"][key]
        if value:
            assert value in plan["prompt"]
    assert FULL_BODY_FRAMING not in plan["prompt"]
    assert "Match image 1's upper-body crop; no legs." in plan["prompt"]


@pytest.mark.parametrize("gender", ["female", "male"])
@pytest.mark.parametrize("shot", [UPPER_FRAMING, "Full-body shot"])
@pytest.mark.parametrize("garment", [False, True])
def test_visibility_instruction_is_independent_of_gender_and_outfit(gender, shot, garment):
    from genai_lab.style_redraw_prompt import compact_flux_prompt
    plan = compact_flux_prompt([], garment, [], "turning slightly", {}, shot, gender, ["white shirt", "shoes"])
    assert "replace their old positions" in plan["sections"]["integrity"]
    assert "Allow hidden or off-frame hands" not in plan["prompt"]
    assert "Visible hands" not in plan["prompt"]
    if shot == UPPER_FRAMING:
        assert "no legs" in plan["sections"]["framing"]
        assert "shoes" not in plan["sections"]["garment"]


def test_compact_parts_preserves_type_and_absence():
    from genai_lab.style_redraw_prompt import compact_parts
    text = compact_parts(["ears", "tail"], {"ears": "dog_ears", "ears_floppy": True, "tail": "dog_tail"})
    assert "floppy dog ears" in text and "dog tail" in text and "No wings" in text
    assert "Keep any original" in compact_parts(None, {})
    with pytest.raises(ValueError):
        compact_parts(["unknown"], {})


def test_prepare_refusal_remains_reviewable(tmp_path, monkeypatch):
    from genai_lab import style_redraw_worker as worker, style_redraw_inputs, character_features
    from genai_lab import studio_controller
    Image.new("RGB", (8,8)).save(tmp_path / "character.png")
    monkeypatch.setattr(worker, "person_and_parts", lambda *a: ({}, {"skin": np.ones((8,8), bool)}))
    monkeypatch.setattr(worker, "count_hands", lambda *a: [None])
    monkeypatch.setattr(character_features, "extract_features", lambda *a, **kw: {})
    monkeypatch.setattr(style_redraw_inputs, "trait_rows", lambda *a: [dict(name="eyes", text="green eyes")])
    monkeypatch.setattr(worker, "prompt_delivery", lambda *a:
                        dict(state="blocked", truncated=True, token_count=513, limit=512))
    worker.prepare_flux(tmp_path, {}, None, np.zeros((8,8,3), np.uint8), ORIGINAL, {}, {}, None, 0)
    p = json.loads((tmp_path / "prepare.json").read_text())
    assert p["state"] == "refused" and p["code"] == "prompt_truncated"
    assert p["prompt_sections"]["identity"] and p["legacy_prompt"]
    assert (tmp_path / "original-masks.npz").exists()


def test_arm_feedback_replaces_conflicting_rule_but_preserves_style_and_crop():
    from genai_lab.style_redraw_prompt import corrected_rejection_prompt, compact_flux_prompt, UPPER_FRAMING
    plan = compact_flux_prompt([dict(name="hair", text="silver hair")], scene="raise one hand", shot=UPPER_FRAMING)
    sections = dict(plan["sections"])
    sections["integrity"] = "Keep arms connected at the shoulders. Allow hidden or off-frame hands. No duplicate limbs, text, logos, watermarks or extra people."
    prepared = dict(prompt=" ".join(sections.values()), prompt_sections=sections,
                    framing=UPPER_FRAMING, prompt_delivery={"stale": True})
    result = corrected_rejection_prompt(prepared, "Repair the cropped arm.", anatomy=True)
    for key in ("style", "identity", "body", "garment", "framing", "scene"):
        assert result["prompt_sections"][key] == sections[key]
    assert "Allow hidden or off-frame hands" not in result["prompt"]
    assert "head-to-waist crop, no legs" in result["prompt"]
    assert "replace their old positions" in result["prompt"]
    assert "prompt_delivery" not in result
    assert "Allow hidden or off-frame hands" in prepared["prompt"]


def test_no_scene_keeps_original_crop_without_forcing_hidden_hands_into_view():
    from genai_lab.style_redraw_prompt import compact_flux_prompt, UPPER_FRAMING
    plan = compact_flux_prompt([], shot=UPPER_FRAMING)
    assert "naturally hidden or off-frame limbs" in plan["prompt"]
    assert "replace their old positions" not in plan["prompt"]
    assert "no legs" in plan["sections"]["framing"]
