from dataclasses import dataclass
from types import SimpleNamespace
import json
import numpy as np
import pytest
from PIL import Image, ImageDraw
from genai_lab.visual_reference import (
    VisualInputs, masked_crop, compare_candidate, generate_visual_batch, configure_visual_condition,
)


@pytest.fixture
def inputs():
    source = Image.new('RGB', (64, 112), 'white')
    head = Image.new('L', source.size)
    body = Image.new('L', source.size)
    ImageDraw.Draw(head).rectangle((20, 2, 44, 24), fill=255)
    ImageDraw.Draw(body).rectangle((10, 25, 54, 100), fill=255)
    value = VisualInputs(source, Image.new('RGB', (32, 32), 'red'),
                         Image.new('RGB', (32, 32), 'blue'), head, body)
    yield value
    value.close()


def test_output_similarity_redetection_enables_parser_foreground_fallback(
        monkeypatch, tmp_path):
    from genai_lab.reference_regions import ReferenceRegions
    from genai_lab.visual_reference import detect_output_similarity_regions

    seen = {}

    def analyze(image, config, root, **kwargs):
        seen.update(kwargs)
        masks = {
            name: Image.new("L", image.size, 255)
            for name in ("identity", "hair", "face", "garment", "foreground")
        }
        return ReferenceRegions(
            masks, {"mode": "reference_regions_v1", "pixel_counts": {}})

    monkeypatch.setattr(
        "genai_lab.reference_regions.analyze_reference_regions", analyze)
    with Image.new("RGB", (16, 16), "white") as image:
        regions, output = detect_output_similarity_regions(
            image, {}, tmp_path, tmp_path, 5)
    try:
        assert seen["allow_parser_foreground_fallback"] is True
        assert seen["analysis_scope"] == "base_output"
        assert output.name == "candidate_5_base"
        assert regions.record["coordinate_space"] == "generated_candidate"
    finally:
        regions.close()



@pytest.mark.parametrize(
    ("suffix", "expected_scope"),
    (
        ("native_approved_base", "base_output"),
        ("flux2_klein", "final_output"),
    ),
)
def test_native_refinement_redetection_uses_stage_specific_scope(
        monkeypatch, tmp_path, suffix, expected_scope):
    from genai_lab.reference_regions import ReferenceRegions
    from genai_lab.visual_reference import detect_output_similarity_regions

    seen = {}

    def analyze(image, config, root, **kwargs):
        seen.update(kwargs)
        masks = {
            name: Image.new("L", image.size, 255)
            for name in ("identity", "hair", "face", "garment", "foreground")
        }
        return ReferenceRegions(
            masks, {"mode": "reference_regions_v1", "pixel_counts": {}})

    monkeypatch.setattr(
        "genai_lab.reference_regions.analyze_reference_regions", analyze)
    with Image.new("RGB", (16, 16), "white") as image:
        regions, output = detect_output_similarity_regions(
            image, {}, tmp_path, tmp_path, 2, suffix=suffix)
    try:
        assert seen["analysis_scope"] == expected_scope
        assert output.name == f"candidate_2_{suffix}"
    finally:
        regions.close()

def test_reference_pipeline_preserves_configured_img2img_mode(monkeypatch, tmp_path):
    import sys
    from genai_lab.model import prepare_pipeline
    calls = []
    pipe = SimpleNamespace(enable_model_cpu_offload=lambda: None)
    factory = SimpleNamespace(from_pretrained=lambda *a, **k: pipe)
    monkeypatch.setitem(sys.modules, 'diffusers', SimpleNamespace(
        AutoPipelineForImage2Image=SimpleNamespace(from_pipe=lambda p: calls.append('img2img') or p),
        ControlNetModel=factory, StableDiffusionPipeline=factory,
        StableDiffusionXLControlNetImg2ImgPipeline=factory, StableDiffusionXLPipeline=factory))
    config = {'model': {'family': 'sdxl', 'id': 'mock', 'cache_dir': str(tmp_path)},
              'generation': {'mode': 'image_to_image'}, 'style': {'enabled': False},
              'clothing_reference_generation': {'enabled': True}}
    assert prepare_pipeline(config) is pipe
    assert calls == ['img2img']
    assert config['generation']['mode'] == 'image_to_image'


def test_no_initial_batch_skips_placeholder_and_preview(inputs, setup_batch):
    from PySide6.QtWidgets import QApplication
    from genai_lab.visual_reference_review import VisualInputReview
    app = QApplication.instance() or QApplication([])
    pipeline, config, request, root, log, *_ = setup_batch
    batch = generate_visual_batch(pipeline, config, request, root, log, inputs, lambda: False, lambda x: None)
    assert not (root / 'input_initial.png').exists()
    assert 'placeholder_diagnostics' not in batch.candidates[0].design_reference_record
    dialog = VisualInputReview(inputs)
    dialog.close()
    batch.close()


def test_crop_removes_unselected_clothing(inputs):
    source = Image.new('RGB', inputs.source.size, 'blue')
    ImageDraw.Draw(source).rectangle((20, 2, 44, 24), fill='red')
    with masked_crop(source, inputs.identity_mask) as crop:
        pixels = np.asarray(crop)
        assert not np.any(np.all(pixels == (0, 0, 255), axis=2))


def test_identity_crop_keeps_face_and_hair_but_not_clothing(inputs):
    source = Image.new('RGB', inputs.source.size, 'green')
    ImageDraw.Draw(source).rectangle((20, 2, 44, 12), fill='blue')
    ImageDraw.Draw(source).rectangle((20, 13, 44, 24), fill='red')
    with masked_crop(source, inputs.identity_mask) as crop:
        pixels = np.asarray(crop)
        assert np.any(np.all(pixels == (0, 0, 255), axis=2))
        assert np.any(np.all(pixels == (255, 0, 0), axis=2))
        assert not np.any(np.all(pixels == (0, 128, 0), axis=2))
    source.close()


def test_empty_crop_is_unavailable_not_zero(inputs):
    with pytest.raises(ValueError, match='비어'):
        masked_crop(inputs.source, Image.new('L', inputs.source.size))


def test_scores_are_separate_and_not_probability(inputs, monkeypatch):
    monkeypatch.setattr('genai_lab.visual_reference.image_feature', lambda *args: np.array([1., 0.]))
    scores = compare_candidate(None, inputs.source, inputs,
                               {'character': np.array([1., 0.]), 'garment': np.array([0., 1.])})
    assert scores['character'] == 1
    assert scores['garment'] == 0
    assert 'probability' in scores['method']


def test_missing_reference_features_are_none(inputs):
    scores = compare_candidate(None, inputs.source, inputs, {})
    assert scores['character'] is None and scores['garment'] is None
    assert len(scores['warnings']) == 3


@dataclass
class Request:
    seed: int = 42
    candidate_number: int = 1
    reference_image_strength: float = .8
    prompt: str = 'blue jacket'
    negative_prompt: str = ''


@dataclass
class Candidate:
    image: Image.Image
    seed: int
    prompt: str = 'blue jacket'
    design_reference_record: dict | None = None


@pytest.fixture
def setup_batch(inputs, monkeypatch, tmp_path):
    seen = []
    scales = []
    monkeypatch.setattr('genai_lab.visual_reference.tempfile.mkdtemp', lambda **kwargs: str(tmp_path))
    monkeypatch.setattr('genai_lab.visual_reference.configure_visual_condition', lambda *args: {'cached': True})
    monkeypatch.setattr('genai_lab.visual_reference.image_feature', lambda *args: np.array([1., 0.]))
    def generate(pipeline, config, request, root, log):
        seen.append(request.seed)
        assert config['clothing_reference_generation']['visual_inputs'] is inputs
        assert config['clothing_reference_generation']['visual_condition'] == {'cached': True}
        return Candidate(inputs.source.copy(), request.seed)
    monkeypatch.setattr('genai_lab.generator.generate_character_candidate', generate)
    return (SimpleNamespace(set_ip_adapter_scale=scales.append),
            {'clothing_reference_generation': {'candidate_count': 3}},
            Request(), tmp_path, SimpleNamespace(write_stage=lambda *args: None), seen, scales)


def test_count_seeds_disk_and_no_auto_approval(inputs, setup_batch):
    pipeline, config, request, root, log, seen, scales = setup_batch
    batch = generate_visual_batch(pipeline, config, request, root, log, inputs, lambda: False, lambda x: None)
    assert seen == [42, 43, 44]
    assert len(batch.candidates) == 3
    assert all(path.exists() for path in batch.paths)
    assert all(c.design_reference_record['approval'] == 'pending' for c in batch.candidates)
    for candidate in batch.candidates:
        record = candidate.design_reference_record
        assert record['identity_reference_scope'] == 'face_hair'
        assert set(record['stage_timings']) == {'후보 생성', '유사도 계산'}
    import json
    saved = json.loads((root / 'candidate_1.json').read_text(encoding='utf-8'))
    assert 'placeholder_diagnostics' not in saved and 'stage_timings' in saved
    assert saved['ip_adapter_input_diagnostics']['stage'] == 'before_ip_adapter_image_embeds'
    assert scales[-1] == .8
    assert request.seed == 42
    batch.close()


def test_semantic_conflict_is_removed_before_similarity_ranking(
        inputs, setup_batch, monkeypatch):
    pipeline, config, request, root, log, seen, _ = setup_batch
    config["candidate_semantic_gate"] = {"enabled": True}
    inputs.approved_generation = SimpleNamespace(record=lambda: {
        "gender": "male",
        "approved_character_tags": ["raccoon ears", "raccoon tail"],
        "approved_tags": ["skirt", "jacket"],
    })
    semantic_actions = iter(("retry", "pass", "pass"))

    class Analyzer:
        def __init__(self, _config):
            pass

        def analyze(self, image, approved_run, settings):
            action = next(semantic_actions)
            return {
                **settings.record(),
                "status": action.upper(),
                "action": action,
                "violations": (["gender_conflict:1girl"]
                               if action == "retry" else []),
                "unresolved": [],
                "checks": {},
                "fallback_quality_score": 0.0,
            }

        def close(self):
            pass

    ranked = []
    monkeypatch.setattr(
        "genai_lab.candidate_semantic_gate.CandidateSemanticAnalyzer", Analyzer)
    monkeypatch.setattr(
        "genai_lab.visual_reference.compare_candidate",
        lambda *args: ranked.append(True) or {
            "method": "test", "character": .8, "garment": .8,
            "vibe": .8, "warnings": [],
        })

    batch = generate_visual_batch(
        pipeline, config, request, root, log, inputs,
        lambda: False, lambda _: None)
    try:
        assert seen == [42, 43, 44]
        assert len(ranked) == 2
        assert [candidate.seed for candidate in batch.candidates] == [43, 44]
        assert any(item["failure_stage"] == "candidate_semantic_gate"
                   for item in batch.quarantined)
    finally:
        batch.close()


def test_all_initial_gender_conflicts_open_extra_seed_budget_and_stop_on_pass(
        inputs, setup_batch, monkeypatch):
    pipeline, config, request, root, log, seen, _ = setup_batch
    config["clothing_reference_generation"]["candidate_count"] = 2
    config["candidate_pipeline"] = {
        "enabled": True,
        "target_valid_candidates": 1,
        "maximum_attempts": 2,
        "gender_retry_attempts": 2,
        "repair_best_candidate_only": False,
    }
    config["candidate_semantic_gate"] = {"enabled": True}
    inputs.approved_generation = SimpleNamespace(record=lambda: {
        "gender": "male",
        "approved_character_tags": [],
        "approved_tags": [],
    })
    semantic_actions = iter(("retry", "retry", "pass"))

    class Analyzer:
        def __init__(self, _config):
            pass

        def analyze(self, image, approved_run, settings):
            action = next(semantic_actions)
            return {
                **settings.record(),
                "status": action.upper(),
                "action": action,
                "violations": (["gender_conflict:1girl"]
                               if action == "retry" else []),
                "unresolved": [],
                "checks": {},
                "fallback_quality_score": 0.0,
            }

        def close(self):
            pass

    monkeypatch.setattr(
        "genai_lab.candidate_semantic_gate.CandidateSemanticAnalyzer", Analyzer)
    batch = generate_visual_batch(
        pipeline, config, request, root, log, inputs,
        lambda: False, lambda _: None)
    try:
        assert seen == [42, 43, 44]
        assert [candidate.seed for candidate in batch.candidates] == [44]
        budget = batch.candidates[0].design_reference_record[
            "candidate_attempt_budget"]
        assert budget["is_gender_retry"] is True
        assert budget["gender_retry_attempts"] == 2
    finally:
        batch.close()


def test_structure_gate_runs_on_every_semantic_failure_without_returning_fallback(
        inputs, setup_batch, monkeypatch):
    pipeline, config, request, root, log, seen, _ = setup_batch
    config["candidate_semantic_gate"] = {"enabled": True}
    config["candidate_structure_gate"] = {"enabled": True}
    inputs.approved_generation = SimpleNamespace(record=lambda: {
        "gender": "male", "approved_character_tags": [],
        "approved_tags": [],
    })
    fallback_scores = iter((0.99, 0.30, 0.20))

    class Analyzer:
        def __init__(self, _config):
            pass

        def analyze(self, image, approved_run, settings):
            score = next(fallback_scores)
            return {
                **settings.record(), "status": "RETRY", "action": "retry",
                "violations": ["ears_species_conflict:cat ears"], "unresolved": [],
                "checks": {}, "fallback_quality_score": score,
            }

        def close(self):
            pass

    structural_actions = iter(("reject", "pass", "pass"))

    def structure(_image, settings):
        action = next(structural_actions)
        return {
            **settings.record(), "status": action.upper(), "action": action,
            "violations": (["large_detached_object_above_character"]
                           if action == "reject" else []),
            "checks": {},
        }

    monkeypatch.setattr(
        "genai_lab.candidate_semantic_gate.CandidateSemanticAnalyzer", Analyzer)
    monkeypatch.setattr(
        "genai_lab.candidate_structure_gate.evaluate_candidate_structure", structure)

    import json
    with pytest.raises(ValueError):
        generate_visual_batch(pipeline, config, request, root, log, inputs,
                              lambda: False, lambda _: None)
    assert seen == [42, 43, 44]
    for number in (1, 2, 3):
        record = json.loads((root / f'candidate_{number}.json').read_text(encoding='utf-8'))
        assert 'candidate_structure_gate' in record
        assert record['candidate_resolution']['returned_image'] is None


def test_adaptive_pipeline_stops_at_two_and_repairs_only_best(
        inputs, setup_batch, monkeypatch):
    pipeline, config, request, root, log, seen, _ = setup_batch
    config["candidate_pipeline"] = {
        "enabled": True,
        "target_valid_candidates": 2,
        "maximum_attempts": 3,
        "repair_best_candidate_only": True,
        "reject_failed_post_audit": False,
    }
    inputs.approved_generation = SimpleNamespace(record=lambda: {
        "gender": "male",
        "approved_tags": [],
        "approved_part_color_descriptions": [],
        "prompt": "blue jacket",
        "negative_prompt": "",
        "part_error_correction": {},
    })
    monkeypatch.setattr(
        'genai_lab.generated_condition_audit.build_generated_condition_audit',
        lambda *args: {'status': 'PASS', 'checks': {}})
    corrected = []

    def correct(pipeline, image, visual_inputs, config, selected_request, **kwargs):
        corrected.append(selected_request.candidate_number)
        return SimpleNamespace(
            image=image,
            report={"status": "completed", "parts": {}},
        )

    monkeypatch.setattr(
        "genai_lab.part_error_correction.correct_detected_parts", correct)
    batch = generate_visual_batch(
        pipeline, config, request, root, log, inputs,
        lambda: False, lambda _: None)
    try:
        assert seen == [42, 43]
        assert corrected == [1]
        assert len(batch.candidates) == len(batch.paths) == 1
        assert batch.candidates[0].design_reference_record[
            "candidate_number"] == 1
        assert batch.candidates[0].design_reference_record[
            "part_error_correction"]["status"] == "completed"
    finally:
        batch.close()


def test_all_semantic_repair_failures_return_nothing_and_persist(
        inputs, setup_batch, monkeypatch):
    import json

    pipeline, config, request, root, log, seen, _ = setup_batch
    config["candidate_pipeline"] = {
        "enabled": True,
        "target_valid_candidates": 2,
        "maximum_attempts": 2,
        "repair_best_candidate_only": True,
        "reject_failed_post_audit": True,
    }
    inputs.approved_generation = SimpleNamespace(record=lambda: {
        "gender": "male",
        "approved_tags": [],
        "approved_part_color_descriptions": [],
        "prompt": "blue jacket",
        "negative_prompt": "",
        "part_error_correction": {},
    })

    def correct(pipeline, image, visual_inputs, config, selected_request,
                **kwargs):
        return SimpleNamespace(
            image=image,
            report={
                "status": "completed",
                "parts": {
                    "ears": {
                        "status": "rejected_keep_original",
                        "reason": "absolute_color_target_not_met",
                    },
                },
            },
        )

    failed_audit = {
        "version": "generated_condition_post_audit_v1",
        "status": "FAIL",
        "checks": {
            "ears": {
                "status": "FAIL",
                "reason": "rejected_keep_original",
            },
            "hair": {
                "status": "UNRESOLVED",
                "reason": "hair_or_face_not_detected",
            },
        },
    }
    monkeypatch.setattr(
        "genai_lab.part_error_correction.correct_detected_parts", correct)
    monkeypatch.setattr(
        "genai_lab.generated_condition_audit.build_generated_condition_audit",
        lambda *args: failed_audit,
    )

    with pytest.raises(ValueError):
        generate_visual_batch(pipeline, config, request, root, log, inputs,
                              lambda: False, lambda _: None)
    assert seen == [42, 43]
    for number in (1, 2):
        record = json.loads((root / f'candidate_{number}.json').read_text(encoding='utf-8'))
        assert record['candidate_resolution']['returned_image'] is None
        assert record['candidate_resolution']['status'] == 'quarantined_after_repair'
        assert record['generated_condition_audit']['status'] == 'FAIL'


def test_similarity_gate_exhaustion_returns_nothing(
        inputs, setup_batch, monkeypatch):
    pipeline, config, request, root, log, seen, _ = setup_batch
    config["candidate_pipeline"] = {
        "enabled": True,
        "target_valid_candidates": 1,
        "maximum_attempts": 2,
        "repair_best_candidate_only": True,
        "minimum_component_similarity": .65,
        "require_all_similarity_scores": True,
    }
    score_rows = iter((
        {"character": .80, "garment": .50, "vibe": .70},
        {"character": .72, "garment": .60, "vibe": .74},
    ))
    monkeypatch.setattr(
        "genai_lab.visual_reference.compare_candidate",
        lambda *args: dict(next(score_rows), method="test", warnings=[]),
    )

    with pytest.raises(ValueError):
        generate_visual_batch(pipeline, config, request, root, log, inputs,
                              lambda: False, lambda _: None)
    assert seen == [42, 43]


def test_img2img_does_not_forward_common_latents(
        inputs, setup_batch, monkeypatch):
    import torch
    pipeline, config, request, root, log, *_ = setup_batch
    pipeline.unet = SimpleNamespace(
        config=SimpleNamespace(in_channels=4), dtype=torch.float32)
    pipeline.vae_scale_factor = 8
    pipeline._execution_device = "cpu"
    request.width, request.height = 64, 112
    config["generation"] = {"mode": "image_to_image"}
    config["common_candidate_latents"] = {
        "enabled": True, "correlation": .96, "variation_seed_offset": 10000,
    }
    received = []

    def generate(pipeline, local, current, root, log):
        section = local["clothing_reference_generation"]
        received.append((
            "candidate_latents" in section,
            "candidate_latent_record" in section,
        ))
        return Candidate(inputs.source.copy(), current.seed)

    monkeypatch.setattr(
        "genai_lab.generator.generate_character_candidate", generate)
    batch = generate_visual_batch(
        pipeline, config, request, root, log, inputs, lambda: False, lambda _: None)
    try:
        assert received == [(False, False), (False, False), (False, False)]
    finally:
        batch.close()


def test_corrupted_img2img_candidate_retries_once_with_identical_seed(
        inputs, setup_batch, monkeypatch):
    import torch
    from genai_lab.generated_image_integrity import GeneratedOutputIntegrityError
    pipeline, config, request, root, log, *_ = setup_batch
    pipeline.unet = SimpleNamespace(
        config=SimpleNamespace(in_channels=4), dtype=torch.float32)
    pipeline.vae_scale_factor = 8
    pipeline._execution_device = "cpu"
    request.width, request.height = 64, 112
    config["clothing_reference_generation"]["candidate_count"] = 2
    config["generation"] = {"mode": "image_to_image"}
    config["common_candidate_latents"] = {
        "enabled": True, "correlation": .96, "variation_seed_offset": 10000,
    }
    config["generated_image_integrity"] = {"enabled": True, "retry_count": 1}
    received = []

    def generate(pipeline, local, current, root, log):
        section = local["clothing_reference_generation"]
        received.append((
            current.seed,
            "candidate_latents" in section,
            "candidate_latent_record" in section,
        ))
        if len(received) == 1:
            raise GeneratedOutputIntegrityError(
                "corrupted", {
                    "status": "invalid",
                    "failure_stage": "final_latent",
                    "retryable": True,
                })
        return Candidate(inputs.source.copy(), current.seed)

    monkeypatch.setattr(
        "genai_lab.generator.generate_character_candidate", generate)
    batch = generate_visual_batch(
        pipeline, config, request, root, log, inputs, lambda: False, lambda _: None)
    try:
        assert len(received) == 3
        assert received[0][0] == received[1][0] == request.seed
        assert received[0][1:] == received[1][1:] == (False, False)
        assert batch.candidates[0].design_reference_record["integrity_retry_count"] == 1
        assert batch.candidates[1].design_reference_record["integrity_retry_count"] == 0
    finally:
        batch.close()


def test_decoded_rgb_corruption_is_quarantined_and_next_candidate_continues(
        inputs, setup_batch, monkeypatch):
    from genai_lab.generated_image_integrity import GeneratedOutputIntegrityError
    pipeline, config, request, root, log, *_ = setup_batch
    config["clothing_reference_generation"]["candidate_count"] = 2
    config["generated_image_integrity"] = {
        "enabled": True,
        "retry_count": 1,
        "retry_decoded_rgb_failure": False,
        "continue_after_corruption": True,
    }
    received = []

    def generate(pipeline, local, current, root, log):
        received.append((
            current.candidate_number,
            current.seed,
            local["clothing_reference_generation"]["integrity_attempt"],
        ))
        if current.candidate_number == 1:
            raise GeneratedOutputIntegrityError(
                "corrupted",
                {
                    "status": "corrupted",
                    "failure_stage": "decoded_rgb",
                    "retryable": False,
                    "debug_image_path": "corrupted/candidate_1_attempt_1.png",
                },
            )
        return Candidate(inputs.source.copy(), current.seed)

    monkeypatch.setattr(
        "genai_lab.generator.generate_character_candidate", generate)
    batch = generate_visual_batch(
        pipeline, config, request, root, log, inputs, lambda: False, lambda _: None)
    try:
        assert received == [(1, request.seed, 1), (2, request.seed + 1, 1)]
        assert len(batch.candidates) == 1
        assert batch.candidates[0].seed == request.seed + 1
        assert len(batch.quarantined) == 1
        assert batch.quarantined[0]["candidate_number"] == 1
        assert batch.quarantined[0]["failure_stage"] == "decoded_rgb"
        assert "후보 1" in batch.warning
    finally:
        batch.close()


def test_adapter_diagnostics_saved_before_encoding_same_image(inputs, setup_batch, monkeypatch):
    pipeline, config, request, root, log, *_ = setup_batch
    def configure(actual_pipeline, actual_inputs, *scales):
        assert actual_inputs.garment is inputs.garment
        with Image.open(root / 'debug_garment_ip_adapter.png') as saved:
            np.testing.assert_array_equal(np.asarray(saved), np.asarray(inputs.garment))
        assert (root / 'debug_garment_ip_adapter.json').exists()
        assert (root / 'input_identity.png').exists()
        return {'cached': True}
    monkeypatch.setattr('genai_lab.visual_reference.configure_visual_condition', configure)
    batch = generate_visual_batch(pipeline, config, request, root, log, inputs, lambda: False, lambda x: None)
    batch.close()


def test_candidate_timers_exclude_previous_generation_preparation_and_scoring(inputs, setup_batch, monkeypatch):
    pipeline, config, request, root, log, seen, scales = setup_batch
    config['reference_analysis'] = {'generation_timeout_seconds': 10, 'preparation_timeout_seconds': 100}
    now = [0.0]
    callbacks = []
    monkeypatch.setattr('genai_lab.visual_reference.perf_counter', lambda: now[0])
    def configure(*args):
        now[0] += 30
        return {'cached': True}
    monkeypatch.setattr('genai_lab.visual_reference.configure_visual_condition', configure)
    def generate(pipeline, local, current, root, log):
        check = local['clothing_reference_generation']['check_running']
        callbacks.append(check)
        check()
        now[0] += 9
        check()
        return Candidate(inputs.source.copy(), current.seed)
    monkeypatch.setattr('genai_lab.generator.generate_character_candidate', generate)
    def score(*args):
        now[0] += 25
        return {}
    monkeypatch.setattr('genai_lab.visual_reference.compare_candidate', score)
    batch = generate_visual_batch(pipeline, config, request, root, log, inputs, lambda: False, lambda x: None)
    assert len(batch.candidates) == 3 and not batch.warning
    assert len({id(callback) for callback in callbacks}) == 3
    assert all(c.design_reference_record['generation_timeout_scope'] == 'per_candidate' for c in batch.candidates)
    with pytest.raises(TimeoutError, match='후보 1 생성'):
        callbacks[0]()  # 이전 콜백의 마감이 다음 후보에서 갱신되지 않아야 한다.
    batch.close()


@pytest.mark.parametrize('check_inside_model', [True, False])
def test_second_candidate_timeout_preserves_first(inputs, setup_batch, monkeypatch, check_inside_model):
    pipeline, config, request, root, log, seen, scales = setup_batch
    config['reference_analysis'] = {'generation_timeout_seconds': 10}
    now = [0.0]
    monkeypatch.setattr('genai_lab.visual_reference.perf_counter', lambda: now[0])
    def generate(pipeline, local, current, root, log):
        now[0] += 9 if current.candidate_number == 1 else 11
        if check_inside_model:
            local['clothing_reference_generation']['check_running']()
        return Candidate(inputs.source.copy(), current.seed)
    monkeypatch.setattr('genai_lab.generator.generate_character_candidate', generate)
    batch = generate_visual_batch(pipeline, config, request, root, log, inputs, lambda: False, lambda x: None)
    assert len(batch.paths) == 1 and batch.paths[0].exists()
    assert '후보 2 생성 시간 제한 10초' in batch.warning
    assert scales[-1] == .8
    batch.close()


def test_preparation_timeout_does_not_start_generation(inputs, setup_batch, monkeypatch):
    pipeline, config, request, root, log, seen, scales = setup_batch
    config['reference_analysis'] = {'preparation_timeout_seconds': 2}
    now = [0.0]
    monkeypatch.setattr('genai_lab.visual_reference.perf_counter', lambda: now[0])
    def configure(*args):
        now[0] += 3
        return {'cached': True}
    monkeypatch.setattr('genai_lab.visual_reference.configure_visual_condition', configure)
    with pytest.raises(TimeoutError, match='공통 참조 준비'):
        generate_visual_batch(pipeline, config, request, root, log, inputs, lambda: False, lambda x: None)
    assert not seen


def test_black_adapter_input_blocks_encoding_and_generation(inputs, setup_batch, monkeypatch):
    pipeline, config, request, root, log, seen, scales = setup_batch
    inputs.garment.paste('black', (0, 0, *inputs.garment.size))
    def forbidden(*args, **kwargs):
        pytest.fail('Rejected image must never be encoded')
    monkeypatch.setattr('genai_lab.visual_reference.configure_visual_condition', forbidden)
    with pytest.raises(ValueError, match='완전히 검은색'):
        generate_visual_batch(pipeline, config, request, root, log, inputs, lambda: False, lambda x: None)
    assert not seen
    assert (root / 'debug_garment_ip_adapter.png').exists()
    assert (root / 'debug_garment_ip_adapter.json').exists()
    assert scales[-1] == .8


def test_legacy_ab_flag_cannot_disable_garment_reference(inputs, setup_batch):
    pipeline, config, request, root, log, seen, scales = setup_batch
    config['clothing_reference_generation'].update(garment_ab_test=True, candidate_count=3)
    batch = generate_visual_batch(pipeline, config, request, root, log, inputs, lambda: False, lambda x: None)
    assert seen == [42, 43, 44]
    assert scales == [.8]
    assert all(c.design_reference_record['garment_scale'] == .45 for c in batch.candidates)
    assert all('garment_ab_test' not in c.design_reference_record for c in batch.candidates)
    batch.close()


def test_custom_reference_scales_reach_condition_and_records(inputs, setup_batch, monkeypatch):
    pipeline, config, request, root, log, seen, scales = setup_batch
    config['clothing_reference_generation'].update(identity_reference_scale=.85,
        garment_reference_scale=.3)
    prepared = []
    def configure(pipeline, value, identity, garment):
        prepared.append((identity, garment))
        pipeline.set_ip_adapter_scale([[identity, garment]])
        return {'cached': True}
    monkeypatch.setattr('genai_lab.visual_reference.configure_visual_condition', configure)
    batch = generate_visual_batch(pipeline, config, request, root, log, inputs, lambda: False, lambda x: None)
    assert prepared == [(.85, .3)]
    assert scales[0] == [[.85, .3]] and scales[-1] == .8
    for index, candidate in enumerate(batch.candidates):
        assert candidate.design_reference_record['identity_scale'] == .85
        assert candidate.design_reference_record['garment_scale'] == .3
    batch.close()


@pytest.mark.parametrize('invalid', [-.1, 1.1, float('nan'), float('inf')])
def test_invalid_reference_scale_blocks_before_preparation(inputs, setup_batch, monkeypatch, invalid):
    pipeline, config, request, root, log, seen, scales = setup_batch
    config['clothing_reference_generation']['identity_reference_scale'] = invalid
    def forbidden(*args):
        pytest.fail('model conditioning must not run for invalid scale')
    monkeypatch.setattr('genai_lab.visual_reference.configure_visual_condition', forbidden)
    with pytest.raises(ValueError, match='참조 강도'):
        generate_visual_batch(pipeline, config, request, root, log, inputs, lambda: False, lambda x: None)
    assert not seen and not scales


def test_cancel_preserves_completed(inputs, setup_batch):
    pipeline, config, request, root, log, seen, _ = setup_batch
    batch = generate_visual_batch(pipeline, config, request, root, log, inputs,
                                  lambda: len(seen) == 1, lambda x: None)
    assert len(batch.candidates) == 1
    assert batch.warning
    batch.close()


def test_later_failure_preserves_first(inputs, setup_batch, monkeypatch):
    pipeline, config, request, root, log, seen, _ = setup_batch
    def generate(*args):
        seen.append(1)
        if len(seen) > 1:
            raise RuntimeError('out of memory')
        return Candidate(inputs.source.copy(), 42)
    monkeypatch.setattr('genai_lab.generator.generate_character_candidate', generate)
    batch = generate_visual_batch(pipeline, config, request, root, log, inputs, lambda: False, lambda x: None)
    assert len(batch.paths) == 1
    assert 'out of memory' in batch.warning
    batch.close()


@pytest.mark.parametrize('invalid_count', [0, 1, 101])
def test_invalid_count_blocks(inputs, setup_batch, invalid_count):
    pipeline, config, request, root, log, *_ = setup_batch
    config['clothing_reference_generation']['candidate_count'] = invalid_count
    with pytest.raises(ValueError):
        generate_visual_batch(pipeline, config, request, root, log, inputs, lambda: False, lambda x: None)


def test_cropped_references_reach_encoder_without_spatial_masks(inputs):
    scales = []
    def prepare(**kwargs):
        assert kwargs['ip_adapter_image'] == [[inputs.identity, inputs.garment]]
        return ['embedding']
    pipeline = SimpleNamespace(set_ip_adapter_scale=scales.append,
        prepare_ip_adapter_image_embeds=prepare, _execution_device='cpu')
    result = configure_visual_condition(pipeline, inputs)
    assert result['ip_adapter_image_embeds'] == ['embedding']
    assert 'cross_attention_kwargs' not in result
    assert scales == [pytest.approx((.7 + .45) / 2)]


def test_candidate_dialog_no_automatic_approval(inputs, setup_batch):
    from PySide6.QtWidgets import QApplication
    from genai_lab.visual_reference_review import VisualCandidateReview
    app = QApplication.instance() or QApplication([])
    pipeline, config, request, root, log, *_ = setup_batch
    batch = generate_visual_batch(pipeline, config, request, root, log, inputs, lambda: False, lambda x: None)
    dialog = VisualCandidateReview(batch)
    dialog.selection.setCurrentIndex(2)
    assert dialog.selection.currentData() == 2
    assert not dialog.result()
    dialog.part_reviews[2]['hair'].setCurrentIndex(2)
    dialog.confirm_selection()
    import json
    record = json.loads((root / 'candidate_3.json').read_text(encoding='utf-8'))
    assert record['part_review']['hair'] == 'different'
    assert record['approval'] == 'pending'
    dialog.close()
    batch.close()


@pytest.mark.parametrize('approve', [False, True])
@pytest.mark.parametrize('tag_approval', [None, False, True])
def test_worker_requires_input_approval_before_model(inputs, monkeypatch, tmp_path, approve, tag_approval):
    import gui_main
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    order = []
    config = {'clothing_reference_generation': {'visual_enabled': True,
              'garment_image': inputs.garment.copy(), 'character_tag_review': tag_approval is not None}}
    request = SimpleNamespace(reference_image=inputs.source, width=64, height=112,
        seed=42, candidate_number=1, inference_steps=20, guidance_scale=7.,
        model_id='test-model', reference_adapter_id='test-adapter')
    config['clothing_reference_generation'].update(
        candidate_count=2, approved_prompt_pair=('approved positive', 'approved negative'))
    log_entries = []
    close_calls = []
    log = SimpleNamespace(
        write_stage=lambda *a: log_entries.append(a),
        close=lambda: close_calls.append(True),
        write_failure=lambda *a: None,
        file_path=tmp_path / 'run.log',
    )
    monkeypatch.setattr(gui_main, 'prepare_visual_inputs', lambda *a: order.append('masks') or inputs)
    monkeypatch.setattr(gui_main, 'save_generation_replay_bundle',
                        lambda *a: tmp_path / 'replay')
    def analyze_tags(image, config, *, face_hair_mask, hair_mask, face_mask,
                     cancelled):
        assert image is inputs.source
        assert face_hair_mask is inputs.identity_mask
        assert hair_mask is inputs.hair_mask
        assert face_mask is inputs.face_mask
        assert callable(cancelled)
        order.append('tags')
        return SimpleNamespace(tag_candidates=(), elapsed_seconds=.1,
            eye_color_report={'status': 'unresolved', 'reasons': ['low_score'], 'debug_dir': str(tmp_path)})
    monkeypatch.setattr(gui_main, 'analyze_character_tags', analyze_tags)
    monkeypatch.setattr(gui_main, 'configure_system_certificates', lambda: None)
    monkeypatch.setattr(gui_main, 'check_environment', lambda: {'vram_bytes': 8 * 1024**3, 'gpu': 'mock'})
    monkeypatch.setattr(gui_main, 'prepare_pipeline', lambda *a, **k: order.append('model') or object())
    monkeypatch.setattr(gui_main, 'generate_visual_batch', lambda *a, **kw: order.append('generate') or object())
    worker = gui_main.GenerationWorker(config, request, tmp_path, log, None, None, None, None)
    failures = []
    worker.failed.connect(lambda message, details, pipeline: failures.append(details))
    def review(value):
        order.append('review')
        if approve:
            worker.orchestrator.approve_visual_inputs(inputs)
        worker.inputs_approved = approve
        worker.review_done.set()
    worker.inputs_ready.connect(review)
    def review_tags(result):
        order.append('tag_review')
        worker.character_tags_approved = tag_approval
        worker.character_tags_done.set()
    worker.character_tags_ready.connect(review_tags)
    worker.run()
    expected = ['review', 'model', 'generate'] if approve else ['review']
    if tag_approval is not None:
        expected = ['tags', 'tag_review'] + (expected if tag_approval else [])
    assert order == ['masks'] + expected, failures
    if tag_approval is not None:
        assert config['clothing_reference_generation']['eye_color_report']['status'] == 'unresolved'
        assert any(
            stage == '이미지 태그 변환 시간' and '상태=completed' in detail
            for stage, detail in log_entries
        )
    else:
        assert (
            '이미지 태그 변환 시간',
            '상태=skipped, 사유=character_tag_review 비활성',
        ) in log_entries
    succeeded = approve and tag_approval is not False
    if succeeded:
        assert any(
            stage == '이미지 생성 시간' and '상태=completed' in detail
            for stage, detail in log_entries
        )
        assert close_calls == []
        worker.orchestrator.close()
        assert close_calls == [True]
    else:
        assert close_calls == [True]


@pytest.mark.parametrize('approved_prompt', [False, True])
def test_worker_checks_prompt_approval_before_loading_model(inputs, monkeypatch, tmp_path, approved_prompt):
    import gui_main
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    order = []
    section = {'visual_enabled': True, 'garment_image': inputs.garment.copy(),
               'character_tag_review': True, 'require_prompt_approval': True, 'candidate_count': 2}
    log = SimpleNamespace(write_stage=lambda *a: None, close=lambda: None,
                          write_failure=lambda *a: None, file_path=tmp_path/'run.log')
    monkeypatch.setattr(gui_main, 'prepare_visual_inputs', lambda *a: inputs)
    monkeypatch.setattr(gui_main, 'save_generation_replay_bundle',
                        lambda *a: tmp_path / 'replay')
    monkeypatch.setattr(gui_main, 'configure_system_certificates', lambda: None)
    monkeypatch.setattr(gui_main, 'load_reference_tokenizers',
                        lambda config: order.append('tokenizers') or ('a', 'b'))
    monkeypatch.setattr(gui_main, 'analyze_character_tags', lambda *a, **kw: SimpleNamespace(
        tag_candidates=(), elapsed_seconds=0., eye_color_report={
            'status': 'unresolved', 'reasons': [], 'debug_dir': str(tmp_path)}))
    monkeypatch.setattr(gui_main, 'check_environment', lambda: {'vram_bytes': 8*1024**3, 'gpu': 'mock'})
    monkeypatch.setattr(gui_main, 'prepare_pipeline', lambda *a, **kw: order.append('model') or object())
    monkeypatch.setattr(gui_main, 'generate_visual_batch', lambda *a, **kw: order.append('generate') or object())
    worker = gui_main.GenerationWorker({'clothing_reference_generation': section},
        SimpleNamespace(reference_image=inputs.source, width=64, height=112,
            seed=42, candidate_number=1, inference_steps=20, guidance_scale=7.,
            model_id='test-model', reference_adapter_id='test-adapter'),
        tmp_path, log, None, None, None, None)
    failures = []
    worker.failed.connect(lambda *args: failures.append(args[1]))
    def approve_tags(result):
        order.append('feature_review')
        worker.character_tags_approved = True
        if approved_prompt:
            section['approved_prompt_pair'] = ('approved positive', 'approved negative')
        worker.character_tags_done.set()
    def approve_inputs(value):
        order.append('input_review')
        worker.orchestrator.approve_visual_inputs(inputs)
        worker.inputs_approved = True
        worker.review_done.set()
    worker.character_tags_ready.connect(approve_tags)
    worker.inputs_ready.connect(approve_inputs)
    worker.run()
    assert order == ['tokenizers', 'feature_review'] + (
        ['input_review', 'model', 'generate'] if approved_prompt else []), failures
    if not approved_prompt:
        assert any('생성 프롬프트가 승인되지' in details for details in failures)


@pytest.mark.parametrize('missing_face', [False, True])
def test_automatic_mask_preparation_never_calls_restoration(inputs, monkeypatch, tmp_path, missing_face):
    from genai_lab.visual_reference import prepare_visual_inputs
    from genai_lab.reference_regions import ReferenceRegions
    checked = []
    masks = {
        "identity": Image.new("L", inputs.source.size) if missing_face else inputs.identity_mask.copy(),
        "hair": inputs.identity_mask.copy(),
        "face": Image.new("L", inputs.source.size),
        "garment": inputs.garment_mask.copy(),
        "foreground": Image.new("L", inputs.source.size, 255),
    }
    regions = ReferenceRegions(masks, {"mode": "reference_regions_v1"})
    original_close = regions.close
    def close():
        checked.append("closed")
        original_close()
    regions.close = close
    def analyze(source, config, root, **kwargs):
        assert source is inputs.source
        assert kwargs["analysis_scope"] == "character_input"
        return regions
    def forbidden(*args, **kwargs):
        pytest.fail("참조 준비에서 기존 합성/의상 제거 함수를 호출하면 안 됩니다.")
    monkeypatch.setattr('genai_lab.reference_regions.analyze_reference_regions', analyze)
    for name in ("execute_character_body_comparison", "create_human_agnostic_image_candidate",
                 "verify_original_clothing_removal"):
        monkeypatch.setattr("genai_lab.body_comparison." + name, forbidden)
    cfg = {"clothing_reference_generation": {"enabled": True}}
    log_entries = []
    run_log = SimpleNamespace(write_stage=lambda *args: log_entries.append(args))
    if missing_face:
        with pytest.raises(ValueError, match='얼굴·헤어 참조'):
            prepare_visual_inputs(
                inputs.source, inputs.garment, cfg, tmp_path, run_log=run_log)
        assert any(
            stage == "이미지 추출 시간" and "상태=failed" in detail
            for stage, detail in log_entries
        )
    else:
        result = prepare_visual_inputs(
            inputs.source, inputs.garment, cfg, tmp_path, run_log=run_log)
        assert not hasattr(result, "initial")
        assert result.analysis_record["additional_parts_status"] == "analyzer_not_connected"
        assert result.color_evidence["hair"].pattern == "unresolved"
        timings = result.analysis_record["preparation_stage_timings"]
        assert timings["image_mask_processing_seconds"] >= 0
        assert timings["image_extraction_seconds"] >= 0
        assert any(stage == "이미지 마스크 처리 시간" for stage, _ in log_entries)
        assert any(stage == "이미지 추출 시간" for stage, _ in log_entries)
        result.close()
    assert checked == ["closed"]


def test_vibe_score_is_separate_and_not_sent_to_adapter(inputs, monkeypatch):
    inputs.full_character_mask = Image.new('L', inputs.source.size, 255)
    inputs.vibe_reference = inputs.source.copy()
    monkeypatch.setattr('genai_lab.visual_reference.image_feature', lambda *a: np.array([1., 0.]))
    scores = compare_candidate(None, inputs.source, inputs,
        {'character': np.array([1., 0.]), 'garment': np.array([0., 1.]), 'vibe': np.array([-1., 0.])})
    assert (scores['character'], scores['garment'], scores['vibe']) == (1., 0., -1.)
    seen = []
    def prepare(**kwargs):
        seen.extend(kwargs['ip_adapter_image'][0])
        return ['embeds']
    configure_visual_condition(SimpleNamespace(set_ip_adapter_scale=lambda *a: None,
        prepare_ip_adapter_image_embeds=prepare, _execution_device='cpu'), inputs)
    assert seen == [inputs.identity, inputs.garment]
    assert all(image is not inputs.vibe_reference for image in seen)


def test_candidate_metadata_records_actual_adapter_order(inputs, setup_batch):
    import json
    from genai_lab.scene_reference import PartReference
    pipeline, config, request, root, log, *_ = setup_batch
    ear_mask = Image.new('L', inputs.source.size)
    ear_mask.paste(255, (0, 0, 5, 10))
    tail_mask = Image.new('L', inputs.source.size)
    tail_mask.paste(255, (0, 35, 8, 90))
    inputs.extra_references = (
        PartReference('tail', Image.new('RGB', (16, 16), 'cyan'), tail_mask, .31),
        PartReference('ears', Image.new('RGB', (16, 16), 'purple'), ear_mask, .19))
    batch = generate_visual_batch(pipeline, config, request, root, log, inputs, lambda: False, lambda _: None)
    try:
        for path in root.glob('candidate_*.json'):
            data = json.loads(path.read_text(encoding='utf-8'))
            assert data['regional_reference_names'] == ['identity', 'ears', 'tail', 'garment']
    finally:
        batch.close()


def test_gender_pass_with_low_similarity_does_not_open_gender_retry(inputs, setup_batch, monkeypatch):
    pipeline, config, request, root, log, seen, _ = setup_batch
    config['clothing_reference_generation']['candidate_count'] = 2
    config['candidate_pipeline'] = {
        'enabled': True, 'target_valid_candidates': 1, 'maximum_attempts': 2,
        'gender_retry_attempts': 2, 'minimum_component_similarity': .65,
        'require_all_similarity_scores': True, 'repair_best_candidate_only': False}
    config['candidate_semantic_gate'] = {'enabled': True}
    inputs.approved_generation = SimpleNamespace(record=lambda: {})
    actions = iter(['retry', 'pass'])
    class Analyzer:
        def __init__(self, config): pass
        def close(self): pass
        def analyze(self, *args):
            action = next(actions)
            return {'action': action, 'violations': ['gender_conflict:1girl'] if action == 'retry' else []}
    monkeypatch.setattr('genai_lab.candidate_semantic_gate.CandidateSemanticAnalyzer', Analyzer)
    monkeypatch.setattr('genai_lab.visual_reference.compare_candidate',
                        lambda *args: dict(character=.48, garment=.57, vibe=.58))
    with pytest.raises(ValueError):
        generate_visual_batch(pipeline, config, request, root, log, inputs, lambda: False, lambda _: None)
    assert seen == [42, 43]
    summary = json.loads((root / 'candidate_failure_summary.json').read_text(
        encoding='utf-8'))
    assert summary['failure_category_counts'] == {
        'gender_conflict': 1, 'similarity_weak': 1}
    assert summary['next_action'] == 'enable_bounded_quality_retry'


def test_similarity_failure_opens_bounded_quality_retry_and_stops_on_pass(
        inputs, setup_batch, monkeypatch):
    pipeline, config, request, root, log, seen, _ = setup_batch
    config['clothing_reference_generation']['candidate_count'] = 2
    config['candidate_pipeline'] = {
        'enabled': True, 'target_valid_candidates': 1, 'maximum_attempts': 2,
        'gender_retry_attempts': 2, 'quality_retry_attempts': 2,
        'minimum_component_similarity': .65,
        'require_all_similarity_scores': True,
        'repair_best_candidate_only': False,
    }
    config['candidate_semantic_gate'] = {'enabled': True}
    inputs.approved_generation = SimpleNamespace(record=lambda: {})

    class Analyzer:
        def __init__(self, config): pass
        def close(self): pass
        def analyze(self, *args):
            return {'action': 'pass', 'violations': [], 'unresolved': []}

    scores = iter((
        dict(character=.48, garment=.57, vibe=.58, warnings=[]),
        dict(character=.61, garment=.62, vibe=.70, warnings=[]),
        dict(character=.76, garment=.74, vibe=.72, warnings=[]),
    ))
    monkeypatch.setattr(
        'genai_lab.candidate_semantic_gate.CandidateSemanticAnalyzer', Analyzer)
    monkeypatch.setattr(
        'genai_lab.visual_reference.compare_candidate',
        lambda *args: next(scores))
    batch = generate_visual_batch(
        pipeline, config, request, root, log, inputs,
        lambda: False, lambda _: None)
    try:
        assert seen == [42, 43, 44]
        assert [candidate.seed for candidate in batch.candidates] == [44]
        budget = batch.candidates[0].design_reference_record[
            'candidate_attempt_budget']
        assert budget['retry_phase'] == 'quality'
        assert budget['is_quality_retry'] is True
        assert budget['is_gender_retry'] is False
    finally:
        batch.close()



def test_mixed_retryable_semantic_failures_open_bounded_quality_retry(
        inputs, setup_batch, monkeypatch):
    pipeline, config, request, root, log, seen, _ = setup_batch
    config['clothing_reference_generation']['candidate_count'] = 2
    config['candidate_pipeline'] = {
        'enabled': True,
        'target_valid_candidates': 1,
        'maximum_attempts': 2,
        'gender_retry_attempts': 2,
        'quality_retry_attempts': 2,
        'minimum_component_similarity': .65,
        'require_all_similarity_scores': True,
        'repair_best_candidate_only': False,
    }
    config['candidate_semantic_gate'] = {'enabled': True}
    inputs.approved_generation = SimpleNamespace(record=lambda: {})
    reports = iter((
        {'action': 'retry', 'violations': ['gender_conflict:1girl'],
         'unresolved': []},
        {'action': 'retry', 'violations': ['ears_species_conflict:cat ears'],
         'unresolved': []},
        {'action': 'pass', 'violations': [], 'unresolved': []},
    ))

    class Analyzer:
        def __init__(self, config):
            pass

        def close(self):
            pass

        def analyze(self, *args):
            return next(reports)

    monkeypatch.setattr(
        'genai_lab.candidate_semantic_gate.CandidateSemanticAnalyzer', Analyzer)
    monkeypatch.setattr(
        'genai_lab.visual_reference.compare_candidate',
        lambda *args: dict(
            character=.76, garment=.74, vibe=.72, warnings=[]))

    batch = generate_visual_batch(
        pipeline, config, request, root, log, inputs,
        lambda: False, lambda _: None)
    try:
        assert seen == [42, 43, 44]
        assert [candidate.seed for candidate in batch.candidates] == [44]
        budget = batch.candidates[0].design_reference_record[
            'candidate_attempt_budget']
        assert budget['retry_phase'] == 'quality'
        assert budget['is_quality_retry'] is True
        assert budget['is_gender_retry'] is False
    finally:
        batch.close()

def test_structure_rejection_prevents_similarity_scoring(inputs, setup_batch, monkeypatch):
    pipeline, config, request, root, log, seen, _ = setup_batch
    checks = []
    def structure(image, settings):
        checks.append(image.size)
        return {'action': 'reject', 'violations': ['large_detached_object_above_character']}
    monkeypatch.setattr('genai_lab.candidate_structure_gate.evaluate_candidate_structure', structure)
    def forbidden(*args):
        raise AssertionError('structurally invalid candidate reached similarity')
    monkeypatch.setattr('genai_lab.visual_reference.compare_candidate', forbidden)
    from genai_lab.generated_image_integrity import GeneratedOutputIntegrityError
    with pytest.raises(GeneratedOutputIntegrityError):
        generate_visual_batch(pipeline, config, request, root, log, inputs, lambda: False, lambda _: None)
    assert len(checks) == 3


def test_missing_or_quarantined_candidates_open_bounded_quality_retry(
        inputs, setup_batch, monkeypatch):
    pipeline, config, request, root, log, seen, _ = setup_batch
    config["clothing_reference_generation"]["candidate_count"] = 2
    config["candidate_pipeline"] = {
        "enabled": True,
        "target_valid_candidates": 1,
        "maximum_attempts": 2,
        "gender_retry_attempts": 2,
        "quality_retry_attempts": 2,
        "minimum_component_similarity": .65,
        "require_all_similarity_scores": True,
        "repair_best_candidate_only": False,
    }

    def generate(_pipeline, local_config, local_request, _root, _log):
        seen.append(local_request.seed)
        if len(seen) <= 2:
            return None
        assert local_config["clothing_reference_generation"][
            "visual_inputs"] is inputs
        return Candidate(inputs.source.copy(), local_request.seed)

    monkeypatch.setattr(
        "genai_lab.generator.generate_character_candidate", generate)
    monkeypatch.setattr(
        "genai_lab.visual_reference.compare_candidate",
        lambda *args, **kwargs: dict(
            character=.76, garment=.74, vibe=.72, warnings=[]),
    )

    batch = generate_visual_batch(
        pipeline, config, request, root, log, inputs,
        lambda: False, lambda _: None,
    )
    try:
        assert seen == [42, 43, 44]
        assert [candidate.seed for candidate in batch.candidates] == [44]
        budget = batch.candidates[0].design_reference_record[
            "candidate_attempt_budget"]
        assert budget["retry_phase"] == "quality"
        assert budget["is_quality_retry"] is True
    finally:
        batch.close()



def test_structure_rejection_opens_bounded_quality_retry(
        inputs, setup_batch, monkeypatch):
    pipeline, config, request, root, log, seen, _ = setup_batch
    config["clothing_reference_generation"]["candidate_count"] = 2
    config["candidate_pipeline"] = {
        "enabled": True,
        "target_valid_candidates": 1,
        "maximum_attempts": 2,
        "gender_retry_attempts": 2,
        "quality_retry_attempts": 2,
        "minimum_component_similarity": .65,
        "require_all_similarity_scores": True,
        "repair_best_candidate_only": False,
    }
    actions = iter(("reject", "reject", "pass"))

    def structure(_image, settings):
        action = next(actions)
        return {
            **settings.record(),
            "status": action.upper(),
            "action": action,
            "violations": (
                ["character_occupancy_too_small"]
                if action == "reject" else []),
            "unresolved": [],
        }

    monkeypatch.setattr(
        "genai_lab.candidate_structure_gate.evaluate_candidate_structure",
        structure,
    )
    monkeypatch.setattr(
        "genai_lab.visual_reference.compare_candidate",
        lambda *args, **kwargs: dict(
            character=.76, garment=.74, vibe=.72, warnings=[]),
    )

    batch = generate_visual_batch(
        pipeline, config, request, root, log, inputs,
        lambda: False, lambda _: None,
    )
    try:
        assert seen == [42, 43, 44]
        assert [candidate.seed for candidate in batch.candidates] == [44]
        budget = batch.candidates[0].design_reference_record[
            "candidate_attempt_budget"]
        assert budget["retry_phase"] == "quality"
        assert budget["is_quality_retry"] is True
    finally:
        batch.close()



def test_output_coordinate_similarity_uses_redetected_face_and_hair(inputs, monkeypatch):
    inputs.identity_mask.close()
    inputs.identity_mask = Image.new('L', inputs.source.size)
    inputs.hair_reference = Image.new('RGB', (32, 32), 'purple')
    full = Image.new('L', inputs.source.size, 255)
    masks = {
        'face': full.copy(),
        'identity': full.copy(),
        'hair': full.copy(),
        'garment': full.copy(),
        'foreground': full.copy(),
    }
    monkeypatch.setattr(
        'genai_lab.visual_reference.image_feature',
        lambda *args: np.array([1., 0.]),
    )
    try:
        scores = compare_candidate(
            None,
            inputs.source,
            inputs,
            {
                'character': np.array([1., 0.]),
                'hair': np.array([1., 0.]),
                'garment': np.array([1., 0.]),
                'vibe': np.array([1., 0.]),
            },
            candidate_masks=masks,
        )
        assert 'redetected output-coordinate ROI' in scores['method']
        assert scores['character'] == 1.
        assert scores['hair'] == 1.
        assert scores['warnings'] == []
    finally:
        full.close()
        for mask in masks.values():
            mask.close()


def test_partial_valid_pool_opens_quality_retry_until_target(inputs, setup_batch, monkeypatch):
    pipeline, config, request, root, log, seen, _ = setup_batch
    config['clothing_reference_generation']['candidate_count'] = 2
    config['candidate_pipeline'] = {
        'enabled': True,
        'target_valid_candidates': 2,
        'maximum_attempts': 2,
        'gender_retry_attempts': 2,
        'quality_retry_attempts': 2,
        'minimum_component_similarity': .65,
        'require_all_similarity_scores': True,
        'repair_best_candidate_only': False,
    }
    config['candidate_semantic_gate'] = {'enabled': True}
    inputs.approved_generation = SimpleNamespace(record=lambda: {})

    class Analyzer:
        def __init__(self, config): pass
        def close(self): pass
        def analyze(self, *args):
            return {'action': 'pass', 'violations': [], 'unresolved': []}

    scores = iter((
        dict(character=.76, garment=.74, vibe=.72, warnings=[]),
        dict(character=.61, garment=.62, vibe=.70, warnings=[]),
        dict(character=.78, garment=.75, vibe=.73, warnings=[]),
    ))
    monkeypatch.setattr(
        'genai_lab.candidate_semantic_gate.CandidateSemanticAnalyzer', Analyzer)
    monkeypatch.setattr(
        'genai_lab.visual_reference.compare_candidate',
        lambda *args: next(scores))

    batch = generate_visual_batch(
        pipeline, config, request, root, log, inputs,
        lambda: False, lambda _: None)
    try:
        assert seen == [42, 43, 44]
        assert [candidate.seed for candidate in batch.candidates] == [42, 44]
        retry_budget = batch.candidates[1].design_reference_record[
            'candidate_attempt_budget']
        assert retry_budget['retry_phase'] == 'quality'
        assert retry_budget['is_quality_retry'] is True
        assert retry_budget['is_gender_retry'] is False
    finally:
        batch.close()


def test_hair_view_failure_still_runs_and_records_final_gates(
        inputs, setup_batch, monkeypatch):
    pipeline, config, request, root, log, seen, _ = setup_batch
    config['clothing_reference_generation']['candidate_count'] = 2
    config['candidate_pipeline'] = {
        'enabled': True,
        'target_valid_candidates': 1,
        'maximum_attempts': 1,
        'repair_best_candidate_only': True,
        'reject_failed_post_audit': True,
    }
    inputs.approved_generation = SimpleNamespace(record=lambda: {
        'gender': 'male',
        'approved_tags': [],
        'approved_part_color_descriptions': [],
        'prompt': 'blue jacket',
        'negative_prompt': '',
        'part_error_correction': {},
    })

    monkeypatch.setattr(
        'genai_lab.selected_garment_correction.correct_selected_garment',
        lambda pipeline, image, *args, **kwargs: (
            image, {'status': 'disabled', 'ip_adapter_inputs': [],
                    'blocked_ip_adapter_inputs': []}),
    )
    monkeypatch.setattr(
        'genai_lab.part_error_correction.correct_detected_parts',
        lambda pipeline, image, *args, **kwargs: SimpleNamespace(
            image=image, report={'status': 'completed', 'parts': {}}),
    )
    monkeypatch.setattr(
        'genai_lab.hair_error_correction.correct_selected_hair',
        lambda pipeline, image, *args, **kwargs: (
            image,
            {
                'status': 'skipped_view_incompatible',
                'reason': 'unobserved_hair_geometry_required',
                'adapter_policy': {},
                'view_compatibility': {
                    'status': 'incompatible',
                    'inpaint_allowed': False,
                },
            },
        ),
    )
    monkeypatch.setattr(
        'genai_lab.generated_condition_audit.build_generated_condition_audit',
        lambda *args: {'status': 'PASS', 'checks': {}},
    )

    with pytest.raises(ValueError, match='최종 게이트'):
        generate_visual_batch(
            pipeline, config, request, root, log, inputs,
            lambda: False, lambda _: None)

    record = json.loads(
        (root / 'candidate_1.json').read_text(encoding='utf-8'))
    assert seen == [42]
    assert 'post_repair_semantic_gate' in record
    assert 'post_repair_structure_gate' in record
    assert 'post_repair_similarity' in record
    assert record['generated_condition_audit']['checks']['hair_view'] == {
        'status': 'FAIL',
        'reason': 'unobserved_hair_geometry_required',
        'correction_status': 'skipped_view_incompatible',
        'view_compatibility': {
            'status': 'incompatible',
            'inpaint_allowed': False,
        },
    }
    assert record['candidate_resolution']['status'] == (
        'quarantined_after_repair')
    assert record['candidate_resolution']['returned_image'] is None

def test_unresolved_hair_mask_skips_tensor_condition_and_keeps_base_flow(
        inputs, monkeypatch, tmp_path):
    from genai_lab.hair_mask_refinement import HairMaskResult
    from genai_lab.reference_regions import ReferenceRegions
    from genai_lab.visual_reference import prepare_visual_inputs

    masks = {
        "identity": inputs.identity_mask.copy(),
        "hair": inputs.identity_mask.copy(),
        "face": Image.new("L", inputs.source.size),
        "garment": inputs.garment_mask.copy(),
        "foreground": Image.new("L", inputs.source.size, 255),
    }
    regions = ReferenceRegions(masks, {"mode": "reference_regions_v1"})
    monkeypatch.setattr(
        "genai_lab.reference_regions.analyze_reference_regions",
        lambda *args, **kwargs: regions,
    )
    monkeypatch.setattr(
        "genai_lab.hair_mask_refinement.refine_hair_mask",
        lambda *args, **kwargs: HairMaskResult(
            "unresolved", None, "hair_mask_refinement_exhausted", (), True),
    )
    config = {"reference_analysis": {
        "color_analysis_enabled": False,
        "hair_visual_reference": {
            "enabled": True,
            "reference_scale": .6,
            "padding_ratio": .15,
            "neutral_background": [128, 128, 128],
        },
    }}
    result = prepare_visual_inputs(
        inputs.source, inputs.garment, config, tmp_path)
    try:
        condition = result.analysis_record["hair_visual_condition"]
        assert condition["status"] == "unresolved_skipped"
        assert condition["automatic_conditioning"] is False
        assert condition["fallback"] == "approved_text_tags_only"
        assert result.hair_reference is None
        assert result.hair_mask is None
    finally:
        result.close()

def test_tail_overlapping_ear_is_unresolved_without_approved_tail_tag():
    from genai_lab.visual_reference import unresolved_cross_class_parts
    report = {
        "parts": {"tail": {"automatic_conditioning": True}},
        "part_overlap": {
            "human_ears:tail": {"smaller_part_ratio": .99},
        },
    }
    unresolved = unresolved_cross_class_parts(report, ("short hair",))
    assert unresolved == {"tail"}
    assert report["parts"]["tail"]["automatic_conditioning"] is False
    assert report["parts"]["tail"]["semantic_status"] == "unresolved"
    assert report["cross_class_conflicts"][0]["raw_detection_preserved"] is True


def test_approved_tail_tag_keeps_tail_candidate_for_later_review():
    from genai_lab.visual_reference import unresolved_cross_class_parts
    report = {
        "parts": {"tail": {"automatic_conditioning": True}},
        "part_overlap": {
            "animal_ears:tail": {"smaller_part_ratio": .99},
        },
    }
    unresolved = unresolved_cross_class_parts(report, ("raccoon tail",))
    assert unresolved == set()
    assert report["parts"]["tail"]["automatic_conditioning"] is True
    assert report["cross_class_conflicts"][0]["action"] == "approved_tag_override"



def test_source_revalidation_uses_character_input_scope(monkeypatch, tmp_path):
    from genai_lab.reference_regions import ReferenceRegions
    from genai_lab.visual_reference import detect_source_validation_regions

    seen = {}

    def analyze(image, config, root, **kwargs):
        seen.update(kwargs)
        masks = {
            name: Image.new("L", image.size, 255)
            for name in ("identity", "hair", "face", "garment", "foreground")
        }
        return ReferenceRegions(
            masks,
            {"mode": "reference_regions_v1", "pixel_counts": {name: 256 for name in masks}},
        )

    monkeypatch.setattr(
        "genai_lab.reference_regions.analyze_reference_regions", analyze)
    with Image.new("RGB", (16, 16), "white") as image:
        regions, output = detect_source_validation_regions(
            image, {}, tmp_path, tmp_path)
    try:
        assert seen["analysis_scope"] == "character_input"
        assert seen["allow_parser_foreground_fallback"] is True
        assert output.name == "source_reference_revalidation"
        assert regions.record["validation_only"] is True
        assert regions.record["source_reference_masks_reused_for_generation"] is False
    finally:
        regions.close()



def test_a6_not_measurable_mask_quarantines_base_before_return(
        inputs, setup_batch, monkeypatch):
    import json
    from genai_lab.reference_regions import ReferenceRegions

    pipeline, config, request, root, log, seen, _ = setup_batch
    config["clothing_reference_generation"]["candidate_count"] = 2
    config["candidate_pipeline"] = {
        "enabled": True,
        "target_valid_candidates": 1,
        "maximum_attempts": 2,
        "repair_best_candidate_only": False,
        "output_coordinate_similarity": True,
    }

    size = inputs.source.size

    def masks(*, tiny_hair=False):
        full = Image.new("L", size, 255)
        hair = Image.new("L", size, 0)
        if tiny_hair:
            hair.putpixel((0, 0), 255)
        else:
            hair.paste(255, (0, 0, max(2, size[0] // 2),
                             max(2, size[1] // 3)))
        return {
            "face": full.copy(),
            "hair": hair,
            "identity": full.copy(),
            "foreground": full.copy(),
            "garment": full,
        }

    source_regions = ReferenceRegions(
        masks(),
        {"mode": "reference_regions_v1", "pixel_counts": {}},
    )

    def source_detection(*args, **kwargs):
        return source_regions, root / "source-validation"

    def output_detection(*args, **kwargs):
        return ReferenceRegions(
            masks(tiny_hair=True),
            {"mode": "reference_regions_v1", "pixel_counts": {}},
        ), root / "base-validation"

    monkeypatch.setattr(
        "genai_lab.visual_reference.detect_source_validation_regions",
        source_detection,
    )
    monkeypatch.setattr(
        "genai_lab.visual_reference.detect_output_similarity_regions",
        output_detection,
    )
    monkeypatch.setattr(
        "genai_lab.visual_reference.compare_candidate",
        lambda *args, **kwargs: {
            "character": .9,
            "hair": .9,
            "garment": .9,
            "vibe": .9,
            "warnings": [],
        },
    )

    with pytest.raises(ValueError, match="필수 게이트"):
        generate_visual_batch(
            pipeline, config, request, root, log, inputs,
            lambda: False, lambda _: None,
        )
    assert seen == [42, 43]
    summary = json.loads(
        (root / "candidate_failure_summary.json").read_text(encoding="utf-8")
    )
    assert summary["failure_category_counts"] == {
        "reference_not_measurable": 2
    }
    candidate = json.loads(
        (root / "candidate_1.json").read_text(encoding="utf-8")
    )
    assert candidate["candidate_resolution"]["status"] == (
        "quarantined_before_native_refinement"
    )
    validation = candidate["reference_base_mask_validation"]
    assert validation["blocking"] is True
    assert validation["parts"]["hair"]["status"] == "not_measurable"
    assert "iou" not in validation["parts"]["hair"]["metrics"]


def test_unapproved_tail_is_preserved_as_diagnostic_without_conditioning():
    from genai_lab.visual_reference import suppress_unapproved_optional_parts
    report = {
        "parts": {
            "tail": {
                "status": "detected",
                "automatic_conditioning": True,
                "accepted_masks": 1,
            },
        },
    }

    suppressed = suppress_unapproved_optional_parts(
        report,
        ("1girl", "black hair"),
    )

    assert suppressed == {"tail"}
    tail = report["parts"]["tail"]
    assert tail["automatic_conditioning"] is False
    assert tail["target_status"] == "unresolved_unapproved_character_part"
    assert tail["raw_detection_preserved"] is True


def test_unapproved_tail_target_status_is_not_relabelled_as_experimental():
    report = {
        "parts": {
            "tail": {
                "status": "detected",
                "automatic_conditioning": False,
                "target_status": "unresolved_unapproved_character_part",
            },
        },
    }
    detected_names = ("tail",)
    generation_part_names = set()
    generation_region_audit = {}

    for part_name in detected_names:
        entry = report["parts"].get(part_name)
        if entry is not None:
            if part_name in generation_part_names:
                entry["target_status"] = "experimental_source_roi"
            entry["generation_region"] = generation_region_audit.get(part_name)

    assert report["parts"]["tail"]["target_status"] == (
        "unresolved_unapproved_character_part"
    )
    assert report["parts"]["tail"]["generation_region"] is None


def test_approved_species_tail_remains_available_for_later_gates():
    from genai_lab.visual_reference import suppress_unapproved_optional_parts
    report = {
        "parts": {
            "tail": {
                "status": "detected",
                "automatic_conditioning": True,
                "accepted_masks": 1,
            },
        },
    }

    suppressed = suppress_unapproved_optional_parts(
        report,
        ("raccoon tail",),
    )

    assert suppressed == set()
    assert report["parts"]["tail"]["automatic_conditioning"] is True
    assert "unapproved_optional_parts" not in report
