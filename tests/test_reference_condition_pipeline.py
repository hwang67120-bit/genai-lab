from dataclasses import dataclass

from PIL import Image

from genai_lab.clothing_reference_generation import prepare_design_reference_request
from genai_lab.generated_condition_audit import build_generated_condition_audit
from genai_lab.part_color_descriptions import PartColorDescription
from genai_lab.reference_condition_snapshot import build_reference_condition_snapshot
from genai_lab.sdxl_long_prompt import plan_phrase_chunks
from genai_lab.visual_reference import VisualInputs
from genai_lab.request import CharacterFramingType


class Tokenizer:
    model_max_length = 16

    def __call__(self, text, **kwargs):
        return {"input_ids": [0, *text.split(), 1]}


def description(part="tail"):
    return PartColorDescription(
        part, ("blue",), "proposed", (), b'{"rgb":[1,2,3]}', "policy")


@dataclass
class Request:
    framing_type: object = CharacterFramingType.FULL_BODY
    prompt: str = ""
    negative_prompt: str = "bad anatomy"


def test_diagnostic_color_is_immutable_and_never_compiled_into_prompt():
    request = Request()
    raw = (description(),)
    baseline, baseline_record = prepare_design_reference_request(
        request, ("jacket",), (Tokenizer(), Tokenizer()),
        character_tags=("1boy", "tail"), part_color_descriptions=raw,
        long_prompt_settings={"enabled": True, "maximum_chunks": 2})
    approved, record = prepare_design_reference_request(
        request, ("jacket",), (Tokenizer(), Tokenizer()),
        character_tags=("1boy", "tail"), part_color_descriptions=raw,
        approved_part_color_descriptions=raw,
        long_prompt_settings={"enabled": True, "maximum_chunks": 2})
    assert "blue tail" not in baseline.prompt
    assert "blue tail" not in approved.prompt
    assert baseline_record["part_color_descriptions"] == record["part_color_descriptions"]
    assert record["approved_part_color_descriptions"][0]["part_name"] == "tail"


def test_phrase_chunk_plan_never_splits_color_from_part():
    tokenizer = Tokenizer()
    plan = plan_phrase_chunks(
        "1boy, purple eyes, short blue hair, dark and light blue animal ears, "
        "dark blue tail, wearing jacket",
        (tokenizer, tokenizer), {"enabled": True, "maximum_chunks": 2})
    assert plan["chunk_count"] == 2
    assert any("dark and light blue animal ears" in item["text"]
               for item in plan["chunks"])


def test_snapshot_keeps_observation_approval_and_compiler_sections_separate():
    first = Image.new("L", (8, 8))
    second = Image.new("L", (8, 8))
    first.paste(255, (0, 0, 8, 4))
    second.paste(255, (0, 4, 8, 8))
    inputs = VisualInputs(
        Image.new("RGB", (8, 8)), Image.new("RGB", (4, 4)),
        Image.new("RGB", (4, 4)), first, second,
        analysis_record={"raw": {"tag": "tail"}},
        part_color_descriptions=(description(),))
    section = {
        "character_gender": "male",
        "approved_character_tags": ("1boy", "tail"),
        "approved_tags": ("jacket",),
        "approved_part_color_descriptions": (description(),),
        "approved_prompt_record": {"positive": {"effective": "1boy"}},
    }
    try:
        snapshot = build_reference_condition_snapshot(inputs, section)
        assert snapshot["observations"]["source_analysis_full"]["raw"]["tag"] == "tail"
        assert set((
            "face", "eyes", "hair", "human_ears", "animal_ears",
            "tail", "garment",
        )).issubset(
            snapshot["observations"])
        assert snapshot["approved_conditions"]["part_colors"][0]["part_name"] == "tail"
        assert snapshot["compiled_prompt"]["positive"]["effective"] == "1boy"
    finally:
        inputs.close()


def test_post_audit_does_not_turn_missing_analyzers_into_pass():
    audit = build_generated_condition_audit(
        {"gender": "male", "approved_tags": ["jacket"],
         "approved_part_color_descriptions": [
             {"part_name": "tail"}, {"part_name": "ears"}]},
        {"part_error_correction": {"parts": {
            "tail": {"status": "accepted_without_correction",
                     "before_color_distance": 2.0},
            "ears": {"status": "unresolved_keep_original",
                     "reason": "output_part_not_detected_keep_original"},
        }}})
    assert audit["checks"]["tail"]["status"] == "PASS"
    assert audit["checks"]["human_ears"]["status"] == "UNRESOLVED"
    assert audit["checks"]["animal_ears"]["status"] == "UNRESOLVED"
    assert audit["checks"]["human_and_animal_ear_coexistence"]["status"] == "UNRESOLVED"
    assert audit["status"] == "UNRESOLVED"


def test_post_audit_fails_when_approved_part_correction_crashes():
    audit = build_generated_condition_audit(
        {
            "gender": "male",
            "approved_tags": [],
            "approved_part_color_descriptions": [{"part_name": "ears"}],
        },
        {
            "part_error_correction": {
                "status": "failed_keep_original",
                "parts": {},
            },
        },
    )
    assert audit["checks"]["human_ears"]["status"] == "FAIL"
    assert audit["checks"]["animal_ears"]["status"] == "FAIL"
    assert audit["status"] == "FAIL"


def test_approved_color_missing_output_audit_is_blocking():
    audit = build_generated_condition_audit(
        {
            "gender": "male",
            "approved_tags": [],
            "approved_part_color_descriptions": [
                {"part_name": "animal_ears"},
            ],
        },
        {"part_error_correction": {"status": "completed", "parts": {}}},
    )
    assert audit["checks"]["animal_ears_color"]["status"] == "UNRESOLVED"
    assert audit["blocking_color_checks"] == {
        "animal_ears_color": "output_part_audit_missing",
    }
