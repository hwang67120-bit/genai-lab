from genai_lab.ear_contract import (
    ANIMAL_EARS,
    HUMAN_EARS,
    build_source_ear_contract,
    evaluate_output_ear_contract,
)


def entry(*, count=1, box=(10, 20, 30, 40), score=.9):
    return {
        "status": "detected",
        "accepted_masks": count,
        "accepted_boxes": [list(box)] * count,
        "detection_scores": [score],
        "mask_path": "mask.png",
        "crop_path": "crop.png",
        "mask_pixels": 100,
    }


def report(**parts):
    return {"parts": parts, "part_overlap": {}, "image_size": [100, 200]}


def test_human_and_animal_ears_are_independent_and_non_detection_is_unknown():
    contract = build_source_ear_contract(
        report(human_ears=entry()), (100, 200))
    assert contract["human"]["state"] == "present"
    assert contract["animal"]["state"] == "unknown"
    assert contract["generation_parts"] == [HUMAN_EARS]
    assert contract["human"]["location_verified"] is True
    assert contract["animal"]["absence_inferred_from_non_detection"] is False


def test_broad_human_ear_false_positive_stays_unknown_and_is_not_conditioned():
    contract = build_source_ear_contract(
        report(human_ears=entry(box=(2, 3, 98, 198))), (100, 200))
    assert contract["human"]["state"] == "unknown"
    assert contract["human"]["visible_count"] == 0
    assert contract["human"]["location_verified"] is False
    assert contract["human"]["evidence"] == "location_unverified"
    assert contract["human"]["ambiguity"]["raw_detection_preserved"] is True
    assert contract["generation_parts"] == []


def test_broad_animal_ear_false_positive_is_not_conditioned():
    contract = build_source_ear_contract(
        report(animal_ears=entry(box=(1, 30, 99, 120))), (100, 200))
    assert contract["animal"]["state"] == "unknown"
    assert contract["animal"]["location_verified"] is False
    assert contract["animal"]["evidence"] == "location_unverified"
    assert contract["animal"]["ambiguity"] == {
        "reason": "animal_ears_mask_outside_local_location_bounds",
        "maximum_normalized_width": .45,
        "maximum_normalized_height": .50,
        "maximum_normalized_center_y": .45,
        "raw_detection_preserved": True,
    }
    assert contract["generation_parts"] == []


def test_approved_animal_species_tag_is_recorded_without_forcing_presence():
    contract = build_source_ear_contract(
        report(), (100, 200), approved_character_tags=("raccoon ears",))
    assert contract["animal"]["approved_species_tags"] == ["raccoon ears"]
    assert contract["animal"]["state"] == "unknown"
    assert contract["generation_parts"] == []


def test_explicit_absence_is_allowed_but_cannot_contradict_detection():
    contract = build_source_ear_contract(
        report(human_ears=entry()), (100, 200),
        explicit_states={ANIMAL_EARS: "absent"})
    assert contract["animal"]["state"] == "absent"


def test_overlapping_unresolved_human_candidate_is_not_promoted_to_generation():
    detected = report(human_ears=entry(), animal_ears=entry())
    detected["part_overlap"]["human_ears:animal_ears"] = {
        "pixels": 100,
        "smaller_part_ratio": 1.0,
    }
    contract = build_source_ear_contract(detected, (100, 200))
    assert contract["human"]["state"] == "unknown"
    assert contract["human"]["evidence"] == "ambiguous_with_animal_ears"
    assert contract["human"]["ambiguity"]["raw_detection_preserved"] is True
    assert contract["animal"]["state"] == "present"
    assert contract["generation_parts"] == [ANIMAL_EARS]


def test_output_audit_checks_each_ear_type_and_count_separately():
    source = build_source_ear_contract(
        report(human_ears=entry(), animal_ears=entry(count=2)),
        (100, 200))
    output = report(human_ears=entry(), animal_ears=entry(count=1))
    audit = evaluate_output_ear_contract(source, output, (100, 200))
    assert audit["checks"][HUMAN_EARS]["status"] == "PASS"
    assert audit["checks"][ANIMAL_EARS]["status"] == "FAIL"
    assert audit["checks"][ANIMAL_EARS]["reason"] == "visible_count_mismatch"
    assert audit["status"] == "FAIL"


def test_output_audit_rejects_cross_type_mask_overlap():
    source = build_source_ear_contract(
        report(human_ears=entry(), animal_ears=entry()), (100, 200))
    output = report(human_ears=entry(), animal_ears=entry())
    output["part_overlap"]["human_ears:animal_ears"] = {
        "pixels": 30, "smaller_part_ratio": .30}
    audit = evaluate_output_ear_contract(source, output, (100, 200))
    assert audit["checks"]["human_animal_overlap"]["status"] == "FAIL"
    assert audit["status"] == "FAIL"


def test_output_audit_rejects_overlap_even_when_human_source_is_unknown():
    source = build_source_ear_contract(
        report(animal_ears=entry()), (100, 200))
    output = report(human_ears=entry(), animal_ears=entry())
    output["part_overlap"]["human_ears:animal_ears"] = {
        "pixels": 100, "smaller_part_ratio": 1.0}
    audit = evaluate_output_ear_contract(source, output, (100, 200))
    assert source["human"]["state"] == "unknown"
    assert audit["checks"]["human_animal_overlap"]["status"] == "FAIL"
    assert audit["checks"]["human_animal_overlap"]["reason"] == (
        "ear_masks_overlapped")
    assert audit["status"] == "FAIL"


def test_output_audit_rejects_large_normalized_location_change():
    source = build_source_ear_contract(
        report(animal_ears=entry(box=(10, 10, 30, 30))), (100, 200))
    output = report(animal_ears=entry(box=(60, 20, 90, 50)))
    audit = evaluate_output_ear_contract(source, output, (100, 200))
    assert audit["checks"][ANIMAL_EARS]["status"] == "FAIL"
    assert audit["checks"][ANIMAL_EARS]["reason"] == "location_mismatch"


def test_human_tag_is_not_recorded_as_animal_species():
    contract = build_source_ear_contract(
        report(human_ears=entry()), (100, 200),
        approved_character_tags=("human_ears",))
    assert contract["animal"]["approved_species_tags"] == []



