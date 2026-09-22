import numpy as np
import pytest
from PIL import Image
from genai_lab.regional_reference import DetectedPart
from genai_lab.source_region_experiment import filter_part_instances, assign_source_regions


def mask(box):
    image = Image.new("L", (32, 48), 0)
    image.paste(255, box)
    return image


def test_sleeve_candidates_rejected_before_union():
    clothes, hair = mask((8, 16, 24, 40)), mask((8, 0, 24, 8))
    tail, sleeve = mask((1, 24, 7, 40)), mask((8, 16, 12, 30))
    candidates = np.stack([np.asarray(tail) == 255, np.asarray(sleeve) == 255])
    accepted, report = filter_part_instances("tail", candidates, {"garment": clothes, "hair": hair})
    assert len(accepted) == 1
    assert np.array_equal(accepted[0], np.asarray(tail) == 255)
    assert [r["accepted"] for r in report] == [True, False]
    for image in (clothes, hair, tail, sleeve):
        image.close()


def test_ear_full_hair_candidate_rejected():
    clothes, hair = mask((8, 16, 24, 40)), mask((8, 4, 24, 10))
    ear = mask((8, 0, 12, 4))
    candidates = np.stack([np.asarray(ear) == 255, np.asarray(hair) == 255])
    accepted, _ = filter_part_instances("animal_ears", candidates, {"garment": clothes, "hair": hair}, hair_context=True)
    assert len(accepted) == 1
    for image in (clothes, hair, ear):
        image.close()


def test_target_priority_does_not_mutate_original_masks():
    identity, clothes = mask((8, 0, 24, 12)), mask((8, 12, 24, 44))
    parts = [DetectedPart("animal_ears", mask((8, 0, 12, 4)), None, .35),
             DetectedPart("tail", mask((1, 24, 7, 40)), None, .35)]
    before = identity.tobytes()
    new_identity, new_clothes, record = assign_source_regions(parts, identity, clothes)
    assert identity.tobytes() == before
    assert record["identity_overlap_removed"] == 16
    assert record["pose_locked"] is False
    assert parts[0].target_region is not parts[0].source_mask
    assert parts[0].target_region.tobytes() == parts[0].source_mask.tobytes()
    assert not np.any((np.asarray(new_identity) > 0) & (np.asarray(parts[0].target_region) > 0))
    for image in (identity, clothes, new_identity, new_clothes):
        image.close()
    for part in parts:
        part.close()


def test_no_parts_is_no_op_but_colliding_parts_still_fail():
    identity, clothes = mask((8, 0, 24, 12)), mask((8, 12, 24, 44))
    new_identity, new_clothes, record = assign_source_regions(
        [],
        identity,
        clothes,
    )
    try:
        assert new_identity.tobytes() == identity.tobytes()
        assert new_clothes.tobytes() == clothes.tobytes()
        assert record["status"] == "no_optional_parts"
        assert record["conditioned_parts"] == []
    finally:
        new_identity.close()
        new_clothes.close()
    parts = [DetectedPart("animal_ears", mask((1, 0, 4, 4)), None, .35),
             DetectedPart("tail", mask((1, 0, 4, 4)), None, .35)]
    with pytest.raises(ValueError, match="서로 겹칩니다"):
        assign_source_regions(parts, identity, clothes)
    assert all(part.target_region is None for part in parts)
    identity.close()
    clothes.close()
    for part in parts:
        part.close()


@pytest.mark.parametrize("ratio,keep", [(0.0, True), (.304122, True), (.899, True),
                                       (.90, False), (.998718, False), (.999486, False)])
def test_partial_tail_overlap_preserves_mask_near_complete_clothing_rejected(ratio, keep):
    # 로그의 중복률을 재현한 합성 마스크. 실제 이미지 검출 품질 시험은 아니다.
    candidate = np.zeros((120, 120), dtype=bool)
    candidate.flat[:10000] = True
    clothing = np.zeros_like(candidate, dtype=np.uint8)
    clothing.flat[:round(10000 * ratio)] = 255
    with Image.fromarray(clothing) as garment, Image.new("L", (120, 120)) as hair:
        before = garment.tobytes()
        accepted, report = filter_part_instances("tail", candidate[None], {"garment": garment, "hair": hair})
        assert len(accepted) == int(keep)
        assert report[0]["accepted"] is keep
        assert garment.tobytes() == before
        if keep:
            assert np.array_equal(accepted[0], candidate)
            assert report[0]["removed_pixels"] == 0
        else:
            assert report[0]["reason"] == "garment_dominated"


@pytest.mark.parametrize("overlap", [.438391, .626050, .780421, .801710, 1.0])
def test_local_ear_hair_overlap_does_not_erase_ear(overlap):
    selected = np.zeros((200, 200), dtype=bool)
    selected.flat[:1000] = True
    hair_values = np.zeros_like(selected, dtype=np.uint8)
    count = round(1000 * overlap)
    hair_values.flat[:count] = 255
    hair_values.flat[1000:1000+10000-count] = 255
    with Image.fromarray(hair_values) as hair, Image.new("L", (200, 200)) as garment:
        accepted, report = filter_part_instances("animal_ears", selected[None], {"hair": hair, "garment": garment}, hair_context=True)
        assert np.array_equal(accepted[0], selected)
        assert report[0]["hair_coverage"] <= .1
        assert report[0]["requires_review"]


def test_broad_head_and_empty_masks_are_rejected_with_distinct_reasons():
    selected = np.zeros((200, 200), dtype=bool)
    selected.flat[:20000] = True
    hair_values = np.zeros_like(selected, dtype=np.uint8)
    hair_values.flat[:15600] = 255
    with Image.fromarray(hair_values) as hair, Image.new("L", (200, 200)) as garment:
        accepted, report = filter_part_instances("animal_ears", np.stack([selected, selected & False]),
                                                {"hair": hair, "garment": garment}, hair_context=True)
        assert len(accepted) == 0
        assert [r["reason"] for r in report] == ["broad_hair_candidate", "empty_mask"]


@pytest.mark.parametrize("setting,value", [
    ("garment_dominance", 0), ("garment_dominance", float("nan")),
    ("broad_hair_coverage", 1.1), ("broad_hair_coverage", -1)])
def test_filter_rejects_invalid_thresholds(setting, value):
    with mask((1, 1, 4, 4)) as garment, mask((5, 1, 8, 4)) as hair:
        with pytest.raises(ValueError):
            filter_part_instances("tail", np.zeros((0, 48, 32)),
                                  {"garment": garment, "hair": hair}, **{setting: value})
