"""Independent human-ear and animal-ear observation contracts.

Detector absence is deliberately UNKNOWN. Only an explicit approved absence
may become ABSENT; this module never invents absence from a missed detection.
"""

from __future__ import annotations

from genai_lab.reference_tag_policy import normalize_tag


HUMAN_EARS = "human_ears"
ANIMAL_EARS = "animal_ears"
EAR_PART_NAMES = (HUMAN_EARS, ANIMAL_EARS)
LEGACY_ANIMAL_EARS = "ears"
ANIMAL_SPECIES = ("cat", "dog", "fox", "wolf", "raccoon", "bear", "rabbit")


def canonical_ear_part_name(name: str) -> str:
    return ANIMAL_EARS if name == LEGACY_ANIMAL_EARS else name


def _entry(parts, name):
    value = parts.get(name)
    if value is None and name == ANIMAL_EARS:
        value = parts.get(LEGACY_ANIMAL_EARS)
    return value if isinstance(value, dict) else {}


def _normalized_boxes(entry, image_size):
    width, height = image_size
    if width < 1 or height < 1:
        raise ValueError("귀 계약 이미지 크기가 올바르지 않습니다.")
    boxes = entry.get("accepted_boxes", entry.get("boxes", ()))
    result = []
    for raw in boxes:
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            continue
        x1, y1, x2, y2 = (float(value) for value in raw)
        result.append({
            "box": [x1, y1, x2, y2],
            "center": [((x1 + x2) / 2) / width,
                       ((y1 + y2) / 2) / height],
            "size": [max(0.0, x2 - x1) / width,
                     max(0.0, y2 - y1) / height],
        })
    return result


def _ear_location_verified(name, evidence):
    """Reject detector masks that cannot be a local face-side ear region.

    This is deliberately only a coarse safety check. It never proves that the
    object is an ear; it prevents a nearly full character/body mask from being
    promoted into automatic conditioning and hair-mask subtraction.
    """
    if not evidence:
        return False
    maximum_width, maximum_height, maximum_center_y = (
        (.35, .35, .55) if name == HUMAN_EARS else (.45, .50, .45)
    )
    return all(
        float(item["size"][0]) <= maximum_width
        and float(item["size"][1]) <= maximum_height
        and float(item["center"][1]) <= maximum_center_y
        for item in evidence
    )


def _observation(name, parts, image_size, explicit_state=None):
    entry = _entry(parts, name)
    count = int(entry.get("accepted_masks", 0) or 0)
    detected = entry.get("status") == "detected" and count > 0
    location_evidence = _normalized_boxes(entry, image_size)
    location_verified = _ear_location_verified(name, location_evidence)
    maximum_width, maximum_height, maximum_center_y = (
        (.35, .35, .55) if name == HUMAN_EARS else (.45, .50, .45)
    )
    allowed = (None, "present", "absent", "occluded", "unknown")
    if explicit_state not in allowed:
        raise ValueError(f"{name} 귀 상태가 올바르지 않습니다.")
    location_rejected = (
        detected
        and explicit_state is None
        and not location_verified
    )
    state = explicit_state or (
        "unknown" if location_rejected else
        "present" if detected else "unknown"
    )
    if state == "absent" and detected:
        raise ValueError(f"{name} 검출 결과와 명시적 부재 승인이 충돌합니다.")
    return {
        "state": state,
        "visible_count": count if detected and not location_rejected else 0,
        "location": "face_side" if name == HUMAN_EARS else "hair_crown",
        "location_evidence": location_evidence,
        "location_verified": location_verified,
        "confidence": max(
            (float(value) for value in entry.get("detection_scores", ())),
            default=None,
        ),
        "mask_path": entry.get("mask_path"),
        "crop_path": entry.get("crop_path"),
        "mask_pixels": entry.get("mask_pixels"),
        "evidence": (
            "location_unverified" if location_rejected else
            "detector_and_segmentation" if detected else "not_observed"
        ),
        "ambiguity": ({
            "reason": f"{name}_mask_outside_local_location_bounds",
            "maximum_normalized_width": maximum_width,
            "maximum_normalized_height": maximum_height,
            "maximum_normalized_center_y": maximum_center_y,
            "raw_detection_preserved": True,
        } if location_rejected else None),
        "absence_inferred_from_non_detection": False,
    }


def build_source_ear_contract(detection_report, image_size, *,
                              approved_character_tags=(), explicit_states=None,
                              ambiguous_overlap_ratio=.80):
    """Build the source contract while keeping both ear classes independent."""
    report = detection_report if isinstance(detection_report, dict) else {}
    parts = report.get("parts", {}) if isinstance(report.get("parts", {}), dict) else {}
    states = explicit_states or {}
    tags = {normalize_tag(tag) for tag in approved_character_tags}
    animal_tags = sorted(
        tag for tag in tags
        if tag == "animal ears"
        or any(tag == f"{species} ears" for species in ANIMAL_SPECIES)
    )
    observations = {
        HUMAN_EARS: _observation(
            HUMAN_EARS, parts, image_size, states.get(HUMAN_EARS)),
        ANIMAL_EARS: _observation(
            ANIMAL_EARS, parts, image_size, states.get(ANIMAL_EARS)),
    }
    overlap = report.get("part_overlap", {}).get(
        f"{HUMAN_EARS}:{ANIMAL_EARS}", {})
    overlap_ratio = (
        overlap.get("smaller_part_ratio")
        if isinstance(overlap, dict) else None
    )
    # Grounding DINO may return the same crown objects for both natural-language
    # queries. That is not evidence that both anatomical ear types are present.
    # Keep the raw observation, but do not promote it to generation unless an
    # explicit state resolved the ambiguity.
    if (states.get(HUMAN_EARS) is None
            and observations[HUMAN_EARS]["state"] == "present"
            and observations[ANIMAL_EARS]["state"] == "present"
            and overlap_ratio is not None
            and float(overlap_ratio) >= ambiguous_overlap_ratio):
        observations[HUMAN_EARS].update(
            state="unknown",
            evidence="ambiguous_with_animal_ears",
            ambiguity={
                "reason": "cross_class_mask_overlap",
                "overlap_ratio": float(overlap_ratio),
                "threshold": float(ambiguous_overlap_ratio),
                "raw_detection_preserved": True,
            },
        )
    return {
        "version": "independent_ear_contract_v1",
        "human": observations[HUMAN_EARS],
        "animal": {
            **observations[ANIMAL_EARS],
            "approved_species_tags": animal_tags,
        },
        "source_masks_are_independent": True,
        "non_detection_policy": "unknown_not_absent",
        "generation_parts": [name for name, value in observations.items()
                             if value["state"] == "present"],
    }


def _mean_center(observation):
    evidence = observation.get("location_evidence", ())
    if not evidence:
        return None
    return tuple(
        sum(float(item["center"][axis]) for item in evidence) / len(evidence)
        for axis in (0, 1)
    )


def evaluate_output_ear_contract(source_contract, detection_report, image_size,
                                 *, maximum_center_delta=.20,
                                 maximum_overlap_ratio=.15):
    """Compare output observations with the approved source ear contract."""
    if not isinstance(source_contract, dict):
        return {
            "version": "generated_ear_contract_audit_v1",
            "status": "UNRESOLVED",
            "reason": "approved_source_ear_contract_missing",
            "checks": {},
        }
    output = build_source_ear_contract(detection_report, image_size)
    checks = {}
    for label, part_name in (("human", HUMAN_EARS), ("animal", ANIMAL_EARS)):
        expected = source_contract.get(label, {})
        actual = output[label]
        expected_state = expected.get("state", "unknown")
        actual_state = actual.get("state", "unknown")
        state, reason = "UNRESOLVED", "source_or_output_state_unknown"
        if expected_state == "present":
            if actual_state == "present":
                expected_count = int(expected.get("visible_count", 0) or 0)
                actual_count = int(actual.get("visible_count", 0) or 0)
                count_ok = not expected_count or expected_count == actual_count
                expected_center = _mean_center(expected)
                actual_center = _mean_center(actual)
                center_delta = (
                    None if expected_center is None or actual_center is None
                    else sum((left - right) ** 2 for left, right in zip(
                        expected_center, actual_center)) ** .5
                )
                location_ok = (
                    center_delta is None
                    or center_delta <= maximum_center_delta
                )
                state = "PASS" if count_ok and location_ok else "FAIL"
                reason = (
                    "visible_count_mismatch" if not count_ok
                    else "location_mismatch" if not location_ok
                    else "matched"
                )
            else:
                state, reason = "FAIL", "approved_ear_type_not_detected"
        elif expected_state == "absent":
            state = "FAIL" if actual_state == "present" else "PASS"
            reason = "unexpected_ear_type" if state == "FAIL" else "approved_absence_preserved"
        elif expected_state == "occluded":
            state, reason = "UNRESOLVED", "source_ear_occluded"
        checks[part_name] = {
            "status": state,
            "reason": reason,
            "expected": expected,
            "actual": actual,
            "center_delta": (
                center_delta if expected_state == "present"
                and actual_state == "present" else None),
            "maximum_center_delta": maximum_center_delta,
        }

    both_expected = all(
        source_contract.get(label, {}).get("state") == "present"
        for label in ("human", "animal")
    )
    overlap = (detection_report or {}).get("part_overlap", {}).get(
        f"{HUMAN_EARS}:{ANIMAL_EARS}")
    overlap_ratio = None if not isinstance(overlap, dict) else overlap.get("smaller_part_ratio")
    overlap_failed = overlap_ratio is not None and overlap_ratio > maximum_overlap_ratio
    checks["human_animal_overlap"] = {
        "status": (
            "FAIL" if overlap_failed else
            "PASS" if both_expected and overlap_ratio is not None else
            "UNRESOLVED" if both_expected else "PASS"),
        "reason": "ear_masks_overlapped" if overlap_failed else (
            "not_applicable" if not both_expected else
            "within_limit" if overlap_ratio is not None
            else "both_masks_not_available"),
        "overlap_ratio": overlap_ratio,
        "maximum_overlap_ratio": maximum_overlap_ratio,
    }
    states = {value["status"] for value in checks.values()}
    overall = "FAIL" if "FAIL" in states else (
        "UNRESOLVED" if "UNRESOLVED" in states else "PASS")
    return {
        "version": "generated_ear_contract_audit_v1",
        "status": overall,
        "checks": checks,
        "output_observation": output,
        "position_policy": "normalized_boxes_recorded_location_not_claimed_without_landmarks",
    }


