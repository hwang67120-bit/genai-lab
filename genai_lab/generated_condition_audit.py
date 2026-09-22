"""Tri-state post audit against the approved condition contract.

Only checks backed by an output analyzer may PASS or FAIL. Missing analyzers are
UNRESOLVED rather than guessed from the generated image.
"""


def _part_result(part_name, approved_colors, correction):
    aliases = {
        "human_ears": {"human_ears"},
        "animal_ears": {"animal_ears", "ears"},
        "tail": {"tail"},
    }
    approved_names = aliases.get(part_name, {part_name})
    if not any(item.get("part_name") in approved_names
               for item in approved_colors):
        return {"status": "UNRESOLVED", "reason": "part_color_not_approved"}
    reports = (correction or {}).get("parts", {})
    report = reports.get(part_name)
    if report is None and part_name == "animal_ears":
        report = reports.get("ears")
    if not report:
        if (correction or {}).get("status") == "failed_keep_original":
            return {
                "status": "FAIL",
                "reason": "part_correction_execution_failed",
            }
        return {"status": "UNRESOLVED", "reason": "output_part_audit_missing"}
    status = report.get("status")
    if status in ("accepted_without_correction", "corrected_and_accepted"):
        return {
            "status": "PASS",
            "reason": status,
            "color_distance": report.get(
                "after_color_distance", report.get("before_color_distance")),
        }
    if status in ("rejected_keep_original", "failed_keep_original"):
        return {"status": "FAIL", "reason": status}
    return {
        "status": "UNRESOLVED",
        "reason": report.get("reason", status or "unknown"),
    }


def _hair_result(correction):
    status = (correction or {}).get("status")
    if status in ("accepted_without_correction", "corrected_and_accepted"):
        return {
            "status": "PASS",
            "reason": status,
            "similarity": correction.get(
                "after_similarity", correction.get("before_similarity")),
        }
    if status in ("rejected_keep_original", "failed_keep_original"):
        return {"status": "FAIL", "reason": status}
    return {
        "status": "UNRESOLVED",
        "reason": (correction or {}).get("reason", status or "audit_missing"),
    }


def build_generated_condition_audit(approved_run, design_record):
    approved_colors = list(
        approved_run.get("approved_part_color_descriptions", ()))
    correction = design_record.get("part_error_correction", {})
    ear_audit = correction.get("ear_output_contract", {})
    ear_checks = ear_audit.get("checks", {})
    correction_failed = correction.get("status") == "failed_keep_original"
    missing_human = {
        "status": "FAIL" if correction_failed else "UNRESOLVED",
        "reason": ("part_correction_execution_failed" if correction_failed
                   else "human_ear_output_audit_missing"),
    }
    missing_animal = {
        "status": "FAIL" if correction_failed else "UNRESOLVED",
        "reason": ("part_correction_execution_failed" if correction_failed
                   else "animal_ear_output_audit_missing"),
    }
    checks = {
        "gender": {
            "status": "UNRESOLVED",
            "expected": approved_run.get("gender"),
            "reason": "generated_gender_analyzer_not_connected",
        },
        "hair": _hair_result(design_record.get("hair_error_correction")),
        "human_ears": ear_checks.get("human_ears", missing_human),
        "animal_ears": ear_checks.get("animal_ears", missing_animal),
        "tail": _part_result("tail", approved_colors, correction),
        "garment": {
            "status": "UNRESOLVED",
            "expected": list(approved_run.get("approved_tags", ())),
            "reason": "generated_garment_structure_analyzer_not_connected",
        },
        "human_and_animal_ear_coexistence": ear_checks.get(
            "human_animal_overlap", {
                "status": "UNRESOLVED",
                "reason": "ear_overlap_output_audit_missing",
            }),
    }
    approved_color_parts = {
        item.get("part_name") for item in approved_colors
        if isinstance(item, dict)
    }
    color_checks = {}
    for part_name in ("human_ears", "animal_ears", "tail"):
        aliases = {part_name}
        if part_name == "animal_ears":
            aliases.add("ears")
        if approved_color_parts.intersection(aliases):
            color_checks[f"{part_name}_color"] = _part_result(
                part_name, approved_colors, correction)
    checks.update(color_checks)
    states = {entry["status"] for entry in checks.values()}
    overall = "FAIL" if "FAIL" in states else (
        "UNRESOLVED" if "UNRESOLVED" in states else "PASS")
    blocking_color_checks = {
        name: entry.get("reason")
        for name, entry in color_checks.items()
        if entry.get("status") in {"FAIL", "UNRESOLVED"}
    }
    return {
        "version": "generated_condition_post_audit_v2",
        "status": overall,
        "checks": checks,
        "blocking_color_checks": blocking_color_checks,
        "approved_color_contract_requires_output_audit": True,
        "input_contract_pass_does_not_imply_visual_match": True,
    }
