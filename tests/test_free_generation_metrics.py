import json


def judged(case, attempt, variant, ok, types=None, note=""):
    return dict(case=case, target=dict(attempt=attempt, variant=variant, sha256="x"), quality_pass=ok, failure_types=types or [], note=note)


def write_run(root, case, items, auto=None):
    run = root / case
    run.mkdir(parents=True)
    candidates = [dict(attempt=1, variant="raw")] + ([dict(attempt=1, variant="hair_auto")] if auto else [])
    (run / "outcome.json").write_text(json.dumps(dict(candidates=candidates, auto_selection=auto)), encoding="utf-8")
    (run / "trace.json").write_text(json.dumps(dict(elapsed_seconds=1800, attempts=[dict(number=1, observation=dict(verification=dict(items=items)))])), encoding="utf-8")


def test_categories_follow_the_fixed_taxonomy():
    from genai_lab.free_generation_metrics import categories
    assert categories(judged("A", 1, "raw", True)) == []
    assert categories(judged("A", 1, "raw", False, ["hair"], "머리색 진한 청록")) == ["hair_color"]
    assert categories(judged("A", 1, "raw", False, ["outfit", "skin"])) == ["outfit_part", "skin"]
    assert categories(judged("A", 1, "raw", False, [])) == ["unstated"]


def test_summary_counts_passes_repeats_detection_and_auto_selection(tmp_path):
    from genai_lab.free_generation_metrics import summarize
    write_run(tmp_path, "M1-S1", dict(skin=dict(status="changed")))
    write_run(tmp_path, "M2-S1", dict(skin=dict(status="kept")))
    write_run(tmp_path, "M3-S1", dict(hair_color=dict(status="kept", review=["x"])), auto=dict(attempt=1, variant="hair_auto"))
    ledger = [judged("M1-S1", 1, "raw", False, ["skin"]), judged("M2-S1", 1, "raw", False, ["skin"]),
              judged("M3-S1", 1, "raw", False, ["hair"], "머리색"), judged("M3-S1", 1, "hair_auto", True)]
    out = summarize(tmp_path, ledger)
    assert out["rows"] == 3 and out["first_pass"] == 0 and out["selection_pass"] == 1 and out["passed_by"]["hair_auto"] == 1
    assert out["failure_categories"]["skin"] == dict(rows=2, characters=["M1", "M2"], repeated=True)
    assert out["detection"]["skin"]["caught"] == 1 and out["detection"]["skin"]["missed"] == 1
    assert out["detection"]["hair_color"]["caught_review_only"] == 1
    assert out["auto_selected"] == 1 and out["auto_accepted"] == 1 and out["auto_worsened"] == 0
    assert out["attempt_minutes"] == 30.0
