"""손 과다 결과의 재사용/저장 차단. 모델을 실행하지 않는 CPU 회귀 검사."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from PIL import Image
from genai_lab.style_redraw_anatomy import assess_hands, candidate_anatomy, gate_correction
from genai_lab.style_redraw_results import StyleRedrawResults
from genai_lab.style_redraw_pipeline import run_style_redraw, run_style_part_combine


@pytest.mark.parametrize("count,eligible", [(1, True), (2, True), (3, False), (5, False),
                                            (None, False), (0, False), (-1, False), (True, False)])
def test_count_never_certifies_complete_anatomy(count, eligible):
    a = assess_hands(count)
    assert a["eligible_for_review"] is eligible
    assert not a["user_quality_approved"] and "arm_attachment" in a["unassessed"]


def test_source_occlusion_does_not_make_two_result_hands_excess():
    assert candidate_anatomy(dict(hands=2, original_hands=1, extra_hands=True))["eligible_for_review"]
    assert not candidate_anatomy(dict(extra_hands=True))["eligible_for_review"]
    assert not candidate_anatomy({})["eligible_for_review"]


@pytest.mark.parametrize("count", [None, 0, 3, 5])
def test_better_clothes_cannot_promote_unchecked_or_excess_hands(count):
    decision = gate_correction(dict(use_correction=True, reason="improved"), assess_hands(count))
    assert not decision["use_correction"]


def make_run(tmp_path, counts):
    paths, records = [], []
    for seed, count in zip((42, 43, 44), counts):
        path = tmp_path / f"result-seed{seed}.png"
        Image.new("RGB", (8, 8), (seed, 0, 0)).save(path)
        paths.append(str(path))
        records.append(dict(seed=seed, file=path.name, hands=count))
    (tmp_path / "generation.json").write_text(json.dumps(dict(candidates=records)))
    outcome = dict(state="review_pending", candidates=paths)
    (tmp_path / "outcome.json").write_text(json.dumps(outcome))
    return outcome


def test_case4_recorded_counts_filter_without_claiming_seed42_pass(tmp_path):
    outcome = make_run(tmp_path, [2, 3, 5])
    results = StyleRedrawResults(outcome)
    assert [c["seed"] for c in results.candidates] == [42]
    assert results.status == "awaiting_user_review"
    assert all(Path(p).exists() for p in outcome["candidates"])
    with pytest.raises(ValueError):
        results.add_candidate(outcome["candidates"][1], "combined", hands=3)
    # Defend save/export even if an external caller mutates the selected record.
    results.candidate["hands"] = 3
    with pytest.raises(ValueError):
        results.approve_saved_by_user()
    results.status = "user_approved"
    with pytest.raises(ValueError):
        results.export(tmp_path / "saved.png")
    assert not (tmp_path / "saved.png").exists()


def test_no_eligible_candidate_is_not_first_candidate_fallback(tmp_path):
    with pytest.raises(ValueError, match="선택 가능한 후보"):
        StyleRedrawResults(make_run(tmp_path, [3, None, 5]))
    assert not (tmp_path / "user-review.json").exists()


@pytest.mark.parametrize("counts,expected", [([3, None, 5], "refused"), ([2, 3, 5], "review_pending")])
def test_pipeline_keeps_excluded_files_and_reports_reason(tmp_path, counts, expected):
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8)).save(source)
    def stage(name, directory):
        if name == "generate":
            make_run(directory, counts)
        else:
            (directory / f"{name}.json").write_text(json.dumps(dict(state="ready")))
        return 0
    runtime = SimpleNamespace(model_cache=Path("unused"), head_semantic_repo=Path("unused"),
                              head_semantic_checkpoint=Path("unused"))
    out = run_style_redraw(source, tmp_path / "run", runtime, run=stage, adult_confirmed=True)
    assert out["state"] == expected
    assert len(out["excluded_candidates"]) == (3 if expected == "refused" else 2)
    assert all(Path(c["path"]).exists() for c in out["excluded_candidates"])
    if expected == "refused":
        assert out["code"] == "anatomy_review_required" and not out.get("candidates")


def test_combine_rejects_bad_input_before_worker(tmp_path):
    outcome = make_run(tmp_path, [2, 3, 5])
    def never_run(*args):
        pytest.fail("invalid candidate reached combine worker")
    with pytest.raises(ValueError):
        run_style_part_combine(tmp_path, *outcome["candidates"][:2], run=never_run)


def test_combine_output_is_rechecked_and_preserved(tmp_path):
    outcome = make_run(tmp_path, [2, 2, 2])
    def combine(stage, directory):
        Image.new("RGB", (8, 8)).save(directory / "combined.png")
        (directory / "combine-result.json").write_text(json.dumps(dict(file="combined.png", hands=3)))
        return 0
    result = run_style_part_combine(tmp_path, *outcome["candidates"][:2], run=combine)
    assert result["state"] == "refused"
    assert (tmp_path / "combined.png").exists()
