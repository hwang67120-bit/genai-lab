"""사람 판정의 항목별 기록과 추적 기록 연결(D-114, GPU 없이)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from genai_lab import outfit_judgement
from genai_lab.outfit_edit import write_json


def run_at(tmp_path, composited=True):
    run = tmp_path / "run"
    run.mkdir()
    write_json(run / "trace.json", dict(schema="outfit_edit_trace/1", judgement=dict(automatic="composited", composited=composited)))
    return run


def test_judgement_values_are_validated_and_unasked_items_stay_not_judged():
    data = outfit_judgement.validate_judgement(dict(judge="user", same_character="yes", outfit="partial"))
    assert data["body_parts"] == "not_judged" and data["pose"] == "not_judged" and data["recorded_at"]
    with pytest.raises(ValueError):
        outfit_judgement.validate_judgement(dict(judge="user", outfit="good"))
    with pytest.raises(ValueError):
        outfit_judgement.validate_judgement(dict(judge="user", severe_defect="present"))     # 이상이 있으면 내용을 적어야 한다
    with pytest.raises(ValueError):
        outfit_judgement.validate_judgement(dict(judge="ai"))


def test_quality_pass_needs_every_core_item_and_fails_on_remaining_garment():
    good = dict(same_character="yes", outfit="partial", original_garment_remaining="none", severe_defect="none")
    assert outfit_judgement.quality_pass(good, composited=True) is True
    assert outfit_judgement.quality_pass(good, composited=False) is False          # 머리 합성 없는 장은 통과로 세지 않는다
    assert outfit_judgement.quality_pass(dict(good, original_garment_remaining="some"), True) is False   # 완성 기준(D-109)
    assert outfit_judgement.quality_pass(dict(good, severe_defect="not_judged"), True) is None            # 불완전 판정
    assert outfit_judgement.quality_pass(dict(original_garment_remaining="some"), True) is False           # 하나만 봐도 실패는 실패
    assert outfit_judgement.quality_pass(dict(good, outfit="accepted"), True) is True


def test_record_links_judgement_to_trace_and_user_wins_over_codex(tmp_path):
    # 요청 15 G11/S08: 사용자 "바지만 변경됐지만 원본 의상과 유사해서 통과" (Codex는 기존 의상 잔존으로 봄)
    run = run_at(tmp_path)
    outfit_judgement.record_judgement(run, dict(judge="codex", same_character="yes", outfit="partial",
                                                original_garment_remaining="some", severe_defect="none", source="report_15 1-3"))
    outfit_judgement.record_judgement(run, dict(judge="user", recorded_by="claude", same_character="yes", outfit="accepted",
                                                original_garment_remaining="none", severe_defect="none",
                                                note="바지만 변경, 원본 의상과 유사해서 통과"))
    records = json.loads((run / "judgement.json").read_text(encoding="utf-8"))
    assert [r["judge"] for r in records] == ["codex", "user"] and records[0]["quality_pass"] is False and records[1]["quality_pass"] is True
    trace = json.loads((run / "trace.json").read_text(encoding="utf-8"))
    assert len(trace["human_judgements"]) == 2 and trace["judgement"]["quality"] == dict(
        judge="user", quality_pass=True, recorded_at=records[1]["recorded_at"])


def test_summary_counts_only_judged_runs_and_keeps_incomplete_separate(tmp_path):
    a, b, c = (tmp_path / n for n in ("a", "b", "c"))
    for d in (a, b, c):
        d.mkdir()
        write_json(d / "trace.json", dict(judgement=dict(composited=d.name != "c")))
    outfit_judgement.record_judgement(a, dict(judge="user", same_character="yes", outfit="same", original_garment_remaining="none",
                                              severe_defect="none"))
    outfit_judgement.record_judgement(b, dict(judge="user", same_character="yes"))      # 불완전
    summary = outfit_judgement.summarize([a, b, c])
    assert (summary["total"], summary["composited"], summary["judged"], summary["quality_pass"], summary["incomplete"]) == (3, 2, 2, 1, 1)
    assert summary["runs"][2]["judged"] is False and summary["runs"][2]["quality_pass"] is None


def test_cli_records_and_summarizes(tmp_path):
    run = run_at(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts" / "record_judgement.py"
    out = subprocess.run([sys.executable, "-B", str(script), "record", str(run), "--judge", "user", "--same-character", "yes",
                          "--outfit", "same", "--garment-remaining", "none", "--severe-defect", "none", "--source", "sheet 1-1"],
                         capture_output=True, text=True, check=True, encoding="utf-8")
    assert json.loads(out.stdout)["quality_pass"] is True
    out = subprocess.run([sys.executable, "-B", str(script), "summary", str(run)], capture_output=True, text=True, check=True,
                         encoding="utf-8")
    assert json.loads(out.stdout)["quality_pass"] == 1


def test_run_without_trace_still_gets_a_judgement_file(tmp_path):
    run = tmp_path / "old"
    (run / "compose").mkdir(parents=True)
    write_json(run / "compose" / "outcome.json", dict(composited=True))
    outfit_judgement.record_judgement(run, dict(judge="user", same_character="yes", outfit="same", original_garment_remaining="none",
                                                severe_defect="none"))
    assert json.loads((run / "judgement.json").read_text(encoding="utf-8"))[0]["quality_pass"] is True
    summary = outfit_judgement.summarize([run])
    assert summary["composited"] == 1 and summary["runs"][0]["composited"] is True     # 집계도 outcome.json을 본다
