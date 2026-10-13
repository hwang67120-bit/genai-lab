"""사람 판정을 항목별로 기록해 추적 기록(trace.json)에 잇는다(D-114).
자동 검사 통과(머리 합성)는 품질 통과가 아니다. 품질은 사람이 항목별로 판정하고, 그 값이 실행 폴더의 judgement.json과
trace.json의 human_judgements에 남는다. 값을 추측해 채우지 않는다 — 묻지 않은 항목은 "not_judged"다.
"""
from datetime import datetime
import json
from pathlib import Path

from genai_lab.outfit_edit import write_json

JUDGES = ("user", "codex", "claude")
ITEMS = {
    "same_character": ("yes", "no", "not_judged"),                       # 얼굴·머리·귀·꼬리·성별 체형이 같은 캐릭터인가
    "outfit": ("same", "partial", "different", "accepted", "not_judged"),  # 참조와 같은가. accepted = 작업실 확인 체크(정도 미기록)
    "body_parts": ("kept", "changed", "missing", "added", "not_applicable", "not_judged"),  # 귀·꼬리·날개 유무·개수·모양
    "original_garment_remaining": ("none", "some", "not_judged"),         # 기존 의상 잔존(완성 기준: 남으면 실패)
    "pose": ("kept", "changed", "not_judged"),
    "severe_defect": ("none", "present", "not_judged"),                   # 얼굴 소실·중복, 일그러짐, 잘못된 팔다리, 잘림
}


def validate_judgement(judgement):
    data = dict(judgement)
    if data.get("judge") not in JUDGES:
        raise ValueError("판정자는 user / codex / claude 중 하나여야 합니다.")
    for key, allowed in ITEMS.items():
        value = data.get(key, "not_judged")
        if value not in allowed:
            raise ValueError(f"{key} 값은 {allowed} 중 하나여야 합니다.")
        data[key] = value
    if data.get("severe_defect") == "present" and not data.get("defect_note"):
        raise ValueError("심한 이상이 있으면 무엇인지(defect_note) 적어 주세요.")
    data.setdefault("note", "")
    data.setdefault("source", "")           # 예: gui_approval, codex_gpu_report_16, comparison-16-1.png 1-1
    data.setdefault("recorded_by", data["judge"])
    data["recorded_at"] = data.get("recorded_at") or datetime.now().isoformat(timespec="seconds")
    return data


def quality_pass(judgement, composited):
    """완성 기준(D-109) 기준의 품질 통과: 머리 합성 + 같은 캐릭터 + 옷 같음/일부/확인 + 기존 의상 잔존 없음 + 심한 이상 없음.
    판정된 핵심 항목 중 하나라도 실패면 False(나머지가 미판정이어도), 전부 통과면 True, 실패 없이 미판정이 남으면 None."""
    passes = {"same_character": ("yes",), "outfit": ("same", "partial", "accepted"), "original_garment_remaining": ("none",),
              "severe_defect": ("none",)}
    if not composited:
        return False
    judged = {k: judgement.get(k, "not_judged") for k in passes}
    if any(v != "not_judged" and v not in passes[k] for k, v in judged.items()):
        return False
    return None if any(v == "not_judged" for v in judged.values()) else True


def record_judgement(run_directory, judgement):
    """실행 폴더에 judgement.json(목록, 추가만)을 쓰고 trace.json이 있으면 human_judgements에 잇는다. 반환: 기록 경로."""
    run_directory = Path(run_directory)
    data = validate_judgement(judgement)
    path = run_directory / "judgement.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    composited = None
    trace_path = run_directory / "trace.json"
    if trace_path.exists():
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        composited = bool(trace.get("judgement", {}).get("composited"))
    else:
        outcome = run_directory / "compose" / "outcome.json"
        if outcome.exists():
            composited = bool(json.loads(outcome.read_text(encoding="utf-8")).get("composited"))
    data["quality_pass"] = quality_pass(data, composited) if composited is not None else None
    existing.append(data)
    write_json(path, existing)
    if trace_path.exists():
        trace["human_judgements"] = existing
        trace["judgement"]["quality"] = latest_quality(existing)
        write_json(trace_path, trace)
    return path


def latest_quality(judgements):
    """같은 실행에 여러 판정이 있으면 사용자 판정이 최우선, 그다음 가장 최근 것."""
    ordered = sorted(judgements, key=lambda j: (j["judge"] != "user", j["recorded_at"]), reverse=False)
    users = [j for j in ordered if j["judge"] == "user"]
    chosen = (users or ordered)[-1]
    return dict(judge=chosen["judge"], quality_pass=chosen.get("quality_pass"), recorded_at=chosen["recorded_at"])


def summarize(run_directories):
    """시험 보고용 집계: 실행별 합성 여부·최신 판정·품질 통과. 판정이 없는 실행은 not_judged."""
    rows = []
    for directory in run_directories:
        directory = Path(directory)
        trace_path, judgement_path = directory / "trace.json", directory / "judgement.json"
        composited = None
        if trace_path.exists():
            composited = bool(json.loads(trace_path.read_text(encoding="utf-8")).get("judgement", {}).get("composited"))
        judgements = json.loads(judgement_path.read_text(encoding="utf-8")) if judgement_path.exists() else []
        latest = latest_quality(judgements) if judgements else None
        rows.append(dict(run=str(directory), composited=composited, judged=bool(judgements),
                         quality_pass=latest["quality_pass"] if latest else None,
                         judge=latest["judge"] if latest else None))
    judged = [r for r in rows if r["judged"]]
    return dict(runs=rows, total=len(rows), composited=sum(1 for r in rows if r["composited"]),
                judged=len(judged), quality_pass=sum(1 for r in judged if r["quality_pass"]),
                incomplete=sum(1 for r in judged if r["quality_pass"] is None))
