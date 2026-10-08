"""Desktop proportion preparation: immutable CPU inputs and explicit head approval."""
from dataclasses import asdict, replace
from pathlib import Path
import json
from genai_lab.onepass_pose import InputDecision
from genai_lab.proportion_inputs import (HeadOutline, ProportionOptions, FOREGROUND_SHA,
    json_value, sha, require, validate_head)
from genai_lab.qwen_record_io import write_json


def verify_pose(pose):
    for field in ("source", "normalized", "control"):
        require(sha(pose[field + "_file"]) == pose[field + "_sha256"], "비율 입력이 변경됐습니다: " + field)
    decision = InputDecision(**{k: tuple(v) if isinstance(v, list) else v for k,v in pose["decision"].items()})
    require("proceed_with_pose" in decision.choices and decision.status != "reject",
            "원본 골격을 사용할 수 없습니다. 비율 참고를 끄거나 다른 이미지를 선택해 주세요.")
    require(pose["ip_early"] == .5, "비율 GUI는 시험에서 검증한 초기 얼굴 참조 강도 0.5를 사용합니다.")
    return decision


def run_cpu_preparation(mode, request, runtime, directory, *, cancelled):
    from genai_lab.studio_generation import cpu_process
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "request.json", json_value(request))
    interpreter = runtime.pose_python if mode == "pose" else runtime.analysis_python
    require(Path(interpreter).is_file(), "CPU 분석 실행 환경이 없습니다.")
    worker = Path(__file__).with_name("studio_proportion_worker.py")
    cpu_process([str(interpreter), str(worker), mode, str(directory / "request.json")],
                directory / "cpu.log", cancelled, runtime.analysis_timeout)
    return json.loads((directory / "result.json").read_text(encoding="utf-8"))


def prepare_studio_pose(analysis, runtime, directory, *, cancelled=lambda: False):
    source = Path(analysis["directory"]) / "character.png"
    result = run_cpu_preparation("pose", {"source_file": source, "source_sha256": sha(source),
        "tag_scores": analysis["character_scores"], "pose_models": runtime.pose_models},
        runtime, directory, cancelled=cancelled)
    verify_pose(result)
    return result


def prepare_studio_head(pose, selection, runtime, directory, *, cancelled=lambda: False):
    verify_pose(pose)
    return run_cpu_preparation("head", {"pose": pose, "box": selection["box"],
        "face_point": selection["face_point"], "model_cache": runtime.model_cache},
        runtime, directory, cancelled=cancelled)


def restore_head(record):
    fields = dict(record["head"])
    for key in ("normalized_file", "mask_file", "contour_file"):
        fields[key] = Path(fields[key])
    for key in ("nose", "neck"):
        fields[key] = tuple(fields[key])
    return HeadOutline(**fields)


def confirmed_options(pose, draft, runtime):
    """Called only after the head review dialog accepts; it cannot confirm another pose."""
    from types import SimpleNamespace
    from genai_lab.studio_background import MODEL_RELATIVE
    verify_pose(pose)
    head = restore_head(draft)
    require(head.confirmed is False, "머리 초안의 확인 상태가 잘못됐습니다.")
    require(head.normalized_sha256 == pose["normalized_sha256"], "다른 정규화 이미지의 머리 윤곽입니다.")
    require(list(head.nose) == pose["nose"] and list(head.neck) == pose["neck"], "코·목 좌표가 변경됐습니다.")
    head = replace(head, confirmed=True)
    validate_head(head, SimpleNamespace(control_sha256=pose["control_sha256"]), (736,1232))
    options = ProportionOptions(True, head, runtime.proportion_sketch_root,
        runtime.model_cache / MODEL_RELATIVE, FOREGROUND_SHA)
    write_json(Path(head.contour_file).parent / "user-confirmation.json", json_value({
        "head": asdict(head), "pose": pose, "box": draft["box"], "face_point": draft["face_point"],
        "reviewer": "user", "automatic_approval": False}))
    return options
