"""Explicit human review; no automatic quality threshold or final selection."""
from pathlib import Path
from genai_lab.qwen_pose_edit import write_json
from genai_lab.qwen_preservation import file_sha

ITEM_STATES = ("유지", "변형", "소실·추가", "가림으로 확인 불가")
GLOBAL_FIELDS = ("자세", "손", "시점", "그림체", "구조", "노출")
GLOBAL_STATES = ("문제 없음", "문제 있음", "확인 불가")
LIMITATIONS = ("B는 보존 보장이 아닙니다. 시험은 캐릭터 2명·seed 1개, 동시 충족 0/6입니다. "
               "목선·끈·장식·그림체가 바뀔 수 있고 손 동작·로우앵글에는 한계가 있습니다. "
               "시험에서 1장 약 27분이었으며 환경에 따라 다릅니다.")


def save_user_review(directory, spec, item_states, global_states, *, adopted, reviewer):
    import json
    directory = Path(directory)
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    result = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if result.get("status") != "completed" or file_sha(directory / "raw.png") != result["raw_sha256"]:
        raise ValueError("검토 가능한 완료 원시 결과가 아닙니다.")
    if request["prompt"]["spec_sha256"] != spec.sha256:
        raise ValueError("실행 때와 다른 명세입니다.")
    ids = {x.id for x in spec.items if x.status == "confirmed"}
    if set(item_states) != ids or any(v not in ITEM_STATES for v in item_states.values()):
        raise ValueError("전달한 보존 항목을 모두 검토해 주세요.")
    if set(global_states) != set(GLOBAL_FIELDS) or any(v not in GLOBAL_STATES for v in global_states.values()):
        raise ValueError("자세·손·시점·그림체·구조·노출을 검토해 주세요.")
    if type(adopted) is not bool or not reviewer.strip():
        raise ValueError("명시적 사용자 결정이 필요합니다.")
    value = {"spec_sha256": spec.sha256, "raw_sha256": result["raw_sha256"],
             "item_states": item_states, "global_states": global_states, "adopted": adopted,
             "reviewer": reviewer, "automatic_verdict": False, "limitations": LIMITATIONS}
    write_json(directory / "user_review.json", value)
    return value
