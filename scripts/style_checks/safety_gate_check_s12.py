"""S12 안전 차단 확인(사용자 규칙: 성인 캐릭터만): 미성년 설정·아이처럼 보이는 그림이 FLUX 경로의 준비 단계에서 거절되는지.
생성은 하지 않는다 — 준비를 통과해도 생성 단계는 실행기에서 막는다(아이처럼 보이는 그림은 차단 판정 평가 자료로만 쓴다).
사용: python safety_gate_check_s12.py <출력 폴더> 이름=그림경로 ...
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab")
from genai_lab.studio_generation import StudioRuntime
from genai_lab.style_redraw_pipeline import default_run, run_style_redraw


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    rt = StudioRuntime.from_environment()
    base = default_run(rt)
    never_generate = lambda stage, d: 1 if stage == "generate" else base(stage, d)   # 생성 금지
    rows = []
    for arg in sys.argv[2:]:
        name, source = arg.split("=", 1)
        o = run_style_redraw(source, out / name, rt, run=never_generate)
        prepared = out / name / "prepare.json"
        p = json.loads(prepared.read_text(encoding="utf-8")) if prepared.is_file() else {}
        row = dict(name=name, outcome=o["state"], code=o.get("code"), message=o.get("message", "")[:200],
                   prepare_state=p.get("state"), safety=p.get("safety"), person_count_refused=p.get("code") == "person_count")
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (out / "safety-gate.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
