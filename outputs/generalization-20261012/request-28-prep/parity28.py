"""요청 28 선행 확인(CPU): 기본값 지시문이 요청 27 실제 지시문과 바이트 같음, 시험 선택지를 켜면 차이가 표의 문장뿐."""
import difflib, json, sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, "Z:/genai-lab")
from genai_lab.free_generation import features_sentence, request_conflicts, staging_text
from genai_lab.qwen_free_generation import FreeGenerationSpec
R = Path("Z:/genai-lab/outputs/generalization-20261012/free-r27")
def spec_of(case, attempt=1):
    q = R / case / ("qwen" if attempt == 1 else f"qwen-{attempt}")
    req = json.loads((q / "request.json").read_text(encoding="utf-8"))
    rec = dict(req["free_spec"]); parts = rec.get("body_parts")
    rec.update(body_parts=None if parts is None else tuple(parts), skeleton_joints=tuple(tuple(r) for r in rec.get("skeleton_joints") or ()))
    return FreeGenerationSpec(**rec), req, (q / "model_inputs" / "prompt.txt").read_text(encoding="utf-8")
bad = 0
for case_dir in sorted(R.glob("*-S*")):
    spec, req, saved = spec_of(case_dir.name)
    if spec.prompt(req["seed"]) != saved or req["prompt"]["negative"] != spec.negative():
        bad += 1; print("DEFAULT MISMATCH", case_dir.name)
print("default parity:", 21 - bad, "/ 21")
def diff(a, b):
    return [d for d in difflib.ndiff(a.split(". "), b.split(". ")) if d[:1] in "+-"]
for case in ("M6-S2", "M6-S3", "M3-S2", "M3-S3"):
    spec, req, saved = spec_of(case)
    f = json.loads((R / case / "character-features" / "features.json").read_text(encoding="utf-8"))
    hidden = request_conflicts(staging_text(spec.request_text, spec.selection_seed))
    assert features_sentence(f, hidden=hidden) == spec.features_text
    new = replace(spec, features_text=features_sentence(f, hidden=hidden, hair_color_words=False))
    print("A", case, diff(saved, new.prompt(req["seed"])), "| negative", repr(new.negative()))
for case, attempt in (("M2-S4", 2), ("K-S4", 2), ("M2-S4", 1), ("K-S4", 1)):
    spec, req, saved = spec_of(case, attempt)
    new = replace(spec, absent_parts_negative=True)
    print("B", case, req["seed"], diff(saved, new.prompt(req["seed"])), "| negative", repr(new.negative()))
