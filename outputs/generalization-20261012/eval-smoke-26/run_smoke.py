"""평가 후보 5명 한 번씩 통과 시험(2026-10-16 사용자 "전신 아닌 거 제외하고 한번 통과시켜줘").

목적: 새 후보가 시작 검사(안전·입력)를 통과하고 정상 이미지가 생성되는지 근거를 얻는다. 품질 평가 세트가 아니다.
조건: 원본 기준, 특징표는 자동 태그(0.5 이상, 사람 확정 아님 — 기록에 automatic_features로 남김), 빈 요청(고정 자세, selection_seed 42),
생성 seed 42(제품 재시도만 43). 거부·실패는 기록하고 다음 후보로 넘어간다. UNC 경로로 실행.
"""
import json, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
P = Path(__file__).resolve().parent
from PIL import Image  # noqa: E402
from genai_lab.character_features import tag_features  # noqa: E402
from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, WdTagSession  # noqa: E402
from genai_lab.free_generation import FreeGenerationRejected  # noqa: E402
import genai_lab.free_generation_pipeline as pipe  # noqa: E402
from genai_lab.qwen_pose_settings import QwenPoseSettings  # noqa: E402
from genai_lab.studio_generation import StudioRuntime  # noqa: E402

CASES = (("M1-horned", "male", ()), ("M2-dreads", "male", ()), ("M3-braid", "male", ()),
         ("M5-dogboy", "male", ("ears", "tail")), ("F5-cheer", "female", ()))


# 이어서 돌리기(2026-10-16, 첫 실행이 M3 생성 중 06:12에 끊김): 인자로 후보를 고르고 --suffix로 새 폴더·상태 파일을 쓴다.
# 예: run_smoke.py M3-braid M5-dogboy F5-cheer --suffix=-resume  (중단된 runs/M3-braid는 기록으로 남김)
ONLY = [a for a in sys.argv[1:] if not a.startswith("--")]
SUFFIX = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--suffix=")), "")
STATUS = f"status{SUFFIX}.json"
if ONLY:
    CASES = tuple(c for c in CASES if c[0] in ONLY)


def write(name, data):
    (P / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


runtime = StudioRuntime.from_environment()
settings = QwenPoseSettings.load(ROOT / "configs/qwen_edit_settings.json")
features = {}
with WdTagSession(ClothingDesignAnalysisSettings(cache_dir=str(runtime.model_cache), local_files_only=True)) as tagger:
    for case, _, _ in CASES:
        with Image.open(P / "sources" / f"{case}.jpg") as image:
            scores = dict(tagger.analyze(image.convert("RGB")).raw_general_scores)
        features[case] = {k: v["values"] for k, v in tag_features(scores).items()}
write(f"automatic-features{SUFFIX}.json", dict(source="automatic_analysis (tag 0.5 이상, 사람 확정 아님)", features=features))
status = dict(status="running", total=len(CASES), done=0, active=None, results=[])
write(STATUS, status)
start = time.time()
for case, gender, parts in CASES:
    status.update(active=case, stage="starting")
    write(STATUS, status)
    def progress(message, _case=case):
        status.update(stage=message, elapsed_seconds=round(time.time() - start, 1))
        write(STATUS, status)
        print(_case, message, flush=True)
    try:
        result = pipe.run_free_generation(P / "sources" / f"{case}.jpg", features[case], "", base_kind="original", gender=gender,
                                          body_parts=parts, confirmed=True, adult_confirmed=True, settings=settings, runtime=runtime,
                                          directory=P / "runs" / (case + SUFFIX), selection_seed=42, progress=progress)
        row = dict(case=case, status=result["status"], attempts=len(result["attempts"]), result=result["result"],
                   retry=[a.get("retry_reason") for a in result["attempts"]])
    except FreeGenerationRejected as error:
        row = dict(case=case, status="rejected", rejection=error.record())
    except Exception as error:
        row = dict(case=case, status="failed", error=str(error)[:500], traceback=traceback.format_exc()[-2000:])
    status["results"].append(row)
    status.update(done=len(status["results"]))
    write(STATUS, status)
status.update(status="completed", active=None, stage="done", elapsed_seconds=round(time.time() - start, 1))
write(STATUS, status)
print("SMOKE COMPLETE", flush=True)
