"""Request 25: three rows x (skeleton on, skeleton off), same seed — D-131 pose guidance. No operational code changes.

Without --run: CPU preflight only (expected prompts, D-128 conflicts, locked snapshot). With --run: generate both rows in order.
Run through the UNC path (like request 22) so ROOT and input paths share one form.
"""
import sys, json, hashlib, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
P = Path(__file__).resolve().parent
O = P.parent
DEST = O / "free-r25"
REQUESTS = (
    ("N1", "sitting on a simple stool with the hands on the knees", "sitting_stool"),
    ("N2", "kneeling on one knee with one hand resting on the raised knee", "kneeling_one_knee"),
    ("K", "walking toward the viewer with the arms swinging naturally", "walking"),
)
# 줄마다 골격 있음 → 없음 순서(같은 seed 42). 비교는 같은 기준·요청에서 골격 하나만 다르다.
CASES = tuple(dict(case=f"{base}-{'skeleton' if guided else 'text'}", base=base, request=text, pose_key=key, guided=guided)
              for base, text, key in REQUESTS for guided in (True, False))


def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name, data): (P / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def snapshot():
    paths = list((ROOT / "genai_lab").glob("*.py")) + [ROOT / "configs/qwen_edit_settings.json", ROOT / "configs/qwen_model_files.json",
                                                       O / "codex_gpu_request_25.md"]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


import genai_lab.free_generation_pipeline as pipe  # noqa: E402
from genai_lab.free_generation import check_request  # noqa: E402
from genai_lab.pose_guidance import classify_pose  # noqa: E402
from genai_lab.qwen_pose_settings import QwenPoseSettings  # noqa: E402
from genai_lab.studio_generation import StudioRuntime  # noqa: E402

rows = {r["case"]: r for r in read(O / "codex-preflight-21-rerun1/proposed-inputs.json")["rows"]}
settings = QwenPoseSettings.load(ROOT / "configs/qwen_edit_settings.json")
plan = []
for c in CASES:
    row = rows[c["base"]]
    assert row["user_confirmed"] and sha(row["base"]) == row["base_sha256"]
    assert check_request(c["request"])["ok"], c
    assert classify_pose(c["request"]) == c["pose_key"], c
    plan.append(dict(case=c["case"], base=row["base"], base_sha256=row["base_sha256"], request=c["request"], pose_key=c["pose_key"],
                     guided=c["guided"],
                     gender=row["gender"], body_parts=row["body_parts"], features=row["proposed_feature_values"]))
audit = dict(status="passed", selection_seed=42, generation_seeds=[42, 43], settings=settings.record(), plan=plan)
if "--run" not in sys.argv:
    write("preflight.json", audit)
    write("locked-snapshot.json", snapshot())
    print("PREFLIGHT PASSED", json.dumps([(p["case"], p["pose_key"], p["guided"]) for p in plan], ensure_ascii=False), flush=True)
    sys.exit(0)

assert not DEST.exists(), "Output already exists; no overwrite"
assert read(P / "preflight.json") == audit
locked = read(P / "locked-snapshot.json")
assert snapshot() == locked, "Code/settings changed after preflight"
original = pipe.make_free_request
checks = []
active = {}


def checked(specification, *, settings, seed):
    assert snapshot() == locked, "Locked code/settings changed"
    req = original(specification, settings=settings, seed=seed)
    assert req["free_spec"]["request_text"] == active["request"] and req["free_spec"]["selection_seed"] == 42
    assert req["settings"] == audit["settings"] and req["seed"] in (42, 43) and req["memory"]["reserved_limit_gib"] == 6.5
    assert req["images"] == (3 if active["guided"] else 2) and (req["free_spec"]["pose_key"] or None) == (active["pose_key"] if active["guided"] else None)
    checks.append(dict(case=active["case"], seed=seed, prompt=req["prompt"]["positive"], spec=req["free_spec"]))
    write("actual-request-checks.json", checks)
    return req


pipe.make_free_request = checked
start = time.time()
status = dict(status="running", completed=0, total=len(plan), active=None, stage="preparing", execution_authorized=True, quality_pass=None)
write("status.json", status)
results = []
try:
    DEST.mkdir(parents=True)
    runtime = StudioRuntime.from_environment()
    for item in plan:
        active.clear(); active.update(item)
        status.update(active=item["case"], stage="starting")
        write("status.json", status)

        def progress(message, _case=item["case"]):
            status.update(stage=message, elapsed_seconds=round(time.time() - start, 1))
            write("status.json", status)
            print(_case, message, flush=True)
        result = pipe.run_free_generation(item["base"], item["features"], item["request"], base_kind="original", gender=item["gender"],
                                          body_parts=item["body_parts"], confirmed=True, adult_confirmed=True, settings=settings,
                                          runtime=runtime, directory=DEST / item["case"], selection_seed=42, progress=progress,
                                          pose_guidance=item["guided"])
        results.append(dict(case=item["case"], status=result["status"], result=result["result"], attempts=len(result["attempts"])))
        status.update(completed=len(results))
        write("status.json", status)
    status.update(status="completed", active=None, stage="awaiting_user_review", elapsed_seconds=round(time.time() - start, 1))
    write("status.json", status)
    write("completion.json", dict(results=results, code_unchanged=snapshot() == locked, user_quality_pass=None))
    print("REQUEST25 COMPLETE", flush=True)
except Exception:
    status.update(status="failed", error=traceback.format_exc(), elapsed_seconds=round(time.time() - start, 1))
    write("status.json", status)
    raise
finally:
    write("after-snapshot.json", snapshot())
