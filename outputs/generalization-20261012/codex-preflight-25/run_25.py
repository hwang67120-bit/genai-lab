"""Request 25: three rows x (skeleton on, skeleton off), same seed — D-131 pose guidance. No operational code changes.

Without --run: CPU preflight only (expected prompts, D-128 conflicts, locked snapshot). With --run: generate both rows in order.
Run through the UNC path (like request 22) so ROOT and input paths share one form.
"""
import sys, json, hashlib, time, traceback, copy, subprocess
from datetime import datetime, timezone
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
def write(name, data):
    path = P / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    if name == "status.json":
        write("batch-status.json", data)


def same_values(a, b):
    return json.dumps(a, ensure_ascii=False, sort_keys=True, separators=(",", ":")) == json.dumps(b, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()


def snapshot():
    paths = list((ROOT / "genai_lab").glob("*.py")) + [ROOT / "configs/qwen_edit_settings.json", ROOT / "configs/qwen_model_files.json",
        ROOT / "docs/FREE_GENERATION_CONTRACT.md", ROOT / "docs/FLOW.md", ROOT / "docs/DECISIONS.md", ROOT / "docs/FREE_GENERATION_POSE_GUIDANCE.md",
        O / "codex_gpu_request_25.md", O / "codex-preflight-21-rerun1/proposed-inputs.json", P / "input-approval.json", P / "criteria.json", Path(__file__).resolve(), P / "validate_25_cpu.py"]
    for base in ("N1", "N2", "K"):
        prev = O / "free-r21-rerun1" / (base + "-1")
        paths += [prev / "inputs/canvas.png", prev / "character-features/features.json", prev / "qwen-inputs/reference.png"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


import genai_lab.free_generation_pipeline as pipe  # noqa: E402
from genai_lab.free_generation import check_request, features_sentence, request_conflicts, free_prompt
from genai_lab.pose_guidance import SKELETON_PROMPT
from genai_lab.qwen_free_generation import validate_free_request  # noqa: E402
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
expected_prompts = {}
expected_refs = {"N1": [313,17,518,287], "N2": [304,33,463,278], "K": [335,51,523,268]}
for c in CASES:
    features = read(O / "free-r21-rerun1" / (c["base"] + "-1") / "character-features/features.json")
    sentence = features_sentence(features, hidden=request_conflicts(c["request"]))
    expected_prompts[c["case"]] = free_prompt(sentence, rows[c["base"]]["gender"], rows[c["base"]]["body_parts"], c["request"], 42, skeleton=c["guided"])
for base in rows:
    if base in expected_refs:
        guided = expected_prompts[base+"-skeleton"]
        assert guided.count(SKELETON_PROMPT) == 1 and guided.replace(SKELETON_PROMPT, "", 1).encode() == expected_prompts[base+"-text"].encode()
assert head() == read(P / "input-approval.json")["approved_head"], "Approved HEAD changed"
assert snapshot() == read(P / "approved-snapshot.json")["snapshot"], "Approved files changed"
audit = dict(status="passed", expected_prompts=expected_prompts, reference_boxes=expected_refs, selection_seed=42, generation_seeds=[42, 43], settings=settings.record(), plan=plan)
if "--run" not in sys.argv:
    write("preflight.json", audit)
    write("locked-snapshot.json", snapshot())
    print("PREFLIGHT PASSED", json.dumps([(p["case"], p["pose_key"], p["guided"]) for p in plan], ensure_ascii=False), flush=True)
    sys.exit(0)

assert not DEST.exists(), "Output already exists; no overwrite"
assert same_values(read(P / "preflight.json"), audit)
locked = read(P / "locked-snapshot.json")
assert snapshot() == locked, "Code/settings changed after preflight"
write("before.json", dict(head=head(), snapshot=snapshot(), recorded_at=datetime.now(timezone.utc).isoformat()))
original = pipe.make_free_request
frozen = {}
pair_inputs = {}
checks = []
active = {}


def checked(specification, *, settings, seed):
    assert snapshot() == locked, "Locked code/settings changed"
    req = original(specification, settings=settings, seed=seed)
    assert req["free_spec"]["request_text"] == active["request"] and req["free_spec"]["selection_seed"] == 42
    assert req["settings"] == audit["settings"] and req["seed"] in (42, 43) and req["memory"]["reserved_limit_gib"] == 6.5
    assert req["images"] == (3 if active["guided"] else 2) and (req["free_spec"]["pose_key"] or None) == (active["pose_key"] if active["guided"] else None)
    validate_free_request(json.loads(json.dumps(req)))
    fs = req["free_spec"]
    assert req["prompt"]["positive"].encode() == expected_prompts[active["case"]].encode(), "Actual prompt differs"
    assert same_values(fs["body_parts"], active["body_parts"]) and fs["gender"] == active["gender"]
    assert same_values(fs["reference_box"], expected_refs[active["base"]]) and fs["reference_scope"] == "head"
    assert fs["image_sha256"] == active["base_sha256"] and sha(fs["image_path"]) == active["base_sha256"]
    assert sha(fs["reference_path"]) == fs["reference_sha256"]
    frozen_value = dict(free_spec=fs, prompt=req["prompt"], settings=req["settings"], memory=req["memory"], images=req["images"])
    if seed == 42:
        assert active["case"] not in frozen, "Repeated initial attempt"
        frozen[active["case"]] = copy.deepcopy(frozen_value)
    else:
        assert active["case"] in frozen and same_values(frozen[active["case"]], frozen_value), "Retry changed conditions"
    inputs = dict(base=fs["image_sha256"], reference=fs["reference_sha256"], box=fs["reference_box"], scope=fs["reference_scope"])
    if active["base"] in pair_inputs:
        assert same_values(pair_inputs[active["base"]], inputs), "Pair input mismatch"
    else: pair_inputs[active["base"]] = copy.deepcopy(inputs)
    checks.append(dict(case=active["case"], seed=seed, status="passed", request=req))
    write("actual-request-checks.json", checks)
    return req


pipe.make_free_request = checked
start = time.time()
write("launch-runtime.json", dict(head=head(), started_at=datetime.now(timezone.utc).isoformat(), python=sys.executable))
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
        assert snapshot() == locked, "Locked SHA changed after generation"
        for attempt in result["attempts"]:
            assert attempt["status"] == "completed" and attempt["seed"] in (42,43)
            worker = attempt["worker_record"]
            peak = max(float(stage.get(k, 0)) for stage in worker["memory_stages"] for k in ("reserved_gib", "stage_peak_reserved_gib"))
            assert peak <= 6.5 and worker["status"] == "completed", "Memory or worker contract failure"
        write(item["case"] + "-exit.json", dict(exit_code=0, status=result["status"], attempts=len(result["attempts"])))
        results.append(dict(case=item["case"], status=result["status"], result=result["result"], attempts=len(result["attempts"])))
        status.update(completed=len(results))
        write("status.json", status)
    assert snapshot() == locked and head() == read(P / "input-approval.json")["approved_head"], "Final SHA/HEAD mismatch"
    status.update(status="completed", active=None, stage="awaiting_user_review", elapsed_seconds=round(time.time() - start, 1))
    write("status.json", status)
    write("completion.json", dict(results=results, code_unchanged=snapshot() == locked, user_quality_pass=None))
    write("finalization.json", dict(status="completed", results=results, code_unchanged=True, user_quality_pass=None))
    (DEST / "summary.json").write_text(json.dumps(dict(status="completed", results=results, user_quality_pass=None), ensure_ascii=False, indent=2), encoding="utf-8")
    print("REQUEST25 COMPLETE", flush=True)
except Exception:
    status.update(status="failed", error=traceback.format_exc(), elapsed_seconds=round(time.time() - start, 1))
    write("status.json", status)
    write("contract-error.json", dict(case=status.get("active"), error=traceback.format_exc(), log="batch-stderr.log"))
    write("finalization.json", dict(status="failed", completed=len(results), error=traceback.format_exc()))
    raise
finally:
    after=snapshot()
    write("after-snapshot.json", after)
    write("after.json", dict(head=head(), snapshot=after, recorded_at=datetime.now(timezone.utc).isoformat()))
    write("unchanged.json", dict(unchanged=after==locked, approved_head=read(P / "input-approval.json")["approved_head"], current_head=head(), files=len(locked)))
