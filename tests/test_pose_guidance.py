"""자유 생성 자세 유도(D-131) — GPU 없이."""
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from genai_lab import pose_guidance as pg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_free_generation import features, fake_models, settings  # noqa: E402,F401
from genai_lab import free_generation_pipeline as pipeline  # noqa: E402
from genai_lab.qwen_pose_worker import validate_request  # noqa: E402


def standing_joints(scale=1.0):
    """정면으로 선 사람 관절(화면 왼쪽 = 캐릭터 오른쪽)."""
    p = dict(nose=(384, 150), neck=(384, 220), right_shoulder=(334, 230), right_elbow=(320, 360), right_wrist=(312, 470),
             left_shoulder=(434, 230), left_elbow=(448, 360), left_wrist=(456, 470), right_hip=(354, 470), right_knee=(350, 660),
             right_ankle=(348, 850), left_hip=(414, 470), left_knee=(418, 660), left_ankle=(420, 850), right_eye=(372, 140),
             left_eye=(396, 140), right_ear=(360, 148), left_ear=(408, 148))
    return [dict(joint_name=k, x=384 + (x - 384) * scale, y=220 + (y - 220) * scale, detected=True) for k, (x, y) in p.items()]


@pytest.mark.parametrize("text,key", [("sitting on a simple stool with the hands on the knees", "sitting_stool"),
                                      ("kneeling on one knee with one hand resting on the raised knee", "kneeling_one_knee"),
                                      ("walking toward the viewer", "walking"), ("standing with the arms crossed", "arms_crossed"),
                                      ("의자에 앉아 있기", "sitting_stool"), ("lying on the floor", None), ("눈 내리는 거리", None)])
def test_requests_are_classified_into_the_supported_list(text, key):
    assert pg.classify_pose(text) == key


def test_skeleton_takes_the_pose_from_the_template_and_lengths_from_the_character():
    small, large = standing_joints(0.8), standing_joints(1.1)
    for joints in (small, large):
        records, info = pg.retarget("sitting_stool", joints, (768, 1024))
        assert info["defaulted"] == [] and len(records) == 18 and all(r["detected"] for r in records)
    thigh = lambda recs: math.dist(*[(r["x"], r["y"]) for r in recs if r["joint_name"] in ("right_hip", "right_knee")])  # noqa: E731
    assert thigh(pg.retarget("sitting_stool", large, (768, 1024))[0]) > thigh(pg.retarget("sitting_stool", small, (768, 1024))[0])
    sit = {r["joint_name"]: r for r in pg.retarget("sitting_stool", standing_joints(), (768, 1024))[0]}
    assert abs(sit["right_knee"]["y"] - sit["right_hip"]["y"]) < 60                 # 앉기: 허벅지가 거의 수평
    assert abs(max(sit["right_ankle"]["y"], sit["left_ankle"]["y"]) - 850) < 1      # 가장 낮은 발 높이는 기준 그림과 같다
    _, fallback = pg.retarget("walking", [], (768, 1024))
    assert set(fallback["defaulted"]) == set(pg.DEFAULT_RATIO)                       # 관절을 못 찾으면 표준 비율


def test_drawing_is_deterministic_and_comparison_flags_a_different_pose():
    records, _ = pg.retarget("sitting_stool", standing_joints(), (768, 1024))
    a, b = pg.draw_skeleton(records, (768, 1024)), pg.draw_skeleton(records, (768, 1024))
    assert a.shape == (1024, 768, 3) and np.array_equal(a, b) and a.any()
    assert pg.compare_pose(records, records)["status"] == "match"
    assert pg.compare_pose(records, standing_joints())["status"] == "mismatch"      # 앉기를 요청했는데 선 자세
    assert pg.compare_pose(records, standing_joints()[:3])["status"] == "unmeasurable"


def test_pipeline_sends_a_verified_skeleton_as_image_3(tmp_path, fake_models, monkeypatch):
    base, calls = fake_models
    monkeypatch.setattr(pipeline, "detect_joints", lambda *a, **kw: standing_joints())
    result = pipeline.run_free_generation(base, features(), "sitting on a simple stool with the hands on the knees", base_kind="original",
        gender="female", body_parts=(), confirmed=True, adult_confirmed=True, settings=settings(tmp_path), runtime=SimpleNamespace(),
        directory=tmp_path / "run", selection_seed=42, pose_guidance=True)
    request = next(c[1] for c in calls if c[0] == "qwen")
    assert request["images"] == 3 and request["free_spec"]["pose_key"] == "sitting_stool"
    assert pg.SKELETON_PROMPT in request["prompt"]["positive"]
    spec = validate_request(json.loads(json.dumps(request)))                         # 작업 프로세스도 다시 그려 대조한다
    assert spec.images == 3
    trace = json.loads(Path(result["trace"]).read_text(encoding="utf-8"))
    assert trace["input_facts"]["pose_guidance"]["status"] == "guided"
    assert len(trace["attempts"][0]["actual_images"]) == 3
    row = next(j for j in result["condition_judgements"] if j["key"] == "pose.skeleton_match")
    assert row["evidence"]["status"] == "mismatch"                                   # 가짜 결과 관절은 선 자세
    Image.new("RGB", (768, 1024), "red").save(spec.skeleton_path)                    # 골격 그림을 바꾸면 거부
    with pytest.raises(ValueError):
        spec.verify_image()


def test_unsupported_pose_and_default_off_keep_two_images(tmp_path, fake_models, monkeypatch):
    base, calls = fake_models
    result = pipeline.run_free_generation(base, features(), "lying on the floor", base_kind="original", gender="female", body_parts=(),
        confirmed=True, adult_confirmed=True, settings=settings(tmp_path), runtime=SimpleNamespace(), directory=tmp_path / "a",
        selection_seed=42, pose_guidance=True)
    trace = json.loads(Path(result["trace"]).read_text(encoding="utf-8"))
    assert trace["input_facts"]["pose_guidance"]["status"] == "unsupported" and any(e["key"] == "pose_skeleton" for e in trace["excluded"])
    off = pipeline.run_free_generation(base, features(), "sitting on a chair", base_kind="original", gender="female", body_parts=(),
        confirmed=True, adult_confirmed=True, settings=settings(tmp_path), runtime=SimpleNamespace(), directory=tmp_path / "b",
        selection_seed=42)
    assert all(c[1]["images"] == 2 for c in calls if c[0] == "qwen")
    assert json.loads(Path(off["trace"]).read_text(encoding="utf-8"))["input_facts"]["pose_guidance"]["status"] == "off"
