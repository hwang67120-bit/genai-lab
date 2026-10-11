"""골격 얼굴 점 빼기: 얼굴 68점·눈·귀(·코) 선만 빠지고 몸통·손은 그대로, 어깨 보정과 이어진다."""
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from genai_lab.pose_face_strip import strip_face_control, stripped_joints
from genai_lab.shoulder_control import NAMES, draw_body, observed_points, prepare_shoulder_control
from genai_lab.proportion_inputs import sha

POINTS = {"nose": (368, 200), "neck": (368, 300), "right_shoulder": (300, 300), "right_elbow": (280, 420),
          "right_wrist": (270, 520), "left_shoulder": (436, 300), "left_elbow": (456, 420), "left_wrist": (466, 520),
          "right_hip": (330, 600), "right_knee": (330, 800), "right_ankle": (330, 1000), "left_hip": (406, 600),
          "left_knee": (406, 800), "left_ankle": (406, 1000), "right_eye": (355, 185), "left_eye": (381, 185),
          "right_ear": (340, 190), "left_ear": (396, 190)}


def joints():
    return [dict(joint_name=n, x=float(POINTS[n][0]), y=float(POINTS[n][1]), confidence_score=.9, detected=True) for n in NAMES]


def scene():
    flipped = draw_body(observed_points(joints()), 1232, 736)
    flipped[170:175, 345:350] = 255          # 얼굴 점(머리 상자 안)
    flipped[515:525, 265:275] = (0, 0, 255)  # 손 그림(손목 근처)
    control = np.ascontiguousarray(flipped[..., ::-1])
    mask = np.zeros((1232, 736), np.uint8); mask[140:260, 320:420] = 255
    return control, mask


@pytest.mark.parametrize("mode,removed", [("no_face", ("right_eye", "left_eye", "right_ear", "left_ear")),
                                          ("no_face_nose", ("nose", "right_eye", "left_eye", "right_ear", "left_ear"))])
def test_face_points_and_head_lines_removed_hands_and_body_kept(mode, removed):
    control, mask = scene()
    stripped, details = strip_face_control(control, joints(), mask, mode)
    assert not stripped[170:175, 345:350].any()                       # 얼굴 점 제거
    assert np.array_equal(stripped[515:525, 265:275], control[515:525, 265:275])  # 손 유지
    expected = draw_body(observed_points(stripped_joints(joints(), mode)), 1232, 736)[..., ::-1]
    outside = np.ones((1232, 736), bool); outside[515:525, 265:275] = False
    assert np.array_equal(stripped[outside], expected[outside])
    assert details["removed_joints"] == list(removed) and details["dropped_face_pixels"] > 0
    assert all(not j["detected"] for j in stripped_joints(joints(), mode) if j["joint_name"] in removed)


def test_all_mode_is_unchanged():
    control, mask = scene()
    same, details = strip_face_control(control, joints(), mask, "all")
    assert np.array_equal(same, control) and details["removed_pixels"] == 0


def test_shoulder_correction_applies_on_stripped_control(tmp_path):
    control, mask = scene()
    stripped, _ = strip_face_control(control, joints(), mask, "no_face")
    Image.fromarray(stripped).save(tmp_path / "control.png")
    Image.fromarray(mask).save(tmp_path / "mask.png")
    head = SimpleNamespace(mask_file=tmp_path / "mask.png", mask_sha256=sha(tmp_path / "mask.png"))
    record = prepare_shoulder_control(tmp_path / "control.png", sha(tmp_path / "control.png"),
                                      stripped_joints(joints(), "no_face"), head, tmp_path / "shoulder")
    assert record["status"] == "applied"
    corrected = np.asarray(Image.open(record["control_file"]).convert("RGB"))
    assert np.array_equal(corrected[515:525, 265:275], control[515:525, 265:275])
    assert not corrected[170:175, 345:350].any()
