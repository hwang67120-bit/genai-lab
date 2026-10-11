"""골격에서 얼굴 방향 정보를 뺀다: 얼굴 68점, 눈·귀 선(선택 시 코). 몸통·손은 원래 픽셀 그대로.

자세 검출이 앞머리에 가린 눈 때문에 정면 얼굴을 돌린 얼굴로 읽으면, 골격의 얼굴 점이 생성 얼굴을 돌려 버린다(D-086).
머리 위치는 사용자가 확인한 머리 윤곽 조건이, 얼굴 모습은 얼굴 참조 이미지가 맡는다.
"""
import math
import cv2
import numpy as np
from genai_lab.proportion_inputs import require
from genai_lab.shoulder_control import NAMES, draw_body, observed_points, validate_original_skeleton

FACE_MODES = ("all", "no_face", "no_face_nose")
HEAD_JOINTS = {"no_face": ("right_eye", "left_eye", "right_ear", "left_ear"),
               "no_face_nose": ("nose", "right_eye", "left_eye", "right_ear", "left_ear")}


def stripped_joints(joints, mode):
    """뺀 관절은 미검출로 표시한다. 어깨 보정이 같은 관절로 몸통을 다시 그릴 때도 얼굴 선이 생기지 않는다."""
    require(mode in FACE_MODES, "알 수 없는 얼굴 점 방식입니다.")
    removed = set(HEAD_JOINTS.get(mode, ()))
    return [dict(joint, detected=False, face_strip=mode) if joint["joint_name"] in removed else dict(joint)
            for joint in joints]


def hand_support(points, shape):
    """손 그림 허용 영역: 손목 중심, 팔뚝 길이 반지름(어깨 보정의 손 영역과 같은 규칙)."""
    support = np.zeros(shape, np.uint8)
    for elbow, wrist in ((3, 4), (6, 7)):
        if points[elbow] is not None and points[wrist] is not None:
            radius = max(8, math.ceil(math.dist(points[elbow], points[wrist])))
            cv2.circle(support, tuple(np.rint(points[wrist]).astype(int)), radius, 1, -1)
    return support.astype(bool)


def strip_face_control(control, joints, head_mask, mode):
    """control은 RGB 배열이다. 원래 골격을 재현해 얼굴·손 픽셀을 분리한 뒤, 손만 남기고 얼굴 관절 없이 몸통을 다시 그린다."""
    require(mode in FACE_MODES, "알 수 없는 얼굴 점 방식입니다.")
    if mode == "all":
        return control.copy(), dict(mode=mode, removed_pixels=0)
    flipped = control[..., ::-1]  # 어깨 보정과 같은 채널 순서로 비교한다.
    points = observed_points(joints)
    extras = validate_original_skeleton(flipped, points, head_mask)
    hands = hand_support(points, head_mask.shape)
    body = draw_body(observed_points(stripped_joints(joints, mode)), *flipped.shape[:2])
    result = np.where((extras & hands)[..., None], flipped, body)
    result = np.ascontiguousarray(result[..., ::-1])
    return result, dict(mode=mode, removed_joints=list(HEAD_JOINTS[mode]),
                        removed_pixels=int(np.any(result != control, axis=2).sum()),
                        kept_hand_pixels=int((extras & hands).sum()),
                        dropped_face_pixels=int((extras & ~hands).sum()))
