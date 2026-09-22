"""관절 기반 체형 후보와 원본 색 표본으로 복원 초기 이미지를 만든다.

관절 폭·볼 위치는 초기 휴리스틱이다. 피부 의미 분할이나 숨은 체형의
정답을 보장하지 않으며 GUI에서 표본과 기본복을 함께 검토한다.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np
from PIL import Image

from genai_lab.image_digest import calculate_image_pixel_sha256
from genai_lab.original_body_pose import OriginalBodyPose
from genai_lab.removal_diagnostics import trace_removal


def binary(image: Image.Image) -> np.ndarray:
    with image.convert("L") as converted:
        return np.asarray(converted, dtype=np.uint8).copy() >= 128


def as_mask(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array.astype(np.uint8) * 255)


@dataclass
class BodyInitialization:
    images: dict[str, Image.Image]
    metadata: dict

    @property
    def initial_image(self) -> Image.Image:
        return self.images["body_initial"]

    def close(self) -> None:
        for image in self.images.values():
            image.close()

    def copy(self) -> "BodyInitialization":
        return BodyInitialization(
            {name: image.copy() for name, image in self.images.items()},
            dict(self.metadata),
        )


@trace_removal("body_initialization")
def create_body_initialization(
    source: Image.Image,
    removal_mask: Image.Image,
    protected_mask: Image.Image,
    foreground_mask: Image.Image,
    pose: OriginalBodyPose,
    basewear_rgb: tuple[int, int, int] = (48, 72, 104),
) -> BodyInitialization:
    """불투명 몸통·골반 기본복, 피부, 배경으로 초기값을 분리한다."""
    if any(im.size != source.size for im in (
        removal_mask, protected_mask, foreground_mask,
    )):
        raise ValueError("초기 이미지의 원본·마스크 크기가 다릅니다.")
    digest = calculate_image_pixel_sha256(source, "RGB")
    if digest != pose.source_image_sha256:
        raise ValueError("체형 초기화 원본이 승인된 DWPose 캐릭터와 다릅니다.")
    if len(basewear_rgb) != 3 or any(
        type(v) is not int or not 0 <= v <= 255 for v in basewear_rgb
    ):
        raise ValueError("기본복 색상은 0~255 정수 RGB여야 합니다.")
    with source.convert("RGB") as rgb:
        pixels = np.array(rgb, dtype=np.uint8)
    removal, protected, foreground = map(
        binary, (removal_mask, protected_mask, foreground_mask)
    )
    if not removal.any() or np.any(removal & (protected | ~foreground)):
        raise ValueError("제거 영역이 비었거나 보호·외곽 영역과 충돌합니다.")
    joints = {j.joint_name: j for j in pose.approved_pose.joint_coordinates}

    def point(name):
        joint = joints.get(name)
        if joint is None or not joint.detected or not all(
            math.isfinite(v) for v in (joint.x, joint.y, joint.confidence_score)
        ) or joint.confidence_score < pose.approved_pose.minimum_pose_confidence:
            raise ValueError(f"체형/피부 표본에 필요한 관절이 부족합니다: {name}")
        if not (0 <= joint.x < source.width and 0 <= joint.y < source.height):
            raise ValueError(f"원본 좌표 밖 관절입니다: {name}")
        return np.array((joint.x, joint.y), dtype=float)

    def xy(p):
        return tuple(np.rint(p).astype(int))

    def disk(canvas, center, radius):
        cv2.circle(canvas, xy(center), max(1, int(round(radius))), 255, -1)

    def capsule(canvas, start, end, radius):
        cv2.line(canvas, xy(start), xy(end), 255, max(2, int(round(radius * 2))))
        disk(canvas, start, radius)
        disk(canvas, end, radius)

    shape = removal.shape
    shoulder = [point(f"{side}_shoulder") for side in ("left", "right")]
    hip = [point(f"{side}_hip") for side in ("left", "right")]
    shoulder_width = float(np.linalg.norm(shoulder[0] - shoulder[1]))
    torso_length = float(np.linalg.norm(np.mean(hip, axis=0) - np.mean(shoulder, axis=0)))
    if shoulder_width < 4 or torso_length < 4:
        raise ValueError("어깨 폭 또는 몸통 길이가 부족해 체형을 만들 수 없습니다.")

    # 몸통은 단순 불투명 원피스 기본복으로 덮는다. 비율은 추정값이다.
    wear = np.zeros(shape, dtype=np.uint8)
    crotch_direction = np.mean(hip, axis=0) - np.mean(shoulder, axis=0)
    crotch_direction /= torso_length
    lower_hip = [p + crotch_direction * torso_length * 0.16 for p in hip]
    polygon = np.array([shoulder[0], shoulder[1], hip[1], lower_hip[1],
                        lower_hip[0], hip[0]], dtype=np.int32)
    cv2.fillPoly(wear, [polygon], 255)
    proxy = wear.copy()
    # 코-어깨 중심선 중 승인 제거 영역에 걸친 부분만 목 연결 힌트로 사용한다.
    capsule(proxy, point("nose"), np.mean(shoulder, axis=0), shoulder_width * 0.09)
    for side in ("left", "right"):
        s, e, w = (point(f"{side}_{part}") for part in ("shoulder", "elbow", "wrist"))
        h, k, a = (point(f"{side}_{part}") for part in ("hip", "knee", "ankle"))
        capsule(proxy, s, e, shoulder_width * 0.14)
        capsule(proxy, e, w, shoulder_width * 0.10)
        capsule(proxy, h, k, shoulder_width * 0.20)
        capsule(proxy, k, a, shoulder_width * 0.13)
        disk(proxy, a, shoulder_width * 0.14)

    # 눈 사이 축의 수직 방향 중 코 쪽을 골라 볼 안쪽 후보를 만든다.
    eyes = [point("left_eye"), point("right_eye")]
    nose = point("nose")
    eye_axis = eyes[1] - eyes[0]
    eye_distance = float(np.linalg.norm(eye_axis))
    if eye_distance < 4:
        raise ValueError("얼굴 피부 표본을 만들 눈 간격이 부족합니다.")
    down = np.array((-eye_axis[1], eye_axis[0])) / eye_distance
    if np.dot(nose - np.mean(eyes, axis=0), down) < 0:
        down *= -1
    face_roi = np.zeros(shape, dtype=np.uint8)
    for eye in eyes:
        disk(face_roi, eye + down * eye_distance * 0.30, eye_distance * 0.13)
    face_roi = (face_roi > 0) & foreground & ~removal
    values = pixels[face_roi]
    if len(values) < 16:
        raise ValueError("볼 피부 표본이 16px 미만입니다. 얼굴 관절과 표본 위치를 확인하세요.")
    # 고정 살색 임계값을 쓰지 않고 볼 후보에서 가장 흔한 색 군집을 고른다.
    bins, counts = np.unique(values // 24, axis=0, return_counts=True)
    dominant = bins[int(np.argmax(counts))]
    center = np.median(values[np.all(values // 24 == dominant, axis=1)], axis=0)
    distance = np.linalg.norm(pixels.astype(float) - center, axis=2)
    face_samples = face_roi & (distance <= 30)
    if np.count_nonzero(face_samples) < 16:
        raise ValueError("일관된 얼굴 피부색 표본이 부족합니다. 표본 위치를 확인하세요.")
    skin = np.rint(np.median(pixels[face_samples], axis=0)).astype(np.uint8)

    hand_roi = np.zeros(shape, dtype=np.uint8)
    for side in ("left", "right"):
        wrist, elbow = point(f"{side}_wrist"), point(f"{side}_elbow")
        disk(hand_roi, wrist + (wrist - elbow) * 0.18, shoulder_width * 0.07)
    hand_samples = (hand_roi > 0) & foreground & ~removal
    hand_values = pixels[hand_samples]
    hand_delta = (float(np.linalg.norm(np.median(hand_values, axis=0) - skin))
                  if len(hand_values) >= 16 else None)

    radius = max(3, int(shoulder_width * 0.15))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    bg_samples = (cv2.dilate(foreground.astype(np.uint8), kernel) > 0) & ~foreground & ~protected
    if np.count_nonzero(bg_samples) < 16:
        raise ValueError("배경 표본이 부족합니다. 캐릭터 외곽 마스크를 확인하세요.")
    bg_values = pixels[bg_samples]
    background = np.rint(np.median(bg_values, axis=0)).astype(np.uint8)
    bg_spread = float(np.percentile(np.linalg.norm(bg_values.astype(float) - background, axis=1), 90))

    body = removal & (proxy > 0)
    wear_region = body & (wear > 0)
    skin_region = body & ~wear_region
    bg_region = removal & ~body
    if not wear_region.any():
        raise ValueError("몸통·골반 기본복이 변경 영역에 없습니다.")
    result = pixels.copy()
    result[skin_region] = skin
    result[wear_region] = basewear_rgb
    result[bg_region] = background
    changed = np.any(result != pixels, axis=2)
    if np.any(changed & (~removal | protected)):
        raise ValueError("초기화가 승인 범위 밖 또는 보호 영역을 변경했습니다.")
    overlay = pixels.copy()
    overlay[face_samples] = (0, 255, 0)
    overlay[hand_samples] = (255, 180, 0)
    return BodyInitialization(
        images={
            "body_initial": Image.fromarray(result),
            "body_proxy": as_mask(proxy > 0),
            "body_skin_region": as_mask(skin_region),
            "body_basewear_region": as_mask(wear_region),
            "body_background_region": as_mask(bg_region),
            "body_skin_samples": as_mask(face_samples),
            "body_sample_overlay": Image.fromarray(overlay),
        },
        metadata={
            "method": "joint_capsules_cheek_dominant_color_v1",
            "source_rgb_sha256": digest,
            "skin_rgb": skin.tolist(), "basewear_rgb": list(basewear_rgb),
            "background_rgb": background.tolist(),
            "skin_sample_count": int(face_samples.sum()),
            "hand_sample_count": int(hand_samples.sum()),
            "hand_face_rgb_distance": hand_delta,
            "hand_comparison": "insufficient" if hand_delta is None else (
                "similar" if hand_delta <= 35 else "different_not_blended"),
            "background_spread_p90": bg_spread,
            "background_is_flat_seed_only": True,
            "shoulder_width": shoulder_width, "torso_length": torso_length,
            "skin_pixels": int(skin_region.sum()),
            "basewear_pixels": int(wear_region.sum()),
            "background_pixels": int(bg_region.sum()),
            "outside_changed_pixels": 0, "protected_changed_pixels": 0,
            "requires_visual_approval": True,
        },
    )
