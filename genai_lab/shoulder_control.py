"""어깨 보정: 입력 확인 → 원래 골격 재현 → 어깨 이동 → 결과 확인."""
from dataclasses import replace
from pathlib import Path
import json
import math
import cv2
import numpy as np
from PIL import Image
from genai_lab.proportion_inputs import sha, require, checked_image, json_value
from genai_lab.qwen_record_io import write_json

NAMES = ("nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow", "left_wrist", "right_hip",
         "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye", "left_eye", "right_ear", "left_ear")
LIMBS = [[2,3],[2,6],[3,4],[4,5],[6,7],[7,8],[2,9],[9,10],[10,11],[2,12],[12,13],[13,14],[2,1],[1,15],[15,17],[1,16],[16,18]]
COLORS = [[255,0,0],[255,85,0],[255,170,0],[255,255,0],[170,255,0],[85,255,0],[0,255,0],[0,255,85],[0,255,170],[0,255,255],
          [0,170,255],[0,85,255],[0,0,255],[85,0,255],[170,0,255],[255,0,255],[255,0,170],[255,0,85]]


def draw_body(pts, H, W):
    canvas = np.zeros((H, W, 3), np.uint8)
    for index, limb in enumerate(LIMBS):
        start, end = np.array(limb) - 1
        if pts[start] is None or pts[end] is None:
            continue
        polygon = limb_polygon(pts[start], pts[end])
        cv2.fillConvexPoly(canvas, polygon, COLORS[index])

    canvas = (canvas * 0.6).astype(np.uint8)
    for index, point in enumerate(pts):
        if point is None:
            continue
        center = (int(point[0]), int(point[1]))
        cv2.circle(canvas, center, 4, COLORS[index], thickness=-1)
    return canvas


def limb_polygon(start, end):
    # 기존 골격과 픽셀이 같아야 하므로 계산 순서와 정수 변환을 유지한다.
    horizontal = [start[0], end[0]]
    vertical = [start[1], end[1]]
    center_y = np.mean(vertical)
    center_x = np.mean(horizontal)
    length = ((vertical[0] - vertical[1]) ** 2
              + (horizontal[0] - horizontal[1]) ** 2) ** .5
    angle = math.degrees(math.atan2(
        vertical[0] - vertical[1], horizontal[0] - horizontal[1]))
    return cv2.ellipse2Poly(
        (int(center_x), int(center_y)), (int(length / 2), 4),
        int(angle), 0, 360, 1)


def observed_points(joints):
    indexed = {j["joint_name"]: j for j in joints}
    points = []
    for name in NAMES:
        joint = indexed.get(name)
        valid = joint and joint.get("detected", True) and joint["confidence_score"] >= .3
        points.append((joint["x"], joint["y"]) if valid else None)
    require(all(points[k] is not None for k in (1, 2, 5)), "목·양 어깨 관절이 부족합니다.")
    require(all(np.isfinite(p).all() for p in points if p is not None), "골격 좌표가 올바르지 않습니다.")
    return points


def extra_point_support(points, head_mask):
    """차이 픽셀과 독립적으로, 확인한 머리와 검출된 손의 허용 영역을 만든다."""
    ys, xs = np.where(head_mask > 0)
    require(len(xs) > 0, "확인한 머리 영역이 비었습니다.")
    support = np.zeros(head_mask.shape, np.uint8)
    cv2.rectangle(support, (max(0, int(xs.min())-4), max(0, int(ys.min())-4)),
                  (int(xs.max())+4, int(ys.max())+4), 1, -1)
    for elbow, wrist in ((3, 4), (6, 7)):
        if points[elbow] is not None and points[wrist] is not None:
            radius = max(8, math.ceil(math.dist(points[elbow], points[wrist])))
            cv2.circle(support, tuple(np.rint(points[wrist]).astype(int)), radius, 1, -1)
    return support.astype(bool)


def correct_pixels(control, joints, head_mask):
    points = observed_points(joints)
    rgb = control[..., ::-1]
    extras = validate_original_skeleton(rgb, points, head_mask)

    adjusted_points = pull_shoulders_toward_neck(points)
    adjusted_body = draw_body(adjusted_points, *rgb.shape[:2])
    corrected = np.where(extras[..., None], rgb, adjusted_body)
    details = shoulder_change_record(points, adjusted_points, rgb, corrected)
    return np.ascontiguousarray(corrected[..., ::-1]), details


def validate_original_skeleton(rgb, points, head_mask):
    original_body = draw_body(points, *rgb.shape[:2])
    extras = np.any(rgb != original_body, axis=2)
    support = extra_point_support(points, head_mask)

    # 차이를 모두 복사하면 손상도 통과하므로, 얼굴·손 밖의 차이는 거부한다.
    require(not np.any(extras & ~support),
            "얼굴·손 영역 밖의 골격을 재현하지 못했습니다.")
    reconstructed = np.where((extras & support)[..., None], rgb, original_body)
    require(np.array_equal(reconstructed, rgb), "원래 골격 재현 검사에 실패했습니다.")
    return extras


def pull_shoulders_toward_neck(points):
    adjusted = list(points)
    neck = np.asarray(points[1])
    for index in (2, 5):
        adjusted[index] = tuple(neck + .85 * (np.asarray(points[index]) - neck))
    return adjusted


def shoulder_change_record(before, after, original, corrected):
    return {
        "reconstruction_equal": True,
        "support_rule": "reviewed_head_box_margin4_hand_radius_forearm",
        "shoulder_dist_before": round(math.dist(before[2], before[5]), 1),
        "shoulder_dist_after": round(math.dist(after[2], after[5]), 1),
        "changed_pixels": int(np.any(corrected != original, axis=2).sum()),
    }


def prepare_shoulder_control(control_file, control_sha, joints, head, directory):
    """보정과 미리보기를 저장하고, 보정 불가 시 원본 유지 상태를 반환한다."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    control = checked_image(control_file, control_sha, (736, 1232), "RGB")
    mask = checked_image(head.mask_file, head.mask_sha256, (736, 1232), "L")
    record = {"shoulder_pull": "0.85", "parent_file": str(control_file), "parent_sha256": control_sha,
        "head_mask_file": str(head.mask_file), "head_mask_sha256": head.mask_sha256, "joints": joints}
    try:
        pixels, details = correct_pixels(control, joints, mask)
        path = directory / "control.png"
        Image.fromarray(pixels).save(path)
        record.update(status="applied", control_file=str(path), control_sha256=sha(path), **details)
    except ValueError as error:
        pixels = control
        record.update(status="not_applied", control_file=str(control_file), control_sha256=control_sha,
            reconstruction_equal=False, warning="어깨 보정을 적용하지 못함: " + str(error))
    preview = np.concatenate((control[..., ::-1], pixels[..., ::-1]), axis=1)
    Image.fromarray(preview).save(directory / "preview.png")
    record.update(preview_file=str(directory / "preview.png"), preview_sha256=sha(directory / "preview.png"))
    write_json(directory / "shoulder.json", json_value(record))
    return json_value(record)


def resolve_shoulder_inputs(inputs, options):
    """확인 기록을 재검증하고, 두 생성 단계에 전달할 골격을 반환한다."""
    if options.shoulder_pull == "off":
        return inputs, None
    record, parent, mask = read_verified_shoulder_record(inputs, options)
    try:
        pixels, details = correct_pixels(parent, record["joints"], mask)
    except ValueError:
        require(record["status"] == "not_applied" and record["control_sha256"] == inputs.control_sha256,
            "어깨 실패 기록과 실제 입력이 다릅니다.")
        return inputs, record
    require(record["status"] == "applied", "어깨 재현 결과와 기록이 다릅니다.")
    actual = checked_image(record["control_file"], record["control_sha256"], (736, 1232), "RGB")
    require(np.array_equal(pixels, actual) and all(record[k] == v for k, v in details.items()), "어깨 보정 픽셀·기록이 달라졌습니다.")
    return replace(inputs, control_file=Path(record["control_file"]), control_sha256=record["control_sha256"]), record


def read_verified_shoulder_record(inputs, options):
    path = options.shoulder_record_file
    require(path is not None and sha(path) == options.shoulder_record_sha256, "어깨 확인 기록이 없거나 변경됐습니다.")
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    require(record["shoulder_pull"] == "0.85" and record["parent_sha256"] == inputs.control_sha256 == options.head.control_sha256,
        "보정 골격과 머리 윤곽의 부모 SHA가 다릅니다.")
    require(record["head_mask_sha256"] == options.head.mask_sha256, "다른 머리의 어깨 보정입니다.")
    parent = checked_image(inputs.control_file, inputs.control_sha256, (736, 1232), "RGB")
    mask = checked_image(options.head.mask_file, options.head.mask_sha256, (736, 1232), "L")
    require(sha(record["preview_file"]) == record["preview_sha256"], "어깨 미리보기가 변경됐습니다.")
    return record, parent, mask
