"""자동 보정 전후의 팔·손 근거를 대조한다. 최초 그림을 정답이나 새 생성 참조로 삼지 않는다."""
import itertools
import math

MAX_JOINT_SHIFT = 0.04  # 캔버스 대각선 비율. 정상 자세 변화도 보류할 수 있어 결함 확정에는 쓰지 않는다.
MAX_HAND_SHIFT = 0.04


def _hand_centres(record):
    detections = record.get("detections")
    if not isinstance(detections, list) or len(detections) != record.get("hands"):
        raise ValueError("손 위치 근거가 검출 수와 일치하지 않습니다.")
    centres = []
    for detection in detections:
        x0, y0, x1, y1 = map(float, detection["box"])
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1)) or x1 <= x0 or y1 <= y0:
            raise ValueError("손 위치 형식 오류")
        centres.append(((x0 + x1) / 2, (y0 + y1) / 2))
    return centres


def _joint_positions(record):
    arms = record.get("arm_evidence", {}).get("arms", [])
    if len(arms) != 2 or {a["side"] for a in arms} != {"left", "right"}:
        raise ValueError("팔 관절 근거가 없습니다.")
    positions = {}
    for arm in arms:
        for name in ("shoulder", "elbow", "wrist"):
            joint = arm["joints"][name]
            if not joint.get("reliable") or not joint.get("in_frame"):
                raise ValueError("가림·경계·낮은 신뢰도의 관절은 보정 전후 일치를 확인할 수 없습니다.")
            point = (float(joint["x"]), float(joint["y"]))
            if not all(math.isfinite(v) for v in point):
                raise ValueError("관절 좌표 형식 오류")
            positions[arm["side"] + "_" + name] = point
    return positions


def compare_correction_anatomy(before, proposed, size):
    """의상/특징/얼굴/색 보정이 팔 배치를 바꾸는지 확인한다. 사용자 자세 변경에는 호출하지 않는다."""
    issues, measurements = [], {}
    if not before.get("automatic_correction_eligible"):
        issues.append("prior_anatomy_not_cleared")
    if before.get("hands") != proposed.get("hands"):
        issues.append("hand_visibility_changed")
    try:
        width, height = size
        if width <= 0 or height <= 0 or not math.isfinite(width + height):
            raise ValueError("캔버스 크기 오류")
        for record in (before, proposed):
            if record.get("image_size") is not None and list(record["image_size"]) != list(size):
                raise ValueError("보정 전후 캔버스 크기 불일치")
        diagonal = math.hypot(width, height)
        original, changed = _joint_positions(before), _joint_positions(proposed)
        shifts = {name: math.dist(point, changed[name]) / diagonal for name, point in original.items()}
        measurements["joint_shift_diagonal_fraction"] = shifts
        if any(value > MAX_JOINT_SHIFT for value in shifts.values()):
            issues.append("arm_position_changed")
        old_hands, new_hands = _hand_centres(before), _hand_centres(proposed)
        if not (0 < len(old_hands) == len(new_hands) <= 2):
            raise ValueError("보정 전후 손 위치의 일대일 대조 불가")
        # 검출 순서가 달라도 같은 손을 중복으로 대응하지 않는다.
        shift = min(max(math.dist(a, b) / diagonal for a, b in zip(old_hands, order))
                    for order in itertools.permutations(new_hands))
        measurements["hand_shift_diagonal_fraction"] = shift
        if shift > MAX_HAND_SHIFT:
            issues.append("hand_position_changed")
    except (TypeError, ValueError, KeyError, IndexError, OverflowError) as error:
        issues.append("correction_geometry_unassessed")
        measurements["error"] = str(error)
    return dict(state="review_required" if issues else "no_detected_geometry_change", issues=issues,
                measurements=measurements,
                thresholds=dict(joint_shift=MAX_JOINT_SHIFT, hand_shift=MAX_HAND_SHIFT),
                reference_role="comparison_only_not_ground_truth", user_quality_approved=False,
                limits=["finger_shape_not_verified", "extra_arm_absence_not_verified", "detector_error_possible"])


def guard_correction_anatomy(before, proposed, size):
    """새 검출이 정상처럼 보여도 기존 인체 의심이나 배치 변화가 있으면 자동 대체를 보류한다."""
    from genai_lab.style_redraw_anatomy import attach_arm_evidence
    comparison = compare_correction_anatomy(before, proposed, size)
    evidence = dict(proposed.get("arm_evidence") or {})
    evidence["correction_comparison"] = comparison
    if comparison["issues"]:
        evidence.update(state="needs_review", automatic_correction_eligible=False,
                        issues=sorted(set(evidence.get("issues", [])) | {"correction_geometry_review_required"}))
    return attach_arm_evidence(proposed, evidence)
