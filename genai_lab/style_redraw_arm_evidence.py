"""손 상자와 CPU 관절의 일치 여부를 검사한다. 두 팔 골격으로 제3의 팔 부재를 증명하지 않는다."""
import math

MIN_CONFIDENCE = 0.35
MAX_SEGMENT_RATIO = 3.0  # 원근/가림도 걸릴 수 있으므로 결함 확정이 아니라 자동 채택 보류다.


def _joint(points, scores, index, width, height):
    x, y = map(float, points[index])
    confidence = float(scores[index])
    finite = all(math.isfinite(v) for v in (x, y, confidence))
    return dict(x=x if finite else None, y=y if finite else None,
                confidence=confidence if finite else None,
                reliable=finite and confidence >= MIN_CONFIDENCE,
                in_frame=finite and 0 <= x < width and 0 <= y < height)


def _box_matches(joint, box, margin):
    if not joint["reliable"] or not joint["in_frame"]:
        return False
    x0, y0, x1, y1 = box
    return x0 - margin <= joint["x"] <= x1 + margin and y0 - margin <= joint["y"] <= y1 + margin


def _arm(side, indices, points, scores, boxes, width, height):
    joints = {name: _joint(points, scores, index, width, height)
              for name, index in zip(("shoulder", "elbow", "wrist"), indices)}
    issues = []
    ratio = None
    if all(j["reliable"] for j in joints.values()):
        s, e, w = joints.values()
        lengths = [math.hypot(a["x"] - b["x"], a["y"] - b["y"]) for a, b in ((s, e), (e, w))]
        ratio = max(lengths) / max(min(lengths), 1e-6)
        if min(lengths) < 0.01 * math.hypot(width, height):
            issues.append("collapsed_arm_chain")
        elif ratio > MAX_SEGMENT_RATIO:
            issues.append("arm_chain_length_imbalance")
    else:
        issues.append("arm_chain_unassessed")
    wrist = joints["wrist"]
    matched = [i for i, b in enumerate(boxes or []) if _box_matches(wrist, b, 0.025 * math.hypot(width, height))]
    near_edge = wrist["reliable"] and (not wrist["in_frame"] or
        min(wrist["x"] / width, wrist["y"] / height, 1 - wrist["x"] / width, 1 - wrist["y"] / height) < 0.04)
    visibility = ("matched_hand" if matched else "frame_edge_or_outside_possible" if near_edge
                  else "occluded_or_detector_missed")
    if not matched:
        issues.append("hand_visibility_uncertain")
    return dict(side=side, joints=joints, segment_ratio=ratio, matched_hands=matched,
                hand_visibility=visibility, issues=issues)



def assess_arm_framing(arms, boxes, width, height):
    """화면 경계의 팔·손 근거를 따로 기록한다. 검출 실패를 잘림 확정으로 바꾸지 않는다."""
    boundary, uncertain = [], []
    for arm in arms:
        edges, joints_at_edge = [], []
        # 손목이 화면 안에 있어도 팔꿈치가 경계 밖으로 꺾인 경우를 확인한다.
        for name in ("elbow", "wrist"):
            joint = arm["joints"][name]
            if not joint["reliable"]:
                uncertain.append(arm["side"])
                continue
            joint_edges = [edge for edge, distance in (
                ("left", joint["x"] / width), ("right", 1 - joint["x"] / width),
                ("top", joint["y"] / height), ("bottom", 1 - joint["y"] / height)) if distance < 0.04]
            if joint_edges:
                edges.extend(joint_edges)
                joints_at_edge.append(name)
        for index in arm["matched_hands"]:
            x0, y0, x1, y1 = boxes[index]
            edges.extend(edge for edge, distance in (("left", x0 / width), ("right", 1 - x1 / width),
                         ("top", y0 / height), ("bottom", 1 - y1 / height)) if distance <= 0.01)
        if edges:
            boundary.append(dict(side=arm["side"], edges=sorted(set(edges)), joints=joints_at_edge))
        elif not arm["matched_hands"]:
            uncertain.append(arm["side"])
    uncertain = sorted(set(uncertain))
    return dict(state="boundary_risk" if boundary else "unassessed" if uncertain else "no_boundary_evidence",
                boundary=boundary, uncertain_sides=uncertain, crop_confirmed=False,
                note="경계 접근은 잘림 의심이며 의도된 구도인지 사용자 확인이 필요하다. 미검출은 잘림 부재를 증명하지 않는다." )


def assess_arm_evidence(pose, hand_boxes, width, height):
    """관절/손 근거를 대조한다. 가림·화면 밖·검출 실패는 분리하되 보이지 않는 팔다리를 추정 생성하지 않는다."""
    base = dict(state="unassessed", arms=[], issues=[], person_count=None,
                automatic_correction_eligible=False, user_quality_approved=False,
                limits=["fixed_two_arm_topology", "extra_arm_absence_not_verified",
                        "occluded_limbs_not_verified", "finger_shape_not_verified", "perspective_can_trigger_length_warning"])
    if not isinstance(pose, dict) or pose.get("state") != "observed":
        return dict(base, issues=["pose_unavailable"], error=(pose or {}).get("error", "관절 검사 기록 없음"))
    try:
        points, scores = pose["points"], pose["scores"]
        base["person_count"] = len(points)
        if len(points) != 1 or len(scores) != 1:
            return dict(base, issues=["single_person_not_confirmed"])
        if width <= 0 or height <= 0 or len(points[0]) < 8 or len(scores[0]) < 8:
            raise ValueError("관절 좌표/이미지 크기 형식 오류")
        boxes = None if hand_boxes is None else [tuple(map(float, b)) for b in hand_boxes]
        if boxes is not None and any(len(b) != 4 or not all(math.isfinite(v) for v in b) or b[2] <= b[0] or b[3] <= b[1] for b in boxes):
            raise ValueError("손 상자 형식 오류")
        arms = [_arm(side, indices, points[0], scores[0], boxes, width, height)
                for side, indices in (("right", (2, 3, 4)), ("left", (5, 6, 7)))]
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as error:
        return dict(base, issues=["invalid_pose_evidence"], error=str(error))
    issues = sorted({issue for arm in arms for issue in arm["issues"]})
    matched = {i for arm in arms for i in arm["matched_hands"]}
    if boxes is None:
        issues.append("hand_detector_unavailable")
    elif len(matched) != len(boxes):
        issues.append("hand_pose_disagreement")
    if set(arms[0]["matched_hands"]) & set(arms[1]["matched_hands"]):
        issues.append("two_wrists_share_one_hand_detection")
    framing = assess_arm_framing(arms, boxes or [], width, height)
    if framing["state"] == "boundary_risk":
        issues.append("arm_frame_boundary_risk")
    return dict(base, state="needs_review" if issues else "evidence_consistent", arms=arms, framing=framing,
                issues=issues, automatic_correction_eligible=not issues)


def apply_rejection_review_guard(evidence, feedback):
    """사용자가 지적한 인체 결함은 두 팔 골격 검사만으로 해소됐다고 간주하지 않는다."""
    reasons = {"팔·손 잘림", "팔·손 중복과 잘림", "팔·손 중복 또는 연결 오류"}
    if not isinstance(feedback, dict) or feedback.get("decision") != "user_rejected" or feedback.get("reason") not in reasons:
        return evidence
    return dict(evidence, state="needs_review", automatic_correction_eligible=False,
                issues=sorted(set(evidence.get("issues", [])) | {"user_anatomy_rejection_not_cleared"}),
                pending_user_reason=feedback["reason"])
