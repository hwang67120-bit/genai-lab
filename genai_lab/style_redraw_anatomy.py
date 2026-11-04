"""단일 캐릭터의 손·팔 검사 근거를 검토와 자동 보정에 연결한다. 인체 전체의 합격 판정은 아니다."""


def assess_hands(count):
    """검출 수로 과다·미확인·과다 미감지를 구분한다. 가려진 손을 누락으로 단정하지 않는다."""
    valid = isinstance(count, int) and not isinstance(count, bool) and count > 0
    state = "unassessed" if not valid else "excess_hands_suspected" if count > 2 else "no_excess_hands_detected"
    return dict(state=state, hands=count, eligible_for_review=valid and count <= 2,
                evidence="hand_detector_count", user_quality_approved=False, automatic_correction_eligible=False,
                unassessed=["arm_attachment", "joint_structure", "body_proportions"])


def attach_arm_evidence(record, evidence):
    """검토와 자동 대체를 분리한다. 미확인 후보도 근거가 있으면 사용자가 볼 수 있으나 자동 채택하지 않는다."""
    result = dict(record, schema_version=2, arm_evidence=evidence,
                  automatic_correction_eligible=False, user_quality_approved=False)
    if record["state"] == "excess_hands_suspected":
        return result
    observed = evidence.get("person_count") == 1 and len(evidence.get("arms", [])) == 2
    if observed or record["eligible_for_review"]:
        consistent = observed and evidence.get("state") == "evidence_consistent" and not evidence.get("issues")
        result.update(state="limited_evidence_consistent" if consistent else "anatomy_review_required",
                      eligible_for_review=True,
                      automatic_correction_eligible=bool(consistent and record["eligible_for_review"]
                                                        and evidence.get("automatic_correction_eligible") is True))
    return result


def candidate_anatomy(candidate):
    """이전 기록도 실제 검출 수로 다시 판단한다. 검사 없는 후보를 정상으로 승격하지 않는다."""
    result = assess_hands(candidate.get("hands"))
    # 과거 extra_hands는 원본보다 늘어난 수였다. 실제 수가 있으면 2개 상한을 우선한다.
    if candidate.get("hands") is None and candidate.get("extra_hands") is True:
        result.update(state="excess_hands_suspected", eligible_for_review=False)
    saved = candidate.get("anatomy") or {}
    if (result["state"] == "excess_hands_suspected" and isinstance(saved, dict)
            and saved.get("hands") == result["hands"]
            and "hand_location" in saved):
        result.update(hand_location=saved["hand_location"], detections=saved.get("detections", []),
                      foreground_box_count=saved.get("foreground_box_count"))
    if isinstance(saved, dict) and saved.get("schema_version") == 2 and saved.get("hands") == result["hands"]:
        evidence = saved.get("arm_evidence")
        if isinstance(evidence, dict):
            result = attach_arm_evidence(result, evidence)
    return result


def gate_correction(decision, proposed):
    """특징/의상 점수 개선만으로 손 과다 또는 미확인 후보를 채택하지 않는다."""
    allowed = proposed.get("automatic_correction_eligible") is True and proposed["eligible_for_review"]
    return dict(decision, use_correction=bool(decision["use_correction"] and allowed),
                reason=decision["reason"] if allowed else proposed["state"], anatomy=proposed)



def require_reviewable(candidate):
    """선택·저장·합성 경계에서 같은 검사를 적용한다. 오류 원본 파일은 삭제하지 않는다."""
    result = candidate_anatomy(candidate)
    if not result["eligible_for_review"]:
        raise ValueError("손 과다 의심 또는 인체 검사 미확인 결과는 선택·저장·합성할 수 없습니다.")
    return result
