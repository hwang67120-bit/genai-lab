"""관절/손 불일치와 검토/자동채택 경계를 CPU로 검증한다."""
import math
import numpy as np
import pytest
from genai_lab.style_redraw_arm_evidence import assess_arm_evidence
from genai_lab.style_redraw_anatomy import assess_hands, attach_arm_evidence, candidate_anatomy, gate_correction


def pose():
    points = [[50, 10], [50, 30], [30, 30], [20, 50], [10, 70], [70, 30], [80, 50], [90, 70]]
    return dict(state="observed", points=[points], scores=[[0.9] * 8])


def hands():
    return [[5, 65, 15, 75], [85, 65, 95, 75]]


def merged(p, boxes):
    return attach_arm_evidence(assess_hands(None if boxes is None else len(boxes)), assess_arm_evidence(p, boxes, 100, 100))


def test_consistent_evidence_is_not_a_quality_pass():
    record = merged(pose(), hands())
    assert record["automatic_correction_eligible"]
    assert record["eligible_for_review"] and not record["user_quality_approved"]
    assert "extra_arm_absence_not_verified" in record["arm_evidence"]["limits"]
    assert gate_correction(dict(use_correction=True, reason="improved"), record)["use_correction"]


def test_wrong_cross_arm_association_cannot_replace_original_or_lose_warning():
    p = pose()
    p["points"][0][3] = [25, 20]
    p["points"][0][4] = [5, 95]
    a = merged(p, [[0, 90, 10, 99], hands()[1]])
    assert "arm_chain_length_imbalance" in a["arm_evidence"]["issues"]
    candidate = dict(hands=2, anatomy=a)
    restored = candidate_anatomy(candidate)
    assert restored["state"] == "anatomy_review_required"
    assert restored["eligible_for_review"] and not restored["automatic_correction_eligible"]
    assert not gate_correction(dict(use_correction=True, reason="better clothes"), restored)["use_correction"]


def test_no_hands_with_pose_can_be_reviewed_without_inventing_visibility():
    a = merged(pose(), [])
    assert a["eligible_for_review"] and not a["automatic_correction_eligible"]
    assert all(arm["hand_visibility"] == "occluded_or_detector_missed" for arm in a["arm_evidence"]["arms"])
    assert candidate_anatomy(dict(hands=0, anatomy=a))["eligible_for_review"]
    assert not candidate_anatomy(dict(hands=0))["eligible_for_review"]


def test_outside_wrist_is_possible_crop_not_missing_limb_verdict():
    p = pose();p["points"][0][4] = [-3, 70]
    a = merged(p, [hands()[1]])
    assert a["arm_evidence"]["arms"][0]["hand_visibility"] == "frame_edge_or_outside_possible"
    assert not a["automatic_correction_eligible"]
    assert "legs" not in str(a)


def test_wrong_hand_box_and_shared_hand_are_not_auto_accepted():
    a = merged(pose(), [[40, 5, 55, 20]])
    assert "hand_pose_disagreement" in a["arm_evidence"]["issues"]
    p = pose();p["points"][0][7] = [10, 70]
    a = merged(p, [hands()[0]])
    assert "two_wrists_share_one_hand_detection" in a["arm_evidence"]["issues"]
    assert not a["automatic_correction_eligible"]


def test_pose_does_not_override_excess_hands():
    a = merged(pose(), hands() + [[40, 40, 50, 50]])
    assert a["state"] == "excess_hands_suspected"
    assert not a["eligible_for_review"] and not a["automatic_correction_eligible"]


@pytest.mark.parametrize("p", [{}, {"state": "unassessed", "error": "missing model"},
    {"state": "observed", "points": [], "scores": []},
    {"state": "observed", "points": [[[0, 0]]], "scores": [[1.0]]}])
def test_missing_or_invalid_pose_never_certifies_hands_only(p):
    a = merged(p, hands())
    assert a["eligible_for_review"] and not a["automatic_correction_eligible"]
    assert a["state"] == "anatomy_review_required"


def test_nonfinite_joint_is_recorded_as_uncertain_and_json_safe():
    p = pose();p["points"][0][3][0] = math.nan
    a = merged(p, hands())
    import json
    json.dumps(a, allow_nan=False)
    assert not a["automatic_correction_eligible"]


def test_legacy_hand_count_alone_no_longer_authorizes_automatic_replacement():
    assert not gate_correction(dict(use_correction=True, reason="improved"), assess_hands(2))["use_correction"]


def test_product_inspector_records_environment_failure_and_keeps_review(tmp_path, monkeypatch):
    from genai_lab import style_redraw_worker as w
    monkeypatch.setattr(w, "hand_detections", lambda images, req: [[((5, 5, 10, 10), "hand", 0.9)]])
    a = w.inspect_anatomy(tmp_path, [np.zeros((100, 100, 3), dtype=np.uint8)], {}, "test")[0]
    assert a["eligible_for_review"] and not a["automatic_correction_eligible"]
    assert a["arm_evidence"]["issues"] == ["pose_unavailable"]
    assert list((tmp_path / "anatomy-pose").glob("*/result.json"))


def test_ui_preserves_review_warning_and_requires_user_approval(tmp_path):
    import json
    from PIL import Image
    from genai_lab.style_redraw_results import StyleRedrawResults
    a = merged(pose(), [])
    path = tmp_path / "result-seed42.png"
    Image.new("RGB", (100, 100)).save(path)
    (tmp_path / "generation.json").write_text(json.dumps(dict(candidates=[dict(file=path.name, seed=42, hands=0, anatomy=a)])))
    result = StyleRedrawResults(dict(state="review_pending", candidates=[str(path)]))
    assert "자동 보정 채택 보류" in result.head_notice
    assert result.status == "awaiting_user_review" and result.approved_sha is None


def test_detected_hand_touching_border_still_holds_automatic_correction():
    a = merged(pose(), [[0, 65, 15, 75], hands()[1]])
    evidence = a["arm_evidence"]
    assert evidence["framing"]["state"] == "boundary_risk"
    assert not evidence["framing"]["crop_confirmed"]
    assert "arm_frame_boundary_risk" in evidence["issues"]
    assert not a["automatic_correction_eligible"]


def test_missing_wrist_is_unassessed_not_crop_or_normal_verdict():
    p = pose(); p["scores"][0][4] = 0.1
    a = merged(p, hands())
    assert a["arm_evidence"]["framing"]["state"] == "unassessed"
    assert not a["arm_evidence"]["framing"]["crop_confirmed"]
    assert not a["automatic_correction_eligible"]


def test_in_frame_hands_only_report_no_boundary_evidence():
    a = merged(pose(), hands())
    assert a["arm_evidence"]["framing"]["state"] == "no_boundary_evidence"
    assert not a["user_quality_approved"]


@pytest.mark.parametrize("phase", ["initial", "autofix", "face-seed44", "colour-seed44"])
def test_user_anatomy_rejection_cannot_be_cleared_by_two_detected_hands(tmp_path, monkeypatch, phase):
    from genai_lab import style_redraw_worker as w
    monkeypatch.setattr(w, "hand_detections", lambda images, req: [[(b, "hand", .9) for b in hands()]])
    monkeypatch.setattr(w, "observe_arm_poses", lambda *args: [pose()])
    req = dict(user_rejection=dict(decision="user_rejected", reason="팔·손 중복과 잘림"))
    a = w.inspect_anatomy(tmp_path, [np.zeros((100,100,3), dtype=np.uint8)], req, phase)[0]
    assert a["eligible_for_review"] and not a["automatic_correction_eligible"]
    assert "user_anatomy_rejection_not_cleared" in a["arm_evidence"]["issues"]
    restored = candidate_anatomy(dict(hands=2, anatomy=a))
    assert not restored["automatic_correction_eligible"]
    assert not gate_correction(dict(use_correction=True, reason="clothes improved"), restored)["use_correction"]


def test_other_feedback_does_not_disable_existing_correction():
    from genai_lab.style_redraw_arm_evidence import apply_rejection_review_guard
    evidence = assess_arm_evidence(pose(), hands(), 100, 100)
    assert apply_rejection_review_guard(evidence, dict(decision="user_rejected", reason="의상 불일치")) == evidence
