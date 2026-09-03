"""가드레일 경고와 최종 차단의 승인 계약을 검증한다."""

import unittest

from genai_lab.guardrails import (
    GuardResult,
    GuardSeverity,
    create_human_agnostic_guard_decision,
    evaluate_guard_results,
)


def create_safe_decision(**overrides):
    values = {
        "remaining_clothing_pixel_count": 0,
        "removal_percent": 100.0,
        "input_change_mask_pixel_count": 1000,
        "input_protected_overlap_pixel_count": 0,
        "input_outside_foreground_pixel_count": 0,
        "processed_mask_pixel_count": 800,
        "model_mask_pixel_count": 750,
        "soft_overlap_pixel_count": 0,
        "hard_overlap_pixel_count": 0,
        "final_protected_overlap_pixel_count": 0,
        "final_outside_foreground_pixel_count": 0,
    }
    values.update(overrides)
    return create_human_agnostic_guard_decision(**values)


class GuardDecisionTest(unittest.TestCase):
    def test_no_target_is_not_a_success(self):
        decision = create_safe_decision(removal_percent=None, removal_status="not_evaluable")
        self.assertFalse(decision.approval_enabled)
        self.assertIn("ORIGINAL_CLOTHING_NOT_EVALUABLE", [r.code for r in decision.blocking_results])

    def test_protected_target_conflict_is_actionable(self):
        decision = create_safe_decision(protected_clothing_overlap_pixel_count=4, removal_status="needs_review")
        self.assertFalse(decision.approval_enabled)
        self.assertTrue(decision.blocking_results[0].recovery_action_ko)

    def test_safe_results_enable_approval(self):
        decision = create_safe_decision()
        self.assertTrue(decision.approval_enabled)
        self.assertEqual(len(decision.blocking_results), 0)
        self.assertEqual(len(decision.warning_results), 0)

    def test_corrected_hard_overlap_is_warning_not_block(self):
        decision = create_safe_decision(hard_overlap_pixel_count=64)
        self.assertTrue(decision.approval_enabled)
        self.assertEqual(len(decision.blocking_results), 0)
        self.assertEqual(len(decision.warning_results), 1)
        warning = decision.warning_results[0]
        self.assertEqual(warning.measured_value, 64)
        self.assertEqual(warning.corrected_value, 0)

    def test_corrected_soft_and_hard_overlap_create_two_warnings(self):
        decision = create_safe_decision(
            soft_overlap_pixel_count=512,
            hard_overlap_pixel_count=64,
        )
        self.assertTrue(decision.approval_enabled)
        self.assertEqual(len(decision.warning_results), 2)

    def test_final_protected_overlap_blocks_approval(self):
        decision = create_safe_decision(
            hard_overlap_pixel_count=64,
            final_protected_overlap_pixel_count=1,
        )
        self.assertFalse(decision.approval_enabled)
        self.assertEqual(len(decision.blocking_results), 1)
        self.assertEqual(
            decision.blocking_results[0].code,
            "FINAL_PROTECTED_OVERLAP",
        )

    def test_empty_model_mask_blocks_approval_with_reason(self):
        decision = create_safe_decision(model_mask_pixel_count=0)
        self.assertFalse(decision.approval_enabled)
        self.assertGreaterEqual(len(decision.blocking_results), 1)
        for result in decision.blocking_results:
            self.assertTrue(result.code)
            self.assertIsNotNone(result.measured_value)
            self.assertIsNotNone(result.threshold_value)
            self.assertTrue(result.recovery_action_ko)

    def test_empty_guard_result_collection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "0개"):
            evaluate_guard_results(())

    def test_disabled_decision_always_has_blocking_reason(self):
        decision = create_safe_decision(remaining_clothing_pixel_count=1)
        self.assertFalse(decision.approval_enabled)
        self.assertGreaterEqual(len(decision.blocking_results), 1)

    def test_unrelated_warning_does_not_change_existing_block(self):
        decision = evaluate_guard_results(
            (
                GuardResult(
                    code="BLOCK_TEST",
                    stage="test",
                    severity=GuardSeverity.BLOCK,
                    measured_value=1,
                    threshold_value=0,
                    unit="px",
                    corrected_value=None,
                    message_ko="차단",
                    recovery_action_ko="수정",
                ),
                GuardResult(
                    code="WARNING_TEST",
                    stage="test",
                    severity=GuardSeverity.WARNING,
                    measured_value=3,
                    threshold_value=0,
                    unit="px",
                    corrected_value=0,
                    message_ko="경고",
                ),
            )
        )
        self.assertFalse(decision.approval_enabled)
        self.assertEqual(len(decision.blocking_results), 1)
        self.assertEqual(len(decision.warning_results), 1)


if __name__ == "__main__":
    unittest.main()
