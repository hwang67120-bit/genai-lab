"""모델 대역을 이용한 데이터 경계 테스트. 실제 검출 정확도 시험이 아니다."""
from dataclasses import FrozenInstanceError
import numpy as np
import pytest
from PIL import Image
from genai_lab.part_detection_candidates import detect_part_candidates
from genai_lab.part_cross_review import PartCrossReviewer
from genai_lab.part_candidate_policy import OverlapProposalPolicy
from genai_lab.reference_conditions import plan_source_regions


class Backend:
    def __init__(self):
        self.calls = 0
        self.boxes = np.array([[1, 1, 5, 5]], dtype=np.float32)
        self.raw = np.zeros((1, 1, 8, 8), dtype=bool)
        self.raw[0, 0, 1:5, 1:5] = True

    def detect(self, image, query, threshold):
        self.calls += 1
        return self.boxes, np.array([.9])

    def segment(self, image, boxes):
        return self.raw, np.array([[.95]])


def test_review_policy_cannot_change_raw_detection_or_rerun_model():
    backend = Backend()
    with Image.new("RGB", (8, 8), "blue") as rgb:
        raw = detect_part_candidates(backend, rgb, "tail", "tail.", .35, lambda: None)
    fingerprint = raw.fingerprint()
    backend.raw[:] = False
    raw.array()[:] = False
    assert raw.array().sum() == 16
    with pytest.raises(FrozenInstanceError):
        raw.masks = ()
    with Image.new("L", (8, 8), 255) as occupied, Image.new("L", (8, 8)) as empty:
        reject = PartCrossReviewer({"garment": occupied, "hair": empty}, OverlapProposalPolicy())
        accept = PartCrossReviewer({"garment": empty, "hair": empty}, OverlapProposalPolicy())
        # 검토자가 받은 다른 부위 마스크도 스냅샷이므로 이후 변경에 영향받지 않는다.
        occupied.paste(0, (0, 0, 8, 8))
        rejected, accepted = reject.review(raw, .8), accept.review(raw, .8)
        assert rejected.selected_indices == ()
        assert accepted.selected_indices == (0,)
        assert accepted.masks(raw).sum() == 16
        decisions = rejected.decisions()
        decisions[0]["reason"] = "modified"
        assert rejected.decisions()[0]["reason"] == "garment_dominated"
    assert raw.fingerprint() == fingerprint
    assert backend.calls == 1


def test_empty_detection_does_not_become_absence():
    backend = Backend()
    backend.detect = lambda *args: (np.empty((0, 4)), np.empty(0))
    with Image.new("RGB", (8, 8)) as rgb:
        raw = detect_part_candidates(backend, rgb, "tail", "tail.", .35, lambda: None)
    assert raw.masks == ()
    assert raw.array().shape == (0, 8, 8)


def test_layout_area_is_measured_and_original_pixels_stay_unchanged():
    with Image.new("L", (8, 8)) as head, Image.new("L", (8, 8)) as garment, Image.new("L", (8, 8)) as tail:
        head.paste(255, (0, 0, 8, 2))
        garment.paste(255, (0, 2, 8, 8))
        tail.paste(255, (0, 3, 2, 7))
        original = garment.tobytes()
        out_head, out_garment, report = plan_source_regions(head, garment, [("tail", tail)])
        try:
            assert report["area_audit"]["garment"] == {"before": 48, "after": 40, "removed": 8}
            assert np.count_nonzero(out_garment) == 40
            assert garment.tobytes() == original
        finally:
            out_head.close()
            out_garment.close()
