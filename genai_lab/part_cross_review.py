"""검출 이후 타 부위와의 겹침을 검토한다. 검출/원본 마스크는 변경하지 않는다."""
from dataclasses import dataclass
import json
import numpy as np
from PIL import Image
from genai_lab.reference_regions import read_binary_mask
from genai_lab.source_region_experiment import filter_part_instances
from genai_lab.part_candidate_policy import EXPERIMENTAL_OVERLAP_POLICY, OverlapProposalPolicy


@dataclass(frozen=True)
class CandidateReview:
    candidate_fingerprint: str
    selected_indices: tuple[int, ...]
    decisions_json: bytes

    def __post_init__(self):
        if (not isinstance(self.candidate_fingerprint, str)
                or not isinstance(self.selected_indices, tuple)
                or not isinstance(self.decisions_json, bytes)
                or any(type(i) is not int or i < 0 for i in self.selected_indices)
                or len(set(self.selected_indices)) != len(self.selected_indices)):
            raise ValueError("교차 검토 결과는 불변 계약이어야 합니다.")

    def masks(self, candidates):
        if candidates.fingerprint() != self.candidate_fingerprint:
            raise ValueError("교차 검토 대상과 원본 검출 결과가 다릅니다.")
        return candidates.array()[list(self.selected_indices)].copy()

    def decisions(self):
        return json.loads(self.decisions_json)


class PartCrossReviewer:
    def __init__(self, source_masks, policy):
        self.policy = policy
        self.size = source_masks["garment"].size
        self._masks = tuple((name, read_binary_mask(source_masks[name], self.size).astype(
            np.uint8).tobytes()) for name in ("garment", "hair"))

    def _images(self):
        return {name: Image.fromarray(np.frombuffer(raw, dtype=np.uint8).reshape(
            self.size[1], self.size[0]) * 255) for name, raw in self._masks}

    def save_sources(self, directory):
        images = self._images()
        try:
            for name, image in images.items():
                image.save(directory / f"source_{name}_mask.png")
        finally:
            for image in images.values():
                image.close()

    def review(self, candidates, mask_threshold, *, hair_context=False):
        if candidates.size != self.size:
            raise ValueError("검출과 교차 검토 좌표 크기가 다릅니다.")
        indices = np.flatnonzero(np.asarray(candidates.mask_scores) >= mask_threshold)
        images = self._images()
        try:
            _, decisions = filter_part_instances(candidates.name, candidates.array()[indices], images,
                garment_dominance=self.policy.garment_dominance,
                broad_hair_coverage=self.policy.broad_hair_coverage,
                minimum_hair_overlap=self.policy.minimum_hair_overlap,
                hair_context=hair_context, proposal_policy=EXPERIMENTAL_OVERLAP_POLICY)
        finally:
            for image in images.values():
                image.close()
        selected = []
        for decision, index in zip(decisions, indices):
            decision["candidate_index"] = int(index)
            if decision["accepted"]:
                selected.append(int(index))
        return CandidateReview(candidates.fingerprint(), tuple(selected),
                               json.dumps(decisions, allow_nan=False).encode())
