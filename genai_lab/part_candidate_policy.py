"""관측 수치와 실험적 후보 제안을 분리한다. 의미 분류의 정답이 아니다."""
from dataclasses import dataclass
import numpy as np
from genai_lab.reference_regions import read_binary_mask

EXPERIMENTAL_OVERLAP_POLICY = "experimental_overlap_v2"


@dataclass(frozen=True)
class OverlapProposalPolicy:
    garment_dominance: float = .90
    broad_hair_coverage: float = .50
    minimum_hair_overlap: float = .50

    def __post_init__(self):
        if not all(np.isfinite(v) and 0 < v <= 1 for v in (
            self.garment_dominance, self.broad_hair_coverage, self.minimum_hair_overlap
        )):
            raise ValueError("부위 후보 제외 임계값은 0 초과 1 이하여야 합니다.")

    def propose(self, evidence, *, hair_context=False):
        if not evidence["area_pixels"]:
            return False, "empty_mask"
        if evidence["garment_overlap"] >= self.garment_dominance:
            return False, "garment_dominated"
        if (hair_context and evidence["hair_overlap"] >= self.minimum_hair_overlap
                and evidence["hair_coverage"] >= self.broad_hair_coverage):
            return False, "broad_hair_candidate"
        return True, "candidate"


def measure_candidate_overlap(masks, source_masks):
    """마스크 간 기하 수치만 측정한다. 선택/제외/종류 판정은 하지 않는다."""
    size = source_masks["garment"].size
    garment = read_binary_mask(source_masks["garment"], size)
    hair = read_binary_mask(source_masks["hair"], size)
    values = np.asarray(masks)
    if values.ndim != 3 or values.shape[1:] != (size[1], size[0]):
        raise ValueError("부위 후보 좌표 크기 오류")
    if not np.isin(values, (0, 1)).all():
        raise ValueError("부위 후보 마스크 값 오류")
    hair_area = int(hair.sum())
    evidence = []
    for index, raw in enumerate(values):
        selected = raw.astype(bool)
        area = int(selected.sum())
        hair_pixels = int((selected & hair).sum())
        evidence.append({
            "index": index, "area_pixels": area,
            "garment_overlap": float((selected & garment).sum() / area) if area else 0.,
            "hair_overlap": hair_pixels / area if area else 0.,
            "hair_coverage": hair_pixels / hair_area if hair_area else 0.,
        })
    return evidence
