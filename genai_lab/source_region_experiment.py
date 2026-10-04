"""명시적인 원본 위치 참조 실험. 자세 고정이나 자동 정렬을 의미하지 않는다."""
import numpy as np
from genai_lab.reference_regions import read_binary_mask
from genai_lab.regional_reference import valid_part_name
from genai_lab.part_candidate_policy import (
    EXPERIMENTAL_OVERLAP_POLICY, OverlapProposalPolicy, measure_candidate_overlap,
)


def filter_part_instances(name, masks, source_masks, *, garment_dominance=.90,
                          broad_hair_coverage=.50, minimum_hair_overlap=.50,
                          hair_context=False, proposal_policy=EXPERIMENTAL_OVERLAP_POLICY):
    """실험 정책이 제안한 후보만 반환한다. 원본 픽셀과 미확정 상태를 보존한다."""
    if not valid_part_name(name):
        raise ValueError("부위 이름 형식 오류")
    if proposal_policy != EXPERIMENTAL_OVERLAP_POLICY:
        raise ValueError("등록되지 않은 부위 후보 제안 정책입니다.")
    policy = OverlapProposalPolicy(garment_dominance, broad_hair_coverage, minimum_hair_overlap)
    evidence = measure_candidate_overlap(masks, source_masks)
    selected, report = [], []
    for raw, item in zip(masks, evidence):
        keep, reason = policy.propose(item, hair_context=hair_context)
        report.append({
            **item, "accepted": keep, "reason": reason,
            "selection_status": "proposed" if keep else "not_proposed",
            "semantic_status": "unresolved",
            "decision_source": proposal_policy,
            "requires_review": True,
            "removed_pixels": 0 if keep else item["area_pixels"],
        })
        if keep:
            selected.append(np.asarray(raw, dtype=bool).copy())
    shape = np.asarray(masks).shape[1:]
    return (np.stack(selected) if selected else np.zeros((0, *shape), dtype=bool)), report


def assign_source_regions(parts, identity_mask, garment_mask):
    """호환 API. 배치 계산은 별도 순수 함수에서 수행한다."""
    from genai_lab.reference_conditions import plan_source_regions
    if any(part.target_region is not None for part in parts):
        raise ValueError("이미 지정된 생성 좌표를 원본 위치로 덮어쓰지 않습니다.")
    result = plan_source_regions(identity_mask, garment_mask,
                                 [(part.name, part.source_mask) for part in parts])
    for part in parts:
        part.target_region = part.source_mask.copy()
    return result
