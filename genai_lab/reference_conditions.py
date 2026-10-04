"""부위 원본 조건과 최종 배치를 분리한다. 생성 모델의 독립성을 보장하지 않는다."""
from dataclasses import dataclass
import hashlib
import json

import numpy as np
from PIL import Image

from genai_lab.reference_regions import read_binary_mask
from genai_lab.regional_reference import valid_part_name, crop_reference
from genai_lab.scene_reference import PartReference

SOURCE_LAYOUT = "source_roi_priority_experiment"
DISJOINT_LAYOUT = "reviewed_disjoint_regions"


@dataclass(frozen=True)
class ReferenceCondition:
    """PIL 공유 참조 대신 불변 bytes를 소유한다. 다른 부위가 수정할 수 없다."""
    name: str
    rgb_size: tuple
    rgb_bytes: bytes
    region_size: tuple
    region_bytes: bytes
    scale: float | None = None  # 얼굴/의상 강도는 기존 생성 설정에서 지정한다.

    @classmethod
    def capture(cls, name, rgb, region, scale=None):
        if (name not in ("identity", "garment", "hair")
                and not valid_part_name(name)):
            raise ValueError("부위 조건 이름 오류")
        if rgb.mode != "RGB" or min(rgb.size) < 1:
            raise ValueError("부위 조건에는 RGB 참조 이미지가 필요합니다.")
        selected = read_binary_mask(region, region.size)
        if not selected.any():
            raise ValueError("부위 조건 영역이 비었습니다.")
        if name in ("identity", "garment") and scale is not None:
            raise ValueError("얼굴·의상 강도는 기존 생성 설정에서 지정합니다.")
        if name not in ("identity", "garment") and (
                scale is None or not np.isfinite(scale) or not 0 <= scale <= 1):
            raise ValueError("부위 조건 강도 오류")
        return cls(name, rgb.size, rgb.tobytes(), region.size, region.tobytes(), scale)

    def image(self):
        return Image.frombytes("RGB", self.rgb_size, self.rgb_bytes)

    def region(self):
        return Image.frombytes("L", self.region_size, self.region_bytes)

    def record(self):
        return {"name": self.name, "rgb_size": self.rgb_size, "region_size": self.region_size,
                "rgb_sha256": hashlib.sha256(self.rgb_bytes).hexdigest(),
                "source_region_sha256": hashlib.sha256(self.region_bytes).hexdigest(),
                "scale": self.scale, "scale_source": "generation_setting" if self.scale is None else "part"}


def plan_source_regions(identity_mask, garment_mask, part_masks):
    """기존 배치 규칙만 수행하는 순수 함수. 원본 조건/검출 객체를 변경하지 않는다."""
    size = identity_mask.size
    identity = read_binary_mask(identity_mask, size)
    garment = read_binary_mask(garment_mask, size)
    occupied = np.zeros((size[1], size[0]), dtype=bool)
    if not part_masks:
        return (
            Image.fromarray(identity.astype(np.uint8) * 255),
            Image.fromarray(garment.astype(np.uint8) * 255),
            {
                "mode": "source_regions_experiment",
                "status": "no_optional_parts",
                "pose_locked": False,
                "region_policy": SOURCE_LAYOUT,
                "requires_input_review": False,
                "semantic_accuracy_verified": "not_applicable",
                "identity_overlap_removed": 0,
                "garment_overlap_removed": 0,
                "area_audit": {
                    "identity": {
                        "before": int(identity.sum()),
                        "after": int(identity.sum()),
                        "removed": 0,
                    },
                    "garment": {
                        "before": int(garment.sum()),
                        "after": int(garment.sum()),
                        "removed": 0,
                    },
                },
                "overlap_policy": "no_optional_parts_no_op",
                "conditioned_parts": [],
            },
        )
    if len(part_masks) > 7:
        raise ValueError("추가 부위 참조는 최대 7개입니다.")
    names = set()
    for name, mask in part_masks:
        if (name != "hair" and not valid_part_name(name)) or name in names:
            raise ValueError("원본 위치 실험은 유효하고 중복 없는 부위 이름이 필요합니다.")
        names.add(name)
        selected = read_binary_mask(mask, size)
        if not selected.any() or np.any(selected & occupied):
            raise ValueError("귀·꼬리 영역이 비었거나 서로 겹칩니다. 자동 합치지 않습니다.")
        occupied |= selected
    new_identity, new_garment = identity & ~occupied, garment & ~occupied
    if not new_identity.any() or not new_garment.any():
        raise ValueError("부위 분리 후 얼굴·의상 영향 영역이 비었습니다. 검출 결과를 확인하세요.")
    return (Image.fromarray(new_identity.astype(np.uint8) * 255),
            Image.fromarray(new_garment.astype(np.uint8) * 255),
            {"mode": "source_regions_experiment", "pose_locked": False,
             "region_policy": SOURCE_LAYOUT, "requires_input_review": True,
             "semantic_accuracy_verified": False,
             "identity_overlap_removed": int((identity & occupied).sum()),
             "garment_overlap_removed": int((garment & occupied).sum()),
             "area_audit": {
                 "identity": {"before": int(identity.sum()), "after": int(new_identity.sum()),
                              "removed": int((identity & occupied).sum())},
                 "garment": {"before": int(garment.sum()), "after": int(new_garment.sum()),
                             "removed": int((garment & occupied).sum())}},
             "overlap_policy": "extra_parts_priority_preserved",
             "conditioned_parts": [name for name, _ in part_masks]})


@dataclass(frozen=True)
class ReferenceConditions:
    conditions: tuple[ReferenceCondition, ...]
    layout_policy: str

    def __post_init__(self):
        if not isinstance(self.conditions, tuple) or not 2 <= len(self.conditions) <= 10:
            raise ValueError(
                "부위 조건은 얼굴·의상·헤어와 최대 7개 추가 부위여야 합니다.")
        names = tuple(c.name for c in self.conditions)
        if names[:2] != ("identity", "garment") or len(set(names)) != len(names):
            raise ValueError("부위 조건 순서 또는 이름 중복 오류")
        if self.layout_policy not in (SOURCE_LAYOUT, DISJOINT_LAYOUT):
            raise ValueError("등록되지 않은 배치 정책")
        size = self.conditions[0].region_size
        for condition in self.conditions:
            if (not isinstance(condition.rgb_bytes, bytes) or not isinstance(condition.region_bytes, bytes)
                    or not isinstance(condition.rgb_size, tuple) or not isinstance(condition.region_size, tuple)
                    or condition.region_size != size):
                raise ValueError("조건은 동일 좌표계의 불변 바이트여야 합니다.")
            with condition.image() as rgb, condition.region() as mask:
                ReferenceCondition.capture(condition.name, rgb, mask, condition.scale)

    def fingerprint(self):
        payload = {"layout_policy": self.layout_policy, "conditions": [c.record() for c in self.conditions]}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def materialize(self):
        """독립 조건을 새 객체로 조립한다. 반환 객체를 수정해도 원본 조건은 유지된다."""
        masks, references = [], []
        try:
            masks = [c.region() for c in self.conditions]
            layout = {}
            if self.layout_policy == SOURCE_LAYOUT:
                head, garment, layout = plan_source_regions(masks[0], masks[1],
                    [(c.name, m) for c, m in zip(self.conditions[2:], masks[2:])])
                masks[0].close()
                masks[1].close()
                masks[0], masks[1] = head, garment
            else:
                occupied = np.zeros((masks[0].height, masks[0].width), dtype=bool)
                for mask in masks:
                    selected = read_binary_mask(mask, masks[0].size)
                    if np.any(occupied & selected):
                        raise ValueError("검토된 부위 조건 영역이 서로 겹칩니다.")
                    occupied |= selected
            for condition, mask in zip(self.conditions, masks):
                references.append(PartReference(condition.name, condition.image(), mask, condition.scale))
            return tuple(references), layout
        except BaseException:
            for ref in references:
                ref.rgb.close()
            for mask in masks:
                mask.close()
            raise

    def validate_materialized(self, inputs):
        refs, _ = self.materialize()
        try:
            actual = [("identity", inputs.identity, inputs.identity_mask, None),
                      ("garment", inputs.garment, inputs.garment_mask, None),
                      *([] if getattr(inputs, "hair_reference", None) is None
                        else [("hair", inputs.hair_reference,
                               inputs.hair_mask,
                               inputs.hair_reference_scale)]),
                      *[(r.name, r.rgb, r.region, r.scale) for r in inputs.extra_references]]
            if len(actual) != len(refs):
                raise ValueError("독립 조건과 생성 참조 개수가 다릅니다.")
            actual_by_name = {item[0]: item for item in actual}
            if len(actual_by_name) != len(actual) or set(actual_by_name) != {ref.name for ref in refs}:
                raise ValueError("독립 조건 부위 이름 누락 또는 중복")
            for expected in refs:
                name, rgb, mask, scale = actual_by_name[expected.name]
                if (expected.name != name or expected.rgb.mode != rgb.mode or expected.rgb.size != rgb.size
                        or expected.rgb.tobytes() != rgb.tobytes() or expected.region.mode != mask.mode
                        or expected.region.size != mask.size or expected.region.tobytes() != mask.tobytes()
                        or expected.scale != scale):
                    raise ValueError("독립 조건과 미리보기/생성 참조가 다릅니다. 다시 준비하고 승인하세요.")
        finally:
            for ref in refs:
                ref.close()

    def record(self):
        return {"version": "independent_reference_conditions_v1", "fingerprint": self.fingerprint(),
                "layout_policy": self.layout_policy, "source_conditions_mutated": False,
                "layout_changed": False, "generation_isolated": False,
                "conditions": [c.record() for c in self.conditions]}

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        for condition in self.conditions:
            with condition.image() as rgb, condition.region() as region:
                rgb.save(directory / (condition.name + "_reference.png"))
                region.save(directory / (condition.name + "_source_region.png"))
        (directory / "conditions.json").write_text(
            json.dumps(self.record(), ensure_ascii=False, indent=2), encoding="utf-8")


def source_part_conditions(source, detected, generation_region_settings=None, audit=None):
    """RGB는 원본 마스크로 자르고 생성 영역은 별도 정책으로 만들 수 있다."""
    from genai_lab.generation_region_strengthening import strengthen_generation_region

    if audit is not None and not isinstance(audit, dict):
        raise ValueError("생성용 영역 보강 기록은 dict여야 합니다.")
    result = []
    for part in detected:
        if part.target_region is not None:
            raise ValueError("이미 지정된 생성 좌표를 원본 위치로 덮어쓰지 않습니다.")
        with crop_reference(source, part.source_mask) as rgb:
            generation_region, record = strengthen_generation_region(
                part.name, part.source_mask, generation_region_settings)
            try:
                result.append(ReferenceCondition.capture(
                    part.name, rgb, generation_region, part.scale))
            finally:
                generation_region.close()
        if audit is not None:
            audit[part.name] = record
    return tuple(result)
