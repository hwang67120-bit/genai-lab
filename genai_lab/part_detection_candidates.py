"""부위 검출만 수행한다. 의상/헤어 마스크와 배치 정책을 입력받지 않는다."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from genai_lab.regional_reference import valid_part_name


@dataclass(frozen=True)
class PartCandidates:
    name: str
    size: tuple[int, int]
    boxes: tuple
    detection_scores: tuple
    mask_scores: tuple
    masks: tuple[bytes, ...]

    def __post_init__(self):
        if not valid_part_name(self.name) or not isinstance(self.size, tuple) or len(self.size) != 2:
            raise ValueError("부위 검출 계약 이름/크기 오류")
        if any(type(v) is not int or v <= 0 for v in self.size):
            raise ValueError("부위 검출 계약 크기 오류")
        values = (self.boxes, self.detection_scores, self.mask_scores, self.masks)
        if not all(isinstance(v, tuple) for v in values) or not all(len(v) == len(self.boxes) for v in values):
            raise ValueError("부위 검출 계약 개수/불변 형식 오류")
        if len(self.boxes) > 8:
            raise ValueError("부위 검출 계약 후보 수 초과")
        for box, score, quality, raw in zip(*values):
            if (not isinstance(box, tuple) or len(box) != 4 or not np.isfinite(box).all()
                    or not np.isfinite(score) or not np.isfinite(quality)
                    or not 0 <= score <= 1):
                raise ValueError("부위 검출 계약 좌표/점수 오류")
            if not (0 <= box[0] < box[2] <= self.size[0] and 0 <= box[1] < box[3] <= self.size[1]):
                raise ValueError("부위 검출 계약 좌표 범위 오류")
            if not isinstance(raw, bytes) or len(raw) != self.size[0] * self.size[1]:
                raise ValueError("부위 검출 계약 마스크 길이 오류")
            if not np.isin(np.frombuffer(raw, dtype=np.uint8), (0, 1)).all():
                raise ValueError("부위 검출 계약 이진 마스크 오류")

    def array(self):
        return np.frombuffer(b"".join(self.masks), dtype=np.uint8).reshape(
            len(self.masks), self.size[1], self.size[0]).astype(bool)

    def fingerprint(self):
        record = (self.name, self.size, self.boxes, self.detection_scores, self.mask_scores,
                  tuple(hashlib.sha256(raw).hexdigest() for raw in self.masks))
        return hashlib.sha256(json.dumps(record, allow_nan=False).encode()).hexdigest()


def detect_part_candidates(backend, rgb, name, query, threshold, check):
    """원본 RGB와 검색어만 사용한다. 취소/시간 제한은 모델 호출 전후 검사한다."""
    check()
    boxes, scores = backend.detect(rgb, query, threshold)
    check()
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    scores = np.asarray(scores).reshape(-1)
    if len(scores) != len(boxes) or not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("귀·꼬리 검출 좌표/점수 오류")
    if len(boxes) > 8:
        raise ValueError(f"{name}: 과다 검출, 자동 확정 중단")
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, rgb.width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, rgb.height)
    valid = (scores >= threshold) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores = boxes[valid], scores[valid]
    quality, raw_masks = (), ()
    if len(boxes):
        masks, quality = backend.segment(rgb, boxes)
        check()
        masks, quality = np.asarray(masks), np.asarray(quality)
        if masks.shape != (len(boxes), 1, rgb.height, rgb.width) or quality.size != len(boxes):
            raise ValueError("SAM2 부위 마스크 크기 오류")
        if not np.isfinite(quality).all() or not np.isin(masks, (0, 1)).all():
            raise ValueError("SAM2 부위 마스크/점수 오류")
        raw_masks = tuple(raw.astype(np.uint8).tobytes() for raw in masks[:, 0])
        quality = tuple(float(v) for v in quality.reshape(-1))
    return PartCandidates(name, rgb.size, tuple(tuple(float(v) for v in box) for box in boxes),
                          tuple(float(v) for v in scores), quality, raw_masks)
