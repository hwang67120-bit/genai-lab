"""Visible 2D proportions and explicit review; no model loading or body inference.

A reviewed head rectangle must exclude hair, ears and horns. The bottom marker
is a visible foot sole, never an ankle substituted for a foot. These are image
projection measurements, not reconstructed anatomy or physical height.
"""
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name}: 유한한 숫자가 필요합니다.')
    return float(value)


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('이미지 SHA-256이 필요합니다.')


def image_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class BodyProportionMeasurement:
    image_sha256: str
    heads_tall: float | None
    shoulder_to_head_width: float | None
    geometry_reviewed: bool = False
    issues: tuple[str, ...] = ()
    method: str = 'visible_head_and_sole_v1'
    head_box: tuple[float, float, float, float] | None = None
    sole_y: float | None = None
    shoulders: tuple[tuple[float, float], tuple[float, float]] | None = None

    def __post_init__(self):
        _digest(self.image_sha256)
        if type(self.geometry_reviewed) is not bool:
            raise ValueError('영역 확인 여부는 bool이어야 합니다.')
        if self.method != 'visible_head_and_sole_v1':
            raise ValueError('발목 지표를 등신 수로 사용할 수 없습니다.')
        for name in ('heads_tall', 'shoulder_to_head_width'):
            value = getattr(self, name)
            if value is not None and _number(value, name) <= 0:
                raise ValueError('비율은 양수여야 합니다.')
        if self.heads_tall is not None and self.heads_tall < 1:
            raise ValueError('전체 높이는 확인한 머리 높이보다 작을 수 없습니다.')
        if not isinstance(self.issues, tuple) or any(not isinstance(x, str) for x in self.issues):
            raise ValueError('측정 문제는 문자열 tuple로 기록합니다.')


def measure_visible_proportions(image_file, *, head_box, sole_y=None,
                                shoulders=None, geometry_reviewed=False):
    """Measure explicit markers on this file. Missing/occluded markers stay None.

    geometry_reviewed affirms a consistent head definition, visible sole and
    shoulder markers, and a sufficiently upright/front view for this comparison.
    Callers must never set it from detector confidence alone.
    """
    from PIL import Image
    from io import BytesIO
    data = Path(image_file).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    with Image.open(BytesIO(data)) as image:
        width, height = image.size
    if head_box is None:
        return BodyProportionMeasurement(digest, None, None, geometry_reviewed,
                                         ('head_region_missing',))
    if len(head_box) != 4:
        raise ValueError('머리 영역에는 좌표 4개가 필요합니다.')
    x0, y0, x1, y1 = (_number(v, 'head_box') for v in head_box)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('머리 영역이 이미지 밖이거나 비어 있습니다.')
    issues = []
    heads = shoulder_ratio = None
    bottom = points = None
    if sole_y is None:
        issues.append('foot_sole_missing_or_occluded')
    else:
        bottom = _number(sole_y, 'sole_y')
        if not y1 <= bottom <= height:
            raise ValueError('발끝은 머리 아래이며 이미지 안에 있어야 합니다.')
        heads = (bottom - y0) / (y1 - y0)
    if shoulders is None:
        issues.append('shoulders_missing_or_occluded')
    else:
        if len(shoulders) != 2 or any(len(p) != 2 for p in shoulders):
            raise ValueError('어깨 좌표는 점 2개가 필요합니다.')
        points = tuple(tuple(_number(v, 'shoulder') for v in p) for p in shoulders)
        if any(not (0 <= x < width and 0 <= y < height) for x, y in points):
            raise ValueError('어깨 좌표가 이미지 밖입니다.')
        span = math.dist(*points)
        if span == 0:
            raise ValueError('어깨 좌표가 겹칩니다.')
        shoulder_ratio = span / (x1 - x0)
    return BodyProportionMeasurement(digest, heads, shoulder_ratio,
                                    geometry_reviewed, tuple(issues),
                                    head_box=(x0, y0, x1, y1), sole_y=bottom, shoulders=points)


def suggest_proportions_from_pose(joints, head_box, image_size, *, confidence=0.30):
    """Inspection-only detector proxy. Never produces a confirmed reference.

    Existing DWPose and head detections may be supplied without another model
    load. The returned ankle ratio deliberately has a different name/unit scope.
    """
    threshold = _number(confidence, 'confidence')
    if not 0 <= threshold <= 1:
        raise ValueError('신뢰도 문턱은 0~1입니다.')
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError('이미지 크기는 양수여야 합니다.')
    result = dict(status='needs_review', heads_tall=None,
                  head_top_to_ankles_per_head_height=None,
                  shoulder_to_head_box_width=None,
                  reasons=['head_region_not_reviewed', 'foot_sole_not_detected',
                           'shoulders_may_be_inferred_under_clothes'])
    if head_box is None:
        result['reasons'].append('head_missing')
        return result
    if len(head_box) != 4:
        raise ValueError('머리 상자 좌표는 4개입니다.')
    x0, y0, x1, y1 = (_number(v, 'head_box') for v in head_box)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('머리 상자 범위가 잘못됐습니다.')
    points = {}
    seen = set()
    for joint in joints:
        if joint.joint_name in seen:
            raise ValueError('중복 관절입니다.')
        seen.add(joint.joint_name)
        score = _number(joint.confidence_score, 'confidence_score')
        x, y = _number(joint.x, 'x'), _number(joint.y, 'y')
        if not 0 <= score <= 1:
            raise ValueError('관절 신뢰도 범위가 잘못됐습니다.')
        if score >= threshold and 0 <= x < width and 0 <= y < height:
            points[joint.joint_name] = (x, y)
    ankles = [points.get(s + '_ankle') for s in ('left', 'right')]
    if all(p is not None for p in ankles):
        bottom = sum(p[1] for p in ankles) / 2
        if bottom >= y1:
            result['head_top_to_ankles_per_head_height'] = (bottom-y0)/(y1-y0)
    shoulders = [points.get(s + '_shoulder') for s in ('left', 'right')]
    if all(p is not None for p in shoulders):
        result['shoulder_to_head_box_width'] = math.dist(*shoulders)/(x1-x0)
    return result


@dataclass(frozen=True)
class BodyProportionReference:
    measurement: BodyProportionMeasurement
    face_sha256: str
    confirmed: bool = False
    basis: str = "reviewed_measurement"

    def __post_init__(self):
        if not isinstance(self.measurement, BodyProportionMeasurement):
            raise TypeError('원본 비율 측정값이 필요합니다.')
        _digest(self.face_sha256)
        if type(self.confirmed) is not bool:
            raise ValueError('확인 상태는 bool이어야 합니다.')
        if self.basis not in ('reviewed_measurement', 'user_declared_target'):
            raise ValueError('비율 근거는 확인한 측정 또는 사용자 지정 목표여야 합니다.')
        if self.basis == 'user_declared_target' and self.measurement.geometry_reviewed:
            raise ValueError('사용자 지정 목표를 실측 확인으로 기록할 수 없습니다.')

    def require_confirmed(self):
        if not self.confirmed:
            raise ValueError('사용할 비율 조건을 먼저 확인해야 합니다.')
        if self.basis == 'reviewed_measurement' and not self.measurement.geometry_reviewed:
            raise ValueError('머리·발끝 기준과 원본 비율을 먼저 확인해야 합니다.')
        if self.measurement.heads_tall is None:
            raise ValueError('발끝이 확인되지 않아 등신 수를 전달할 수 없습니다.')

    def prompt_phrases(self):
        self.require_confirmed()
        values = [f'body proportions of {self.measurement.heads_tall:.3g} head lengths tall']
        if self.measurement.shoulder_to_head_width is not None:
            values.append(f'shoulders {self.measurement.shoulder_to_head_width:.3g} head widths wide')
        return tuple(values)


def compare_body_proportions(reference, observed, *, relative_tolerance=0.05):
    """Comparison evidence only; tolerance is provisional, never product approval."""
    reference.require_confirmed()
    if not isinstance(observed, BodyProportionMeasurement):
        raise TypeError('생성 결과의 비율 측정값이 필요합니다.')
    tolerance = _number(relative_tolerance, 'relative_tolerance')
    if not 0 <= tolerance < 1:
        raise ValueError('비교 허용 오차는 0 이상 1 미만입니다.')
    fields = {}
    for name in ('heads_tall', 'shoulder_to_head_width'):
        target, actual = getattr(reference.measurement, name), getattr(observed, name)
        if target is None:
            continue
        error = None if actual is None else (actual-target)/target
        fields[name] = dict(target=target, observed=actual, relative_error=error,
                            within_tolerance=None if error is None else abs(error) <= tolerance)
    complete = observed.geometry_reviewed and all(v['observed'] is not None for v in fields.values())
    status = ('within_tolerance' if all(v['within_tolerance'] for v in fields.values())
              else 'outside_tolerance') if complete else 'unmeasurable_or_unreviewed'
    return dict(status=status, fields=fields, observed=asdict(observed), target_basis=reference.basis,
                relative_tolerance=tolerance, tolerance_validated=False,
                user_review_required=True, product_approved=False)
