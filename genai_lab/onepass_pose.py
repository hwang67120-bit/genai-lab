"""Deterministic one-pass pose preprocessing; no generation or model loading.

An estimator supplies the *actual* person count as well as the existing DWPose
coordinates. A legacy selected-person result cannot prove K1 and is not silently
assumed to contain one person. Backend installation/GUI approval are separate.
"""
from dataclasses import dataclass
from itertools import combinations
import math
from typing import Callable

from PIL import Image

from genai_lab.onepass_input_settings import OnePassInputSettings, PoseCheckSettings, InputPolicies
from genai_lab.pose_estimation import (
    PoseJointCoordinateCandidate, PoseEstimationApprovedInput, prepare_pose_control_input,
)
from scripts.pose_reference_runner import BODY_JOINT_NAMES

SIDES = ('left', 'right')
LIMBS = ('shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle')
REQUIRED = tuple(f'{s}_{n}' for s in SIDES for n in ('shoulder', 'hip', 'knee', 'ankle'))
DIRECTION_TAGS = ('looking_at_viewer', 'profile', 'from_side', 'from_behind', 'from_above')


@dataclass(frozen=True)
class Finding:
    code: str
    values: dict


@dataclass(frozen=True)
class PoseAssessment:
    findings: tuple[Finding, ...]
    torso_length: float | None
    guessed_joints: tuple[str, ...]
    direction_scores: dict[str, float]
    nonfrontal: bool
    high_angle: bool


@dataclass(frozen=True)
class InputDecision:
    status: str
    rejects: tuple[str, ...]
    warnings: tuple[str, ...]
    choices: tuple[str, ...]


def apply_input_policy(assessment: PoseAssessment, policies=InputPolicies()) -> InputDecision:
    policy = dict(policies.rules)
    rejects = tuple(dict.fromkeys(f.code for f in assessment.findings if policy[f.code] == 'reject'))
    warns = tuple(dict.fromkeys(f.code for f in assessment.findings if policy[f.code] == 'warn'))
    if assessment.high_angle:
        if policies.high_angle == 'out_of_scope':
            rejects += ('HIGH_ANGLE',)
        else:
            warns += ('HIGH_ANGLE',)
    status = 'reject' if rejects else ('warn_user_choice' if warns else 'pass')
    return InputDecision(status, rejects, warns,
                         ('choose_other_image',) if rejects else ('proceed', 'choose_other_image'))


def joints_from_records(records) -> tuple[PoseJointCoordinateCandidate, ...]:
    joints = tuple(PoseJointCoordinateCandidate(
        joint_name=str(r['joint_name']), x=float(r['x']), y=float(r['y']),
        confidence_score=float(r['confidence_score']), detected=bool(r['detected']),
    ) for r in records)
    validate_joints(joints)
    return joints


def validate_joints(joints):
    names = [j.joint_name for j in joints]
    if len(set(names)) != len(names) or any(n not in BODY_JOINT_NAMES for n in names):
        raise ValueError('관절 이름 중복/알 수 없는 이름')
    if any(not all(math.isfinite(v) for v in (j.x, j.y, j.confidence_score))
           or not 0 <= j.confidence_score <= 1 for j in joints):
        raise ValueError('관절 좌표/신뢰도는 유한한 값이어야 합니다.')


def normalize_scores(scores):
    result = {}
    for tag, score in scores.items():
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('태그 점수는 유한한 0~1 값이어야 합니다.')
        key = tag.strip().lower().replace(' ', '_')
        result[key] = max(float(score), result.get(key, 0.0))
    return result


def assess_pose(joints, person_count, tag_scores, settings=PoseCheckSettings()) -> PoseAssessment:
    """K1..K7, guessed joints and direction facts only; no policy decisions.

    K geometry uses the shoulder midpoint to hip midpoint (e2e check), while
    character slimness separately uses neck to hip midpoint (e2e ratios).
    GUESS is [0.30, 0.50), matching the recorded executable, not rounded scores.
    """
    validate_joints(joints)
    if type(person_count) is not int or person_count < 0:
        raise ValueError('실제 검출 인원 수가 필요합니다. 1명으로 가정하지 않습니다.')
    scores = normalize_scores(tag_scores)
    index = {j.joint_name: j for j in joints}
    def valid(*names):
        return all(n in index and index[n].confidence_score >= settings.confidence for n in names)
    def point(n):
        return index[n].x, index[n].y
    def distance(a, b):
        return math.dist(point(a), point(b))
    def midpoint(a, b):
        return tuple((x + y) / 2 for x, y in zip(point(a), point(b)))
    findings = []
    def flag(code, **values):
        findings.append(Finding(code, values))
    if person_count != 1:
        flag('K1', person_count=person_count)
    missing = [n for n in REQUIRED if not valid(n)]
    if missing:
        flag('K2', missing=missing)
    torso = None
    if valid('left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'):
        torso = math.dist(midpoint('left_shoulder', 'right_shoulder'), midpoint('left_hip', 'right_hip'))
        if torso <= 0:
            flag('K6', reason='zero_torso', torso_length=torso)
        else:
            segments = {'upper_arm': ('shoulder', 'elbow'), 'forearm': ('elbow', 'wrist'),
                        'thigh': ('hip', 'knee'), 'shank': ('knee', 'ankle')}
            for seg, (a, b) in segments.items():
                lengths = {}
                for side in SIDES:
                    if valid(f'{side}_{a}', f'{side}_{b}'):
                        length = distance(f'{side}_{a}', f'{side}_{b}')
                        lengths[side] = length
                        if length / torso > getattr(settings, seg):
                            flag('K3', segment=f'{side}_{seg}', ratio=length / torso,
                                 threshold=getattr(settings, seg))
                if len(lengths) == 2:
                    low, high = min(lengths.values()), max(lengths.values())
                    ratio = high / low if low else None
                    if low == 0 or ratio > settings.bilateral_length_ratio:
                        flag('K5', segment=seg, ratio=ratio, lengths=lengths,
                             threshold=settings.bilateral_length_ratio)
            for side in SIDES:
                for a, b in combinations(('hip', 'knee', 'ankle'), 2):
                    if valid(f'{side}_{a}', f'{side}_{b}'):
                        ratio = distance(f'{side}_{a}', f'{side}_{b}') / torso
                        if ratio < settings.collapse_ratio:
                            flag('K4', side=side, pair=[a, b], ratio=ratio, threshold=settings.collapse_ratio)
            for name in ('shoulder', 'hip'):
                ratio = distance('left_' + name, 'right_' + name) / torso
                if ratio < settings.torso_width_ratio:
                    flag('K6', part=name, ratio=ratio, threshold=settings.torso_width_ratio)
            if valid('left_knee', 'right_knee', 'left_ankle', 'right_ankle'):
                knee = distance('left_knee', 'right_knee') / torso
                ankle = distance('left_ankle', 'right_ankle') / torso
                if max(knee, ankle) < settings.overlap_ratio:
                    flag('K7', knee_ratio=knee, ankle_ratio=ankle, threshold=settings.overlap_ratio)
    guessed = tuple(j.joint_name for j in joints if j.joint_name in
                    {f'{s}_{n}' for s in SIDES for n in LIMBS} and j.detected
                    and settings.confidence <= j.confidence_score < settings.guessed_confidence_upper)
    if len(guessed) >= settings.guessed_count:
        flag('GUESS', joints=list(guessed), count=len(guessed))
    direction = {t: scores.get(t, 0.0) for t in DIRECTION_TAGS}
    nonfront = (direction['looking_at_viewer'] < settings.tag_threshold or
                any(direction[t] >= settings.tag_threshold for t in DIRECTION_TAGS[1:]))
    if nonfront:
        flag('FACE_DIR', scores=direction, threshold=settings.tag_threshold)
    return PoseAssessment(tuple(findings), torso, guessed, direction, nonfront,
                          direction['from_above'] >= settings.tag_threshold)


def normalize_pose_image(source, joints, settings=OnePassInputSettings()):
    """Return padded body-bbox image and exact rounded box; never mutate source."""
    validate_joints(joints)
    points = [(j.x, j.y) for j in joints if j.detected and j.confidence_score >= settings.checks.confidence]
    if len(points) < 2:
        raise ValueError('정규화에 필요한 원본 관절이 부족합니다.')
    x0, y0 = map(min, zip(*points))
    x1, y1 = map(max, zip(*points))
    height = y1 - y0
    if height <= 0:
        raise ValueError('원본 관절 bbox 높이가 0입니다.')
    cfg = settings.normalization
    left, top = x0 - cfg.side_margin * height, y0 - cfg.top_margin * height
    right, bottom = x1 + cfg.side_margin * height, y1 + cfg.bottom_margin * height
    width, height = right - left, bottom - top
    cx, cy = (left + right) / 2, (top + bottom) / 2
    aspect = cfg.width / cfg.height
    if width / height < aspect:
        width = height * aspect
    else:
        height = width / aspect
    box = tuple(round(v) for v in (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2))
    size = box[2] - box[0], box[3] - box[1]
    if min(size) < 1 or size[0] * size[1] > cfg.maximum_canvas_pixels:
        raise ValueError('정규화 캔버스가 입력 크기 제한을 벗어났습니다.')
    with Image.new('RGB', size, (cfg.background,) * 3) as canvas:
        with source.convert('RGB') as rgb:
            canvas.paste(rgb, (-box[0], -box[1]))
        return canvas.resize((cfg.width, cfg.height), Image.Resampling.LANCZOS), box


def prepare_onepass_control(control_map, joints, settings=OnePassInputSettings()):
    """Reuse the approved resize/padding function, then RGB -> BGR per model card."""
    validate_joints(joints)
    if len(joints) != 18:
        raise ValueError('제어 지도에는 몸 관절 기록 18개가 필요합니다.')
    count = sum(j.detected for j in joints)
    approved = PoseEstimationApprovedInput(control_map, tuple(joints), count, 18 - count,
                                          settings.checks.confidence, ())
    cfg = settings.normalization
    prepared = prepare_pose_control_input(approved, cfg.width, cfg.height)
    try:
        channels = prepared.control_map_image.split()
        try:
            return Image.merge('RGB', tuple(reversed(channels)))
        finally:
            for channel in channels:
                channel.close()
    finally:
        prepared.close()


@dataclass(frozen=True)
class PoseObservation:
    joints: tuple[PoseJointCoordinateCandidate, ...]
    person_count: int
    overlay: Image.Image
    control_map: Image.Image
    # Selected person, pixel coordinates; empty means face landmarks unavailable.
    face_points: tuple[tuple[float, float], ...] = ()
    face_scores: tuple[float, ...] = ()

    def close(self):
        self.overlay.close()
        self.control_map.close()


class PoseInputError(ValueError):
    """Input cannot be normalized; carry measured rule evidence for the review UI."""
    def __init__(self, message, assessment, policies):
        super().__init__(message)
        self.assessment = assessment
        self.decision = apply_input_policy(assessment, policies)


@dataclass(frozen=True)
class PreparedPose:
    normalized_image: Image.Image
    overlay: Image.Image
    control_image: Image.Image
    crop_box: tuple[int, int, int, int]
    observation_joints: tuple[PoseJointCoordinateCandidate, ...]
    assessment: PoseAssessment
    decision: InputDecision

    def close(self):
        self.normalized_image.close()
        self.overlay.close()
        self.control_image.close()


def prepare_onepass_pose(source, *, estimate: Callable[[Image.Image], PoseObservation],
                         tag: Callable[[Image.Image], dict], settings=OnePassInputSettings()):
    """Two detections with externally supplied backends; no hidden model fallback.

    Geometry failure raises an input error. Rule rejection returns review evidence
    (including overlay) but never authorizes generation. The caller owns the result.
    """
    first = estimate(source)
    try:
        if type(first.person_count) is not int or first.person_count < 0:
            raise ValueError('실제 검출 인원 수가 필요합니다.')
        try:
            normalized, box = normalize_pose_image(source, first.joints, settings)
        except ValueError as error:
            assessment = assess_pose(first.joints, first.person_count, tag(source), settings.checks)
            raise PoseInputError(str(error), assessment, settings.policies) from error
    finally:
        first.close()
    second = None
    try:
        second = estimate(normalized)
        assessment = assess_pose(second.joints, second.person_count, tag(source), settings.checks)
        # Normalization must not silently turn a multi-person input into a single person.
        if first.person_count != 1:
            from dataclasses import replace
            assessment = replace(assessment, findings=(Finding('K1', {
                'person_count': first.person_count, 'stage': 'original'}),) + assessment.findings)
        control = prepare_onepass_control(second.control_map, second.joints, settings)
        return PreparedPose(normalized, second.overlay.copy(), control, box, second.joints,
                            assessment, apply_input_policy(assessment, settings.policies))
    except Exception:
        normalized.close()
        raise
    finally:
        if second is not None:
            second.close()
