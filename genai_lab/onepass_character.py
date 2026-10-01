"""Character tags and the adopted C5 segmentation + C6 near-head arm guard.

Source: outputs/face-crop-c5-20260928/{methods_c5,methods_c6,decision}.md.
Accepted crops require user preview. Missing detections never trigger a fallback.
"""
from dataclasses import dataclass, field
import math
import re
from PIL import Image

from genai_lab.onepass_input_settings import OnePassInputSettings
from genai_lab.onepass_pose import normalize_scores, validate_joints
from genai_lab.reference_tag_policy import gender_tag_kind
from genai_lab.onepass_garment_vocabulary import filter_pocket_tags

POSE_ALLOW = (
    'hands_in_pockets', 'hand_in_pocket', 'holding_gun', 'holding_weapon', 'gun',
    'rifle', 'aiming', 'kneeling', 'on_one_knee', 'squatting', 'sitting',
    'from_above', 'from_below', 'holding_phone', 'cellphone', 'looking_at_phone',
    'arms_up', 'hands_up', 'arm_up', 'hand_on_own_head', 'hands_on_own_head',
    'crossed_arms', 'hand_up',
)
CHARACTER_PATTERN = re.compile(
    r'^(.*_hair|.*_eyes|dark_skin|dark-skinned_.*|tan|muscular|muscular_male|'
    r'muscular_female|abs|tall|thick_thighs|flat_chest|small_breasts|medium_breasts|'
    r'large_breasts|pectorals|broad_shoulders|beard|facial_hair|mustache|stubble|'
    r'animal_ears|tail|.*_ears|.*_tail|old|old_man)$')
CHARACTER_EXCLUDED = frozenset((
    'hair_ornament', 'hair_between_eyes', 'closed_eyes', 'hair_over_one_eye',
    'hair_bun', 'hair_ribbon', 'hairband', 'hairclip', 'one_eye_closed',
    'hair_flower', 'looking_at_viewer',
))


@dataclass(frozen=True)
class CharacterTags:
    appearance: tuple[tuple[str, float], ...]
    gender_candidates: tuple[tuple[str, float], ...]
    excluded: tuple[tuple[str, float], ...]


def select_character_tags(scores, settings=OnePassInputSettings()):
    """Detector gender is a candidate record only; selection is stage 1's job."""
    normalized = normalize_scores(scores)
    selected, genders, excluded = [], [], []
    for tag, score in sorted(normalized.items(), key=lambda item: -item[1]):
        if score < settings.checks.tag_threshold:
            continue
        if gender_tag_kind(tag):
            genders.append((tag, score))
        elif CHARACTER_PATTERN.fullmatch(tag) and tag not in CHARACTER_EXCLUDED:
            selected.append((tag, score))
        else:
            excluded.append((tag, score))
    return CharacterTags(tuple(selected), tuple(genders), tuple(excluded))


def select_pose_tags(scores, approved_garment_tags, settings=OnePassInputSettings()):
    """Keep allowlisted pose tags; pocket instructions require approved pocket garments."""
    scores = normalize_scores(scores)
    selected = tuple(t for t in POSE_ALLOW if scores.get(t, 0) >= settings.checks.tag_threshold)
    retained, _ = filter_pocket_tags(selected, approved_garment_tags)
    return tuple(sorted(((t, scores[t]) for t in retained), key=lambda item: -item[1]))



def measure_slimness(joints, settings=OnePassInputSettings()):
    """Return observed shoulder/neck-hip ratio, or None when unmeasurable.

    ps/ns/off does not change the observation; stage 3 decides whether to use it.
    """
    validate_joints(joints)
    index = {j.joint_name: j for j in joints if j.confidence_score >= settings.checks.confidence}
    required = ('neck', 'left_hip', 'right_hip', 'left_shoulder', 'right_shoulder')
    if any(n not in index for n in required):
        return {'shoulder_torso_ratio': None, 'slim': None, 'reason': 'missing_joints'}
    point = lambda n: (index[n].x, index[n].y)
    hips = tuple((a + b) / 2 for a, b in zip(point('left_hip'), point('right_hip')))
    torso = math.dist(point('neck'), hips)
    if torso == 0:
        return {'shoulder_torso_ratio': None, 'slim': None, 'reason': 'zero_torso'}
    ratio = math.dist(point('left_shoulder'), point('right_shoulder')) / torso
    return {'shoulder_torso_ratio': ratio, 'slim': ratio < settings.character.slim_ratio,
            'threshold': settings.character.slim_ratio, 'reason': None}


@dataclass(frozen=True)
class HeadDetection:
    box: tuple[float, float, float, float]
    score: float


@dataclass(frozen=True)
class FaceCropReview:
    image: Image.Image | None
    box: tuple[int, int, int, int] | None
    head_box: tuple[float, float, float, float] | None
    arm_points: tuple[str, ...]
    status: str
    reason: str | None
    method: str
    requires_user_review: bool = True
    mask: Image.Image | None = None
    details: dict = field(default_factory=dict)

    def close(self):
        if self.image is not None:
            self.image.close()
        if self.mask is not None:
            self.mask.close()


def prepare_c6_crop(source, heads, joints, settings=OnePassInputSettings(), *,
                    foreground_mask=None, face_points=(), face_scores=()):
    """C6 from recorded or live detections in *source* pixel coordinates.

    foreground_mask must be the isnet-anime result, not a head-box proxy.
    Invalid data raises; unavailable/low-confidence detections return rejection.
    No image resizing, inferred landmarks, model loading, or silent manual fallback.
    """
    import cv2
    import numpy as np

    validate_joints(joints)
    cfg = settings.character
    threshold = cfg.face_confidence
    details = {'notes': [], 'failures': []}
    def reject(reason, head_box=None, arms=()):
        details['failures'].append(reason)
        return FaceCropReview(None, None, head_box, arms, 'reject', reason, 'c6', details=details)

    heads = tuple(heads)
    for head in heads:
        if (len(head.box) != 4 or not all(math.isfinite(v) for v in (*head.box, head.score))
                or not 0 <= head.score <= 1 or head.box[2] <= head.box[0] or head.box[3] <= head.box[1]):
            raise ValueError('머리 검출 상자/점수 형식 오류')
    heads = tuple(h for h in heads if h.score >= cfg.head_confidence)
    details['head_count'] = len(heads)
    if not heads:
        return reject('머리 검출 없음')
    index = {j.joint_name: j for j in joints}
    nose = index.get('nose')
    containing = [h for h in heads if nose is not None and nose.confidence_score >= threshold
                  and h.box[0] <= nose.x <= h.box[2] and h.box[1] <= nose.y <= h.box[3]]
    head = max(containing or heads, key=lambda h: h.score)
    details['choice'] = '코 포함 상자 중 최고 신뢰도' if containing else '전체 최고 신뢰도'
    x0, y0, x1, y1 = head.box
    w, h = x1-x0, y1-y0
    box = (max(0, math.floor(x0-cfg.head_margin*w)), max(0, math.floor(y0-cfg.head_margin*h)),
           min(source.width, math.ceil(x1+cfg.head_margin*w)), min(source.height, int(y1)))
    details.update(head_box=head.box, head_conf=head.score, expanded_box=box)
    if box[2] <= box[0] or box[3] <= box[1]:
        return reject('머리 상자가 이미지 밖', head.box)
    zone = (max(0, box[0]-cfg.arm_zone_side*w), max(0, box[1]-cfg.arm_zone_top*h),
            min(source.width, box[2]+cfg.arm_zone_side*w), min(source.height, box[3]+cfg.arm_zone_bottom*h))
    arm_names = ('right_elbow', 'right_wrist', 'left_elbow', 'left_wrist')
    arms = tuple(n for n in arm_names if n in index and index[n].confidence_score >= threshold
                 and zone[0] <= index[n].x <= zone[2] and zone[1] <= index[n].y <= zone[3])
    details['zone'] = zone
    # Reject before producing a crop; absence of arm keypoints is not proof of cleanliness.
    if arms:
        return reject('팔이 머리 근처 — 팔이 머리를 가리지 않는 캐릭터 이미지를 넣어 주세요', head.box, arms)
    fp, fs = np.asarray(face_points, dtype=float), np.asarray(face_scores, dtype=float)
    if fp.size == 0 and fs.size == 0:
        return reject('얼굴 68점 없음', head.box)
    if fp.shape != (68, 2) or fs.shape != (68,) or not np.isfinite(fp).all() or not np.isfinite(fs).all():
        raise ValueError('얼굴 68점/신뢰도 형식 오류')
    details['face_median'] = float(np.median(fs))
    if details['face_median'] < threshold:
        return reject('얼굴 68점 신뢰도 중앙값<문턱값', head.box)
    if foreground_mask is None:
        return reject('isnet-anime 전경 마스크 없음', head.box)
    if foreground_mask.size != source.size:
        raise ValueError('전경 마스크와 입력 좌표 크기가 다릅니다.')
    with foreground_mask.convert('L') as gray:
        foreground = np.asarray(gray) >= cfg.foreground_threshold
    keep = np.zeros((source.height, source.width), dtype=bool)
    keep[box[1]:box[3], box[0]:box[2]] = foreground[box[1]:box[3], box[0]:box[2]]
    hull = np.zeros(keep.shape, dtype=np.uint8)
    valid = fs >= threshold
    if valid.sum() >= 3:
        contour = cv2.convexHull(fp[valid].astype(np.float32)).reshape(-1, 2)
        cv2.fillConvexPoly(hull, np.rint(contour).astype(np.int32), 255)
    arm_mask = np.zeros(keep.shape, dtype=np.uint8)
    shoulders = [index.get(n) for n in ('right_shoulder', 'left_shoulder')]
    width = math.dist((shoulders[0].x, shoulders[0].y), (shoulders[1].x, shoulders[1].y)) if all(shoulders) else 0
    details.update(arm_thickness=0, arm_segments=[])
    if width > 0 and all(j.confidence_score >= threshold for j in shoulders):
        thickness = max(1, math.ceil(cfg.arm_band_ratio*width))
        details['arm_thickness'] = thickness
        for n1, n2 in (('right_shoulder','right_elbow'), ('right_elbow','right_wrist'),
                       ('left_shoulder','left_elbow'), ('left_elbow','left_wrist')):
            j1, j2 = index.get(n1), index.get(n2)
            if j1 is None or j2 is None or min(j1.confidence_score, j2.confidence_score) < threshold:
                continue
            p1, p2 = (tuple(np.rint((j.x, j.y)).astype(int)) for j in (j1, j2))
            cv2.line(arm_mask, p1, p2, 255, thickness=thickness)
            for point in (p1, p2):
                cv2.circle(arm_mask, point, math.ceil(thickness/2), 255, -1)
            details['arm_segments'].append((n1, n2))
        keep &= ~((arm_mask > 0) & ~(hull > 0))
    else:
        details['notes'].append('팔 띠 없음(어깨 신뢰도/폭 미달)')
    if fs[8] >= threshold:
        cy = float(fp[8, 1] + cfg.chin_margin*h)
        keep[max(0, math.floor(cy)+1):, :] = False
        details['chin_cut_y'] = cy
    else:
        details['notes'].append('턱 규칙 미적용(턱 신뢰도 미달)')
    ys, xs = np.where(keep)
    if not len(xs):
        return reject('보존 픽셀 0', head.box)
    margin = max(1, math.ceil(cfg.crop_margin*max(xs.max()-xs.min()+1, ys.max()-ys.min()+1)))
    crop_box = (max(0,int(xs.min())-margin), max(0,int(ys.min())-margin),
                min(source.width,int(xs.max())+1+margin), min(source.height,int(ys.max())+1+margin))
    with source.convert('RGB') as rgb:
        arr = np.array(rgb)
    arr[~keep] = 255
    with Image.fromarray(arr) as masked, masked.crop(crop_box) as crop:
        side = max(crop.size)
        preview = Image.new('RGB', (side, side), 'white')
        preview.paste(crop, ((side-crop.width)//2, (side-crop.height)//2))
    details['kept_pixels'] = int(keep.sum())
    return FaceCropReview(preview, crop_box, head.box, (), 'review', None, 'c6',
                          mask=Image.fromarray(keep.astype(np.uint8)*255), details=details)


def prepare_manual_face_crop(cropped_image):
    """Explicit user-provided clean crop; never a silent replacement for C6 failure."""
    if min(cropped_image.size) < 1:
        raise ValueError('수동 크롭 이미지가 비어 있습니다.')
    return FaceCropReview(cropped_image.convert('RGB'), None, None, (), 'review', None, 'manual')
