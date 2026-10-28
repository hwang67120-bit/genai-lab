"""CPU 측정 2차 방식이다. 픽셀을 관찰·비교하며 생성 조건은 바꾸지 않는다."""
import math
from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image, ImageDraw

def to_lab(rgb):
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32); L[..., 0] *= 100 / 255; L[..., 1:] -= 128; return L

def largest(mask):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))) if n > 1 else mask.astype(bool)

def kmeans3(lab_pixels):
    if len(lab_pixels) < 30: return None, None
    data = lab_pixels.astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    cv2.setRNGSeed(0)
    _, labels, centers = cv2.kmeans(data, 3, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    frac = np.bincount(labels.ravel(), minlength=3) / len(labels)
    return centers, frac

def measure_pixels(image, pose, alpha, head_box, *, use_arm_parts=False, handwear_mask=None):
    """현재 이미지 좌표로 측정한다. 팔 분할을 요청하면 전달된 마스크만 쓰며 추론은 하지 않는다."""
    if type(use_arm_parts) is not bool or (handwear_mask is not None and not use_arm_parts):
        raise ValueError("팔 마스크는 팔 분할 측정을 명시적으로 켠 경우에만 전달합니다.")
    rgb = np.asarray(image.convert("RGB")); h_img, w_img = rgb.shape[:2]
    mask = largest(np.asarray(alpha) > 127)
    out = {"size": [w_img, h_img], "notes": []}
    draw = {"lines": [], "boxes": []}
    if mask.shape != rgb.shape[:2] or not mask.any():
        out["notes"].append("empty_foreground"); return out, draw
    if head_box is None:
        out["notes"].append("no_head"); return out, draw
    bx0, by0, bx1, by1 = head_box; H = by1 - by0
    if not np.isfinite(head_box).all() or H <= 0 or bx1 <= bx0:
        out["notes"].append("invalid_head"); return out, draw
    out.update(head_box=list(head_box), H=H)
    J = {j['joint_name']: (j['x'], j['y']) for j in pose['joints'] if j['confidence_score'] >= 0.3}
    fp, fs = pose.get('face_points') or [], pose.get('face_scores') or []
    def fpt(i):
        return tuple(fp[i]) if i < len(fp) and i < len(fs) and fs[i] >= 0.3 else None
    nose = J.get('nose') or fpt(30)
    if nose is None or not (0 <= nose[0] < w_img and 0 <= nose[1] < h_img):
        out['notes'].append('no_nose'); return out, draw
    lab = to_lab(rgb)
    skin = sample_skin(lab, mask, nose, fpt, out)
    dE = np.linalg.norm(lab - skin, axis=2)
    skin_body = (dE < 20) & mask; skin_face = (dE < 14) & mask
    out['skin_lab'] = [round(float(v), 1) for v in skin]
    arm_region = prepare_arm_region(mask, handwear_mask, J) if use_arm_parts else None
    measure_body(mask, skin_body, J, H, out, draw, arm_region=arm_region)
    measure_hair(lab, mask, skin_face, skin, J, fpt, head_box, out, draw)
    draw['mask'] = mask
    draw['overlay'] = draw_overlay(rgb, mask, skin_face, head_box, draw)
    return out, draw


def sample_skin(lab, mask, nose, fpt, out):
    patches = []
    h_img, w_img = mask.shape
    for a, b in ((2, 31), (14, 35)):
        pa, pb = fpt(a), fpt(b)
        if pa and pb:
            cx, cy = int(round((pa[0] + pb[0]) / 2)), int(round((pa[1] + pb[1]) / 2))
            if 0 <= cx < w_img and 0 <= cy < h_img and mask[cy, cx]:
                patches.append(lab[max(0,cy-3):cy+4,max(0,cx-3):cx+4].reshape(-1,3))
    if len(patches) == 2:
        return np.median(np.concatenate(patches), axis=0)
    nx, ny = int(round(nose[0])), int(round(nose[1]))
    nx, ny = min(nx,w_img-1), min(ny,h_img-1)
    out['notes'].append('skin_fallback')
    return np.median(lab[max(0,ny-3):ny+4,max(0,nx-3):nx+4].reshape(-1,3), axis=0)


def seg_dist(p, a, b):
    a, b, p = np.asarray(a, float), np.asarray(b, float), np.asarray(p, float)
    ab = b - a; t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    return np.linalg.norm(p - (a + np.outer(t, ab) if p.ndim > 1 else a + t * ab), axis=-1)


def walk(mask, P, Q, t):
    h, w = mask.shape
    P, Q = np.array(P, float), np.array(Q, float); ax = Q - P; L = np.linalg.norm(ax)
    if L < 1: return None
    ax /= L; n = np.array([-ax[1], ax[0]]); M = P + t * (Q - P)
    if not (0 <= M[0] < w and 0 <= M[1] < h) or not mask[int(M[1]), int(M[0])]: return None
    pts = [M]
    for sgn in (1, -1):
        k = 1
        while True:
            p = M + sgn * 0.5 * k * n
            if not (0 <= p[0] < w and 0 <= p[1] < h) or not mask[int(p[1]), int(p[0])]: break
            pts.append(p); k += 1
    return np.array(pts)


def arm_bands(mask, J, H):
    bands, notes = [], []
    for side in ('right', 'left'):
        s, e, wr = J.get(f'{side}_shoulder'), J.get(f'{side}_elbow'), J.get(f'{side}_wrist')
        if s is None or e is None: notes.append(f'{side}_arm_unknown'); continue
        chain = [s, e]
        if wr is not None: chain += [wr, tuple(np.array(wr) + 0.4 * (np.array(wr) - np.array(e)))]
        pts = walk(mask, s, e, 0.5)
        width = 0.5 * (len(pts) - 1) if pts is not None else None
        if width is None or width > 0.8 * H: width = 0.35 * H; notes.append(f'{side}_width_fallback')
        bands.append({'side': side, 'chain': [tuple(map(float, c)) for c in chain], 'r': 0.55 * width})
    return bands, notes




def touching_arms(points, bands):
    return [band["side"] for band in bands if any(
        (seg_dist(points, a, b) < band["r"]).any()
        for a, b in zip(band["chain"], band["chain"][1:]))]


@dataclass(frozen=True)
class ArmRegion:
    """분할 입력의 유무와 사용 가능 여부를 구분한다. 없으면 기존 계산으로 돌아가지 않는다."""
    mask: np.ndarray | None
    usable: bool
    reason: str | None

    def record(self):
        return {"available": self.mask is not None, "usable": self.usable,
                "reason": self.reason, "pixels": int(self.mask.sum()) if self.mask is not None else 0}


def torso_axis(start, finish, fraction):
    """목·골반 좌표에서 측정 중심과 수직 방향을 구한다. 없는 축은 추측하지 않는다."""
    if start is None or finish is None:
        return None
    a, b = np.asarray(start, float), np.asarray(finish, float)
    if a.shape != (2,) or b.shape != (2,) or not np.isfinite([a, b]).all():
        return None
    length = np.linalg.norm(b - a)
    if length < 1:
        return None
    direction = (b - a) / length
    return a + fraction * (b - a), np.array([-direction[1], direction[0]])


def prepare_arm_region(silhouette, handwear, joints):
    """팔 마스크를 전경 안으로 제한하고 몸통 중점을 덮는 오분할을 표시한다."""
    if handwear is None:
        return ArmRegion(None, False, "arm_parts_pending")
    values = np.asarray(handwear)
    if values.shape != silhouette.shape or values.ndim != 2:
        raise ValueError("팔 마스크와 전경의 크기가 다릅니다. 좌표 변환은 자동으로 하지 않습니다.")
    if values.dtype.kind not in "bufi" or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("팔 마스크는 음수·무한값 없는 이차원 수치여야 합니다.")
    arm = (values > 0) & silhouette
    hips = joints.get("right_hip"), joints.get("left_hip")
    midhip = (np.asarray(hips[0]) + np.asarray(hips[1])) / 2 if all(p is not None for p in hips) else None
    axis = torso_axis(joints.get("neck"), midhip, 0.5)
    if axis is None:
        return ArmRegion(arm, False, "no_axis")
    center, _ = axis
    h, w = silhouette.shape
    if not (0 <= center[0] < w and 0 <= center[1] < h):
        return ArmRegion(arm, False, "torso_center_outside")
    if arm[int(center[1]), int(center[0])]:
        return ArmRegion(arm, False, "torso_center_on_arm")
    return ArmRegion(arm, True, None)


def trace_torso_ray(silhouette, center, direction, arm_region, bands):
    """0.5픽셀 간격을 유지한다. 팔 경계 뒤도 검사해 누락된 팔 띠를 놓치지 않는다."""
    h, w = silhouette.shape
    points, arm_stop = [], None
    step = 0
    while True:
        point = center + 0.5 * step * direction
        if not (0 <= point[0] < w and 0 <= point[1] < h) or not silhouette[int(point[1]), int(point[0])]:
            break
        if arm_region.usable and arm_stop is None and arm_region.mask[int(point[1]), int(point[0])]:
            arm_stop = step
        points.append(point)
        step += 1
    points = np.asarray(points)
    end = arm_stop if arm_stop is not None else len(points) - 1
    touched = touching_arms(points, bands)
    samples = points[:arm_stop] if arm_stop is not None else points
    return {"end": end, "sil_end": len(points) - 1, "samples": samples,
            "boundary": "arm_part" if arm_stop is not None else "silhouette",
            "touch": touched, "unresolved": bool(touched) and (not arm_region.usable or arm_stop is None)}


def measure_torso_with_arms(silhouette, skin, start, finish, fraction, H, bands, arm_region, shoulder_width):
    """몸통 폭과 경계 기록·검토선을 반환한다. 분할 대기는 성공한 숫자로 바꾸지 않는다."""
    axis = torso_axis(start, finish, fraction)
    if axis is None:
        return None, None
    center, direction = axis
    h, w = silhouette.shape
    if not (0 <= center[0] < w and 0 <= center[1] < h) or not silhouette[int(center[1]), int(center[0])]:
        return {"invalid": "outside"}, None
    if arm_region.mask is None:
        return {"invalid": "arm_parts_pending", "status": "pending", "arm_usable": False}, None
    rays = [trace_torso_ray(silhouette, center, sign * direction, arm_region, bands) for sign in (1, -1)]
    width = 0.5 * sum(ray["end"] for ray in rays)
    samples = np.concatenate((rays[0]["samples"], rays[1]["samples"][1:]))
    exposure = float(skin[samples[:, 1].astype(int), samples[:, 0].astype(int)].mean()) if len(samples) else 0.0
    result = {"w": round(width / H, 3), "w_px": width,
              "sil_w_px": 0.5 * sum(ray["sil_end"] for ray in rays), "exposure": round(exposure, 2),
              "bare": exposure >= 0.6, "touch": sorted({side for ray in rays for side in ray["touch"]}),
              "boundary": [ray["boundary"] for ray in rays], "ends": [ray["end"] for ray in rays],
              "arm_usable": arm_region.usable, "arm_reason": arm_region.reason,
              "M": center.tolist(), "nvec": direction.tolist(), "status": "measured"}
    if any(ray["unresolved"] for ray in rays) or not len(samples) or width <= 0:
        result.update(invalid="arm_touch", status="unmeasurable")
    elif not arm_region.usable and shoulder_width and width > 1.5 * shoulder_width:
        result.update(invalid="arm_merge", status="unmeasurable")
    ends = [tuple(center + sign * 0.5 * ray["end"] * direction) for sign, ray in zip((1, -1), rays)]
    return result, (ends[0], ends[1], "invalid" in result)


def measure_body(mask, skin_body, J, H, out, draw, *, arm_region=None):
    h_img, w_img = mask.shape
    ys, xs = np.nonzero(mask)
    out['B1'] = round(float((ys.max() - ys.min()) / H), 3) if ('right_ankle' in J or 'left_ankle' in J) else None
    sh = (J.get('right_shoulder'), J.get('left_shoulder'))
    sw = math.dist(*sh) if all(sh) else None
    out['B2'] = round(sw / H, 3) if sw else None
    bands, notes = arm_bands(mask, J, H)
    out['notes'].extend(notes)
    out['arm_bands'] = bands
    draw['arm_bands'] = bands
    if arm_region is not None:
        out['arm_region'] = arm_region.record()
        if arm_region.mask is not None:
            draw['arm_parts'] = arm_region.mask
    def width(P, Q, t, kind):
        if kind == 'torso' and arm_region is not None:
            result, line = measure_torso_with_arms(mask, skin_body, P, Q, t, H, bands, arm_region, sw)
            if line is not None:
                draw['lines'].append(line)
            return result
        if P is None or Q is None: return None
        P, Q = np.array(P, dtype=float), np.array(Q, dtype=float); ax = Q - P; L = np.linalg.norm(ax)
        if L < 1: return None
        ax /= L; n = np.array([-ax[1], ax[0]]); M = P + t * (Q - P)
        if not (0 <= M[0] < w_img and 0 <= M[1] < h_img) or not mask[int(M[1]), int(M[0])]: return {'invalid': 'outside'}
        pts = [M]
        for sgn in (1, -1):
            k = 1
            while True:
                p = M + sgn * 0.5 * k * n
                if not (0 <= p[0] < w_img and 0 <= p[1] < h_img) or not mask[int(p[1]), int(p[0])]: break
                pts.append(p); k += 1
        pts = np.array(pts); wpx = 0.5 * (len(pts) - 1)
        expo = float(np.mean([skin_body[int(p[1]), int(p[0])] for p in pts]))
        res = {'w': round(wpx / H, 3), 'exposure': round(expo, 2), 'bare': expo >= 0.6}
        if kind == 'torso':
            res['touch'] = touching_arms(pts, bands)
            if sw and wpx > 1.5 * sw: res['invalid'] = 'arm_merge'
            elif res['touch']: res['invalid'] = 'arm_touch'
        if kind == 'limb' and wpx > 0.8 * H: res['invalid'] = 'merge'
        a, b = pts[np.argmin(pts @ n)], pts[np.argmax(pts @ n)]
        draw['lines'].append((tuple(a), tuple(b), 'invalid' in res))
        return res
    hips = (J.get('right_hip'), J.get('left_hip')); neck = J.get('neck')
    midhip = tuple((np.array(hips[0]) + np.array(hips[1])) / 2) if all(hips) else None
    out['B3'] = width(neck, midhip, 0.3, 'torso'); out['B4'] = width(neck, midhip, 0.6, 'torso'); out['B5'] = width(neck, midhip, 1.0, 'torso')
    for key, (a, b, t) in {'B6': (('right_hip', 'right_knee'), ('left_hip', 'left_knee'), 0.3), 'B7': (('right_knee', 'right_ankle'), ('left_knee', 'left_ankle'), 0.4),
                           'B8': (('right_shoulder', 'right_elbow'), ('left_shoulder', 'left_elbow'), 0.5)}.items():
        sides = [width(J.get(p), J.get(q), t, 'limb') for p, q in (a, b)]
        ok = [s for s in sides if s and 'invalid' not in s]
        out[key] = {'w': round(float(np.mean([s['w'] for s in ok])), 3), 'exposure': round(float(np.mean([s['exposure'] for s in ok])), 2),
                    'bare': all(s['bare'] for s in ok), 'sides': len(ok)} if ok else {'invalid': 'no_valid_side'}


def measure_hair(lab, mask, skin_face, skin, J, fpt, head_box, out, draw):
    bx0, by0, bx1, by1 = head_box; H = by1-by0
    h_img, w_img = mask.shape
    eyes = [J.get('right_eye'), J.get('left_eye')]
    if not all(eyes):
        e1 = [fpt(i) for i in range(36, 42)]; e2 = [fpt(i) for i in range(42, 48)]
        if all(e1) and all(e2): eyes = [tuple(np.mean(e1, axis=0)), tuple(np.mean(e2, axis=0))]
    if all(eyes):
        ex = (eyes[0][0] + eyes[1][0]) / 2; ey = (eyes[0][1] + eyes[1][1]) / 2; ed = abs(eyes[0][0] - eyes[1][0])
        # 머리카락 (v2)
        rx0, rx1 = int(max(0, bx0 - 0.2 * H)), int(min(w_img, bx1 + 0.2 * H)); ry0, ry1 = int(max(0, by0 - 0.05 * H)), int(min(h_img, by0 + 2.2 * H))
        region = np.zeros_like(mask); region[ry0:ry1, rx0:rx1] = True
        band = np.zeros_like(mask); band[int(max(0, by0)):int(max(0, ey - 0.35 * H)), int(max(0, bx0)):int(bx1)] = True; band &= mask
        centers, cfrac = kmeans3(lab[band])
        separable = None
        if centers is not None:
            dom = centers[int(np.argmax(cfrac))]; sep_dE = float(np.linalg.norm(dom - skin)); separable = sep_dE >= 20
            out['hair_skin_dE'] = round(sep_dE, 1)
        if not separable:
            out['notes'].append('hair_skin_inseparable' if centers is not None else 'no_hair_band')
        else:
            near = np.min(np.stack([np.linalg.norm(lab - c, axis=2) for c in centers]), axis=0) < 15
            hair = region & mask & ~skin_face & near
            n, labm = cv2.connectedComponents(hair.astype(np.uint8), connectivity=8)
            keep = set(np.unique(labm[band & hair])) - {0}
            hair = np.isin(labm, list(keep)) if keep else np.zeros_like(hair)
            if hair.sum() >= 50:
                q = np.percentile(lab[hair], np.linspace(0, 100, 101), axis=0)
                draw['hair_quantiles'] = q
            draw['hair'] = hair
    else:
        out['notes'].append('no_eyes')


def draw_overlay(rgb, mask, skin_face, head_box, draw):
    bx0, by0, bx1, by1 = head_box
    ov = rgb.copy()
    if 'hair' in draw: ov[draw['hair']] = (ov[draw['hair']] * .5 + np.array([0, 200, 0]) * .5).astype(np.uint8)
    face_tint = skin_face & (np.indices(mask.shape)[0] < by1)
    ov[face_tint] = (ov[face_tint] * .6 + np.array([255, 140, 0]) * .4).astype(np.uint8)
    tint = np.zeros(mask.shape, np.uint8)
    for band in draw.get('arm_bands', []):
        radius = max(1, int(round(band['r'])))
        chain = [tuple(np.rint(p).astype(int)) for p in band['chain']]
        for a, b in zip(chain, chain[1:]): cv2.line(tint, a, b, 255, radius * 2)
        for p in chain: cv2.circle(tint, p, radius, 255, -1)
    if 'arm_parts' in draw:
        region = draw['arm_parts']
        ov[region] = (ov[region] * .5 + np.array([150, 80, 255]) * .5).astype(np.uint8)
    region = tint > 0
    ov[region] = (ov[region] * .7 + np.array([50, 100, 240]) * .3).astype(np.uint8)
    im = Image.fromarray(ov); dr = ImageDraw.Draw(im)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts: dr.line([tuple(p[0]) for p in c] + [tuple(c[0][0])], fill=(0, 120, 255), width=2)
    dr.rectangle((bx0, by0, bx1, by1), outline=(255, 0, 255), width=2)
    for a, b, bad in draw['lines']: dr.line([a, b], fill=(160, 160, 160) if bad else (255, 0, 0), width=3)
    for z in draw['boxes']: dr.rectangle(z, outline=(255, 220, 0), width=2)
    return im
