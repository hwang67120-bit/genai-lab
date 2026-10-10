"""검증한 H2·G3 규칙: 머리 범위 → 영역 보정 → 원본 픽셀 붙이기. 모델 실행은 없다."""
import cv2
import numpy as np
from PIL import Image, ImageFilter
from genai_lab.proportion_inputs import require
TAGS = ('hair', 'headwear', 'face', 'eyes', 'eyewear', 'ears', 'earwear', 'nose', 'mouth', 'neck', 'neckwear', 'topwear', 'handwear', 'bottomwear', 'legwear', 'footwear', 'tail', 'wings', 'objects')
N = 1024
DE_LIMIT = 15.0
# 긴 머리 경로(L1). 짧은 머리는 검증된 G3+H2 그대로 둔다. 아래 값은 GPU 측정 뒤 잠근다.
FACIAL_HAIR = frozenset({'facial hair', 'beard', 'mustache', 'goatee', 'stubble'})
SHIFT_LIMIT = 0.08
ROTATE_LIMIT = 15.0
SCALE_RANGE = (0.85, 1.15)
EYE_COVER_LIMIT = 0.5
HAIR_COLOR_DE = 25.0
BG_NEAR_DE = 10.0
BG_PLAIN_MIN = 0.6

def k(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

def dil(mask, radius):
    return cv2.dilate(mask.astype(np.uint8), k(radius)) > 0

def ero(mask, radius):
    return cv2.erode(mask.astype(np.uint8), k(radius)) > 0

def head_crop(rgb, hmask):
    """시험 때와 같은 머리 상자를 원본과 생성 그림에 함께 사용한다."""
    require(rgb.shape == (1232, 736, 3) and hmask.shape == (1232, 736), '비율 이미지 크기가 다릅니다.')
    ys, xs = np.nonzero(hmask > 127)
    require(len(xs) > 0, '확인한 머리 윤곽이 비었습니다.')
    cx, cy = ((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2)
    side = int(round(1.3 * max(xs.max() - xs.min(), ys.max() - ys.min())))
    require(0 < side <= 736, '머리 확대 범위가 이미지 크기를 벗어납니다.')
    x0 = min(max(0, int(round(cx - side / 2))), 736 - side)
    y0 = min(max(0, int(round(cy - side / 2))), 1232 - side)
    box = (x0, y0, x0 + side, y0 + side)
    return (crop_head(rgb, box), crop_head(hmask, box, Image.Resampling.BILINEAR) > 127, box)

def crop_head(image, box, mode=Image.Resampling.LANCZOS):
    return np.asarray(Image.fromarray(image).crop(box).resize((N, N), mode))

def padding_mask(orig):
    """테두리와 연결된 정확한 회색 여백만 제외한다. 내부 회색 머리는 보존한다."""
    gray = np.all(orig == 128, axis=2).astype(np.uint8)
    _, labels, _, _ = cv2.connectedComponentsWithStats(gray, 4)
    border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))) - {0}
    return np.isin(labels, list(border)) & (gray > 0)

def validate_parts(parts):
    require(set(parts) == set(TAGS), '머리 분할 항목이 다릅니다.')
    require(all((a.shape == (N, N) and a.dtype == np.bool_ for a in parts.values())), '머리 분할 크기·형식이 다릅니다.')

def lab(a):
    return cv2.cvtColor(np.clip(a, 0, 255).astype(np.float32) / 255, cv2.COLOR_RGB2LAB)

def rgb_of(l):
    return np.clip(cv2.cvtColor(l.astype(np.float32), cv2.COLOR_LAB2RGB) * 255, 0, 255)

def hair_mask(orig, hm, po, pad):
    """머리·피부 씨앗을 대칭으로 선택하고 사용자 확인용 H2 영역을 만든다."""
    validate_parts(po)
    require(hm.any(), '머리 윤곽이 비었습니다.')
    lab = cv2.cvtColor(orig.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    scope = dil(hm, 9) & ~pad
    feats = po['eyes'] | po['nose'] | po['mouth']
    cand = po['face'] & ~dil(feats, 6) & scope
    hs = po['hair'] & scope
    require(cand.any() and hs.any(), '머리·피부 씨앗을 구분하지 못했습니다. 원래 후보를 유지합니다.')
    d_skin = np.linalg.norm(lab - lab[cand].mean(0), axis=2)
    d_hair = np.linalg.norm(lab - lab[hs].mean(0), axis=2)
    hair_seed = hs & (d_skin >= np.quantile(d_skin[hs], 0.3))
    skin_seed = cand & (d_hair >= np.quantile(d_hair[cand], 0.3))
    rng = np.random.default_rng(0)

    def sample(m):
        p = lab[m]
        return p[rng.choice(len(p), min(20000, len(p)), replace=False)]
    from sklearn.neighbors import KNeighborsClassifier
    X = np.concatenate([sample(hair_seed), sample(skin_seed)])
    y = np.r_[np.ones(min(20000, int(hair_seed.sum()))), np.zeros(min(20000, int(skin_seed.sum())))]
    knn = KNeighborsClassifier(n_neighbors=15).fit(X, y)
    idx = np.nonzero(scope)
    is_hair = np.zeros((N, N), np.uint8)
    is_hair[idx] = knn.predict(lab[idx]).astype(np.uint8)
    med = np.median(lab[cand], axis=0)
    dark = scope & (lab[..., 0] < 25)
    vote = cv2.blur(is_hair.astype(np.float32), (7, 7))
    is_hair[dark] = (vote[dark] >= 0.5).astype(np.uint8)
    H = cv2.morphologyEx(is_hair, cv2.MORPH_OPEN, k(2))
    H = cv2.morphologyEx(H, cv2.MORPH_CLOSE, k(2)) > 0
    n, lab_c, st, _ = cv2.connectedComponentsWithStats(H.astype(np.uint8), 8)
    H = np.isin(lab_c, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 2000])
    unclassified = hm & ~np.logical_or.reduce([po[t] for t in TAGS])
    H |= (po['ears'] | po['headwear'] | po['earwear'] | unclassified) & scope
    H &= ~feats & scope
    return (H, {'skin_seed_px': int(skin_seed.sum()), 'hair_seed_px': int(hair_seed.sum()), 'H_px': int(H.sum()), 'seethrough_hair_px': int(hs.sum()), 'skin_median_lab': med.round(2).tolist()})

def split_features(features):
    n, labels, st, _ = cv2.connectedComponentsWithStats(features.astype(np.uint8), 8)
    balls = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 3000]
    if not balls:
        raise ValueError('눈알 덩어리를 찾지 못했습니다(규칙 실패)')
    keep, brow = (np.zeros_like(features, bool), np.zeros_like(features, bool))
    for i in range(1, n):
        x, y, w, h, a = st[i]
        comp = labels == i
        if i in balls:
            keep |= comp
            continue
        near = min(balls, key=lambda b: abs(st[b, 0] + st[b, 2] / 2 - (x + w / 2)))
        (brow if y + h <= st[near, 1] + 15 else keep)[comp] = True
    return (keep, brow, int(min((st[b, 1] for b in balls))), int(max((st[b, 1] + st[b, 3] for b in balls))))

def paste_profile(appearance):
    """확인된 외형 태그로 경로를 고른다. short_hair면 None(검증된 기존 경로)이다."""
    tags = {str(tag).strip().replace('_', ' ') for tag in appearance}
    if 'short hair' in tags:
        return None
    return dict(rule='G3+H2+L1', facial_hair=bool(FACIAL_HAIR & tags))

def rule_name(profile):
    return profile['rule'] if profile else 'G3+H2'

def background_measure(orig, hm, parts):
    """분할·윤곽 6px 밖, 정규화 여백 제외 픽셀로 원본 배경색과 단색 정도를 잰다."""
    pixels = ~dil(np.logical_or.reduce(list(parts.values())) | hm, 6) & ~padding_mask(orig)
    if not pixels.any():
        return None
    median = np.median(orig[pixels], axis=0).astype(np.float32)
    de = np.linalg.norm(lab(orig)[pixels] - lab(median[None, None])[0, 0], axis=1)
    return dict(color=median, white_distance=float(np.linalg.norm(median - 255)),
                plain_ratio=float((de < BG_NEAR_DE).mean()), pixels=int(pixels.sum()))

def prepare_original(orig, hm, parts, H, profile=None):
    """확인한 머리 영역으로 분할을 보완하고, 피부·수염·배경 출처를 준비한다."""
    validate_parts(parts)
    po = {name: mask.copy() for name, mask in parts.items()}
    pad = padding_mask(orig)
    background = np.full(3, 255, np.float32)
    near_bg = np.zeros((N, N), bool)
    if profile:
        # 흰색이 아닌 배경: 윤곽 안이라도 어느 분할에도 없고 배경색에 가까우면 배경으로 본다.
        measured = background_measure(orig, hm, parts)
        require(measured is not None, '원본 배경색을 확인하지 못했습니다.')
        background = measured['color']
        close = np.linalg.norm(lab(orig) - lab(background[None, None])[0, 0], axis=2) < BG_NEAR_DE
        near_bg = hm & ~np.logical_or.reduce([parts[t] for t in TAGS]) & ~H & close
    po['face'] = (po['face'] | hm & ~H & ~near_bg) & ~H
    po['hair'] = H
    fg_o = (np.logical_or.reduce([po[t] for t in TAGS]) | hm & ~near_bg) & ~pad
    feat_o_all = po['eyes'] | po['nose'] | po['mouth']
    feat_o_keep, _, _, o_ball_bottom = split_features(feat_o_all)
    body = po['face'] | po['neck'] | po['neckwear'] | po['topwear'] | po['handwear'] | po['bottomwear']
    loose = hm & ~body & ~near_bg
    if profile:
        loose[o_ball_bottom:] = False  # 원본 턱·목 외곽선이 생성 목 위에 붙지 않게 한다.
    o_head = ((po['hair'] | po['headwear'] | po['ears'] | po['earwear']) & dil(hm, 9) | loose) & ~pad
    hull = np.zeros((N, N), np.uint8)
    pts = cv2.findNonZero(po['face'].astype(np.uint8))
    if pts is not None:
        cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
    beard = po['hair'] & (hull > 0)
    beard[:o_ball_bottom] = False
    if profile and not profile['facial_hair']:
        # 수염 태그가 없으면 얼굴 옆 긴 머리는 붙인다. 머리로 분류된 턱 외곽선만 뺀다:
        # 폭 7px 미만의 얇은 선이거나, 눈 위 머리의 대표색 4개 중 어느 것과도 Lab ΔE 25 이상 다른 픽셀.
        thick = cv2.morphologyEx(beard.astype(np.uint8), cv2.MORPH_OPEN, k(3)) > 0
        above_hair = po['hair'].copy()
        above_hair[o_ball_bottom:] = False
        unlike = np.zeros_like(beard)
        if above_hair.any() and beard.any():
            from sklearn.cluster import KMeans
            lo_all = lab(orig)
            source = lo_all[above_hair]
            sample = source[np.random.default_rng(0).choice(len(source), min(20000, len(source)), replace=False)]
            centers = KMeans(n_clusters=min(4, len(sample)), n_init=4, random_state=0).fit(sample).cluster_centers_
            near = np.linalg.norm(lo_all[beard][:, None] - centers[None], axis=2).min(1)
            unlike[beard] = near >= HAIR_COLOR_DE
        beard &= ~thick | unlike
    o_head &= ~beard
    o_skin = po['face'] & fg_o
    o_bg = ~fg_o
    lo = cv2.cvtColor(orig.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    chroma = np.hypot(lo[..., 1], lo[..., 2])
    skin_ref_o = po['face'] & ~dil(feat_o_all, 6) & ~pad
    require(skin_ref_o.any(), '원본 피부 기준 영역이 비었습니다.')
    lowc = po['face'] & (chroma < 0.5 * float(np.median(chroma[skin_ref_o])))
    clean = cv2.inpaint(orig, dil(feat_o_keep, 6).astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
    dist = np.linalg.norm(clean.astype(np.float32) - background, axis=2)
    alpha = np.zeros((N, N), np.float32)
    band = dil(fg_o, 4) & ~ero(fg_o, 3)
    alpha[ero(fg_o, 3)] = 1
    alpha[band] = np.clip(dist[band] / 60, 0, 1)
    a3 = alpha[..., None]
    fgcol = np.where(a3 > 0.02, np.clip((clean.astype(np.float32) - (1 - a3) * background) / np.maximum(a3, 0.02), 0, 255), clean.astype(np.float32))
    return dict(po=po, pad=pad, o_head=o_head, o_skin=o_skin, o_bg=o_bg, lowc=lowc, fgcol=fgcol, a3=a3, feat_o_all=feat_o_all,
                beard=beard, background=background, near_bg=near_bg)

def paste_weights(o_head, o_skin, o_bg, lowc, pad, hm, g_hair, g_skin, region, ball_top, dE, keep_hair=None):
    """머리는 원본, 눈 아래 피부는 생성, 피부색이 가까운 이마만 40px 섞는다."""
    above = np.zeros((N, N), bool)
    above[:ball_top] = True
    ramp = np.clip((ball_top - np.arange(N, dtype=np.float32)) / 40, 0, 1)[:, None] * np.ones((1, N), np.float32)
    head_g3 = o_head | lowc & above & dil(hm, 9) & ~pad
    skin_g3 = o_skin & ~(lowc & above)
    erase = skin_g3 & g_hair
    w = np.zeros((N, N), np.float32)
    w[head_g3] = 1
    w[erase] = 1
    if dE <= DE_LIMIT:
        m = skin_g3 & g_skin & above
        w[m] = ramp[m]
        w[skin_g3 & ~g_skin & ~g_hair & above] = 1
    erase_hair = o_bg & g_hair
    if keep_hair is not None:
        erase_hair &= ~keep_hair  # 자른 범위 밖으로 이어지는 생성 긴 머리는 남기고 이음은 다시 그리기에 맡긴다.
    w[erase_hair] = 1
    w[o_bg & ~g_hair & ~g_skin & above] = 1
    w[~region] = 0
    return (w, skin_g3, erase)

def eye_anchor(features):
    """눈알 덩어리 묶음의 가로 중심과 아래끝. 앞머리가 눈 위를 가려도 아래끝은 덜 흔들린다."""
    n, labels, st, _ = cv2.connectedComponentsWithStats(features.astype(np.uint8), 8)
    balls = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 3000]
    if not balls:
        raise ValueError('눈알 덩어리를 찾지 못했습니다(규칙 실패)')
    x0 = min(st[i, 0] for i in balls)
    x1 = max(st[i, 0] + st[i, 2] for i in balls)
    return (x0 + x1) / 2, max(st[i, 1] + st[i, 3] for i in balls)

def warp_head(orig, hm, parts, H, matrix):
    """원본 머리 재료를 같은 2×3 행렬로 옮긴다. 들어오는 가장자리 마스크는 비운다."""
    matrix = np.float32(matrix)
    plain = np.allclose(matrix[:, :2], np.eye(2))
    def move(mask):
        return cv2.warpAffine(mask.astype(np.uint8), matrix, (N, N), flags=cv2.INTER_NEAREST, borderValue=0) > 0
    moved = cv2.warpAffine(orig, matrix, (N, N), flags=cv2.INTER_NEAREST if plain else cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
    return moved, move(hm), {name: move(mask) for name, mask in parts.items()}, move(H)

def shift_head(orig, hm, parts, H, dx, dy):
    """원본 머리 재료를 정수 픽셀만큼 옮긴다."""
    return warp_head(orig, hm, parts, H, [[1, 0, dx], [0, 1, dy]])

def eye_points(features):
    """가장 큰 눈알 덩어리 두 개의 무게중심, 왼쪽부터. 둘이 아니면 None.
    아래끝은 눈 밑 그림자·볼 표시에 따라 흔들려서 기울기·크기는 무게중심으로 잰다."""
    n, labels, st, cen = cv2.connectedComponentsWithStats(features.astype(np.uint8), 8)
    balls = sorted((i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 3000), key=lambda i: -st[i, cv2.CC_STAT_AREA])[:2]
    if len(balls) != 2:
        return None
    return sorted((float(cen[i][0]), float(cen[i][1])) for i in balls)

def align_long_hair(orig, hm, parts, H, pg):
    """원본 두 눈을 생성 두 눈에 맞춘다: 기울기·크기는 눈 무게중심, 이동은 눈 묶음 기준점.
    눈이 하나면 이동만 한다. 범위를 넘으면 추측하지 않고 붙이지 않는다."""
    o_pts = eye_points(parts['eyes'] | parts['nose'] | parts['mouth'])
    g_pts = eye_points(pg['eyes'] | pg['nose'] | pg['mouth'])
    o_anchor = np.float32(eye_anchor(parts['eyes'] | parts['nose'] | parts['mouth']))
    g_anchor = np.float32(eye_anchor(pg['eyes'] | pg['nose'] | pg['mouth']))
    if o_pts and g_pts:
        (o1, o2), (g1, g2) = np.float32(o_pts), np.float32(g_pts)
        # 회전·크기 중심은 두 눈 가운데다.
        o_mid = (o1 + o2) / 2
        angle = float(np.degrees(np.arctan2(*(g2 - g1)[::-1]) - np.arctan2(*(o2 - o1)[::-1])))
        scale = float(np.linalg.norm(g2 - g1) / max(1e-6, np.linalg.norm(o2 - o1)))
    else:
        o_mid = o_anchor
        angle, scale = 0.0, 1.0
    matrix = cv2.getRotationMatrix2D((float(o_mid[0]), float(o_mid[1])), -angle, scale)
    # cv2 각도는 화면 기준 반시계 방향이 양수라 이미지 좌표 각도의 부호를 바꾼다.
    moved_anchor = matrix[:, :2] @ o_anchor + matrix[:, 2]
    matrix[:, 2] += g_anchor - moved_anchor
    dx, dy = (g_anchor - o_anchor).round().astype(int).tolist()
    if max(abs(dx), abs(dy)) > SHIFT_LIMIT * N:
        raise ValueError('원본과 생성 얼굴 위치 차이가 커서 머리 붙이기를 건너뛰었습니다.')
    if abs(angle) > ROTATE_LIMIT or not SCALE_RANGE[0] <= scale <= SCALE_RANGE[1]:
        raise ValueError('원본과 생성 얼굴의 기울기·크기 차이가 커서 머리 붙이기를 건너뛰었습니다.')
    if angle == 0.0 and scale == 1.0:
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return (*warp_head(orig, hm, parts, H, matrix),
            dict(shift_px=[dx, dy], rotate_deg=round(angle, 2), scale=round(scale, 3)))

def uncover_eyes(feat_g, H):
    """원본 앞머리가 덮는 생성 눈알 부분은 보호에서 뺀다. 눈알마다 덮이는 비율이 한도를 넘으면 그 눈은 그대로 보호한다."""
    n, labels, st, _ = cv2.connectedComponentsWithStats(feat_g.astype(np.uint8), 8)
    keep, ratios = feat_g.copy(), []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 3000:
            continue
        ball = labels == i
        ratio = float((ball & H).sum()) / float(ball.sum())
        ratios.append(round(ratio, 3))
        if ratio <= EYE_COVER_LIMIT:
            keep &= ~(ball & H)
    return keep, ratios

def hair_palette(lab_pixels, centers):
    distance = np.linalg.norm(lab_pixels[:, None] - centers[None], axis=2)
    return np.bincount(distance.argmin(1), minlength=len(centers)) / max(1, len(lab_pixels))

def hair_measurements(orig, H, po_face, gen, pg, M):
    """표시용 측정. 자동 거부에 쓰지 않는다(기준은 머리 모양별 시험 뒤 정한다)."""
    g_hair = pg['hair'] | pg['headwear']
    lo, lg = lab(orig), lab(gen)
    result = dict(hair_area_ratio=round(float(g_hair.sum()) / max(1, int(H.sum())), 3),
                  overflow_px=int((g_hair & ~dil(H, 6)).sum()),
                  generated_hair_on_original_face_px=int((g_hair & po_face & ~H).sum()),
                  paste_on_generated_clothes_px=int((M & (pg['topwear'] | pg['neckwear']) & ~dil(H, 6)).sum()))
    if H.any() and g_hair.any():
        from sklearn.cluster import KMeans
        source = lo[H]
        rng = np.random.default_rng(0)
        sample = source[rng.choice(len(source), min(20000, len(source)), replace=False)]
        centers = KMeans(n_clusters=4, n_init=4, random_state=0).fit(sample).cluster_centers_
        result.update(hair_color_dE=round(float(np.linalg.norm(lg[g_hair].mean(0) - source.mean(0))), 2),
                      hair_palette_difference=round(float(np.abs(hair_palette(source, centers)
                          - hair_palette(lg[g_hair], centers)).sum() / 2), 3))
    return result

def continuing_hair(g_hair, o_hair, chin):
    """원본 머리 밖 생성 머리 중 아래 가장자리, 또는 턱 아래 좌우 가장자리에 닿는 덩어리.
    가로 줄로 자르면 떠 있는 머리 조각이 생겨서 덩어리 단위로 남긴다."""
    outside = g_hair & ~dil(o_hair, 6)
    n, labels = cv2.connectedComponents(outside.astype(np.uint8), connectivity=8)
    edge = np.concatenate([labels[-1], labels[chin:, 0], labels[chin:, -1]])
    return np.isin(labels, list(set(np.unique(edge)) - {0}))

def figure_measurements(o_hair, g_hair, orig, gen, head_bottom):
    """전체 그림(정규화 좌표)에서 원본·생성 머리 모양과 색을 비교한다. 표시용이며 자동 거부는 없다."""
    def tails(mask):
        below = mask.copy()
        below[:head_bottom] = False
        n, _, st, _ = cv2.connectedComponentsWithStats(below.astype(np.uint8), 8)
        return sum(1 for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 1500)
    def lowest(mask):
        rows = np.nonzero(mask.any(1))[0]
        return int(rows.max()) if len(rows) else None
    union = int((o_hair | g_hair).sum())
    result = dict(hair_iou=round(float((o_hair & g_hair).sum()) / max(1, union), 3),
                  hair_area_ratio=round(float(g_hair.sum()) / max(1, int(o_hair.sum())), 3),
                  lowest_hair_row=[lowest(o_hair), lowest(g_hair)], tails_below_head=[tails(o_hair), tails(g_hair)],
                  overflow_px=int((g_hair & ~dil(o_hair, 12)).sum()))
    if o_hair.any() and g_hair.any():
        from sklearn.cluster import KMeans
        source, target = lab(orig)[o_hair], lab(gen)[g_hair]
        rng = np.random.default_rng(0)
        sample = source[rng.choice(len(source), min(20000, len(source)), replace=False)]
        centers = KMeans(n_clusters=4, n_init=4, random_state=0).fit(sample).cluster_centers_
        result.update(hair_color_dE=round(float(np.linalg.norm(target.mean(0) - source.mean(0))), 2),
                      hair_palette_difference=round(float(np.abs(hair_palette(source, centers) - hair_palette(target, centers)).sum() / 2), 3))
    return result

def paste_canvas(orig, hm, parts, H, gen, pg, profile=None):
    """생성 앞머리만 원본으로 교체하고 눈알·코·입과 생성 수염을 보호한다."""
    validate_parts(pg)
    alignment, eye_cover = None, None
    if profile:
        orig, hm, parts, H, alignment = align_long_hair(orig, hm, parts, H, pg)
    prepared = prepare_original(orig, hm, parts, H, profile)
    po, pad = (prepared['po'], prepared['pad'])
    o_head, o_skin, o_bg = (prepared[key] for key in ('o_head', 'o_skin', 'o_bg'))
    lowc, fgcol, a3, feat_o_all = (prepared[key] for key in ('lowc', 'fgcol', 'a3', 'feat_o_all'))
    feats_all = pg['eyes'] | pg['nose'] | pg['mouth']
    feat_g, _, ball_top, _ = split_features(feats_all)
    if profile:
        feat_g, eye_cover = uncover_eyes(feat_g, H)
    fg_g = np.logical_or.reduce([pg[t] for t in TAGS])
    g_hair = pg['hair'] | pg['headwear']
    g_skin = pg['face'] & ~g_hair
    require((~dil(fg_g, 6)).any(), '생성 머리 배경을 측정하지 못했습니다.')
    bg = np.median(gen[~dil(fg_g, 6)], axis=0).astype(np.float32)
    top_fg = np.linalg.norm(gen.astype(np.float32) - bg, axis=2) > 25
    top_fg[ball_top:] = False
    top_fg = cv2.morphologyEx(top_fg.astype(np.uint8), cv2.MORPH_OPEN, k(1)) > 0
    region = (dil(po['hair'] | po['headwear'] | g_hair | hm, 9) | top_fg) & ~dil(feat_g, 2)
    ref = g_skin & ~dil(feats_all, 6)
    src = o_skin & ~dil(feat_o_all, 6) & ~pad
    require(ref.any() and src.any(), '원본·생성 피부색 비교 영역이 비었습니다.')
    shift = lab(gen)[ref].mean(0) - lab(fgcol)[src].mean(0)
    dE = float(np.linalg.norm(shift))
    keep_hair, chin = None, None
    if profile:
        rows = np.nonzero(parts['face'].any(1))[0]
        chin = int(rows.max()) + 1 if len(rows) else N
        keep_hair = continuing_hair(g_hair, po['hair'] | po['headwear'], chin)
    w, skin_g3, erase = paste_weights(o_head, o_skin, o_bg, lowc, pad, hm, g_hair, g_skin, region, ball_top, dE, keep_hair)
    M = w > 0
    soft = cv2.GaussianBlur(M.astype(np.float32), (0, 0), 1.5)
    soft[~M] = 0
    soft = (soft * w)[..., None]
    tint = skin_g3 & ~lowc
    col = fgcol.copy()
    l = lab(col)
    l[tint] += shift
    col[tint] = rgb_of(l)[tint]
    fill = a3 * col + (1 - a3) * bg
    if dE > DE_LIMIT:
        gen_fill = cv2.inpaint(gen, (erase & region).astype(np.uint8) * 255, 9, cv2.INPAINT_TELEA).astype(np.float32)
        fill[erase] = gen_fill[erase]
    canvas = np.clip(np.rint(gen * (1 - soft) + fill * soft), 0, 255).astype(np.uint8)
    require(np.array_equal(canvas[~M], gen[~M]), '붙이기 영역 밖 픽셀 변경')
    measurements = dict(skin_dE=round(dE, 2), forehead_blend=dE <= DE_LIMIT, erase_fill='gen_inpaint' if dE > DE_LIMIT else 'orig_skin', beard_px=int(prepared['beard'].sum()), erase_px=int((erase & M).sum()))
    if profile:
        measurements.update(rule=profile['rule'], facial_hair_rule=profile['facial_hair'], **alignment, eye_cover_ratio=eye_cover,
            original_background_rgb=prepared['background'].round(1).tolist(), background_as_hair_removed_px=int(prepared['near_bg'].sum()),
            chin_row=chin, kept_generated_hair_px=int(keep_hair.sum()), hair=hair_measurements(orig, H, po['face'], gen, pg, M))
    return (canvas, M, feat_g, measurements)

def protected_support(mask, features, box):
    """최종 4px 확장 영역에서 눈·코·입을 빼고 합성 허용 범위를 반환한다."""
    size = (box[2] - box[0], box[3] - box[1])
    local = np.asarray(Image.fromarray(mask).resize(size, Image.Resampling.NEAREST)) > 127
    face = np.asarray(Image.fromarray(features).resize(size, Image.Resampling.NEAREST)) > 127
    support = cv2.dilate(local.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    require(not np.any(local & face), '원래 편집 마스크가 얼굴 특징을 덮습니다')
    expanded_overlap = int((support & face).sum())
    support &= ~face
    return support, face, expanded_overlap

def blend_result(raw, redrawn, mask, features, box):
    """최종 4px 확장에서도 눈·코·입을 제외하고 영역 밖 픽셀을 보존한다."""
    size = (box[2] - box[0], box[3] - box[1])
    support, face, expanded_overlap = protected_support(mask, features, box)
    soft = np.asarray(Image.fromarray(support.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(2))).astype(float) / 255
    soft[~support] = 0
    patch = np.asarray(redrawn.resize(size, Image.Resampling.LANCZOS), dtype=float)
    result = raw.copy()
    old = raw[box[1]:box[3], box[0]:box[2]]
    result[box[1]:box[3], box[0]:box[2]] = np.rint(old * (1 - soft[..., None]) + patch * soft[..., None]).astype(np.uint8)
    allowed = np.zeros(raw.shape[:2], bool)
    allowed[box[1]:box[3], box[0]:box[2]] = support
    require(np.array_equal(raw[~allowed], result[~allowed]), '편집 영역 밖 픽셀 변경')
    require(np.array_equal(old[face], result[box[1]:box[3], box[0]:box[2]][face]), '얼굴 특징 픽셀 변경')
    return (result, {'outside_equal': True, 'face_equal': True, 'expanded_face_overlap_removed': expanded_overlap})


def blend_white_product(before, raw_redraw, alpha, mask, features, box):
    """다시 그린 전경을 흰색에 합성하고, 허용 영역만 검토용 결과에 옮긴다."""
    from genai_lab.proportion_inputs import white_background
    require(before.shape == raw_redraw.shape, '흰 배경 머리 합성 크기가 다릅니다.')
    support, face, overlap = protected_support(mask, features, box)
    white = white_background(raw_redraw, alpha)
    allowed = np.zeros(before.shape[:2], bool)
    allowed[box[1]:box[3], box[0]:box[2]] = support
    result = before.copy()
    result[allowed] = white[allowed]
    old = before[box[1]:box[3], box[0]:box[2]]
    updated = result[box[1]:box[3], box[0]:box[2]]
    background = allowed & np.all(before >= 250, axis=2) & (alpha == 0)
    require(np.array_equal(before[~allowed], result[~allowed]), '편집 영역 밖 픽셀 변경')
    require(np.array_equal(old[face], updated[face]), '얼굴 특징 픽셀 변경')
    require(np.all(result[background] == 255), '머리 주변 흰 배경에 회색이 남았습니다.')
    return result, dict(outside_equal=True, face_equal=True,
        expanded_face_overlap_removed=overlap, white_background_equal=True,
        white_background_checked_pixels=int(background.sum()))
