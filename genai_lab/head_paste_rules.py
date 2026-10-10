"""검증한 H2·G3 규칙: 머리 범위 → 영역 보정 → 원본 픽셀 붙이기. 모델 실행은 없다."""
import cv2
import numpy as np
from PIL import Image, ImageFilter
from genai_lab.proportion_inputs import require
TAGS = ('hair', 'headwear', 'face', 'eyes', 'eyewear', 'ears', 'earwear', 'nose', 'mouth', 'neck', 'neckwear', 'topwear', 'handwear', 'bottomwear', 'legwear', 'footwear', 'tail', 'wings', 'objects')
N = 1024
DE_LIMIT = 15.0

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

def prepare_original(orig, hm, parts, H):
    """확인한 머리 영역으로 분할을 보완하고, 피부·수염·배경 출처를 준비한다."""
    validate_parts(parts)
    po = {name: mask.copy() for name, mask in parts.items()}
    pad = padding_mask(orig)
    po['face'] = (po['face'] | hm & ~H) & ~H
    po['hair'] = H
    fg_o = (np.logical_or.reduce([po[t] for t in TAGS]) | hm) & ~pad
    feat_o_all = po['eyes'] | po['nose'] | po['mouth']
    feat_o_keep, _, _, o_ball_bottom = split_features(feat_o_all)
    body = po['face'] | po['neck'] | po['neckwear'] | po['topwear'] | po['handwear'] | po['bottomwear']
    o_head = ((po['hair'] | po['headwear'] | po['ears'] | po['earwear']) & dil(hm, 9) | hm & ~body) & ~pad
    hull = np.zeros((N, N), np.uint8)
    pts = cv2.findNonZero(po['face'].astype(np.uint8))
    if pts is not None:
        cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
    beard = po['hair'] & (hull > 0)
    beard[:o_ball_bottom] = False
    o_head &= ~beard
    o_skin = po['face'] & fg_o
    o_bg = ~fg_o
    lo = cv2.cvtColor(orig.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    chroma = np.hypot(lo[..., 1], lo[..., 2])
    skin_ref_o = po['face'] & ~dil(feat_o_all, 6) & ~pad
    require(skin_ref_o.any(), '원본 피부 기준 영역이 비었습니다.')
    lowc = po['face'] & (chroma < 0.5 * float(np.median(chroma[skin_ref_o])))
    clean = cv2.inpaint(orig, dil(feat_o_keep, 6).astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
    dist = np.linalg.norm(clean.astype(np.float32) - 255, axis=2)
    alpha = np.zeros((N, N), np.float32)
    band = dil(fg_o, 4) & ~ero(fg_o, 3)
    alpha[ero(fg_o, 3)] = 1
    alpha[band] = np.clip(dist[band] / 60, 0, 1)
    a3 = alpha[..., None]
    fgcol = np.where(a3 > 0.02, np.clip((clean.astype(np.float32) - (1 - a3) * 255) / np.maximum(a3, 0.02), 0, 255), clean.astype(np.float32))
    return dict(po=po, pad=pad, o_head=o_head, o_skin=o_skin, o_bg=o_bg, lowc=lowc, fgcol=fgcol, a3=a3, feat_o_all=feat_o_all, beard=beard)

def paste_weights(o_head, o_skin, o_bg, lowc, pad, hm, g_hair, g_skin, region, ball_top, dE):
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
    w[o_bg & g_hair] = 1
    w[o_bg & ~g_hair & ~g_skin & above] = 1
    w[~region] = 0
    return (w, skin_g3, erase)

def paste_canvas(orig, hm, parts, H, gen, pg):
    """생성 앞머리만 원본으로 교체하고 눈알·코·입과 생성 수염을 보호한다."""
    validate_parts(pg)
    prepared = prepare_original(orig, hm, parts, H)
    po, pad = (prepared['po'], prepared['pad'])
    o_head, o_skin, o_bg = (prepared[key] for key in ('o_head', 'o_skin', 'o_bg'))
    lowc, fgcol, a3, feat_o_all = (prepared[key] for key in ('lowc', 'fgcol', 'a3', 'feat_o_all'))
    feats_all = pg['eyes'] | pg['nose'] | pg['mouth']
    feat_g, _, ball_top, _ = split_features(feats_all)
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
    w, skin_g3, erase = paste_weights(o_head, o_skin, o_bg, lowc, pad, hm, g_hair, g_skin, region, ball_top, dE)
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
    return (canvas, M, feat_g, dict(skin_dE=round(dE, 2), forehead_blend=dE <= DE_LIMIT, erase_fill='gen_inpaint' if dE > DE_LIMIT else 'orig_skin', beard_px=int(prepared['beard'].sum()), erase_px=int((erase & M).sum())))

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
