"""그림체 참조의 부위 색을 원본 캐릭터 색으로 맞춘다(D-187 다음 단계, 사용자 10/29 (가)).

GPU 비교 v2~v4에서 그림체 참조에 남은 색·내용(머리색·청바지·회색)이 결과로 그대로 들어왔다. 그래서 참조를
그림마다 손으로 고치지 않고, 원본에서 읽은 부위 색으로 참조의 같은 부위를 칠한다. 참조의 명암·주름(밝기 변화)은
유지하고 평균 밝기와 색만 원본 쪽으로 옮긴다. 분할은 호출하는 쪽이 한다(이 모듈은 마스크만 받는다, 모델 없음).

- 색 공간: CIE Lab(sRGB D65). L은 (참조 - 참조 평균) × 비율 + 원본 평균, a·b는 원본 평균 + 참조 편차 × 0.3.
- 한 부위에 색이 둘 이상이면(조끼 + 셔츠) 원본·참조를 각각 k개로 나누고 밝기 순서로 짝을 맞춘다. 두 무리의 색상(a·b)이
  가까우면(같은 옷의 밝은 면·어두운 면) 한 색으로 합친다. 합치기는 원본 쪽에만 한다 — 참조 쪽은 흰 셔츠와 검은 무늬처럼
  색상이 비슷해도 밝고 어두운 옷을 나눠야 원본의 두 옷(셔츠·조끼)에 짝지을 수 있다.
- 분할기의 부위 이름은 그림마다 다른 옷을 가리킨다(T014: bottomwear = 조끼 아랫부분, legwear = 바지, handwear = 셔츠 소매 /
  target-02: legwear·handwear = 맨살). 그래서 계획은 이름이 아니라 역할로 묶는다: 역할마다 참조 부위들·원본 부위들,
  원본 색 고르는 법(pair = 밝기 순서 짝, largest = 가장 넓은 색 하나).
- 피부처럼 보이는 픽셀은 옷 역할에서 뺀다(분할기가 맨살을 옷으로 잡는 경우, D-187 v4 첫 시도). 그늘진 피부도 잡도록
  범위를 넓게 둔다(V001 시험에서 팔·어깨 그늘이 옷으로 칠해짐, 10/29).
- 아래옷은 윗옷에서 이미 쓴 색을 원본 후보에서 뺀다(exclude_colours_of). 긴 코트 아랫자락이 bottomwear로 잡혀 바지보다
  넓게 나온 경우(V001) 코트 색이 아래옷으로 뽑히는 것을 막는다. 빼고 남는 게 없으면(원피스) 원래 후보를 쓴다.
- 확실하지 않으면 중립색(사용자 10/29 1번): "largest"로 고른 원본 색이 그림의 min_fraction(기본 2%) 미만이거나 남은
  후보의 절반 미만이면 그 색을 믿지 않고 참조 부위를 회색(명암만 유지)으로 둔다. 옷이 거의 가려진 경우(V001 바지는
  긴 코트에 가려 약 1%) 틀린 색을 밀어 넣지 않기 위함이다. 문장(특징표)이 그 부위 색을 맡는다.
- 눈(D-189): 다섯 번 모두 결과 눈이 참조의 파란 눈이 됐다. 눈 부위 안에서 색이 있는 홍채 픽셀(채도 C* ≥ 12, 흰자·동공·
  속눈썹 제외)만 골라 화면 왼쪽·오른쪽 눈을 따로 원본의 같은 쪽 눈 색으로 맞춘다(짝눈). 한쪽 눈이 안 보이면 다른 쪽 색을 쓴다.
  눈 부위 마스크에는 눈두덩 피부가 섞이므로 양쪽 모두 피부처럼 보이는 픽셀을 홍채에서 뺀다(T014 파란 눈이 갈색 피부와
  평균되어 회색, 참조 분홍 눈두덩까지 칠해짐 — 첫 시도 style-reference-v6/-V001b).
- 원본의 선화(밝기 L < 8인 검은 윤곽선)는 옷 색 후보에서 뺀다. 좁은 바지 영역에서는 윤곽선이 바지보다 많아
  "가장 넓은 색"이 검정이 되는 경우가 있었다(V001). 남는 게 50픽셀 미만이면 빼지 않는다.
"""
import numpy as np

_M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])
INK_L = 8.0          # source pixels darker than this are outlines, not garment colour


def rgb_to_lab(rgb):
    c = np.asarray(rgb, np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ _M.T / _WHITE
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_rgb(lab):
    lab = np.asarray(lab, np.float64)
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * _WHITE
    c = xyz @ np.linalg.inv(_M).T
    c = np.where(c > 0.0031308, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055, 12.92 * c)
    return np.clip(c * 255.0 + 0.5, 0, 255).astype(np.uint8)


def skin_like(rgb, face):
    """얼굴 부위의 색상·채도 범위(5~95%)에 드는 픽셀. 얼굴이 너무 작으면 빈 마스크."""
    import PIL.Image
    hsv = np.asarray(PIL.Image.fromarray(np.asarray(rgb, np.uint8)).convert("HSV")).astype(np.int16)
    face = np.asarray(face, bool) & (hsv[..., 2] > 60)
    if face.sum() < 50:
        return np.zeros(face.shape, bool)
    h5, h95 = np.percentile(hsv[..., 0][face], [5, 95])
    s5, s95 = np.percentile(hsv[..., 1][face], [5, 95])
    return ((hsv[..., 0] >= h5 - 8) & (hsv[..., 0] <= h95 + 8) & (hsv[..., 1] >= max(s5 - 20, 18))
            & (hsv[..., 1] <= s95 + 60) & (hsv[..., 2] > 35))


def clusters(lab, k, iterations=12, merge_ab=12.0):
    """Lab 픽셀을 k개로 나누고 밝기 오름차순 번호를 붙인다(재현되게 밝기 분위수로 시작).
    k=2에서 두 무리의 a·b 거리가 merge_ab보다 가까우면 같은 색의 명암 차이로 보고 하나로 합친다."""
    lab = np.asarray(lab, np.float64)
    if k == 1 or len(lab) < k * 20:  # noqa: tiny parts stay one colour
        return np.zeros(len(lab), int), lab.mean(0)[None]
    centers = np.stack([lab[np.argsort(lab[:, 0])[int(len(lab) * (i + 0.5) / k)]] for i in range(k)])
    for _ in range(iterations):
        label = np.argmin(((lab[:, None] - centers[None]) ** 2).sum(-1), 1)
        centers = np.stack([lab[label == i].mean(0) if (label == i).any() else centers[i] for i in range(k)])
    if k == 2 and np.linalg.norm(centers[0, 1:] - centers[1, 1:]) < merge_ab:
        return np.zeros(len(lab), int), lab.mean(0)[None]
    order = np.argsort(centers[:, 0])
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    return remap[label], centers[order]


def neutral_part(reference_lab, reference_mask, lightness=(30.0, 82.0)):
    """참조 부위를 무채색으로: 참조의 밝기 순서(2~98%)를 좁힌 회색 범위에 옮긴다(D-187 v4와 같은 뜻)."""
    out = reference_lab.copy()
    px = reference_lab[reference_mask]
    lo, hi = np.percentile(px[:, 0], [2, 98])
    t = np.clip((px[:, 0] - lo) / max(hi - lo, 1e-3), 0, 1)
    px = np.stack([lightness[0] + (lightness[1] - lightness[0]) * t, np.zeros(len(px)), np.zeros(len(px))], -1)
    out[reference_mask] = px
    return out


def match_part(reference_lab, reference_mask, source_lab, source_mask, k=1, source="pair", chroma_keep=0.1,
               lightness_ratio=(0.5, 1.2), min_fraction=None, min_pixels=50):
    """참조 부위의 Lab을 원본 부위의 색으로 옮긴다. source="largest"면 원본을 둘로 나눠 넓은 쪽 색 하나만 쓴다.
    반환: 새 Lab(부위 밖은 그대로)과 기록."""
    out = reference_lab.copy()
    ref_px, src_px = reference_lab[reference_mask], source_lab[source_mask]
    if (src_px[:, 0] >= INK_L).sum() >= min_pixels:
        src_px = src_px[src_px[:, 0] >= INK_L]                       # line art is not a garment colour
    if len(ref_px) >= min_pixels and len(src_px) < min_pixels and source == "largest" and min_fraction is not None:
        return neutral_part(reference_lab, reference_mask), dict(          # no evidence at all: grey, never the reference colour
            state="neutral_uncertain", reason="source part too small to read a colour", source_pixels=int(len(src_px)))
    if len(ref_px) < min_pixels or len(src_px) < min_pixels:
        return out, dict(state="skipped_too_small", reference_pixels=int(len(ref_px)), source_pixels=int(len(src_px)))
    ref_label, ref_centers = clusters(ref_px, k, merge_ab=0.0)
    src_label, src_centers = clusters(src_px, 2 if source == "largest" else k)
    counts = np.bincount(src_label, minlength=len(src_centers))
    largest = int(counts.argmax())
    evidence = dict(source_cluster_pixels=counts.tolist(), image_pixels=int(reference_mask.size),
                    chosen_fraction_of_image=round(float(counts[largest]) / source_mask.size, 4),
                    chosen_share_of_candidates=round(float(counts[largest]) / max(len(src_px), 1), 3))
    if source == "largest" and min_fraction is not None and (
            evidence["chosen_fraction_of_image"] < min_fraction or evidence["chosen_share_of_candidates"] < 0.5):
        return neutral_part(reference_lab, reference_mask), dict(
            state="neutral_uncertain", reason=f"chosen colour below {min_fraction:.0%} of the image or half of the candidates",
            source_centers=np.round(src_centers, 2).tolist(), **evidence)
    new = ref_px.copy()
    pairs = []
    for i in range(len(ref_centers)):
        sel = ref_label == i
        src_sel = src_label == (largest if source == "largest" else min(i, len(src_centers) - 1))
        r, s = ref_px[sel], src_px[src_sel]
        if not len(r) or not len(s):
            continue
        ratio = float(np.clip(s[:, 0].std() / max(r[:, 0].std(), 1e-3), *lightness_ratio))
        new[sel, 0] = (r[:, 0] - r[:, 0].mean()) * ratio + s[:, 0].mean()
        new[sel, 1:] = s[:, 1:].mean(0) + (r[:, 1:] - r[:, 1:].mean(0)) * chroma_keep
        pairs.append(dict(reference_lab=np.round(r.mean(0), 1).tolist(), source_lab=np.round(s.mean(0), 1).tolist(),
                          pixels=int(sel.sum()), lightness_ratio=round(ratio, 3)))
    new[:, 0] = np.clip(new[:, 0], 0, 100)
    out[reference_mask] = new
    return out, dict(state="matched", k=k, source=source, reference_colours=len(ref_centers),
                     source_colours=len(src_centers), source_centers=np.round(src_centers, 2).tolist(), pairs=pairs, **evidence)


def without_colours(lab, mask, centers, max_ab=12.0, max_l=25.0):
    """mask에서 centers(다른 역할에서 이미 쓴 원본 색)와 같은 색으로 보이는 픽셀을 뺀다(색상 가깝고 밝기도 가까움)."""
    keep = np.asarray(mask, bool).copy()
    for c in np.asarray(centers, np.float64).reshape(-1, 3):
        same = (np.linalg.norm(lab[..., 1:] - c[1:], axis=-1) < max_ab) & (np.abs(lab[..., 0] - c[0]) < max_l)
        keep &= ~same
    return keep


def iris(lab, mask, chroma_min=12.0):
    """눈 부위 중 색이 있는 픽셀(홍채). 흰자(채도 낮음)·동공·속눈썹(아주 어두움)·반사광(아주 밝음)은 뺀다."""
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    return np.asarray(mask, bool) & (chroma >= chroma_min) & (lab[..., 0] > 10) & (lab[..., 0] < 92)


def split_sides(mask):
    """마스크를 가로 범위의 가운데에서 화면 왼쪽·오른쪽으로 나눈다. 픽셀이 너무 적으면 None."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 40:
        return None
    middle = int(round((xs.min() + xs.max()) / 2))
    left = np.asarray(mask, bool).copy()
    left[:, middle:] = False
    return left, np.asarray(mask, bool) & ~left


def match_eyes(reference_lab, reference_mask, source_lab, source_mask, chroma_min=12.0, reference_skin=None, source_skin=None):
    """참조의 왼쪽·오른쪽 홍채를 원본의 같은 쪽 홍채 색으로. 원본 한쪽이 비면 다른 쪽을 쓴다. 피부 마스크는 홍채에서 뺀다."""
    ref_iris, src_iris = iris(reference_lab, reference_mask, chroma_min), iris(source_lab, source_mask, chroma_min)
    if reference_skin is not None:
        ref_iris &= ~np.asarray(reference_skin, bool)
    if source_skin is not None:
        src_iris &= ~np.asarray(source_skin, bool)
    ref_sides, src_sides = split_sides(ref_iris), split_sides(src_iris)
    if ref_sides is None or src_sides is None:
        return reference_lab, dict(state="skipped_no_iris", reference_found=ref_sides is not None, source_found=src_sides is not None)
    out, sides = reference_lab, {}
    for name, ref_side, src_side, other in (("image_left", ref_sides[0], src_sides[0], src_sides[1]),
                                            ("image_right", ref_sides[1], src_sides[1], src_sides[0])):
        use = src_side if src_side.sum() >= 15 else other
        out, sides[name] = match_part(out, ref_side, source_lab, use, min_pixels=15)
        sides[name]["source_side"] = name if use is src_side else "other_side"
    return out, dict(state="matched", sides=sides)


def union(parts, names):
    return np.logical_or.reduce([np.asarray(parts[n], bool) for n in names])


def recolor_reference(reference_rgb, reference_parts, source_rgb, source_parts, plan, background_alpha=None,
                      background_rgb=(214, 210, 202)):
    """plan: {역할: dict(reference=[부위…], source=[부위…], k=색 수, pick="pair"|"largest", keep_skin=False,
    exclude_colours_of=다른 역할, min_fraction=확신 기준, split="left_right"(눈: 홍채를 화면 좌우로 따로))}. keep_skin이 아니면(옷 역할) 양쪽 모두 피부처럼 보이는 픽셀을 뺀다.
    exclude_colours_of가 있으면 그 역할에서 쓴 원본 색을 원본 후보에서 뺀다(남는 게 50픽셀 미만이면 빼지 않음).
    배경 알파가 있으면 배경을 단색으로."""
    ref_skin = skin_like(reference_rgb, reference_parts["face"])
    src_skin = skin_like(source_rgb, source_parts["face"])
    lab, src_lab = rgb_to_lab(reference_rgb), rgb_to_lab(source_rgb)
    record = {}
    for role, rule in plan.items():
        ref_mask, src_mask = union(reference_parts, rule["reference"]), union(source_parts, rule["source"])
        if rule.get("split") == "left_right":
            lab, record[role] = match_eyes(lab, ref_mask, src_lab, src_mask, rule.get("chroma_min", 12.0), ref_skin, src_skin)
            record[role].update(reference_parts=rule["reference"], source_parts=rule["source"])
            continue
        if not rule.get("keep_skin", False):
            ref_mask, src_mask = ref_mask & ~ref_skin, src_mask & ~src_skin
        excluded = None
        other = rule.get("exclude_colours_of")
        if other and record.get(other, {}).get("source_centers"):
            narrowed = without_colours(src_lab, src_mask, record[other]["source_centers"])
            excluded = int(src_mask.sum() - narrowed.sum())
            if narrowed.sum() >= 50:
                src_mask = narrowed
            else:
                excluded = f"kept_all ({excluded} matched {other}; one-piece)"
        lab, record[role] = match_part(lab, ref_mask, src_lab, src_mask, rule.get("k", 1), rule.get("pick", "pair"),
                                       min_fraction=rule.get("min_fraction"))
        record[role].update(reference_parts=rule["reference"], source_parts=rule["source"], excluded_source_pixels=excluded)
    result = lab_to_rgb(lab).astype(np.float64)
    if background_alpha is not None:
        a = np.asarray(background_alpha, np.float64)[..., None]
        result = result * a + np.asarray(background_rgb, np.float64) * (1 - a)
    return np.clip(result + 0.5, 0, 255).astype(np.uint8), record
