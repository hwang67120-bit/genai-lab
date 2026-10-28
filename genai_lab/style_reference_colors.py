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
- 피부처럼 보이는 픽셀은 옷 역할에서 뺀다(분할기가 맨살을 옷으로 잡는 경우, D-187 v4 첫 시도).
"""
import numpy as np

_M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


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
    return ((hsv[..., 0] >= h5 - 6) & (hsv[..., 0] <= h95 + 6) & (hsv[..., 1] >= max(s5 - 20, 18))
            & (hsv[..., 1] <= s95 + 40) & (hsv[..., 2] > 50))


def clusters(lab, k, iterations=12, merge_ab=12.0):
    """Lab 픽셀을 k개로 나누고 밝기 오름차순 번호를 붙인다(재현되게 밝기 분위수로 시작).
    k=2에서 두 무리의 a·b 거리가 merge_ab보다 가까우면 같은 색의 명암 차이로 보고 하나로 합친다."""
    lab = np.asarray(lab, np.float64)
    if k == 1 or len(lab) < k * 20:
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


def match_part(reference_lab, reference_mask, source_lab, source_mask, k=1, source="pair", chroma_keep=0.1,
               lightness_ratio=(0.5, 1.2)):
    """참조 부위의 Lab을 원본 부위의 색으로 옮긴다. source="largest"면 원본을 둘로 나눠 넓은 쪽 색 하나만 쓴다.
    반환: 새 Lab(부위 밖은 그대로)과 기록."""
    out = reference_lab.copy()
    ref_px, src_px = reference_lab[reference_mask], source_lab[source_mask]
    if len(ref_px) < 50 or len(src_px) < 50:
        return out, dict(state="skipped_too_small", reference_pixels=int(len(ref_px)), source_pixels=int(len(src_px)))
    ref_label, ref_centers = clusters(ref_px, k, merge_ab=0.0)
    src_label, src_centers = clusters(src_px, 2 if source == "largest" else k)
    largest = int(np.bincount(src_label).argmax())
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
                     source_colours=len(src_centers), pairs=pairs)


def union(parts, names):
    return np.logical_or.reduce([np.asarray(parts[n], bool) for n in names])


def recolor_reference(reference_rgb, reference_parts, source_rgb, source_parts, plan, background_alpha=None,
                      background_rgb=(214, 210, 202)):
    """plan: {역할: dict(reference=[부위…], source=[부위…], k=색 수, pick="pair"|"largest", keep_skin=False)}.
    keep_skin이 아니면(옷 역할) 양쪽 모두 피부처럼 보이는 픽셀을 뺀다. 배경 알파가 있으면 배경을 단색으로."""
    ref_skin = skin_like(reference_rgb, reference_parts["face"])
    src_skin = skin_like(source_rgb, source_parts["face"])
    lab, src_lab = rgb_to_lab(reference_rgb), rgb_to_lab(source_rgb)
    record = {}
    for role, rule in plan.items():
        ref_mask, src_mask = union(reference_parts, rule["reference"]), union(source_parts, rule["source"])
        if not rule.get("keep_skin", False):
            ref_mask, src_mask = ref_mask & ~ref_skin, src_mask & ~src_skin
        lab, record[role] = match_part(lab, ref_mask, src_lab, src_mask, rule.get("k", 1), rule.get("pick", "pair"))
        record[role].update(reference_parts=rule["reference"], source_parts=rule["source"])
    result = lab_to_rgb(lab).astype(np.float64)
    if background_alpha is not None:
        a = np.asarray(background_alpha, np.float64)[..., None]
        result = result * a + np.asarray(background_rgb, np.float64) * (1 - a)
    return np.clip(result + 0.5, 0, 255).astype(np.uint8), record
