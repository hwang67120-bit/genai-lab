"""자유 생성 머리색 되돌리기(D-130, 선택 마무리). 생성 결과의 머리 영역에서 색감(Lab a·b)만 기준 머리의 분포로 맞추고
명암(L)은 그대로 둔다. 원본 픽셀을 붙이지 않는다. 결과는 따로 저장한 '보정 후보'이고 사용자가 고를 때만 쓴다.

근거: 요청 21·24 N2-2 눈길에서 옅은 분홍 머리가 배경색 쪽으로 끌려 회색이 됨(머리 채도 14.6 → 8.2·11.7, 흰 배경은 유지).
분위수 대응이라 기준에서 분홍이 진하던 부분은 결과에서도 상대적으로 진한 쪽에 대응된다(위치 대응은 아님).
"""
import numpy as np

QUANTILES = np.linspace(0, 100, 101)
MIN_PIXELS = 200


def hair_ab_quantiles(image, mask):
    """기준 머리의 a·b 분위수(특징표에 저장). 픽셀이 적으면 None."""
    from genai_lab.character_features import to_lab
    pixels = to_lab(image)[mask]
    if len(pixels) < MIN_PIXELS:
        return None
    return dict(a=[round(float(v), 2) for v in np.percentile(pixels[:, 1], QUANTILES)],
                b=[round(float(v), 2) for v in np.percentile(pixels[:, 2], QUANTILES)])


def _lab_to_rgb(lab):
    import cv2
    raw = lab.copy()
    raw[..., 0] *= 255 / 100
    raw[..., 1:] += 128
    return cv2.cvtColor(np.clip(raw, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def restore_hair_color(image, hair_mask, reference_quantiles, strength=1.0):
    """image: RGB uint8, hair_mask: 같은 크기 bool(배경 뺀 머리), reference_quantiles: hair_ab_quantiles 결과.
    반환 (보정 RGB, 기록) 또는 (None, 이유)."""
    import cv2
    from genai_lab.character_features import to_lab
    if not reference_quantiles:
        return None, dict(status="unmeasurable", reason="기준 머리색 분포 없음")
    if int(hair_mask.sum()) < MIN_PIXELS:
        return None, dict(status="unmeasurable", reason="결과 머리 영역이 너무 작음")
    lab = to_lab(image)
    out = lab.copy()
    for channel, key in ((1, "a"), (2, "b")):
        source = lab[..., channel][hair_mask]
        mapped = np.interp(source, np.percentile(source, QUANTILES), np.array(reference_quantiles[key], np.float32))
        out[..., channel][hair_mask] = source + (mapped - source) * strength
    soft = cv2.GaussianBlur(hair_mask.astype(np.float32), (5, 5), 0)[..., None]       # 경계를 부드럽게(띠 방지)
    restored = (_lab_to_rgb(out) * soft + image * (1 - soft)).round().astype(np.uint8)
    before, after = lab[hair_mask], to_lab(restored)[hair_mask]
    chroma = lambda px: round(float(np.hypot(px[:, 1], px[:, 2]).mean()), 1)  # noqa: E731
    reference_chroma = round(float(np.hypot(np.array(reference_quantiles["a"]), np.array(reference_quantiles["b"])).mean()), 1)
    return restored, dict(status="restored", pixels=int(hair_mask.sum()), chroma_before=chroma(before), chroma_after=chroma(after),
                          reference_chroma=reference_chroma, lightness_change=round(float(np.abs(after[:, 0] - before[:, 0]).mean()), 2),
                          strength=strength, note="색감(a·b)만 기준 분포로, 명암(L) 유지. 원본 픽셀 합성 아님")
