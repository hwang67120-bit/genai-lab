"""자유 생성 머리색 자동 보정 판정(D-137, 오프라인 검증용 — 아직 파이프라인에 연결하지 않음).

흐름: 승인된 기준(특징표·기준 캔버스) 대비 결과 비교 → 조명 끼임인지 고유색 손실인지 구분 → 대상이면 기존 머리색 보정(D-130)
1회 → 독립 재검사(머리 밖 훼손·위치별 색 분포) → 통과한 보정본만 '자동 선택' 후보. 하나라도 실패·측정 불가면 원 결과 유지.
기준은 늘 승인된 기준 그림이다. 보정 결과를 다음 기준으로 올리지 않는다.

근거(2026-10-19, 요청 27 저장 결과 + 사용자 판정): M6 세 줄은 머리만 피부와 반대 방향으로 이동(cos −0.2~−0.55, 이동량 8.5~15.5)
→ 사람 실패, 보정본 2개 사람 통과. M3는 피부와 같은 방향(cos ≈ 1) → 사람 통과. 문턱은 이 자료를 보고 정했으므로 요청 26 결과로만
다시 채점한다(scripts/replay_hair_auto_restore.py).
"""
import numpy as np

SHIFT_MIN = 8.0          # 기준 대비 머리 a·b 중앙값 이동이 이보다 작으면 문제 아님
CAST_COS_MAX = 0.0       # 머리 이동과 피부 이동의 방향 코사인이 이 값 이하(반대·직각)일 때만 고유색 손실 후보. 같은 방향이면 장면 조명
MIN_CHROMA = 12.0        # 기준 머리의 평균 채도. 무채색(검정·흰·회색)은 색조가 잡음이라 대상 아님
MAX_CHROMA_RANGE = 15.0  # 주요 색 무리(비율 0.1 이상) 채도 폭. 넘으면 여러 색 머리(브릿지·끝 색) — 위치를 무시하는 보정이 망칠 수 있다
MAX_HUE_SPREAD = 30.0    # 주요 색 무리 색조 폭(도)
MINOR_FRACTION = 0.02    # 이 비율 이상인 작은 색 무리도 강한 색이면(채도 20↑) 여러 색으로 본다(F5 빨간 줄 0.02)
BAND_ERROR_MAX = 8.0     # 보정본 위·가운데·끝 a·b 중앙값이 기준과 평균 이만큼 안이어야 한다
COLLATERAL_MAX = 0.02    # 눈·얼굴 영역 중 바뀐 픽셀 비율 상한
CHANGE_TOLERANCE = 2     # RGB 차이 이하이면 안 바뀐 것으로 본다


def _median_ab(lab, mask):
    px = lab[mask]
    return np.array([np.median(px[:, 1]), np.median(px[:, 2])], np.float64)


def _hue(ab):
    return float(np.degrees(np.arctan2(ab[1], ab[0])) % 360)


def _hue_gap(a, b):
    gap = abs(a - b) % 360
    return min(gap, 360 - gap)


def reference_eligibility(hair_item):
    """특징표 머리색 항목에서 자동 보정 대상인지(채도 있는 한 가지 색 계열). 반환 dict(eligible, reason, …)."""
    palette = [c for c in (hair_item or {}).get("palette") or [] if c.get("fraction", 0) >= MINOR_FRACTION]
    if not palette or not (hair_item or {}).get("ab_quantiles"):
        return dict(eligible=False, reason="기준 머리 색 분포 없음")
    chroma = lambda c: float(np.hypot(c["lab"][1], c["lab"][2]))  # noqa: E731
    major = [c for c in palette if c["fraction"] >= 0.1]
    total = sum(c["fraction"] for c in major) or 1.0
    mean_chroma = sum(chroma(c) * c["fraction"] for c in major) / total
    if mean_chroma < MIN_CHROMA:
        return dict(eligible=False, reason=f"무채색 계열(평균 채도 {mean_chroma:.1f})", mean_chroma=round(mean_chroma, 1))
    chromas = [chroma(c) for c in major]
    hues = [_hue(c["lab"][1:]) for c in major]
    spread = max(_hue_gap(h1, h2) for h1 in hues for h2 in hues)
    main_hue = _hue(np.average([c["lab"][1:] for c in major], axis=0, weights=[c["fraction"] for c in major]))
    minor = [c for c in palette if c["fraction"] < 0.1 and chroma(c) >= 20 and _hue_gap(_hue(c["lab"][1:]), main_hue) > MAX_HUE_SPREAD]
    detail = dict(mean_chroma=round(mean_chroma, 1), chroma_range=round(max(chromas) - min(chromas), 1), hue_spread=round(spread, 1))
    if max(chromas) - min(chromas) > MAX_CHROMA_RANGE or spread > MAX_HUE_SPREAD or minor:
        return dict(eligible=False, reason="여러 색 머리(위치 무시 보정 위험)", **detail)
    return dict(eligible=True, reason="채도 있는 한 가지 색 계열", **detail)


def cast_test(reference_hair_ab, reference_skin_lab, result_lab, hair_mask, result_skin_lab):
    """머리 이동과 피부 이동을 비교해 장면 조명(같은 방향)인지 머리만의 변화인지. 피부를 못 재면 판단 불가."""
    if result_skin_lab is None or reference_skin_lab is None or int(hair_mask.sum()) < 200:
        return dict(status="unmeasurable", reason="피부 또는 머리 영역 측정 불가")
    hair_shift = _median_ab(result_lab, hair_mask) - np.asarray(reference_hair_ab, np.float64)
    skin_shift = np.asarray(result_skin_lab[1:], np.float64) - np.asarray(reference_skin_lab[1:], np.float64)
    size = float(np.linalg.norm(hair_shift))
    cos = float(hair_shift @ skin_shift / (size * float(np.linalg.norm(skin_shift)) + 1e-6))
    issue = size >= SHIFT_MIN and cos <= CAST_COS_MAX
    return dict(status="hair_only_shift" if issue else ("small_shift" if size < SHIFT_MIN else "follows_lighting"),
                hair_shift=round(size, 1), skin_shift=round(float(np.linalg.norm(skin_shift)), 1), cos=round(cos, 2))


def band_medians(lab, mask, bands=3):
    """머리 마스크를 세로로 위·가운데·끝으로 나눈 a·b 중앙값(위치별 색 분포). 픽셀이 적은 띠는 None."""
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) == 0:
        return [None] * bands
    edges = np.linspace(rows[0], rows[-1] + 1, bands + 1).astype(int)
    out = []
    for top, bottom in zip(edges[:-1], edges[1:]):
        part = np.zeros_like(mask)
        part[top:bottom] = mask[top:bottom]
        out.append(_median_ab(lab, part) if part.sum() >= 50 else None)
    return out


def band_error(reference_bands, bands):
    pairs = [(r, b) for r, b in zip(reference_bands, bands) if r is not None and b is not None]
    if len(pairs) < 2:
        return None
    return float(np.mean([np.linalg.norm(r - b) for r, b in pairs]))


def collateral(raw, restored, hair_mask, guard_masks):
    """보정이 머리 밖(경계 3px 제외)과 눈·얼굴을 건드렸는지."""
    import cv2
    changed = np.abs(raw.astype(np.int16) - restored.astype(np.int16)).max(axis=2) > CHANGE_TOLERANCE
    grown = cv2.dilate(hair_mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    outside = int((changed & ~grown).sum())
    guarded = {}
    for name, mask in guard_masks.items():
        if mask is not None and mask.sum() > 0:
            guarded[name] = round(float((changed & mask).sum() / mask.sum()), 3)
    return dict(outside_hair=outside, guarded=guarded, ok=outside == 0 and all(v <= COLLATERAL_MAX for v in guarded.values()))


def decide(features, base_canvas, base_parts, raw, result_parts, restore=None):
    """한 시도의 자동 보정 판정. features: 특징표, base_canvas/base_parts: 승인된 기준 그림과 그 분할(같은 좌표),
    raw/result_parts: 결과 그림과 그 결과에서 다시 찾은 분할(raw 크기로 맞춘 bool 마스크). restore: (image, mask, quantiles) → (보정본, 기록).
    반환 dict(decision ∈ not_applicable | no_issue | unmeasurable | candidate_only | auto_selected, …). 원 결과는 바꾸지 않는다."""
    from genai_lab.character_features import hair_mask_without_background, skin_color, to_lab
    if restore is None:
        from genai_lab.hair_color_restore import restore_hair_color as restore
    items = (features or {}).get("items", {})
    hair_item = items.get("hair_color") or {}
    record = dict(eligibility=reference_eligibility(hair_item))
    if not record["eligibility"]["eligible"]:
        return dict(record, decision="not_applicable", reason=record["eligibility"]["reason"])
    if "hair" not in result_parts:
        return dict(record, decision="unmeasurable", reason="결과 머리 영역 없음")
    hair = hair_mask_without_background(raw, result_parts["hair"])
    lab = to_lab(raw)
    quantiles = hair_item["ab_quantiles"]
    reference_ab = (quantiles["a"][50], quantiles["b"][50])
    skin = skin_color(raw, result_parts)
    record["cast"] = cast_test(reference_ab, ((items.get("skin") or {}).get("color") or {}).get("lab"), lab, hair,
                               skin["lab"] if skin else None)
    if record["cast"]["status"] == "unmeasurable":
        return dict(record, decision="unmeasurable", reason=record["cast"]["reason"])
    if record["cast"]["status"] != "hair_only_shift":
        return dict(record, decision="no_issue", reason=record["cast"]["status"])
    # 앞머리가 눈·얼굴을 덮으면 분할의 머리와 눈 영역이 겹친다(요청 26·27 M6: 눈 영역 3~10% 변경). 눈은 보정에서 뺀다.
    import cv2
    eyes = result_parts.get("eyes")
    protect = cv2.dilate(eyes.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) if eyes is not None else np.zeros_like(hair)
    restored, report = restore(raw, hair & ~protect, quantiles)
    record["restore"] = report
    if restored is None:
        return dict(record, decision="unmeasurable", reason=report.get("reason", "보정 실패"))
    base_hair = hair_mask_without_background(base_canvas, base_parts["hair"])
    reference_bands = band_medians(to_lab(base_canvas), base_hair)
    before, after = band_error(reference_bands, band_medians(lab, hair)), band_error(reference_bands, band_medians(to_lab(restored), hair))
    record["bands"] = dict(raw_error=None if before is None else round(before, 1), restored_error=None if after is None else round(after, 1))
    face = result_parts.get("face")
    skin_like = None
    if face is not None and skin:
        skin_like = face & (np.linalg.norm(lab - np.asarray(skin["lab"], np.float32), axis=2) < 10)    # 얼굴 중 피부색 픽셀
    record["collateral"] = collateral(raw, restored, hair, dict(eyes=eyes, face_skin=skin_like))
    if before is None or after is None:
        return dict(record, decision="candidate_only", reason="위치별 색 분포 측정 불가", restored=restored)
    if not record["collateral"]["ok"]:
        return dict(record, decision="candidate_only", reason="머리 밖·눈·얼굴 훼손", restored=restored)
    if not (after < before and after <= BAND_ERROR_MAX):
        return dict(record, decision="candidate_only", reason="위치별 색 분포가 기준에 충분히 가까워지지 않음", restored=restored)
    return dict(record, decision="auto_selected", reason="머리만의 색 변화 + 재검사 통과", restored=restored)
