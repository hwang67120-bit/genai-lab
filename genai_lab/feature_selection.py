"""특징 선택 단계(D-140): 특징표 항목마다 생성 조건 / 관찰·검토 기록 / 요청 때문에 제외로 나누고, 이미지 근거와의 충돌을 기록한다.

생성 조합은 고정(D-140)이라 이 단계는 지시문을 바꾸지 않는다 — 분류와 충돌 경고를 trace에 남겨 사람이 입력 단계에서 보게 하고,
운영 집계(20건마다)에서 '근거가 약한 특징을 조건으로 쓴 실패'를 따로 셀 수 있게 한다. 검증 기준에서는 어떤 항목도 지우지 않는다.
충돌 규칙은 색 이름별 규칙을 늘리지 않으려고 굵게만 둔다: 태그 점수가 약함, 유채색/무채색 어긋남, 밝기 어긋남.
근거: 요청 26 전 통과 시험 F5 dark_skin(실제 피부 L* 71) — 낱말이 피부를 지나치게 어둡게 끈 사례(D-134).
"""
import numpy as np

from genai_lab.character_features import WEAK_THRESHOLD

ACHROMATIC_HAIR = ("black_hair", "white_hair", "grey_hair", "silver_hair")
CHROMATIC_HAIR = ("blue_hair", "pink_hair", "red_hair", "green_hair", "purple_hair", "orange_hair", "aqua_hair", "light_blue_hair")
CHROMATIC_MIN = 8.0          # 유채색 머리 태그인데 실제 채도가 이보다 낮으면 충돌
ACHROMATIC_MAX = 15.0        # 무채색 머리 태그인데 실제 채도가 이보다 높으면 충돌
DARK_MAX_L = 50.0            # black_hair / dark_skin 인데 이보다 밝으면 충돌
LIGHT_MIN_L = 55.0           # white_hair·silver_hair·pale_skin 인데 이보다 어두우면 충돌
DARK_SKIN_MAX_L = 60.0
TAG_ITEMS = ("hair_length", "hair_style", "hair_color", "eye_color", "head_accessories", "skin")


def _tags(item, key):
    if key == "hair_length":
        return [item["value"]] if item.get("value") else []
    return list(item.get("tags") if key in ("hair_color", "eye_color", "skin") else item.get("values") or [])


def _hair_clusters(item):
    """주요 머리 색 무리(비율 0.1 이상)의 밝기·채도. 여러 색 머리(흰 브릿지·검은 뿌리)는 무리 하나라도 맞으면 충돌로 보지 않는다."""
    return [dict(L=float(c["lab"][0]), chroma=float(np.hypot(c["lab"][1], c["lab"][2])))
            for c in item.get("palette") or [] if c.get("fraction", 0) >= 0.1]


def color_conflicts(key, tags, item):
    """색 태그와 측정 픽셀의 굵은 어긋남. 반환: 충돌 문장 목록."""
    found = []
    if key == "hair_color":
        clusters = _hair_clusters(item)
        if not clusters:
            return found
        top_chroma = round(max(c["chroma"] for c in clusters), 1)
        for tag in tags:
            if tag in CHROMATIC_HAIR and top_chroma < CHROMATIC_MIN:
                found.append(f"{tag}인데 측정 채도 최대 {top_chroma}(무채색에 가까움)")
            if tag in ACHROMATIC_HAIR and all(c["chroma"] > ACHROMATIC_MAX for c in clusters):
                found.append(f"{tag}인데 모든 색 무리가 채도 {ACHROMATIC_MAX} 초과(색이 있음)")
            if tag == "black_hair" and all(c["L"] > DARK_MAX_L for c in clusters):
                found.append(f"black_hair인데 가장 어두운 무리 L* {round(min(c['L'] for c in clusters), 1)}")
            if tag in ("white_hair", "silver_hair") and all(c["L"] < LIGHT_MIN_L for c in clusters):
                found.append(f"{tag}인데 가장 밝은 무리 L* {round(max(c['L'] for c in clusters), 1)}")
    if key == "skin":
        lab = (item.get("color") or {}).get("lab")
        if lab:
            if "dark_skin" in tags and lab[0] > DARK_SKIN_MAX_L:
                found.append(f"dark_skin인데 측정 밝기 L* {round(lab[0], 1)}(밝은 쪽)")
            if "pale_skin" in tags and lab[0] < LIGHT_MIN_L:
                found.append(f"pale_skin인데 측정 밝기 L* {round(lab[0], 1)}")
    return found


def select_features(features, base_scores, hidden=()):
    """반환 dict(items={항목: dict(use, reason, conflicts)}, warnings=[...]). use ∈ condition | observation | excluded.
    condition은 지금 지시문에 들어가는 항목(바꾸지 않음), 충돌이 있으면 warnings에 올린다."""
    items, warnings = {}, []
    hidden = set(hidden or ())
    for key in TAG_ITEMS:
        item = (features or {}).get("items", {}).get(key) or {}
        if item.get("state") != "measured":
            items[key] = dict(use="observation", reason="모름·약한 특징은 조건에 쓰지 않고 관찰만")
            continue
        if key in hidden:
            items[key] = dict(use="excluded", reason="요청이 가리는 특징 — 지시문에서 제외, 검증 기준에는 남김(D-127)")
            continue
        tags = _tags(item, key)
        conflicts = [f"{t} 태그 점수 {round(float(base_scores.get(t, 0.0)), 2)}(이미지 근거 약함)"
                     for t in tags if float((base_scores or {}).get(t, 0.0)) < WEAK_THRESHOLD]
        conflicts += color_conflicts(key, tags, item)
        items[key] = dict(use="condition", reason="확정 특징", source=item.get("source"), conflicts=conflicts)
        warnings += [f"{key}: {c}" for c in conflicts]
    return dict(schema="feature_selection/1", items=items, warnings=warnings,
                note="생성 조합 고정(D-140): 분류·경고만 기록하고 지시문은 바꾸지 않는다. 검증 기준은 그대로.")
