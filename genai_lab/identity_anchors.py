"""고정 특징표(D-228, 사용자 10/31 "그림체와 의상이 바뀌어도 알아볼 수 있는 특징을 판단할 기준이 없다").

동일성 = 픽셀이 아니라 캐릭터를 알아보게 하는 특징의 유지(완성 기준 2). 원본 그림의 태그(WD 태거)에서 '바뀌면 안 되는 것'을 뽑고,
결과 그림의 태그와 항목마다 비교한다: 유지(kept) · 바뀜(changed, 같은 묶음의 다른 값이 강함) · 사라짐(lost) · 판단 불가(unclear,
예: 눈 색 태그가 결과에 하나도 없음 = 눈이 안 보임). 그림체·옷(옷 그림을 넣었을 때)·자세·배경·조명·표정은 비교하지 않는다.
자동 판정은 사람 판정의 근거를 보여 주는 것이지 대신하지 않는다.
"""
from genai_lab.character_features import (EYE_COLOR_TAGS, HAIR_COLOR_TAGS, HAIR_LENGTH_TAGS, HAIR_STYLE_TAGS, HEAD_ACCESSORY_TAGS,
                                          SKIN_TAGS)

ANCHOR_MIN = 0.5            # 원본에서 이만큼 확실해야 고정 특징으로 삼는다
KEPT_MIN = 0.35             # 결과에서 이 이상이면 유지(그림체가 바뀌면 태그 확신이 낮아지는 것을 감안)
# 머리 모양 중 그림체·각도에 따라 흔들리는 일반 낱말은 고정 특징에서 뺀다(앞머리 유무·직모 등).
UNSTABLE_STYLE = ("bangs", "straight_hair", "messy_hair", "hair_between_eyes", "sidelocks", "hair_intakes", "hair_flaps",
                  "swept_bangs", "parted_bangs", "blunt_bangs")
MARK_TAGS = ("scar", "scar_on_face", "scar_across_eye", "mole", "mole_under_eye", "facial_mark", "facial_tattoo", "freckles",
             "whisker_markings", "beauty_mark")
# 묶음: (이름, 화면 이름, 태그들, 한 가지만 맞는 묶음인가)
HAIR_PATTERN_TAGS = ("multicolored_hair", "gradient_hair", "streaked_hair", "two-tone_hair", "colored_inner_hair", "colored_tips")
GROUPS = (
    ("hair_color", "머리색", tuple(t for t in HAIR_COLOR_TAGS if t not in HAIR_PATTERN_TAGS), True),
    ("hair_pattern", "머리 무늬", HAIR_PATTERN_TAGS, False),
    ("hair_length", "머리 길이", HAIR_LENGTH_TAGS, True),
    ("hair_style", "머리 모양", tuple(t for t in HAIR_STYLE_TAGS if t not in UNSTABLE_STYLE), False),
    ("eye_color", "눈 색", EYE_COLOR_TAGS, True),
    ("skin", "피부", SKIN_TAGS, True),
    ("head_accessories", "머리 장식", HEAD_ACCESSORY_TAGS, False),
    ("marks", "고유 표시", MARK_TAGS, False),
)
BODY_PART_LABELS = {"ears": "동물 귀", "tail": "꼬리", "wings": "날개"}


def anchors(scores):
    """원본 태그 → 고정 특징 목록 [dict(group, label, tag, score)]. 동물 귀·꼬리·날개는 사용자 화면과 같은 규칙(guessed_body_parts)."""
    from genai_lab.studio_controller import guessed_body_parts
    scores = scores or {}
    found = []
    for group, label, tags, exclusive in GROUPS:
        strong = sorted(((float(scores.get(t, 0.0)), t) for t in tags if float(scores.get(t, 0.0)) >= ANCHOR_MIN), reverse=True)
        # 한 가지 값만 맞는 묶음(머리색·눈 색·길이·피부)은 가장 강한 하나만 — 두 개를 다 잡으면 서로를 '바뀜'으로 읽는다(U004 눈 색)
        for value, tag in (strong[:1] if exclusive else strong):
            found.append(dict(group=group, label=label, tag=tag, score=round(value, 2)))
    for part in guessed_body_parts(dict(character_scores=scores)):
        found.append(dict(group="body_parts", label=BODY_PART_LABELS[part], tag=part, score=None))
    return found


def compare(original_anchors, result_scores):
    """고정 특징마다 결과에서 유지·바뀜·사라짐·판단 불가. 반환: dict(items, kept, changed, lost, unclear, score)."""
    from genai_lab.studio_controller import guessed_body_parts
    result_scores = result_scores or {}
    result_parts = guessed_body_parts(dict(character_scores=result_scores))
    exclusive = {g: tags for g, _, tags, one in GROUPS if one}
    items = []
    for a in original_anchors:
        if a["group"] == "body_parts":
            status, now = ("kept", None) if a["tag"] in result_parts else ("lost", None)
        else:
            value = float(result_scores.get(a["tag"], 0.0))
            now = None
            others = (sorted(((float(result_scores.get(t, 0.0)), t) for t in exclusive[a["group"]] if t != a["tag"]), reverse=True)
                      if a["group"] in exclusive else [])
            if value >= KEPT_MIN and not (others and others[0][0] >= ANCHOR_MIN and others[0][0] > value + 0.3):
                status = "kept"
            elif a["group"] in exclusive:
                if others and others[0][0] >= ANCHOR_MIN:
                    status, now = "changed", others[0][1]
                elif a["group"] == "eye_color" and max([value] + [s for s, _ in others]) < 0.2:
                    status = "unclear"          # 눈 색 태그가 하나도 없다 = 눈이 안 보이는 각도(뒷모습 등)
                else:
                    status = "lost"
            else:
                status = "lost"
            a = dict(a, result_score=round(value, 2))
        items.append(dict(a, status=status, now=now))
    count = lambda s: sum(1 for i in items if i["status"] == s)
    judged = count("kept") + count("changed") + count("lost")
    return dict(items=items, kept=count("kept"), changed=count("changed"), lost=count("lost"), unclear=count("unclear"),
                score=round(count("kept") / judged, 3) if judged else None)


def summary_ko(report):
    """화면·판정표용 한 줄: '고정 특징 5/6 유지 · 바뀜: 머리색(blonde→brown) · 사라짐: 리본'."""
    if not report or report.get("score") is None:
        return "고정 특징: 판단할 특징 없음"
    judged = report["kept"] + report["changed"] + report["lost"]
    parts = [f"고정 특징 {report['kept']}/{judged} 유지"]
    changed = [f"{i['label']}({i['tag']}→{i['now']})" for i in report["items"] if i["status"] == "changed"]
    lost = [f"{i['label']}({i['tag']})" for i in report["items"] if i["status"] == "lost"]
    if changed:
        parts.append("바뀜: " + ", ".join(changed))
    if lost:
        parts.append("사라짐: " + ", ".join(lost))
    return " · ".join(parts)
