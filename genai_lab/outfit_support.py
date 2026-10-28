"""원본 편집형 옷 바꾸기의 지원 범위 경고(D-113). 자동 태그 점수에서 알려진 실패 유형의 특징을 찾아 시작 전에 알린다.
조건을 바꾸지 않는다(그림별 조정 금지, D-104). 경고는 "이 그림은 지원 범위 밖이라 실패할 수 있고, 실패하면 알려 드린다"는 뜻이다.
규칙마다 근거 사례(요청 번호)를 적는다. 근거 없는 특징은 넣지 않는다.
"""

# (key, level, tags, threshold, message, basis)
#  level: outside = 지원 범위 밖(실패 유형이 반복 확인됨), notice = 달라질 수 있음(1회 이상 관찰)
SUPPORT_RULES = (
    ("pose_not_standing", "outside",
     ("sitting", "crouching", "squatting", "kneeling", "lying", "seiza", "wariza", "all_fours"), 0.5,
     "앉거나 웅크린 자세: Qwen이 선 자세로 다시 그려 머리 크기가 맞지 않아 원본 머리를 붙이지 못할 수 있습니다.",
     "S08(앉음) 요청 14·16 두 벌 모두 머리 합성 거부, 요청 15는 한 벌만 유지"),
    ("draped_outerwear", "outside",
     ("jacket_on_shoulders", "coat_on_shoulders", "clothes_on_shoulders", "cape"), 0.35,
     "어깨에 걸친 겉옷·망토: 원래 겉옷이 새 옷 위에 남을 수 있습니다.",
     "K(걸친 코트) 요청 14·15·16 모두 코트 잔존"),
    ("mechanical_body", "outside",
     ("mechanical_tail", "mechanical_arms", "mechanical_parts", "mechanical_legs", "cyborg", "android", "robot_joints"), 0.35,
     "기계 몸·기계 의상: 몸의 일부로 읽혀 옷이 바뀌지 않을 수 있습니다.",
     "S06(기계 몸) 요청 14·15·16 모두 의상 미교체"),
    ("multiple_tails", "notice",
     ("multiple_tails",), 0.5,
     "꼬리가 여럿: 개수가 줄거나 합쳐져 보일 수 있습니다.",
     "S08(꼬리 4개) 요청 15·16에서 2개처럼 보임"),
)
TAIL_NOTICE = ("tail_shape", "notice", "꼬리 모양·색이 원본과 달라질 수 있습니다(꼬리는 Qwen이 그립니다).",
               "S04(가는 보라 꼬리) 요청 15·16에서 굵은 꼬리·흰 끝")


def support_warnings(character_scores, body_parts=None):
    """태그 점수(0~1)와 확인된 부위로 경고 목록을 만든다. 각 항목: key, level, message, evidence(태그: 점수), basis."""
    scores = character_scores or {}
    found = []
    for key, level, tags, threshold, message, basis in SUPPORT_RULES:
        evidence = {tag: round(float(scores[tag]), 3) for tag in tags if float(scores.get(tag, 0.0)) >= threshold}
        if evidence:
            found.append(dict(key=key, level=level, message=message, evidence=evidence, basis=basis))
    if body_parts and "tail" in body_parts and not any(w["key"] == "multiple_tails" for w in found):
        key, level, message, basis = TAIL_NOTICE
        found.append(dict(key=key, level=level, message=message, evidence={"body_parts": "tail"}, basis=basis))
    return found


def support_summary(warnings):
    """상태 줄·확인 창용 한 문장. 경고가 없으면 빈 문자열."""
    outside = [w["message"] for w in warnings if w["level"] == "outside"]
    notice = [w["message"] for w in warnings if w["level"] == "notice"]
    parts = []
    if outside:
        parts.append("지원 범위 밖 특징 " + str(len(outside)) + "개: " + " / ".join(outside) + " 실패하면 알려 드립니다.")
    if notice:
        parts.append("참고: " + " / ".join(notice))
    return " ".join(parts)
