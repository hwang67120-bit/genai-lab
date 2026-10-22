"""자유 생성(D-123) 공통 논리 — GUI·HTTP·GPU에 독립: 요청 문장 검사, 무작위 전신 자세, 조건 조립(D-112 구조), 부위 개수.

계약: docs/FREE_GENERATION_CONTRACT.md. 지시문은 코드 상수 + 확정 특징표 값 + 요청 문장이며 LLM·RAG는 없다.
요청 문장은 연출(자세·구도·시점·배경·상황)에 한정한다. 외형·의상을 바꾸는 말과 안전 정책에 걸리는 말은 거부한다(이유 표시).
"""
import re

from genai_lab.outfit_edit import body_parts_sentence

SEEDS = (42, 43)                       # 편집과 같음: 검사에 걸리면 두 번째 seed로 한 번 더
POSES = (                              # 코드 고정 목록. seed로 하나를 고르고 기록한다(LLM 없음, 결정 ⑤)
    "standing with the weight on one leg, one hand on the hip, looking at the viewer",
    "walking toward the viewer with the arms swinging naturally",
    "standing with the arms crossed, looking slightly to the side",
    "standing with the hands clasped behind the back and the head tilted a little",
    "standing in a three-quarter view, one hand raised in a small wave",
    "sitting on a simple stool with the knees together and the hands resting on the knees",
    "kneeling on one knee with one hand resting on the raised knee",
    "standing with the arms relaxed at the sides, looking over one shoulder toward the viewer",
)
# 명백한 변경 지시만 차단한다. 단어 출현은 의미 판정이 아니다.
# 이 검사에 걸리지 않았다는 사실도 허용 의미를 증명하지 않는다(사람이 연출 범위를 확인).
ACCESSORY = r"(?:hat|cap|beret|glasses|sunglasses|crown|tiara|ribbon|headband|hairband|hair ornament|earrings|mask|모자|안경|선글라스|왕관|티아라|리본|머리띠|귀걸이|가면)"
APPEARANCE_TARGET = r"(?:hair(?:style| color)?|eye color|skin(?: color)?|tail|ears|wings|horns?|outfit|clothes|clothing|dress|shirt|skirt|pants|jacket|coat|gender|" + ACCESSORY[3:-1] + r"|머리카락|머리색|머리 모양|눈 색|눈색|눈동자|피부|꼬리|귀|날개|뿔|옷|의상|치마|바지|성별)"
# 연출 대상(자세·각도·배경 등)을 바꾸는 change는 외형 변경이 아니다(검토 3번 오거부).
STAGING_OBJECT = r"(?:\s+(?:the|her|his|their|its)?\s*(?:pose|posture|angle|view|camera|composition|background|scene|position|expression|lighting|framing|direction))"
CHANGE = (r"(?:\b(?:change|replace|remove|add|recolor|dye|shorten|lengthen)\b(?!" + STAGING_OBJECT + r")"
          r"|바꾸|바꿔|변경|제거|없애|추가|염색|입혀|갈아입)")
# 길게·짧게·잘라는 풍경 묘사("노을이 길게 드리운")에도 쓰이므로 외형 대상 바로 뒤에 붙을 때만 변경으로 본다.
ATTACHED_LENGTH = r"(?:머리카락|머리|꼬리|귀|날개|뿔)[을를이가은는도]?\s{0,2}(?:더\s*)?(?:길게|짧게|잘라)"
SAFETY_PATTERNS = (
    r"\b(?:nude|naked|topless|undress|nsfw|erotic|genitals|nipples|loli|shota|childlike|toddler)\b",
    r"(?:알몸|누드|벗겨|벗은|노출시키|야한|어린이처럼|아동처럼|아동 외형|어린 외형|미성년)",
    r"\b(?:make|turn|change).{0,25}\b(?:child|kid|baby|young girl|young boy)\b",
)
# 배경 요청 낱말(2026-10-16 보강: flower field·garden·meadow가 인식되지 않아 흰 배경 문장과 충돌, 요청 26 준비 중 발견).
# 모호한 낱말은 뺀다: flower(머리 꽃 장식), sea-green 같은 붙임표 낱말, 한국어 다리(bridge/leg)·산(산책)·강(강하게)·역(역할)·성·밤(밤색)은 붙은 꼴만.
BACKGROUND_PATTERN = (r"\b(?:background|scene(?:ry)?|landscape|room|street|city|town|village|alley|road|forest|woods|garden|meadow|field|grassland"
                      r"|beach|shore|sea|ocean|lake|river|waterfall|mountain|hill|desert|cave|ruins|countryside|sky|night|indoors?|outdoors?"
                      r"|inside|outside|interior|park|cafe|office|library|classroom|school|kitchen|bedroom|castle|temple|shrine|station"
                      r"|rooftop|balcony|stage|market|shop|store|bridge|harbor|window|sunset|sunrise|dusk|dawn|rain|snow)(?:s|es)?(?![\w-])"
                      r"|\b(?:snowy|rainy|starry|moonlit|sunlit)\b"
                      r"|(?:배경|거리|도시|마을|골목|숲|정원|꽃밭|초원|들판|눈밭|해변|바닷가|바다|호수|강가|강변|폭포|산속|산길|산꼭대기|사막|동굴|폐허"
                      r"|하늘|실내|야외|공원|카페|도서관|교실|학교|부엌|침실|옥상|무대|기차역|지하철역|노을|해\s*질\s*녘|밤하늘|밤거리|밤에"
                      r"|눈(?:이)?\s*내리|비(?:가)?\s*내리)")
# 결과·기준 그림의 자동 차단 태그(태거 어휘에 실제로 있는 이름만, 검토 2번). 노출은 성인 캐릭터에도 차단한다.
EXPOSURE_TAGS = ("nude", "completely_nude", "topless", "bottomless", "nipples", "nipple_slip", "breasts_out", "pussy", "penis",
                 "anus", "pubic_hair", "sex", "cum", "pussy_juice")
AGE_BLOCK_TAGS = ("loli", "shota")
MALE_TOPLESS_ALLOWED = ("nipples", "topless")
# 나이 관련 태그는 차단하지 않고 근거로만 남긴다. SD·4등신 같은 그림체가 있어(사용자 결정 2026-10-14) child·chibi만으로 막지 않는다.
AGE_REVIEW_TAGS = ("child", "aged_down", "chibi")
SAFETY_THRESHOLD = 0.5
# 노출 검사 방식(D-158, 2026-10-23 사용자 "배포 단계가 아니면 제약을 두지 않는다 — 투명하게 다 작동하는지 알아야 나중에 제한할 때 오염을 막는다").
# "record"(배포 전): 노출 태그는 막지 않고 기록만 한다(trace의 exposure). "block"(배포 때): 이전 규칙(D-125·D-134) 그대로.
# 어느 방식이든 막는 것: loli·shota, 그리고 어려 보이는 신호(AGE_REVIEW_TAGS)와 노출이 함께 잡힌 경우.
EXPOSURE_POLICY = "record"
EXPOSURE_POLICIES = ("record", "block")
NO_PROPS = "Do not add weapons or held props unless the request asks for them. "
KEEP_OUTFIT = "Keep the character's outfit exactly as in image 1: the same garments, colors, patterns and details. "
KEEP_FACE = "Keep the face, hairstyle, hair color and head accessories the same as in image 1 and as in the head close-up in image 2. "
# 참조 크롭이 머리보다 넓으면(D-126 ③) "head close-up"이라고 부르지 않는다.
KEEP_FACE_WIDE = "Keep the face, hairstyle, hair color and head accessories the same as in image 1 and as in the cropped reference in image 2. "
# 어디에 무슨 색인지 말하지 않는 다색 태그는 문장에서 뺀다(N2-2 원색 추가 가설, D-126 ②). 특징표·검증에는 남는다.
PLACEMENT_FREE_HAIR = ("multicolored_hair", "gradient_hair", "streaked_hair", "two-tone_hair")
HAIR_PLACEMENT = "Keep the exact hair colors and where each color appears as in image 1; do not add new hair colors. "
# 요청과 충돌하는 자동 검사(D-126 ⑤): 눈 감기면 눈 색, 뒷모습이면 얼굴 항목을 재지 않는다.
EYES_CLOSED_REQUEST = (r"\b(?:eyes?\s+(?:are\s+)?closed|closed\s+eyes|clos(?:e|es|ing)\s+(?:her|his|their|the)\s+eyes|wink(?:s|ing)?)\b"
                       r"|눈(?:을|은)?\s*감|윙크")
BACK_VIEW_REQUEST = r"\b(?:from behind|back view|facing away|back to the viewer|seen from the back)\b|뒷모습|뒤돌아|등을\s*보"
# 손·꼬리를 가리는 연출(D-128, 사용자 "손·꼬리에도 같은 원칙 적용"). 명시적인 가림 표현만 본다.
HANDS_HIDDEN_REQUEST = (r"\b(?:(?:hands?|arms?)\s+(?:clasped\s+|folded\s+|held\s+)?behind\s+(?:her|his|their|the)\s+back"
                        r"|hands?\s+in\s+(?:her|his|their|the\s+)?\s*pockets?|hands?\s+(?:are\s+)?hidden|hiding\s+(?:her|his|their|the)\s+hands?)\b"
                        r"|뒷짐|손을?\s*(?:등\s*)?뒤로|손을?\s*주머니|손을?\s*(?:숨|가리)")
TAIL_HIDDEN_REQUEST = (r"\b(?:tail\s+(?:is\s+)?(?:hidden|not\s+visible|out\s+of\s+(?:view|frame))|hid(?:e|es|ing)\s+(?:her|his|their|the)\s+tail)\b"
                       r"|꼬리(?:를|가|는)?\s*(?:숨|가려|가리|안\s*보)")
# 요청이 가리는 특징은 지시문에서 '보이게 유지'를 요구하지 않는다(D-127, 요청 21 K-2: 눈 색 설명 + eyes closed → 눈을 뜸).
EYE_NOTE = r"\b(?:eyes?|iris|irises|pupils?|heterochromia)\b"
HAND_NOTE = r"\b(?:hands?|fingers?|fingernails?|nails?|gloves?|palms?|wrists?)\b"
TAIL_NOTE = r"\btails?\b"
HIDDEN_NOTES = {"eye_color": EYE_NOTE, "hands": HAND_NOTE, "tail": TAIL_NOTE}
# 요청 반영의 자동 관찰(태그). 관찰일 뿐 판정·재시도 근거가 아니다 — 요청 21 K-2 closed_eyes 0.00, 요청 20 K-2 1차 0.93.
REQUEST_TAGS = ((EYES_CLOSED_REQUEST, "eyes_closed", ("closed_eyes",)),
                (BACK_VIEW_REQUEST, "back_view", ("from_behind",)),
                (HANDS_HIDDEN_REQUEST, "hands_hidden", ("arms_behind_back", "hands_in_pockets", "hand_in_pocket")))
WHOLE_FIGURE = "Show the whole character inside the frame. "
# 요청 26(D-136): 연출 요청에 구도 말이 없으면 공원 서기 등에서 얼굴·상반신 확대로 당겨 그렸다(K·F5·M3 공원, M6 카페 잘림).
# 구도·시점을 말한 요청(클로즈업·상반신 등)은 덮어쓰지 않는다.
FULL_BODY = "Full-body shot: show the whole character from head to feet inside the frame; do not crop or zoom in on the face or upper body. "
COMPOSITION_REQUEST = (r"\b(?:close-?ups?|portraits?|headshots?|upper\s+body|half\s+body|bust|waist[\s-]+up|cowboy\s+shot|full[\s-]+body"
                       r"|face|crop(?:ped)?|zoom(?:ed)?|from\s+(?:above|below)|low\s+angle|high\s+angle)\b"
                       r"|클로즈업|얼굴|상반신|반신|허리\s*위|확대|전신|위에서|아래에서")
WHITE_BACKGROUND = "Place the character alone on a plain pure white background with nothing else. "
SCENE_COLORS = "Keep the character's own {} colors exactly as in image 1; the scene lighting must not change or wash out these colors. "
INTENT = "free_generation: keep the same character and outfit, change only the requested staging"


# 외부 호출자(사람·LLM 프롬프트 작성기)가 처리할 수 있는 고정 거부 코드(D-133, docs/FREE_GENERATION_API.md).
REJECTION_CODES = {
    "unsafe": "노출·아동 외형 등 금지된 요청",
    "multiple_characters": "여러 캐릭터 요청(한 캐릭터만 지원)",
    "outfit_change": "의상을 바꾸거나 입히는 요청(의상 교체는 편집 단계)",
    "appearance_change": "머리·눈·피부·부위·장식 등 외형을 바꾸는 요청",
    "request_too_long": "요청 문장 2000자 초과",
    "invalid_input": "형식이 맞지 않는 입력(기준 종류·성별·부위·seed·문자열 아님)",
    "invalid_features": "특징표 형식 오류(태그 항목·외형 태그·고유 형태 메모)",
    "confirmation_required": "성인 캐릭터·특징표·연출 범위 확인이 없음",
    "approval_required": "편집본 기준인데 같은 이미지의 사용자 의상 편집 승인이 없음",
    "unsafe_input": "기준 이미지에서 금지 태그가 관찰됨",
}


class FreeGenerationRejected(ValueError):
    """입력·요청 거부. code는 REJECTION_CODES의 키(고정), 문장은 사람용 설명."""

    def __init__(self, code, message, detail=None):
        if code not in REJECTION_CODES:
            raise KeyError(code)
        super().__init__(message)
        self.code, self.detail = code, detail

    def record(self):
        return dict(code=self.code, message=str(self), detail=self.detail)


REQUEST_RULES = (     # (코드, 패턴) — 앞의 코드가 우선
    ("unsafe", SAFETY_PATTERNS),
    ("multiple_characters", (r"\b(?:two|three|multiple)\s+(?:people|characters|persons)\b|(?:두|세)\s*(?:명|사람|캐릭터)",)),
    ("outfit_change", (r"\b(?:wear|wearing|put on)\b[^.;!?\n]{0,25}\b(?:dress|shirt|skirt|pants|jacket|coat|uniform|costume)\b",
                       r"(?:옷|의상|치마|바지|수영복|속옷)[^.;!?\n]{0,20}(?:입고|입은|입기)")),
    ("appearance_change", (CHANGE + r"[^.;!?\n]{0,35}" + APPEARANCE_TARGET, APPEARANCE_TARGET + r"[^.;!?\n]{0,35}" + CHANGE,
                           ATTACHED_LENGTH, r"\b(?:give|put|wear|wearing|put on)\b[^.;!?\n]{0,25}\b" + ACCESSORY + r"\b")),
)


def check_request(text):
    """명백한 계약 충돌의 문장 근거를 반환. 통과는 의미 승인이나 안전 보장이 아니다.
    거부면 code(고정 거부 코드)·codes(걸린 코드 전부)를 함께 준다."""
    if not isinstance(text, str):
        raise FreeGenerationRejected("invalid_input", "연출 요청은 문자열이어야 합니다.")
    text = text.strip()
    if len(text) > 2000:
        raise FreeGenerationRejected("request_too_long", "연출 요청은 2000자 이내로 써 주세요.")
    found = [(code, m.group(0)) for code, patterns in REQUEST_RULES for pattern in patterns for m in re.finditer(pattern, text, re.I)]
    matches = [m for _, m in found]
    codes = list(dict.fromkeys(code for code, _ in found))
    return dict(ok=not matches, reason="외형·의상 변경 또는 금지된 외형 요청은 자유 생성 범위 밖입니다." if matches else "",
                matched=matches, code=codes[0] if codes else None, codes=codes, empty=not text, semantic_status="not_reviewed",
                judge="rule", note="문장 패턴 검사는 보조 정보이며 사용자가 연출 범위를 확인해야 합니다.")


def safety_observation(scores, policy=None):
    """태그 점수에서 차단 근거·사람 확인용 근거·노출 기록을 나눈다. 통과는 안전 보장이 아니다(사람 체크가 최종).
    policy: EXPOSURE_POLICY(기본 "record") 또는 "block". 반환 dict(blocked, review, exposure, policy)."""
    policy = policy or EXPOSURE_POLICY
    if policy not in EXPOSURE_POLICIES:
        raise ValueError("노출 검사 방식 오류")
    scores = scores or {}
    exposure = {k: round(float(scores[k]), 3) for k in EXPOSURE_TAGS if float(scores.get(k, 0.0)) >= SAFETY_THRESHOLD}
    review = {k: round(float(scores[k]), 3) for k in AGE_REVIEW_TAGS if float(scores.get(k, 0.0)) >= SAFETY_THRESHOLD}
    age_blocked = sorted(k for k in AGE_BLOCK_TAGS if float(scores.get(k, 0.0)) >= SAFETY_THRESHOLD)
    if policy == "record":
        # 배포 전: 노출은 기록만. 단 어려 보이는 신호(loli·shota 또는 child·aged_down·chibi)와 노출이 함께면 막는다 — 단계와 무관.
        blocked = sorted({*age_blocked, *(exposure if (age_blocked or review) else ())})
        return dict(blocked=blocked, review=review, exposure=exposure, policy=policy)
    blocked = sorted({*exposure, *age_blocked})
    # 성인 남성 상의 탈의는 흔하고 노출 차단 대상이 아니다(사용자 결정 2026-10-16). 남성·상의 탈의 남성 태그가 확실할 때만
    # nipples·topless를 차단에서 빼고 기록한다. 다른 노출 태그와 나이 차단(loli·shota)은 그대로.
    if (float(scores.get("1boy", 0.0)) >= 0.7 and float(scores.get("topless_male", 0.0)) >= SAFETY_THRESHOLD
            and float(scores.get("1girl", 0.0)) < SAFETY_THRESHOLD):
        for tag in MALE_TOPLESS_ALLOWED:
            if tag in blocked:
                blocked.remove(tag)
                review[tag] = round(float(scores[tag]), 3)
    return dict(blocked=blocked, review=review, exposure=exposure, policy=policy)


def mentions_background(text):
    return bool(re.search(BACKGROUND_PATTERN, text, re.I))


def random_pose(seed):
    return POSES[int(seed) % len(POSES)]


def _plain(tag):
    return tag.replace("_", " ")


def distinctive_notes(notes):
    """고유 형태 메모 검사(D-126 ④). 반환: prompt(영어 메모만 — Qwen 전달), review(모두 — 결과 확인 창 체크 줄).
    외형·의상 변경 문장, 금지 표현은 거부한다. None(모름)은 빈 목록."""
    notes = list(notes or ())
    from genai_lab.character_features import MAX_DISTINCTIVE
    if len(notes) > MAX_DISTINCTIVE or any(not isinstance(n, str) or not n.strip() or len(n) > 60 for n in notes):
        raise FreeGenerationRejected("invalid_features", f"고유 형태 메모는 {MAX_DISTINCTIVE}개까지, 하나에 60자 이내로 써 주세요.")
    for note in notes:
        checked = check_request(note)
        if not checked["ok"]:
            raise FreeGenerationRejected("invalid_features", "고유 형태 메모에는 지금 모습만 적어 주세요(바꾸는 말·금지 표현 불가): "
                                         + ", ".join(checked["matched"]), detail=dict(codes=checked["codes"]))
    english = [n.strip() for n in notes if re.fullmatch(r"[A-Za-z0-9 ,'()/+-]+", n.strip())]
    return dict(prompt=english, review=[n.strip() for n in notes])


def staging_text(request_text, seed):
    """실제 연출 문장: 요청이 비면 seed로 고른 고정 자세(가림 판단도 실제 연출 기준, D-128)."""
    return request_text.strip() if (request_text or "").strip() else random_pose(seed)


def request_conflicts(text):
    """요청 문장이 일부러 가리는 특징 → {항목: 이유}. 그 항목은 지시문에서 보이게 유지하라고 하지 않고(features_sentence·
    free_conditions), 자동 검사에서 빼고(verify_features skip — 검증 항목 이름만 쓰인다) 사람이 본다."""
    text = text or ""
    skip = {}
    if re.search(EYES_CLOSED_REQUEST, text, re.I):
        skip["eye_color"] = "요청과 충돌: 눈 감기"
    if re.search(BACK_VIEW_REQUEST, text, re.I):
        skip.update(eye_color="요청과 충돌: 뒷모습", skin="요청과 충돌: 뒷모습")
    if re.search(HANDS_HIDDEN_REQUEST, text, re.I):
        skip["hands"] = "요청과 충돌: 손 가림"
    if re.search(TAIL_HIDDEN_REQUEST, text, re.I):
        skip.update(tail="요청과 충돌: 꼬리 가림", body_parts="요청과 충돌: 꼬리 가림")
    return skip


def request_observations(text, scores, threshold=SAFETY_THRESHOLD):
    """요청 문장 중 태그로 볼 수 있는 연출(눈 감기·뒷모습)이 결과에 보이는지. observed / not_observed / unmeasurable."""
    found = []
    for pattern, key, tags in REQUEST_TAGS:
        if not re.search(pattern, text or "", re.I):
            continue
        if scores is None:
            found.append(dict(key=key, status="unmeasurable", tags=list(tags), reason="tagger unavailable"))
            continue
        best = max(float(scores.get(t, 0.0)) for t in tags)
        found.append(dict(key=key, status="observed" if best >= threshold else "not_observed", tags=list(tags), score=round(best, 3),
                          note="태그 관찰. 요청 반영 판정은 사람이 한다"))
    return found


# 의상으로 분류되지만 의상이 아닌 태그(요청 23 N2-hands: 뒷짐이 hands_in_pockets로 잡혀 '의상 생김'에 섞임).
# 자세·드는 동작 태그와, 특징표가 따로 검사하는 머리 장식(hair_bow·hat 등)은 의상 차이에서 뺀다.
POSE_TAG = r"^(?:holding|hands?_|arms?_|own_|crossed_)|_in_pockets?$|_behind_back$"


def garment_tag_changes(base_scores, result_scores, threshold=SAFETY_THRESHOLD):
    """기준·결과 의상 태그(tag_scope garment)의 잃음·생김. 관찰만 — 자세 변화로 가려지기 쉬워 재시도 근거가 아니다(D-126 ④)."""
    from genai_lab.character_features import HEAD_ACCESSORY_TAGS
    from genai_lab.reference_features import tag_scope
    def garments(scores):
        return {k for k, v in (scores or {}).items() if float(v) >= threshold and tag_scope(k) == "garment"
                and not re.search(POSE_TAG, k) and k not in HEAD_ACCESSORY_TAGS}
    if base_scores is None or result_scores is None:
        return dict(status="unmeasurable", reason="tagger unavailable", lost=[], added=[])
    base, result = garments(base_scores), garments(result_scores)
    return dict(status="observed", lost=sorted(base - result), added=sorted(result - base), judge="automatic_analysis",
                note="태그 차이는 관찰 근거다. 가림·태그 노이즈가 많아 재시도·판정에 쓰지 않는다(사람이 의상 확인)")


SKIN_TONE = "Keep the exact skin tone as in image 1; do not make the skin lighter or darker. "
# D-136: 톤 문장만으로는 짙은 피부가 장면에서 밝아졌다(요청 26 M2 L* 31 → 70, 3줄 모두 실패). 기준 피부 픽셀 밝기(Lab L*)가
# 낮을 때만 낱말을 붙인다. 밝은 쪽(F5 L* 71)은 낱말 없이 톤 문장만 — dark skin 낱말이 더 어둡게 끌었다(D-134).
SKIN_WORDS = ((45, "dark brown skin"), (60, "brown skin"))


def skin_word(skin):
    lab = ((skin or {}).get("color") or {}).get("lab")
    if not lab:
        return None
    return next((word for limit, word in SKIN_WORDS if float(lab[0]) < limit), None)


def features_sentence(features, hidden=(), hair_color_words=True):
    """확정·측정된 특징표 값으로 한 문장. unknown/weak 항목은 뺀다. 아무것도 없으면 빈 문자열.
    hidden: 요청이 가리는 항목(request_conflicts의 키). 그 항목과 그에 관한 고유 형태 메모는 문장에 넣지 않는다(D-127)."""
    hidden = set(hidden or ())
    items = (features or {}).get("items", {})
    parts = []
    hair = []
    for key in ("hair_length", "hair_color", "hair_style"):
        item = items.get(key) or {}
        if item.get("state") != "measured":
            continue
        values = [item["value"]] if key == "hair_length" and item.get("value") else (item.get("tags") if key == "hair_color" else item.get("values")) or []
        if key == "hair_color" and not hair_color_words:      # 요청 28 시험 A: 머리색 낱말만 뺀다(색 배치 문장·검증은 그대로)
            values = []
        hair.extend(_plain(v) for v in values if v not in PLACEMENT_FREE_HAIR)
    if hair:
        parts.append(" ".join(hair) + " hair" if not any(h.endswith("hair") for h in hair) else " ".join(hair))
    eyes = (items.get("eye_color") or {})
    if eyes.get("state") == "measured" and eyes.get("tags") and "eye_color" not in hidden:
        parts.append(", ".join(_plain(v) for v in eyes["tags"]))
    acc = items.get("head_accessories") or {}
    if acc.get("state") == "measured" and acc.get("values"):
        parts.append("head accessories: " + ", ".join(_plain(v) for v in acc["values"]))
    # 피부는 태그 낱말(dark skin 등)로 적지 않는다(2026-10-16 통과 시험 F5: 자동 태그 dark_skin이 실제(RGB 212,163,145)보다
    # 어둡게 끌어 두 시도 모두 실패). 낱말은 기준 픽셀이 실제로 짙을 때만(D-136), 톤 문장은 늘 붙인다.
    skin = items.get("skin") or {}
    skin_kept = skin.get("state") == "measured" and skin.get("tags") and "skin" not in hidden
    if skin_kept and skin_word(skin):
        parts.append(skin_word(skin))
    text = ("The character has " + "; ".join(parts) + "; keep these exactly. ") if parts else ""
    if skin_kept:
        text += SKIN_TONE
    if (items.get("hair_color") or {}).get("state") == "measured":
        text += HAIR_PLACEMENT
    notes = (items.get("distinctive") or {})
    english = distinctive_notes(notes.get("values"))["prompt"] if notes.get("state") == "measured" else []
    english = [n for n in english if not any(re.search(HIDDEN_NOTES[k], n, re.I) for k in hidden if k in HIDDEN_NOTES)]
    if english:
        text += "Keep these distinctive details exactly as in image 1: " + "; ".join(english) + ". "
    return text


def free_conditions(features_text, gender, body_parts, request_text, seed, base_kind="edited", reference_scope="head", skeleton=False,
                    absent_parts_negative=False):
    """조건 목록(D-112 구조). features_text는 features_sentence()의 결과(확정 특징표에서 온 문장)."""
    conditions, excluded = [], []
    checked = check_request(request_text)
    if not checked["ok"]:
        raise FreeGenerationRejected(checked["code"], checked["reason"] + " (" + ", ".join(checked["matched"]) + ")",
                                     detail=dict(codes=checked["codes"], matched=checked["matched"]))
    conditions.append(dict(key="keep_character", text="Keep the same character as in image 1. " + (features_text or ""),
                           source="user_confirmed" if features_text else "policy",
                           reason="특징 유지 요청이며 동일성 보장은 아니다", basis="확정 특징표" if features_text else "특징표 없음(일반문만)"))
    if reference_scope not in ("head", "wide"):
        raise ValueError("참조 범위 오류")
    conditions.append(dict(key="keep_face", text=KEEP_FACE if reference_scope == "head" else KEEP_FACE_WIDE, source="policy",
                           reason="같은 기준 그림의 머리 크롭을 추가 시각 참조로 전달(동일성 보장 없음)" if reference_scope == "head"
                           else "머리보다 넓은 크롭이라 head close-up이라고 부르지 않는다(D-126 ③)", basis=dict(contract="D-123", scope=reference_scope)))
    conditions.append(dict(key="keep_outfit", text=KEEP_OUTFIT, source="policy", reason="자유 생성은 연출만 바꾼다", basis="D-098"))
    if gender in ("male", "female"):
        word = "male" if gender == "male" else "female"
        pronoun = "his" if gender == "male" else "her"
        conditions.append(dict(key="gender", text=f"The character is {word}. Keep {pronoun} body shape and proportions as in image 1. ",
                               source="user", reason="확정 성별", basis=gender))
    else:
        excluded.append(dict(key="gender", reason="성별 미지정"))
    parts_text = body_parts_sentence(None if body_parts is None else tuple(body_parts)).replace("color and position", "color and attachment to the body")
    if body_parts and "tail" in body_parts and "tail" in request_conflicts(staging_text(request_text, seed)):
        # 꼬리를 가리는 연출이면 '보이게 그대로'를 요구하지 않는다(D-128). 없애거나 더하지 말라는 조건은 남긴다.
        parts_text = parts_text.replace("keep them exactly as in image 1 with the same number, shape, color and attachment to the body",
                                        "keep them as in image 1 with the same number, shape, color and attachment to the body where visible; "
                                        "the tail may be hidden by the requested pose, but do not remove it or add extra tails")
    if absent_parts_negative and body_parts is not None:
        # 요청 28 시험 B: "no tail … do not add any" 부정 문장을 지시문에서 빼고 부정 지시문 칸으로(absent_parts_negative_prompt)
        parts_text = "" if not tuple(body_parts) else re.sub(r"The character has no [^;]*; do not add any\. ", "", parts_text)
        excluded.append(dict(key="absent_body_parts_sentence", reason="없는 부위는 부정 지시문 칸으로(요청 28 시험 B)",
                             negative=absent_parts_negative_prompt(body_parts)))
    conditions.append(dict(key="body_parts", text=parts_text,
                           source="user" if body_parts is not None else "policy", reason="확정 부위만 유지·추가 금지(D-111)", basis=list(body_parts) if body_parts else body_parts))
    if checked["empty"]:
        pose = random_pose(seed)
        conditions.append(dict(key="request", text=f"Full body: the character is {pose}. ", source="random_pose",
                               reason="요청 문장 없음 → 코드 고정 목록에서 seed로 선택(D-098)", basis=dict(seed=seed, index=int(seed) % len(POSES))))
    else:
        pose = None
        conditions.append(dict(key="request", text=request_text.strip().rstrip(".") + ". ", source="user", reason="사용자 요청(연출)", basis=request_text.strip()))
    if skeleton:      # D-131: 지원 자세의 골격 그림(image 3)으로 구조를 정한다(글보다 그림, 요청 25 시험)
        from genai_lab.pose_guidance import SKELETON_PROMPT
        conditions.append(dict(key="pose_skeleton", text=SKELETON_PROMPT, source="policy",
                               reason="지원 자세 목록의 골격을 캐릭터 비율로 맞춰 image 3으로 전달", basis="D-131"))
    if checked["empty"] or not mentions_background(request_text):
        conditions.append(dict(key="background", text=WHITE_BACKGROUND, source="policy", reason="배경 요청 없음 → 흰 배경(결정 ②)", basis=None))
    else:
        conditions.append(dict(key="background", text="", source="user", reason="배경 표현 후보가 있어 흰 배경 강제를 제외(실제 반영은 사람 검토)", basis=request_text))
        excluded.append(dict(key="white_background", reason="배경 요청 후보; 사용자 확인"))
        # 장면 조명이 캐릭터 고유 색을 덮지 않게 한다(D-129, 요청 21 N2-2 눈길에서 분홍 머리가 회색으로; 흰 배경 요청 23 N2-hands는 유지).
        # 요청이 가리는 특징(눈 감기 → 눈, 뒷모습 → 피부)은 보이게 요구하지 않는다(D-127).
        hidden = request_conflicts(request_text)
        own = [w for w, key in (("hair", None), ("eye", "eye_color"), ("skin", "skin"), ("outfit", None)) if key not in hidden]
        conditions.append(dict(key="scene_colors", text=SCENE_COLORS.format(", ".join(own[:-1]) + " and " + own[-1]), source="policy",
                               reason="배경·장면 요청이 있을 때만: 장면 조명과 관계없이 고유 색 유지", basis=dict(decision="D-129", items=own)))
    conditions.append(dict(key="no_props", text=NO_PROPS, source="policy", reason="요청에 없는 소품 금지", basis=None))
    if checked["empty"]:
        conditions.append(dict(key="framing", text=WHOLE_FIGURE, source="policy", reason="기본 무작위 자세는 전신", basis=None))
    elif not re.search(COMPOSITION_REQUEST, request_text, re.I):
        conditions.append(dict(key="framing", text=FULL_BODY, source="policy", basis="D-136",
                               reason="요청에 구도·시점 말이 없으면 전신(요청 26: 공원 서기 3명이 얼굴·상반신 확대로 실패)"))
    else:
        excluded.append(dict(key="framing", reason="사용자 연출의 구도·시점을 전신 강제로 덮어쓰지 않는다"))
    return dict(conditions=conditions, excluded=excluded, pose=pose, base_kind=base_kind)


def free_prompt(features_text, gender, body_parts, request_text, seed, reference_scope="head", skeleton=False, absent_parts_negative=False):
    return "".join(c["text"] for c in free_conditions(features_text, gender, body_parts, request_text, seed,
                                                       reference_scope=reference_scope, skeleton=skeleton,
                                                       absent_parts_negative=absent_parts_negative)["conditions"])


NEGATIVE_PART_WORDS = (("tail", "animal tail"), ("ears", "animal ears"), ("wings", "wings"))


def absent_parts_negative_prompt(body_parts):
    """요청 28 시험 B: 확인된 부위 목록에 없는 부위를 부정 지시문으로. 부위가 모름(None)이면 쓰지 않는다(기본 " ")."""
    if body_parts is None:
        return " "
    absent = [word for key, word in NEGATIVE_PART_WORDS if key not in tuple(body_parts)]
    return ", ".join(absent) if absent else " "


def part_counts(parts, min_px=200):
    """분할 연결 요소는 관찰값이다. 가림·합침·잘못된 분할 때문에 실제 부위 개수는 확인 불가."""
    import cv2
    import numpy as np
    report = {}
    for key in ("tail", "wings", "ears"):
        mask = parts.get(key)
        components = None
        if mask is not None:
            n, _, stats, _ = cv2.connectedComponentsWithStats(np.ascontiguousarray(mask).astype(np.uint8), connectivity=8)
            components = int(sum(stats[i, cv2.CC_STAT_AREA] >= min_px for i in range(1, n)))
        report[key] = dict(mask_components=components, count=None, status="unmeasurable",
                           reason="연결 요소 수는 해부학적 개수가 아니다. 가림·분할 오류·연결은 사람 검토 필요")
    return report
