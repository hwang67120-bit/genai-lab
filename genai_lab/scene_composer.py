"""사용자 화면의 장면 자동 조합(D-153, 2판 D-155).

고정 장면 몇 개에서 고르지 않고, 장소 · 동작 · 표정 · 시간/빛 · 조명 설계 · 색 구성 · 날씨 · 카메라를 각각 무작위로 골라 매번 새 장면 문장을 만든다.
늘 지키는 것: 전신(머리부터 발끝까지), 배경 있음, 동일성 확인 가능(얼굴이 보이는 자세만 — 뒷모습·얼굴 가림·눈 감기·손 숨기기 제외).
2판(2026-10-22 사용자 "색감·빛·비율·표정·해상도 차이, 한눈에 예쁜 작품"):
- 비율을 망가뜨린 카메라(높은 각도·넓게 잡은 화면)와 형광처럼 번진 안개를 뺐다(첫 GUI 결과: 머리 크고 다리 가늘게 왜곡, 푸른 안개 빛남).
- 조명 설계(역광·테두리 빛·창가 햇빛 등)와 색 구성을 넣었다. 색 구성은 '배경과 조명에만' — 캐릭터 고유 색은 지시문의 색 유지 문장이 지킨다.
- 표정과 몸짓(무게중심·손 위치·고개 기울기)을 구체적으로 썼다.
광택·반짝이를 줄이는 문장은 넣지 않았다 — 그림체 단계에서 한 요인으로 따로 시험한다(STYLE_STAGE_DESIGN 3-2).
장면 문장은 기존 요청 검사(free_generation.check_request)를 그대로 지난다. 평가 세트·개발 화면의 '요청 없음 = 고정 자세 + 흰 배경'(D-098)은 그대로다.
"""
import random

VERSION = 2
FULL_BODY_TEXT = "Full-body shot: the whole character is visible from head to feet, with a detailed background."

OUTDOOR = (
    "on a quiet city street with neon signs", "in a forest clearing", "on a rooftop overlooking the city",
    "on a stone bridge over a river", "in a park with autumn trees", "on a sandy beach by the sea", "in a snowy mountain village",
    "in a field of tall grass under a wide sky", "on an empty train station platform", "in an old town alley with paper lanterns",
    "by a calm lake surrounded by mountains", "in a desert with distant dunes", "in a castle courtyard", "at a seaside harbor with small boats",
    "in a bamboo forest", "on a hill overlooking the countryside", "in a garden with stone paths", "at a shrine with red gates",
    "in front of ancient stone ruins", "on a balcony above a canal town",
)
INDOOR = (
    "in a cozy cafe with wooden tables", "in a library with tall bookshelves", "in a bright classroom with large windows",
    "in a greenhouse full of plants", "in a quiet museum hall", "in a kitchen with morning light through the window",
    "in an artist studio with canvases", "in a grand castle hall with tall pillars", "in a small bookstore with warm shelves",
    "in an aquarium tunnel with blue water light",
)
ACTIONS = (
    "standing with the weight on one leg, one hand on the hip and the head slightly tilted",
    "giving a peace sign near the cheek with the other hand relaxed at the side",
    "walking with a light step, one arm swinging forward",
    "turning slightly with a glance over the shoulder, the face toward the viewer",
    "leaning against a wall with one foot crossed over the other and the arms loosely folded in front",
    "sitting sideways on a chair with one arm resting on the backrest",
    "sitting on stone steps with the elbows on the knees and the chin resting on one hand",
    "crouching down with the arms resting on the knees, looking up at the viewer",
    "kneeling on one knee with one hand on the raised knee",
    "stretching one arm up toward the sky with a lively pose",
    "tucking a strand of hair behind one ear with one hand",
    "holding an open umbrella over one shoulder",
    "holding a book against the chest with both hands",
    "holding a cup of tea in both hands near the chest",
    "jumping lightly with both feet off the ground and the arms spread",
    "spinning in place with the arms out and a light motion",
    "stepping forward with one hand raised in a small wave",
    "pointing ahead with one hand while the other rests on the hip",
    "petting a small cat that sits beside the feet",
    "carrying a paper shopping bag in one hand, the other hand at the side",
    "reaching up to touch a hanging branch",
    "standing on tiptoes with the hands joined in front",
    "leaning forward slightly with the hands clasped in front and a playful pose",
    "sitting on a low wall with the legs crossed at the ankles",
)
EXPRESSIONS = (
    "with a confident smile", "with a playful grin", "with a gentle smile", "with a calm, cool expression",
    "with a slightly shy smile", "with a cheerful open-mouthed smile", "with a curious look", "with a soft, relaxed expression",
    "with a proud, smug smile", "with a dreamy look",
)
# 시간대(낮/밤)를 먼저 고르고 그에 맞는 시간·조명만 고른다(1판 견본에 '햇빛 숲 + 밤 가로등' 같은 모순이 있었다).
# 조명 문장에 'from behind'를 쓰지 않는다 — 요청 규칙이 뒷모습으로 읽어 눈 색·피부 검사를 뺀다(free_generation.BACK_VIEW_REQUEST).
TIME = {
    ("outdoor", "day"): ("in soft morning light", "under the bright midday sun", "at golden hour", "at sunset", "on an overcast day"),
    ("outdoor", "night"): ("at dusk", "at night under street lamps", "under a starry night sky"),
    ("indoor", "day"): ("in the morning", "at midday", "in the late afternoon"),
    ("indoor", "night"): ("in the evening", "at night"),
}
LIGHTING = {
    ("outdoor", "day"): ("warm backlight with a soft rim light outlining the character", "dappled sunlight falling through leaves",
                         "strong side light with clear, soft shadows", "golden backlight with a gentle lens glow",
                         "soft diffused light with gentle contrast"),
    ("outdoor", "night"): ("cool ambient light with warm accent lights in the background", "colorful neon accent light on one side",
                           "lantern glow mixed with cool evening light", "moonlight with warm lights in the distance"),
    ("indoor", "day"): ("warm sunlight streaming through a window", "soft window light from one side",
                        "bright clean daylight with soft bounce light", "a soft rim light outlining the character"),
    ("indoor", "night"): ("warm lamp light with gentle shadows", "cozy mixed light from lamps and candles",
                          "a soft rim light outlining the character", "warm string lights in the background"),
}
PALETTES = (
    "warm orange and teal", "pastel pink and mint green", "deep blue with gold accents", "soft cream with emerald accents",
    "sunny yellow and sky blue", "lavender and peach", "crimson and charcoal", "fresh green and white", "amber and navy",
    "coral and turquoise",
)
WEATHER = ("", "", "", "", "with light rain falling", "with snow falling gently", "with cherry petals drifting in the wind",
           "with autumn leaves blowing", "with a gentle breeze moving the scene")
CAMERA = ("eye-level view", "eye-level view", "eye-level view", "slightly low camera angle", "three-quarter view at eye level")


def compose_scene(seed):
    """seed마다 같은 장면(재현용). 반환: dict(text, parts, seed, version, source)."""
    rng = random.Random(seed)
    indoor = rng.random() < 0.3
    key = ("indoor" if indoor else "outdoor", "night" if rng.random() < 0.3 else "day")
    parts = dict(indoor=indoor, period=key[1], place=rng.choice(INDOOR if indoor else OUTDOOR), action=rng.choice(ACTIONS),
                 expression=rng.choice(EXPRESSIONS), time=rng.choice(TIME[key]),
                 lighting=rng.choice(LIGHTING[key]), palette=rng.choice(PALETTES),
                 weather="" if indoor else rng.choice(WEATHER), camera=rng.choice(CAMERA))
    scene = f"{parts['action']}, {parts['expression']}, {parts['place']} {parts['time']}" + (f", {parts['weather']}" if parts["weather"] else "")
    text = (f"The character is {scene}. Lighting: {parts['lighting']}. "
            f"Color design: a {parts['palette']} palette in the background and lighting only. "
            f"{parts['camera'][0].upper() + parts['camera'][1:]}, natural body proportions. {FULL_BODY_TEXT}")
    return dict(text=text, seed=seed, version=VERSION, parts=parts, source="scene_composer")


def complete_user_scene(text, seed):
    """사람이 쓴 장면. 배경 말이 없으면 장소·빛을 덧붙인다(흰 배경 대신 장면). 구도는 기존 규칙(전신 문장)에 맡긴다."""
    from genai_lab.free_generation import mentions_background
    text = (text or "").strip()
    if not text:
        raise ValueError("장면 문장이 비어 있습니다.")
    if mentions_background(text):
        return dict(text=text, seed=seed, version=VERSION, parts=dict(user=text), source="user")
    auto = compose_scene(seed)["parts"]
    return dict(text=f"{text.rstrip('.')}. Setting: {auto['place']} {auto['time']}, lighting: {auto['lighting']}, with a detailed background.",
                seed=seed, version=VERSION, parts=dict(user=text, place=auto["place"], time=auto["time"], lighting=auto["lighting"]),
                source="user+scene_composer")
