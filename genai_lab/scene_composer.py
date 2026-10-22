"""사용자 화면의 장면 자동 조합(D-153, 2026-10-22 사용자 "동일성만 유지된다면 어떤 자세든 전신으로 나오기만 하면 되고 배경은 어떤 거든 상관없다").

고정 장면 몇 개에서 고르지 않고, 장소 · 동작 · 빛/시간 · 날씨 · 카메라를 각각 무작위로 골라 매번 새 장면 문장을 만든다(조합 수십만).
늘 지키는 것은 둘뿐이다: 전신(머리부터 발끝까지)과 동일성 확인 가능(얼굴이 보이는 자세만 — 뒷모습·얼굴 가림·눈 감기·손 숨기기 제외).
장면 문장은 기존 요청 검사(free_generation.check_request)를 그대로 지난다 — 외형·의상을 바꾸는 말은 넣지 않는다.
사람이 쓴 장면(장면 입력 칸)에 배경 말이 없으면 장소·빛만 덧붙여 흰 배경이 되지 않게 한다.
평가 세트·개발 화면의 '요청 없음 = 고정 자세 + 흰 배경'(D-098)은 바꾸지 않는다(시험 비교 유지).
"""
import random

FULL_BODY_TEXT = "Full-body shot: the whole character is visible from head to feet, with a detailed background."

OUTDOOR = (
    "on a quiet city street at night lit by neon signs", "in a sunlit forest clearing", "on a rooftop overlooking the city",
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
    "standing and smiling at the viewer", "walking forward with a relaxed stride", "running lightly toward the viewer",
    "jumping with both feet off the ground", "turning halfway around with the face toward the viewer",
    "leaning against a wall with one foot crossed over the other", "sitting on a low wall with the legs dangling",
    "sitting on a bench with both hands resting on the knees", "kneeling on one knee and looking at the viewer",
    "crouching down and looking up at the viewer", "stretching both arms above the head", "waving one hand at the viewer",
    "holding an open umbrella", "holding a book in both hands", "reaching up toward something above",
    "dancing with the arms spread wide", "spinning in place with a cheerful look", "stepping forward with one hand raised",
    "standing with one hand on the hip", "pointing ahead with a bright expression", "sitting on stone steps",
    "standing on tiptoes", "looking up at the sky with a smile", "petting a small cat", "holding a cup of tea with both hands",
    "carrying a paper shopping bag", "picking up a fallen leaf", "resting the chin on one hand while sitting",
    "tossing a coin into the air", "skipping along with light steps",
)
OUTDOOR_LIGHT = (
    "in soft morning light", "under the bright midday sun", "at golden hour with warm sunlight", "at sunset with orange light",
    "at dusk with blue evening light", "at night under street lamps", "under a starry night sky", "on an overcast day with soft shadows",
)
INDOOR_LIGHT = ("with warm lamp light", "with soft window light", "with bright even daylight", "in the evening with cozy lights")
WEATHER = ("", "", "", "with light rain falling", "with snow falling gently", "with cherry petals drifting in the wind",
           "with autumn leaves blowing", "with thin mist in the air", "with a gentle breeze")
CAMERA = ("eye-level view", "eye-level view", "slightly low camera angle", "slightly high camera angle", "wide shot")


def compose_scene(seed):
    """seed마다 같은 장면(재현용). 반환: dict(text, parts, seed)."""
    rng = random.Random(seed)
    indoor = rng.random() < 0.3
    place = rng.choice(INDOOR if indoor else OUTDOOR)
    action = rng.choice(ACTIONS)
    light = rng.choice(INDOOR_LIGHT if indoor else OUTDOOR_LIGHT)
    weather = "" if indoor else rng.choice(WEATHER)
    camera = rng.choice(CAMERA)
    scene = f"{action} {place}, {light}" + (f", {weather}" if weather else "")
    text = f"The character is {scene}. {camera.capitalize()}. {FULL_BODY_TEXT}"
    return dict(text=text, seed=seed, parts=dict(place=place, indoor=indoor, action=action, light=light, weather=weather, camera=camera),
                source="scene_composer")


def complete_user_scene(text, seed):
    """사람이 쓴 장면. 배경 말이 없으면 장소·빛을 덧붙인다(흰 배경 대신 장면). 구도는 기존 규칙(전신 문장)에 맡긴다."""
    from genai_lab.free_generation import mentions_background
    text = (text or "").strip()
    if not text:
        raise ValueError("장면 문장이 비어 있습니다.")
    if mentions_background(text):
        return dict(text=text, seed=seed, parts=dict(user=text), source="user")
    auto = compose_scene(seed)["parts"]
    return dict(text=f"{text.rstrip('.')}. Setting: {auto['place']}, {auto['light']}, with a detailed background.",
                seed=seed, parts=dict(user=text, place=auto["place"], light=auto["light"]), source="user+scene_composer")
