"""장면 자동 조합(D-153, 2판 D-155): 매번 다른 장면, 기존 요청 검사 통과, 배경 있음, 전신, 얼굴이 보이는 자세만,
비율을 망가뜨리는 카메라·번지는 안개 없음, 조명 설계·색 구성(배경과 조명에만)·표정 포함."""
import pytest

from genai_lab.free_generation import check_request, free_conditions, mentions_background, request_conflicts
from genai_lab.scene_composer import ACTIONS, CAMERA, EXPRESSIONS, LIGHTING, TIME, VERSION, WEATHER, compose_scene, complete_user_scene


def test_same_seed_same_scene_and_many_different_scenes():
    assert compose_scene(7) == compose_scene(7)
    texts = {compose_scene(seed)["text"] for seed in range(300)}
    assert len(texts) >= 295 and compose_scene(1)["version"] == VERSION == 2


@pytest.mark.parametrize("seed", range(0, 400, 3))
def test_every_scene_passes_the_existing_rules(seed):
    scene = compose_scene(seed)
    text = scene["text"]
    assert check_request(text)["ok"], check_request(text)
    assert mentions_background(text)                      # 흰 배경 문장이 붙지 않는다
    assert request_conflicts(text) == {}                  # 눈 감기·뒷모습·손 숨기기·꼬리 가림 없음 → 동일성 확인 가능
    assert "head to feet" in text and "natural body proportions" in text
    assert "Lighting: " in text and "palette in the background and lighting only" in text
    assert scene["parts"]["expression"] in text
    conditions = free_conditions("", "female", (), text, 42)["conditions"]
    background = next(c for c in conditions if c["key"] == "background")
    assert "white" not in background["text"].lower()
    assert any(c["key"] == "scene_colors" for c in conditions)          # 캐릭터 고유 색 유지 문장이 함께 붙는다


def test_no_face_hiding_actions_no_distorting_camera_no_glowing_mist():
    for action in ACTIONS:
        assert not any(word in action for word in ("from behind", "back to", "facing away", "covering", "closed", "pocket", "behind the back"))
    assert not any(word in " ".join(CAMERA) for word in ("high", "wide"))
    assert not any("mist" in w or "fog" in w for w in WEATHER)
    assert not any("wink" in e for e in EXPRESSIONS)
    assert not any("from behind" in light for lights in LIGHTING.values() for light in lights)


def test_time_and_lighting_match_the_period():
    for seed in range(200):
        parts = compose_scene(seed)["parts"]
        key = ("indoor" if parts["indoor"] else "outdoor", parts["period"])
        assert parts["time"] in TIME[key] and parts["lighting"] in LIGHTING[key]


def test_user_scene_gets_a_setting_only_when_it_has_none():
    with_place = complete_user_scene("비 오는 밤거리에서 우산을 들고 서 있는", 3)
    assert with_place["text"] == "비 오는 밤거리에서 우산을 들고 서 있는" and with_place["source"] == "user"
    bare = complete_user_scene("jumping happily", 3)
    assert bare["text"].startswith("jumping happily. Setting: ") and mentions_background(bare["text"]) and "lighting: " in bare["text"]
    with pytest.raises(ValueError):
        complete_user_scene("  ", 3)
