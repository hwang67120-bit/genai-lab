"""장면 자동 조합(D-153): 매번 다른 장면, 기존 요청 검사 통과, 배경 있음, 전신, 얼굴이 보이는 자세만."""
import pytest

from genai_lab.free_generation import check_request, free_conditions, mentions_background, request_conflicts
from genai_lab.scene_composer import ACTIONS, compose_scene, complete_user_scene


def test_same_seed_same_scene_and_many_different_scenes():
    assert compose_scene(7) == compose_scene(7)
    texts = {compose_scene(seed)["text"] for seed in range(300)}
    assert len(texts) >= 280


@pytest.mark.parametrize("seed", range(0, 400, 3))
def test_every_scene_passes_the_existing_rules(seed):
    scene = compose_scene(seed)
    text = scene["text"]
    assert check_request(text)["ok"], check_request(text)
    assert mentions_background(text)                      # 흰 배경 문장이 붙지 않는다
    assert request_conflicts(text) == {}                  # 눈 감기·뒷모습·손 숨기기·꼬리 가림 없음 → 동일성 확인 가능
    assert "head to feet" in text
    keys = [c["key"] for c in free_conditions("", "female", (), text, 42)["conditions"]]
    assert "background" in keys and "white" not in free_conditions("", "female", (), text, 42)["conditions"][keys.index("background")]["text"].lower()


def test_no_face_hiding_actions():
    for action in ACTIONS:
        assert not any(word in action for word in ("from behind", "back to", "facing away", "covering", "closed", "pocket", "behind"))


def test_user_scene_gets_a_setting_only_when_it_has_none():
    with_place = complete_user_scene("비 오는 밤거리에서 우산을 들고 서 있는", 3)
    assert with_place["text"] == "비 오는 밤거리에서 우산을 들고 서 있는" and with_place["source"] == "user"
    bare = complete_user_scene("jumping happily", 3)
    assert bare["text"].startswith("jumping happily. Setting: ") and mentions_background(bare["text"])
    with pytest.raises(ValueError):
        complete_user_scene("  ", 3)
