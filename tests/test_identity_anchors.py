"""고정 특징표(D-228): 원본 태그 → 바뀌면 안 되는 특징, 결과와 비교(유지·바뀜·사라짐·판단 불가). 모델 불필요."""
from genai_lab.identity_anchors import anchors, compare, summary_ko


def test_anchors_take_one_value_for_exclusive_groups_and_skip_unstable_style():
    found = anchors({"blonde_hair": 0.9, "brown_hair": 0.6, "yellow_eyes": 0.7, "brown_eyes": 0.55, "long_hair": 0.8,
                     "braid": 0.7, "hair_bun": 0.6, "ahoge": 0.8, "bangs": 0.9, "hair_ribbon": 0.7, "multicolored_hair": 0.6,
                     "animal_ears": 0.9, "cat_ears": 0.9, "tail": 0.8, "smile": 0.9})
    tags = {(a["group"], a["tag"]) for a in found}
    assert ("hair_color", "blonde_hair") in tags and ("hair_color", "brown_hair") not in tags     # 가장 강한 하나(U004 눈 색 거짓 경보)
    assert ("eye_color", "yellow_eyes") in tags and ("eye_color", "brown_eyes") not in tags
    assert {("hair_style", "braid"), ("hair_style", "hair_bun"), ("hair_style", "ahoge")} <= tags
    assert ("hair_style", "bangs") not in tags                                                    # 그림체에 흔들리는 낱말
    assert ("hair_pattern", "multicolored_hair") in tags and ("head_accessories", "hair_ribbon") in tags
    assert {("body_parts", "ears"), ("body_parts", "tail")} <= tags
    assert not any(a["tag"] == "smile" for a in found)                                           # 표정은 고정 특징이 아님


def test_compare_kept_changed_lost_and_unclear():
    original = anchors({"black_hair": 0.9, "purple_eyes": 0.8, "short_hair": 0.7, "animal_ears": 0.9, "tail": 0.9, "hair_ribbon": 0.6})
    same = compare(original, {"black_hair": 0.8, "purple_eyes": 0.6, "short_hair": 0.5, "animal_ears": 0.9, "tail": 0.95,
                              "hair_ribbon": 0.4})
    assert same["score"] == 1.0 and same["lost"] == same["changed"] == 0
    tail_lost = compare(original, {"black_hair": 0.8, "purple_eyes": 0.6, "short_hair": 0.5, "animal_ears": 0.9, "cat_tail": 0.2})
    assert {i["tag"] for i in tail_lost["items"] if i["status"] == "lost"} == {"tail", "hair_ribbon"}
    grey = compare(original, {"grey_hair": 0.8, "black_hair": 0.2, "short_hair": 0.6, "animal_ears": 0.9, "tail": 0.9})
    changed = [i for i in grey["items"] if i["status"] == "changed"]
    assert changed[0]["tag"] == "black_hair" and changed[0]["now"] == "grey_hair"
    eyes = [i for i in grey["items"] if i["group"] == "eye_color"][0]
    assert eyes["status"] == "unclear"                                    # 눈 색 태그가 하나도 없음 = 눈이 안 보임(뒷모습)
    assert grey["score"] == round(3 / 5, 3)                               # 판단 불가는 점수에서 뺀다


def test_summary_line():
    report = compare(anchors({"black_hair": 0.9, "tail": 0.9}), {"grey_hair": 0.9})
    line = summary_ko(report)
    assert line.startswith("고정 특징 0/2 유지") and "black_hair→grey_hair" in line and "꼬리(tail)" in line
    assert summary_ko(compare([], {})) == "고정 특징: 판단할 특징 없음"


def test_body_part_kinds_and_ear_kind_change_is_caught():
    # D-236: 그림체 LoRA가 늘어진 개 귀를 서 있는 늑대 귀로 바꿨는데 부위 유무만 봐서 '유지'로 읽었다
    from genai_lab.identity_anchors import body_part_kinds
    scores = {"animal_ears": 0.94, "dog_ears": 0.85, "wolf_ears": 0.3, "tail": 0.91, "dog_tail": 0.85, "fox_tail": 0.6}
    assert body_part_kinds(scores, ("ears", "tail")) == {"ears": "dog_ears", "tail": "dog_tail", "ears_floppy": True}
    assert body_part_kinds(scores, ("tail",)) == {"tail": "dog_tail"}                 # 없는 부위의 종류는 넣지 않는다
    assert body_part_kinds({"wolf_ears": 0.8}, ("ears",)) == {"ears": "wolf_ears"}   # 선 개과 귀는 wolf_ears — '늘어진' 없음
    assert body_part_kinds({"animal_ears": 0.9, "cat_ears": 0.4}, ("ears",)) == {}     # 확실하지 않으면 종류 없음(기존 문장)
    assert body_part_kinds({"cat_ears": 0.9, "floppy_ears": 0.7}, ("ears",)) == {"ears": "cat_ears", "ears_floppy": True}
    original = anchors(scores)
    assert ("ear_kind", "dog_ears") in {(a["group"], a["tag"]) for a in original}
    drifted = compare(original, {"animal_ears": 0.98, "dog_ears": 0.18, "wolf_ears": 0.46, "tail": 0.96, "wolf_tail": 0.58,
                                 "dog_tail": 0.24})
    assert {i["tag"] for i in drifted["items"] if i["status"] in ("lost", "changed")} == {"dog_ears", "dog_tail"}
    kept = compare(original, {"animal_ears": 0.95, "dog_ears": 0.82, "tail": 0.95, "dog_tail": 0.87})
    assert kept["score"] == 1.0
