"""자동 동일성 고치기(S32): 잃은 태그 찾기·장면 태그 빼기·'빠진 특징' 문장. 모델 불필요."""
from genai_lab.identity_autofix import fixed_prompt, fix_sentence, lost_tags, remaining, to_fix

ORIGINAL = {"furry": 0.97, "animal_nose": 0.65, "standing": 0.67, "high_heel_boots": 0.71, "long_hair": 0.91, "smile": 0.4}
RESULTS = [{"furry": 0.03, "animal_nose": 0.0, "standing": 0.2, "high_heel_boots": 0.26, "long_hair": 0.9, "smile": 0.0},
           {"furry": 0.07, "animal_nose": 0.01, "standing": 0.8, "high_heel_boots": 0.46, "long_hair": 0.89},
           {"furry": 0.01, "animal_nose": 0.0, "standing": 0.1, "high_heel_boots": 0.61, "long_hair": 0.9}]
KINDS = {"furry": dict(kind="body"), "animal_nose": dict(kind="body"), "standing": dict(kind="scene"), "high_heel_boots": dict(kind="outfit")}


def test_lost_tags_need_a_majority_of_candidates_and_a_sure_original():
    assert lost_tags(ORIGINAL, RESULTS) == ["furry", "animal_nose", "standing"]   # 부츠는 1/3만 잃음(seed 탓), smile은 원본이 0.4
    assert lost_tags(ORIGINAL, []) == []


def test_scene_tags_are_not_fixed_and_garment_keeps_only_body():
    lost = ["furry", "standing", "high_heel_boots", "unknown_tag"]
    assert to_fix(lost, KINDS) == ["furry", "high_heel_boots", "unknown_tag"]       # 모르는 태그는 고친다
    assert to_fix(lost, KINDS, with_garment=True) == ["furry", "unknown_tag"]        # 옷을 바꿨으면 원래 옷은 잃는 게 맞다


def test_fix_sentence_goes_before_the_keep_instruction():
    p = "gnlbq4 style illustration. Show the same adult character. Keep the same face, hairstyle."
    fixed = fixed_prompt(p, ["furry", "animal_nose"])
    assert "must be shown clearly: furry, animal nose. Keep the same face," in fixed
    assert fixed_prompt(p, []) == p and fix_sentence([]) == ""
    assert fixed_prompt("no anchor here.", ["furry"]).endswith("must be shown clearly: furry.")
    assert remaining(ORIGINAL, {"furry": 0.88, "animal_nose": 0.14}, ["furry", "animal_nose"]) == ["animal_nose"]
