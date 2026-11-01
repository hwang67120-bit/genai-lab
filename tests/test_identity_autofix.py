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


def test_next_round_keeps_asking_for_everything_and_adds_new_losses():
    from genai_lab.identity_autofix import next_round
    asked = next_round(["furry", "animal_nose"], ["animal_nose", "rabbit_tail", "standing"], KINDS | {"rabbit_tail": dict(kind="body")})
    assert asked == ["furry", "animal_nose", "rabbit_tail"]                        # 안 돌아온 코도 계속(빼면 털까지 다시 잃음)


def test_best_per_seed_prefers_fewer_lost_then_the_earlier_round():
    from genai_lab.identity_autofix import best_per_seed
    r0 = [{"furry": 0.0}, {"furry": 0.0}, {"furry": 0.9}]
    r1 = [{"furry": 0.9}, {"furry": 0.8}, {"furry": 0.9}]
    r2 = [{"furry": 0.9}, {"furry": 0.0}, {"furry": 0.1}]
    assert best_per_seed(ORIGINAL, [r0, r1, r2], ["furry"]) == [1, 1, 0]


def test_unsure_kinds_are_not_fixed_and_the_shipped_table_knows_the_furry_case():
    import json
    from pathlib import Path
    kinds = {"qr_code": dict(kind="outfit", outfit=0.66), "cleavage": dict(kind="outfit", outfit=0.95), "plant": dict(kind="body", body=0.47),
             "furry": dict(kind="body", body=1.0)}
    assert to_fix(["qr_code", "cleavage", "plant", "furry"], kinds) == ["cleavage", "furry"]
    table = json.loads((Path(__file__).resolve().parents[1] / "configs" / "tag_kinds_wd_v3.json").read_text(encoding="utf-8"))["table"]
    assert len(table) > 8000
    assert {table[t]["kind"] for t in ("furry", "body_fur", "animal_nose", "dog_ears", "very_dark_skin", "scar")} == {"body"}
    assert {table[t]["kind"] for t in ("realistic", "sky", "smile", "sitting", "hand_on_own_hip", "simple_background")} == {"scene"}
