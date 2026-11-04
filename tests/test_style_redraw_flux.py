"""모드 B FLUX 본체(문장·부위 마스크·색 맞춤·손 표시·진입점 선택) 시험. 모델·GPU 불필요."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from genai_lab.style_redraw_flux import (COLOUR_SHIFT_CAP, body_part_tags, colour_match, flux_prompt, hand_flag, missing_body_parts,
                                         part_masks)
from genai_lab.style_redraw_pipeline import run_style_redraw
from genai_lab.style_reference_colors import rgb_to_lab


class TestFluxHelpers(unittest.TestCase):
    def test_prompt_keeps_pose_and_skin_and_lists_traits(self):
        p = flux_prompt([dict(text="green eyes"), dict(text=""), dict(text="short pink hair, undercut")])
        self.assertIn("(green eyes; short pink hair, undercut)", p)
        self.assertIn("Keep the same pose, arms, hands and framing", p)
        self.assertIn("do not lighten or darken the skin", p)
        self.assertNotIn("relaxed arms", p)                    # 팔짱 원본에서 팔이 늘어난 원인(D-209)
        self.assertNotIn("()", flux_prompt([]))

    def test_outfit_prompt_drops_old_clothes_words(self):
        traits = [dict(name="eyes", text="brown eyes"), dict(name="skin", text="brown skin"),
                  dict(name="clothes", text="green vest, grey trousers")]
        p = flux_prompt(traits, with_garment=True)
        self.assertIn("(brown eyes; brown skin)", p)
        self.assertNotIn("green vest", p)                      # 원래 옷 낱말은 새 옷과 다툼
        self.assertIn("outfit shown in image 2", p)
        self.assertIn("do not lighten or darken the skin", p)

    def test_body_part_sentence_like_the_qwen_path(self):
        with_tail = flux_prompt([dict(text="black hair")], body_parts=("ears", "tail"))
        self.assertIn("The character has animal ears and a tail or tails; keep them exactly", with_tail)
        self.assertIn("no wings; do not add any", with_tail)
        self.assertIn("If the character has animal ears, a tail or wings", flux_prompt([]))          # 모름 = 조건문
        self.assertIn("no animal ears, no tail and no wings", flux_prompt([], with_garment=True, body_parts=()))

    def test_framing_follows_how_much_of_the_body_the_original_shows(self):
        # D-247: 상반신 원본에 전신을 강제로 그리면 없는 다리를 지어내 비율·옷이 오염됨 → 원본이 보여 준 만큼만
        from genai_lab.scene_composer import compose_scene
        from genai_lab.style_redraw_flux import COWBOY_FRAMING, FULL_BODY_FRAMING, UPPER_FRAMING, framing
        self.assertEqual(framing({"full_body": 0.9}), FULL_BODY_FRAMING)
        self.assertEqual(framing({"pink_footwear": 0.97}), FULL_BODY_FRAMING)
        self.assertEqual(framing({"upper_body": 0.8, "belt": 0.7, "pants": 0.6}), COWBOY_FRAMING)      # T009: 허리·바지 보임
        self.assertEqual(framing({"portrait": 0.9}), UPPER_FRAMING)
        # U002의 상체 자르기. 손 자세만으로 허벅지까지 그리게 해서는 안 된다.
        self.assertEqual(framing({"upper_body": .442, "hand_on_own_hip": .831}), UPPER_FRAMING)
        self.assertEqual(framing({"hand_on_own_hip": .831, "thighs": .90}), COWBOY_FRAMING)
        scene = compose_scene(39001)["text"]
        p = flux_prompt([dict(text="black hair")], body_parts=(), scene=scene, shot=COWBOY_FRAMING)
        self.assertNotIn("head to feet", p)
        self.assertNotIn("Full body, the whole character", p)
        self.assertNotIn("whole body, arms and legs", p)
        self.assertIn("do not show the lower legs or feet", p)
        self.assertIn("head to feet", flux_prompt([dict(text="black hair")], body_parts=(), scene=scene))      # 기본은 전신

    def test_male_and_garment_words_are_written_out(self):
        # D-249: 남성이 여성 얼굴로, 옷 그림의 옷이 안 입혀짐 → 남성 문장·옷 그림 낱말
        from genai_lab.style_redraw_flux import MALE_SENTENCE
        p = flux_prompt([dict(text="black hair")], with_garment=True, body_parts=(), scene="Walking.", gender="male",
                        garment_words=["black vest", "white shirt"])
        self.assertIn(MALE_SENTENCE, p)
        self.assertIn("The outfit in image 2 is: black vest, white shirt.", p)
        q = flux_prompt([dict(text="black hair")], body_parts=(), scene="Walking.", gender="female", garment_words=["black vest"])
        self.assertNotIn("masculine", q)
        self.assertNotIn("The outfit in image 2", q)                          # 옷 그림이 없으면 옷 낱말 문장 없음

    def test_furry_characters_get_a_weaker_style_lora(self):
        # D-245: 사람 캐릭터로 학습한 그림체 LoRA가 털 수인 얼굴을 사람 쪽으로 당김 → 원본 태그로 판별해 세기 0.3
        from genai_lab.style_redraw_flux import FURRY_STYLE_LORA_WEIGHT, STYLE_LORA_WEIGHT, style_lora_weight
        self.assertEqual(style_lora_weight({"furry": 0.97, "furry_female": 0.98}), FURRY_STYLE_LORA_WEIGHT)
        self.assertEqual(style_lora_weight({"furry_male": 0.6}), FURRY_STYLE_LORA_WEIGHT)
        self.assertEqual(style_lora_weight({"animal_ears": 0.97, "dog_ears": 0.85, "furry": 0.1}), STYLE_LORA_WEIGHT)   # 동물 귀 사람은 그대로
        self.assertEqual(style_lora_weight(None), STYLE_LORA_WEIGHT)

    def test_prompt_starts_with_style_lora_trigger_and_names_part_kinds(self):
        # D-236: 그림체 LoRA 트리거 낱말이 맨 앞, 귀·꼬리 종류를 적으면 LoRA가 귀 모양을 바꾸지 못했다(9/9)
        from genai_lab.style_redraw_flux import STYLE_LORA_TRIGGER
        for p in (flux_prompt([]), flux_prompt([], with_garment=True), flux_prompt([], scene="Walking in a park.")):
            self.assertTrue(p.startswith(STYLE_LORA_TRIGGER + " "))
        kinds = dict(ears="dog_ears", tail="dog_tail")
        p = flux_prompt([dict(text="red hair")], body_parts=("ears", "tail"), scene="Standing in a greenhouse.", body_part_kinds=kinds)
        self.assertIn("The character has dog ears (the same dog ear shape, size and angle as in image 1) and a dog tail; keep them", p)
        floppy = flux_prompt([], body_parts=("ears",), body_part_kinds=dict(ears="dog_ears", ears_floppy=True))
        self.assertIn("has floppy dog ears (", floppy)
        self.assertIn("The character has animal ears and a tail or tails;",
                      flux_prompt([], body_parts=("ears", "tail"), body_part_kinds={}))      # 종류를 모르면 기존 문장

    def test_scene_prompt_changes_pose_and_keeps_identity(self):
        traits = [dict(name="eyes", text="purple eyes"), dict(name="clothes", text="black bodysuit")]
        p = flux_prompt(traits, body_parts=("ears", "tail"), scene="The character is sitting on a bench in a park.")
        self.assertIn("in a new scene. Scene: The character is sitting on a bench in a park.", p)
        self.assertIn("(purple eyes; black bodysuit)", p)
        self.assertIn("skin tone and body proportions and outfit exactly as in image 1", p)
        self.assertIn("a tail or tails", p)
        self.assertNotIn("Keep the same pose", p)                                  # 장면이 있으면 원본 자세를 고정하지 않음
        self.assertNotIn("concrete wall", p)                                       # 배경은 장면 문장이 정함
        self.assertIn("must not change the character's own colours", p)            # 역광에 머리색이 바뀌지 않게(D-225)
        self.assertNotIn("must not change the character's own colours", flux_prompt(traits))   # 장면 없을 때는 그대로
        g = flux_prompt(traits, with_garment=True, body_parts=(), scene="Walking in the rain.")
        self.assertIn("wearing the complete outfit shown in image 2", g)
        self.assertNotIn("black bodysuit", g)                                      # 원래 옷 낱말은 빼서 새 옷과 다투지 않게
        self.assertIn("Do not copy the mannequin", g)

    def test_scene_lighting_never_brightens_and_keeps_contrast(self):
        # U003(D-225): 역광으로 어두운 검은 머리를 밝은 셀 원본 평균 쪽으로 올리면 회색이 된다
        orig = np.zeros((40, 40, 3), np.uint8); orig[:] = (90, 90, 100)                  # 원본 평균이 더 밝음
        res = np.zeros((40, 40, 3), np.uint8); res[:] = (25, 25, 30)                     # 어두운 장면의 검은 머리
        m = np.ones((40, 40), bool); e = np.zeros_like(m)
        masks = dict(skin=e, hair=m, upper=e, lower=e)
        out, shifts = colour_match(res, orig, masks, dict(masks), feather=0, scene_lighting=True)
        self.assertEqual(shifts["hair"][0], 0.0)
        self.assertLessEqual(int(out[20, 20].sum()), int(res[20, 20].sum()) + 3)        # 밝아지지 않음
        _, plain = colour_match(res, orig, masks, dict(masks), feather=0)
        self.assertGreater(plain["hair"][0], 0)                                           # 장면이 아니면 예전처럼 올린다
        dark = np.zeros((40, 40, 3), np.uint8); dark[:] = (40, 20, 25)                    # 원본이 더 어두우면 어둡게 되돌리기는 함
        _, down = colour_match(orig, dark, masks, dict(masks), feather=0, scene_lighting=True)
        self.assertLess(down["hair"][0], 0)

    def test_output_size_is_the_chosen_high_resolution(self):
        from genai_lab.style_redraw_flux import OUTPUT_SIZE
        self.assertEqual(OUTPUT_SIZE, (1024, 1360))                               # D-230 사용자가 고정한 그림체(처음부터 고해상도)
        self.assertTrue(all(v % 16 == 0 for v in OUTPUT_SIZE))                    # FLUX 잠재 격자에 맞는 크기

    def test_missing_parts_by_result_tags(self):
        kept = {"animal_ears": 0.98, "cat_ears": 0.95, "tail": 0.95, "cat_tail": 0.94}
        lost = {"animal_ears": 0.98, "cat_ears": 0.95, "cat_tail": 0.2}                  # 고치기 전 U003(꼬리 없음)
        self.assertEqual(missing_body_parts(("ears", "tail"), kept), [])
        self.assertEqual(missing_body_parts(("ears", "tail"), lost), ["tail"])
        self.assertEqual(missing_body_parts(("ears",), {"pointy_ears": 0.9}), ["ears"])   # 엘프 귀는 동물 귀가 아님
        self.assertEqual(missing_body_parts(None, {}), [])
        self.assertEqual(body_part_tags({"mechanical_tail": 0.86, "striped_tail": 0.1, "smile": 0.9}), {"mechanical_tail": 0.86})

    def test_identity_only_colour_match_leaves_clothes(self):
        orig = np.zeros((40, 40, 3), np.uint8); orig[:] = (90, 60, 40)
        res = np.zeros((40, 40, 3), np.uint8); res[:] = (200, 160, 140)
        skin = np.zeros((40, 40), bool); skin[:20] = True
        cloth = np.zeros((40, 40), bool); cloth[20:] = True
        m = dict(skin=skin, hair=np.zeros_like(skin), upper=cloth, lower=np.zeros_like(skin))
        out, shifts = colour_match(res, orig, m, dict(m), feather=0, parts=("skin", "hair"), cap=(40, 20, 20))
        self.assertNotIn("upper", shifts)
        self.assertEqual(tuple(out[30, 30]), (200, 160, 140))                 # 옷은 그대로
        self.assertLess(int(out[10, 10].sum()), int(res[10, 10].sum()))       # 피부는 원본(짙은) 쪽으로

    def test_brightening_stops_before_highlights_blow_out(self):
        orig = np.zeros((40, 40, 3), np.uint8); orig[:] = (252, 246, 242)              # 흰 셀 피부
        res = np.zeros((40, 40, 3), np.uint8); res[:] = (200, 180, 170); res[:5] = (240, 232, 228)   # 그늘 + 밝은 곳
        skin = np.ones((40, 40), bool); empty = np.zeros_like(skin)
        m = dict(skin=skin, hair=empty, upper=empty, lower=empty)
        out, shifts = colour_match(res, orig, m, dict(m), feather=0, parts=("skin",), cap=(40, 20, 20))
        self.assertLess(shifts["skin"][0], 10)                                         # +20 넘게 밝히지 않음
        self.assertLess(int(out[2, 2].min()), 250)                                     # 밝은 곳이 하얗게 날아가지 않음
        self.assertGreater(int(out[20, 20].sum()), int(res[20, 20].sum()))            # 그래도 원본 쪽(밝게)으로는 감

    def test_part_masks_stay_inside_the_person(self):
        rgb = np.zeros((40, 40, 3), np.uint8); rgb[:] = (200, 160, 130)      # 배경도 피부와 비슷한 색
        face = np.zeros((40, 40), bool); face[5:15, 10:20] = True
        top = np.zeros((40, 40), bool); top[15:30, 10:20] = True
        person = np.zeros((40, 40), bool); person[0:35, 8:22] = True
        m = part_masks(rgb, dict(face=face, topwear=top), person)
        self.assertFalse(m["skin"][:, :8].any())                             # 사람 밖(배경)은 피부가 아님
        self.assertTrue(m["skin"][10, 15])
        self.assertTrue(m["upper"][20, 15]); self.assertFalse(m["skin"][20, 15])   # 분할이 옷이라 하면 같은 색이어도 옷

    def test_colour_match_moves_part_colour_with_cap_and_keeps_rest(self):
        orig = np.zeros((60, 60, 3), np.uint8); orig[:] = (120, 30, 60)      # 와인색
        res = np.zeros((60, 60, 3), np.uint8); res[:] = (200, 40, 40)        # 빨강으로 바뀐 결과
        res[:, :10] = (90, 90, 90)                                           # 부위 밖(배경)
        m = np.zeros((60, 60), bool); m[:, 20:] = True
        masks = dict(skin=np.zeros_like(m), hair=np.zeros_like(m), upper=m, lower=np.zeros_like(m))
        out, shifts = colour_match(res, orig, masks, dict(masks), feather=0)
        self.assertIsNone(shifts["skin"])
        self.assertTrue(all(abs(v) <= c + 1e-6 for v, c in zip(shifts["upper"], COLOUR_SHIFT_CAP)))
        before, after, target = rgb_to_lab(res)[30, 40], rgb_to_lab(out)[30, 40], rgb_to_lab(orig)[30, 40]
        self.assertLess(np.abs(after - target).sum(), np.abs(before - target).sum())     # 원본 쪽으로 감
        self.assertEqual(tuple(out[30, 5]), (90, 90, 90))                                  # 부위 밖 그대로

    def test_hand_flag(self):
        self.assertTrue(hand_flag(1, 2)); self.assertFalse(hand_flag(2, 2)); self.assertIsNone(hand_flag(None, 2))


class TestEngineChoice(unittest.TestCase):
    def test_engine_recorded_and_garment_fitted(self):
        with TemporaryDirectory() as temp:
            root = Path(temp); src = root / "c.png"; Image.new("RGB", (100, 150)).save(src)
            runtime = SimpleNamespace(model_cache=Path("c"), head_semantic_repo=Path("r"), head_semantic_checkpoint=Path("k"))
            def run(stage, directory):
                if stage == "pose": (directory / "pose.json").write_text("{}", encoding="utf-8")
                if stage == "prepare": (directory / "prepare.json").write_text(json.dumps(dict(state="refused", code="x", message="m")), encoding="utf-8")
                return 0
            run_style_redraw(src, root / "a", runtime, run=run, adult_confirmed=True)
            req = json.loads((root / "a" / "request.json").read_text(encoding="utf-8"))
            self.assertEqual(req["engine"], "flux")
            self.assertTrue({"flux_model", "hand_cache"} <= set(req))
            garment = root / "g.png"; Image.new("RGB", (300, 500), (10, 120, 200)).save(garment)
            run_style_redraw(src, root / "c", runtime, run=run, garment_path=garment, adult_confirmed=True)
            with Image.open(root / "c" / "garment.png") as g:
                self.assertEqual(g.size, (768, 1024))
            self.assertEqual(json.loads((root / "c" / "request.json").read_text(encoding="utf-8"))["garment"], str(garment))
            run_style_redraw(src, root / "s", runtime, run=run, scene="  walking in a park at dusk  ", adult_confirmed=True)
            self.assertEqual(json.loads((root / "s" / "request.json").read_text(encoding="utf-8"))["scene"], "walking in a park at dusk")
            with self.assertRaises(ValueError):
                run_style_redraw(src, root / "x", runtime, run=run, scene="walking in the rain wearing a red dress", adult_confirmed=True)   # 의상 변경


if __name__ == "__main__":
    unittest.main()
