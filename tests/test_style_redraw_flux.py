"""모드 B FLUX 본체(문장·부위 마스크·색 맞춤·손 표시·진입점 선택) 시험. 모델·GPU 불필요."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from genai_lab.style_redraw_flux import COLOUR_SHIFT_CAP, colour_match, flux_prompt, hand_flag, part_masks
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
    def test_engine_recorded_and_unknown_rejected(self):
        with TemporaryDirectory() as temp:
            root = Path(temp); src = root / "c.png"; Image.new("RGB", (100, 150)).save(src)
            runtime = SimpleNamespace(model_cache=Path("c"), head_semantic_repo=Path("r"), head_semantic_checkpoint=Path("k"))
            def run(stage, directory):
                if stage == "pose": (directory / "pose.json").write_text("{}", encoding="utf-8")
                if stage == "prepare": (directory / "prepare.json").write_text(json.dumps(dict(state="refused", code="x", message="m")), encoding="utf-8")
                return 0
            run_style_redraw(src, root / "a", runtime, run=run)
            req = json.loads((root / "a" / "request.json").read_text(encoding="utf-8"))
            self.assertEqual(req["engine"], "flux")
            self.assertTrue({"flux_model", "hand_cache"} <= set(req))
            with self.assertRaises(ValueError):
                run_style_redraw(src, root / "b", runtime, run=run, engine="other")
            garment = root / "g.png"; Image.new("RGB", (300, 500), (10, 120, 200)).save(garment)
            run_style_redraw(src, root / "c", runtime, run=run, garment_path=garment)
            with Image.open(root / "c" / "garment.png") as g:
                self.assertEqual(g.size, (768, 1024))
            self.assertEqual(json.loads((root / "c" / "request.json").read_text(encoding="utf-8"))["garment"], str(garment))
            with self.assertRaises(ValueError):
                run_style_redraw(src, root / "d", runtime, run=run, engine="sdxl_lora", garment_path=garment)


if __name__ == "__main__":
    unittest.main()
