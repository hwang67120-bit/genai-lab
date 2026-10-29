"""모드 B 다듬기(색 되돌리기·영역·문장·진입점) 시험. 모델·GPU 불필요."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from genai_lab.style_redraw_pipeline import run_style_refine
from genai_lab.style_redraw_refine import CONTENT_NEGATIVE, STYLE_WORDS, colour_back, refine_prompts, style_mask_without_garments
from genai_lab.style_reference_colors import rgb_to_lab


class TestStyleRedrawRefine(unittest.TestCase):
    def test_colour_back_takes_lightness_from_refined_and_colour_from_chosen(self):
        chosen = np.zeros((40, 40, 3), np.uint8); chosen[:] = (200, 160, 40)      # 겨자색
        refined = np.zeros((40, 40, 3), np.uint8); refined[:] = (110, 110, 120)   # 회색 쪽으로 바뀐 색, 더 어두움
        out = colour_back(chosen, refined, sigma=0)
        lab_out, lab_ref, lab_cho = rgb_to_lab(out), rgb_to_lab(refined), rgb_to_lab(chosen)
        self.assertAlmostEqual(float(lab_out[..., 0].mean()), float(lab_ref[..., 0].mean()), delta=2)   # 밝기 = 다듬기
        self.assertGreater(float(lab_out[..., 2].mean()), 20)              # 노란 기운(b) = 고른 후보
        self.assertGreater(float(lab_cho[..., 2].mean()), float(lab_ref[..., 2].mean()) + 20)
        with self.assertRaises(ValueError):
            colour_back(chosen, refined[:20])

    def test_style_mask_skips_garment_boxes_only(self):
        profile = {"outfits": [dict(name="upper", box=[10, 30, 50, 60]), dict(name="lower", box=[10, 55, 50, 90]),
                               dict(name="hair", box=[5, 0, 55, 30])]}
        mask = np.asarray(style_mask_without_garments(profile, (64, 96)))
        self.assertEqual(int(mask[40, 20]), 0)          # 윗옷
        self.assertEqual(int(mask[70, 20]), 0)          # 아래옷
        self.assertEqual(int(mask[10, 20]), 255)        # 머리 영역은 그림체 적용
        self.assertEqual(int(mask[40, 60]), 255)        # 배경

    def test_prompts_add_style_words_and_block_target_content(self):
        profile = dict(scene_prompt="Knee-up shot", traits=[dict(text="green eyes"), dict(text="")], negative_terms="man, male")
        positive, negative = refine_prompts(profile)
        self.assertEqual(positive, "Knee-up shot. green eyes. " + STYLE_WORDS)
        self.assertIn(CONTENT_NEGATIVE, negative)
        self.assertTrue(negative.endswith(", man, male"))


class TestRefineEntry(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name) / "run"; self.dir.mkdir()
        self.candidates = []
        for seed in (42, 43, 44):
            path = self.dir / f"result-seed{seed}.png"; Image.new("RGB", (8, 8), (seed, 0, 0)).save(path)
            self.candidates.append(str(path))
        (self.dir / "outcome.json").write_text(json.dumps(dict(state="review_pending", candidates=self.candidates)), encoding="utf-8")

    def fake(self, calls, fail=False):
        def run(stage, directory):
            calls.append(stage)
            if fail:
                (directory / "refine-error.json").write_text(json.dumps({"message": "sentinel"}), encoding="utf-8"); return 1
            ask = json.loads((directory / "refine-request.json").read_text(encoding="utf-8"))
            Image.new("RGB", (8, 8)).save(directory / f"refined-seed{ask['seed']}.png")
            (directory / "refine-result.json").write_text(json.dumps(dict(file=f"refined-seed{ask['seed']}.png",
                                                                          raw_file=f"refined-raw-seed{ask['seed']}.png", peak_reserved_gib=6.3)), encoding="utf-8")
            return 0
        return run

    def test_refines_the_chosen_candidate_only(self):
        calls = []
        before = Path(self.candidates[1]).read_bytes()
        out = run_style_refine(self.dir, self.candidates[1], run=self.fake(calls))
        self.assertEqual((calls, out["state"]), (["refine"], "review_pending"))
        self.assertTrue(out["refined"].endswith("refined-seed43.png"))
        ask = json.loads((self.dir / "refine-request.json").read_text(encoding="utf-8"))
        self.assertEqual((ask["candidate"], ask["seed"]), ("result-seed43.png", 43))
        self.assertEqual(Path(self.candidates[1]).read_bytes(), before)
        self.assertTrue((self.dir / "refine-outcome-seed43.json").is_file())

    def test_rejects_pictures_outside_the_candidates_and_reports_failure(self):
        other = Path(self.temp.name) / "other.png"; Image.new("RGB", (8, 8)).save(other)
        with self.assertRaises(ValueError):
            run_style_refine(self.dir, other, run=self.fake([]))
        out = run_style_refine(self.dir, self.candidates[0], run=self.fake([], fail=True))
        self.assertEqual((out["state"], out["stage"]), ("failed", "refine"))
        self.assertIn("sentinel", out["message"])


if __name__ == "__main__":
    unittest.main()
