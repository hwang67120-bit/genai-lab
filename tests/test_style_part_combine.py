"""모드 B 부위별 고르기(얼굴·머리 후보 + 몸·옷 후보) 시험. 모델·GPU 불필요."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from genai_lab.style_part_combine import MAX_SHIFT_PX, combine_parts, head_mask
from genai_lab.style_redraw_pipeline import run_style_part_combine


def textured(seed, shape=(120, 100)):
    from scipy.ndimage import gaussian_filter
    noise = np.random.default_rng(seed).random(shape)
    return (np.repeat(gaussian_filter(noise, 2)[..., None], 3, 2) * 255).astype(np.uint8)


class TestCombineParts(unittest.TestCase):
    def setUp(self):
        self.head = np.zeros((120, 100), bool); self.head[10:50, 30:70] = True

    def test_head_from_face_body_from_body(self):
        face = textured(1)
        body = face.copy(); body[60:] = textured(2)[60:]                      # 같은 머리, 다른 몸·옷
        out, info = combine_parts(face, body, self.head, self.head)
        self.assertTrue(np.array_equal(out[110, 50], body[110, 50]))          # 몸은 몸 후보 그대로
        self.assertLess(np.abs(out[30, 50].astype(int) - face[30, 50]).max(), 3)   # 머리 안쪽은 얼굴 후보
        self.assertTrue(info["aligned"])
        self.assertLess(max(map(abs, info["shift"])), 0.5)

    def test_background_outside_person_stays_from_body(self):
        face = textured(1)
        body = face.copy(); body[:, :25] = 30; body[60:] = textured(2)[60:]     # 몸 후보는 배경 빛이 다름(왼쪽 어두움)
        person = np.zeros_like(self.head); person[5:120, 32:68] = True
        out, _ = combine_parts(face, body, self.head, self.head, face_person=person)
        self.assertTrue(np.array_equal(out[30, 20], body[30, 20]))            # 머리 옆 배경 = 몸 후보(테두리 없음)
        self.assertLess(np.abs(out[30, 50].astype(int) - face[30, 50]).max(), 3)
        out_all, _ = combine_parts(face, body, self.head, self.head)
        self.assertFalse(np.array_equal(out_all[30, 24], body[30, 24]))       # 사람 영역이 없으면 넓힌 머리 영역이 배경까지 덮음

    def test_weak_match_is_not_moved(self):
        out, info = combine_parts(textured(1), textured(2), self.head, self.head)   # 머리가 전혀 다름 → 잰 어긋남은 잡음
        self.assertFalse(info["aligned"])
        self.assertEqual(info["shift"], [0.0, 0.0])

    def test_small_shift_is_undone(self):
        face = textured(3)
        body = np.roll(face, (2, 3), axis=(0, 1))                            # 몸 후보의 머리가 (3, 2)px 옮겨져 있음
        moved = np.roll(self.head, (2, 3), axis=(0, 1))
        out, info = combine_parts(face, body, self.head, moved)
        self.assertAlmostEqual(info["shift"][0], 3, delta=0.5)
        self.assertAlmostEqual(info["shift"][1], 2, delta=0.5)
        self.assertLess(np.abs(out[32, 53].astype(int) - body[32, 53]).max(), 6)  # 옮긴 뒤 몸 후보 자리와 맞음

    def test_large_shift_and_missing_head_refused(self):
        face = textured(4)
        body = np.roll(face, (0, int(MAX_SHIFT_PX) + 6), axis=(0, 1))
        with self.assertRaises(ValueError):
            combine_parts(face, body, self.head, self.head)
        empty = np.zeros_like(self.head)
        with self.assertRaises(ValueError):
            combine_parts(face, face, empty, empty)

    def test_head_mask_keys(self):
        hair = np.zeros((4, 4), bool); hair[0] = True
        neck = np.zeros((4, 4), bool); neck[3] = True
        m = head_mask(dict(hair=hair, neck=neck), (4, 4))
        self.assertTrue(m[0].all()); self.assertFalse(m[3].any())            # 목은 몸 후보 것


class TestPartCombineEntry(unittest.TestCase):
    def test_entry_checks_candidates_and_records(self):
        with TemporaryDirectory() as temp:
            d = Path(temp)
            files = []
            for seed in (42, 43, 44):
                path = d / f"result-seed{seed}.png"; Image.new("RGB", (8, 8), (seed, 0, 0)).save(path); files.append(str(path))
            (d / "outcome.json").write_text(json.dumps(dict(state="review_pending", candidates=files)), encoding="utf-8")
            def run(stage, directory):
                ask = json.loads((directory / "combine-request.json").read_text(encoding="utf-8"))
                self.assertEqual((ask["face_seed"], ask["body_seed"]), (43, 44))
                Image.new("RGB", (8, 8)).save(directory / "combined-face43-body44.png")
                (directory / "combine-result.json").write_text(json.dumps(dict(file="combined-face43-body44.png", shift=[0.1, -0.2])),
                                                               encoding="utf-8")
                return 0
            result = run_style_part_combine(d, files[1], files[2], run=run)
            self.assertEqual(result["state"], "review_pending")
            self.assertEqual(result["shift"], [0.1, -0.2])
            self.assertTrue((d / "combine-outcome-face43-body44.json").is_file())
            with self.assertRaises(ValueError):
                run_style_part_combine(d, files[1], files[1], run=run)       # 같은 후보 둘
            with self.assertRaises(ValueError):
                run_style_part_combine(d, files[1], str(d / "other.png"), run=run)

    def test_worker_failure_returns_failed(self):
        with TemporaryDirectory() as temp:
            d = Path(temp)
            files = []
            for seed in (42, 43):
                path = d / f"result-seed{seed}.png"; Image.new("RGB", (8, 8)).save(path); files.append(str(path))
            (d / "outcome.json").write_text(json.dumps(dict(state="review_pending", candidates=files)), encoding="utf-8")
            def run(stage, directory):
                (directory / "combine-error.json").write_text(json.dumps(dict(message="머리 위치가 너무 다릅니다")), encoding="utf-8")
                return 1
            result = run_style_part_combine(d, files[0], files[1], run=run)
            self.assertEqual(result["state"], "failed")
            self.assertIn("머리 위치", result["message"])


if __name__ == "__main__":
    unittest.main()
