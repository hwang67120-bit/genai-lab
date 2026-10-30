"""모드 B 얼굴 다듬기(상자·붙이기·눈 색 되돌리기·진입점) 시험. 모델·GPU 불필요."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from genai_lab.style_redraw_face import eye_colour_back, face_crop_box, face_prompts, feather_mask, paste_face
from genai_lab.style_redraw_pipeline import run_style_face_detail
from genai_lab.style_reference_colors import rgb_to_lab


class TestStyleRedrawFace(unittest.TestCase):
    def test_face_box_square_enlarged_and_clipped(self):
        self.assertEqual(face_crop_box([300, 100, 460, 260], (768, 1024)), [236, 36, 524, 324])     # 160 x 1.8 = 288
        self.assertEqual(face_crop_box([0, 0, 100, 100], (768, 1024)), [0, 0, 140, 140])

    def test_paste_keeps_edges_and_changes_centre_only(self):
        full = Image.new("RGB", (400, 400), (10, 10, 10))
        out = paste_face(full, Image.new("RGB", (1024, 1024), (250, 250, 250)), [100, 100, 300, 300])
        arr = np.asarray(out)
        self.assertEqual(tuple(arr[200, 200]), (250, 250, 250))       # 가운데는 다시 그린 얼굴
        self.assertEqual(tuple(arr[101, 101]), (10, 10, 10))          # 상자 모서리는 원래 그림
        self.assertEqual(tuple(arr[50, 50]), (10, 10, 10))            # 상자 밖은 그대로
        self.assertEqual(np.asarray(full)[200, 200].tolist(), [10, 10, 10])   # 원본 불변
        self.assertGreater(np.asarray(feather_mask((100, 100)))[50, 50], 250)

    def test_eye_colour_back_only_inside_eyes(self):
        detailed = np.zeros((40, 40, 3), np.uint8); detailed[:] = (90, 160, 90)     # 초록 눈으로 바뀜
        chosen = np.zeros((40, 40, 3), np.uint8); chosen[:] = (90, 120, 200)        # 원래 파란 눈
        eyes = np.zeros((40, 40), bool); eyes[10:14, 10:20] = True
        out, n = eye_colour_back(detailed, chosen, eyes, dilate=0)
        self.assertEqual(n, 40)
        self.assertLess(float(rgb_to_lab(out)[12, 15, 2]), -10)                     # 눈 안: 파란 기운(b<0)
        self.assertAlmostEqual(float(rgb_to_lab(out)[12, 15, 0]), float(rgb_to_lab(detailed)[12, 15, 0]), delta=2)   # 밝기 유지
        self.assertEqual(tuple(out[30, 30]), (90, 160, 90))                          # 눈 밖 그대로
        same, zero = eye_colour_back(detailed, chosen, np.zeros((40, 40), bool))
        self.assertEqual(zero, 0); self.assertTrue((same == detailed).all())

    def test_eye_whites_are_not_tinted(self):
        detailed = np.zeros((40, 40, 3), np.uint8); detailed[:] = (120, 110, 140)     # 흐려진 홍채
        detailed[10:14, 10:14] = (245, 245, 248)                                       # 흰자
        chosen = np.zeros((40, 40, 3), np.uint8); chosen[:] = (170, 40, 220)           # 보라가 넓게 퍼진 후보
        eyes = np.zeros((40, 40), bool); eyes[10:14, 10:24] = True
        out, n = eye_colour_back(detailed, chosen, eyes, dilate=0)
        self.assertEqual(n, 4 * 10)                                                    # 흰자 4x4는 빠짐
        self.assertEqual(tuple(out[12, 12]), (245, 245, 248))                          # 흰자 그대로
        self.assertGreater(float(rgb_to_lab(out)[12, 20, 1]), float(rgb_to_lab(detailed)[12, 20, 1]) + 10)   # 홍채는 보라 쪽

    def test_prompts(self):
        positive, negative = face_prompts(dict(traits=[dict(text="violet eyes"), dict(text="")], negative_terms="woman"))
        self.assertTrue(positive.startswith("gnlstyle, close-up portrait of the face, violet eyes"))
        self.assertTrue(negative.endswith(", woman"))


class TestFaceEntry(unittest.TestCase):
    def test_runs_face_stage_on_the_chosen_candidate(self):
        with TemporaryDirectory() as temp:
            d = Path(temp)
            cands = []
            for s in (42, 43, 44):
                Image.new("RGB", (8, 8), (s, 0, 0)).save(d / f"result-seed{s}.png"); cands.append(str(d / f"result-seed{s}.png"))
            (d / "outcome.json").write_text(json.dumps(dict(state="review_pending", candidates=cands)), encoding="utf-8")
            calls = []
            def run(stage, directory):
                calls.append(stage)
                ask = json.loads((directory / "face-request.json").read_text(encoding="utf-8"))
                (directory / "face-result.json").write_text(json.dumps(dict(file=f"face-detail-seed{ask['seed']}.png",
                    raw_file=f"face-detail-raw-seed{ask['seed']}.png", peak_reserved_gib=5.6)), encoding="utf-8")
                return 0
            out = run_style_face_detail(d, cands[2], run=run)
            self.assertEqual((calls, out["state"]), (["face"], "review_pending"))
            self.assertTrue(out["result"].endswith("face-detail-seed44.png"))
            self.assertTrue((d / "face-outcome-seed44.json").is_file())
            with self.assertRaises(ValueError):
                run_style_face_detail(d, str(d / "other.png"), run=run)


if __name__ == "__main__":
    unittest.main()
