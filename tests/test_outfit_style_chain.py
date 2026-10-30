"""모드 A → B 연결 시험(가짜 실행기). 모델·GPU 불필요."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from genai_lab.outfit_style_chain import run_outfit_then_style


def fake_outfit(composited=True, make_file=True):
    def run(character, garment, *, runtime, directory, **kwargs):
        directory = Path(directory); (directory / "compose").mkdir(parents=True)
        result = directory / "compose" / ("result.png" if composited else "qwen_view.png")
        if make_file:
            Image.new("RGB", (8, 8), (10, 120, 90)).save(result)
        import hashlib
        out = dict(directory=str(directory), composited=composited, raw=str(directory / "raw.png"), trace=str(directory / "trace.json"))
        if composited:
            out.update(result=str(result), result_sha256=hashlib.sha256(result.read_bytes()).hexdigest() if make_file else None)
        else:
            out.update(qwen_view=str(result), reason="머리 검사")
        return out
    return run


class TestChain(unittest.TestCase):
    def test_outfit_result_goes_to_style_with_gender(self):
        calls = []
        def style(base, directory, runtime, *, confirmed_gender, paths):
            calls.append((Path(base).name, confirmed_gender)); return dict(state="review_pending", candidates=[])
        with TemporaryDirectory() as temp:
            out = run_outfit_then_style("c.png", "g.png", Path(temp) / "run", outfit_kwargs=dict(gender="female", settings=None),
                                        outfit_runner=fake_outfit(), style_runner=style)
            self.assertEqual(out["state"], "review_pending")
            self.assertEqual(calls, [("edit-result.png", "female")])
            review = json.loads((Path(temp) / "run" / "edit-result.review.json").read_text(encoding="utf-8"))
            self.assertEqual(review["reviewer"], "automatic_chain")                # 사람 승인이 아님을 기록
            self.assertTrue((Path(temp) / "run" / "chain.json").is_file())

    def test_uncomposited_outfit_still_passes_the_edited_view(self):
        calls = []
        def style(base, directory, runtime, *, confirmed_gender, paths):
            calls.append(confirmed_gender); return dict(state="review_pending")
        with TemporaryDirectory() as temp:
            out = run_outfit_then_style("c.png", "g.png", Path(temp) / "run", outfit_kwargs=dict(gender="unspecified"),
                                        outfit_runner=fake_outfit(composited=False), style_runner=style)
            self.assertEqual((out["state"], calls), ("review_pending", [None]))
            self.assertFalse(out["outfit"]["composited"])

    def test_missing_outfit_picture_stops_before_style(self):
        def style(*a, **k): raise AssertionError("must not run")
        with TemporaryDirectory() as temp:
            out = run_outfit_then_style("c.png", "g.png", Path(temp) / "run", outfit_kwargs={},
                                        outfit_runner=fake_outfit(composited=False, make_file=False), style_runner=style)
            self.assertEqual((out["state"], out["stage"]), ("failed", "outfit"))


if __name__ == "__main__":
    unittest.main()
