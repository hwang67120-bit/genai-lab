"""모드 B 공통 진입점·작업 프로세스 도우미 시험. 하위 프로세스는 가짜 실행기로 바꾼다(모델·GPU 불필요)."""
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from PIL import Image

from genai_lab.style_redraw_pipeline import run_style_redraw
from genai_lab.style_redraw_worker import joints_from_pose, support_warnings

RUNTIME = SimpleNamespace(model_cache=Path("cache"), head_semantic_repo=Path("repo"), head_semantic_checkpoint=Path("ckpt"))


def fake_run(calls, prepare_state="ready", fail=None):
    def run(stage, directory):
        calls.append(stage)
        if stage == fail:
            (directory / f"{stage}-error.json").write_text(json.dumps({"message": "sentinel"}), encoding="utf-8")
            return 1
        if stage == "pose":
            (directory / "pose.json").write_text("{}", encoding="utf-8")
        elif stage == "prepare":
            body = dict(state=prepare_state, warnings=[dict(code="upper_body_only")], traits=[], gender="male")
            if prepare_state == "refused":
                body.update(code="safety", message="진행할 수 없습니다.")
            (directory / "prepare.json").write_text(json.dumps(body), encoding="utf-8")
        else:
            for seed in (42, 43, 44):
                Image.new("RGB", (768, 1024)).save(directory / f"result-seed{seed}.png")
            body = {"peak_reserved_gib": 6.2, "candidates": [{"seed": s, "file": f"result-seed{s}.png"} for s in (42, 43, 44)]}
            (directory / "generation.json").write_text(json.dumps(body), encoding="utf-8")
        return 0
    return run


class TestCancel(unittest.TestCase):
    """D-224: 작업실 '취소'가 실행 중인 작업 프로세스를 실제로 끝낸다."""

    def test_running_process_is_stopped_on_cancel(self):
        import sys
        from genai_lab.style_redraw_pipeline import StyleRedrawCancelled, run_process
        with TemporaryDirectory() as temp:
            t0 = time.time()
            asked = time.time() + 1.0
            with self.assertRaises(StyleRedrawCancelled):
                run_process([sys.executable, "-c", "import time; time.sleep(60)"], Path(temp) / "log.txt", 120,
                            cancelled=lambda: time.time() > asked, stage="generate")
            self.assertLess(time.time() - t0, 15)                 # 60초를 기다리지 않는다
            time.sleep(0.5)
            (Path(temp) / "log.txt").unlink()                     # 자식(실제 Python)까지 끝나 로그를 놓았다

    def test_timeout_stops_the_process_and_normal_exit_returns_code(self):
        import subprocess
        import sys
        from genai_lab.style_redraw_pipeline import run_process
        with TemporaryDirectory() as temp:
            with self.assertRaises(subprocess.TimeoutExpired):
                run_process([sys.executable, "-c", "import time; time.sleep(60)"], Path(temp) / "a.txt", 1.0)
            self.assertEqual(run_process([sys.executable, "-c", "print('ok'); raise SystemExit(3)"], Path(temp) / "b.txt", 30), 3)
            self.assertIn("ok", (Path(temp) / "b.txt").read_text(encoding="utf-8"))

    def test_cancel_between_stages_skips_generation(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "char.png"
            Image.new("RGB", (400, 600)).save(source)
            calls, flag = [], []
            def run(stage, directory):
                calls.append(stage)
                if stage == "prepare":
                    flag.append(True)                              # 준비가 끝날 즈음 사용자가 취소
                return fake_run([])(stage, directory)
            out = run_style_redraw(source, root / "run", RUNTIME, run=run, cancelled=lambda: bool(flag), adult_confirmed=True)
            self.assertEqual(out["state"], "cancelled")
            self.assertEqual(calls, ["pose", "prepare"])
            self.assertEqual(json.loads((root / "run" / "outcome.json").read_text(encoding="utf-8"))["state"], "cancelled")


class TestAdultConfirmation(unittest.TestCase):
    def test_nothing_runs_without_adult_confirmation(self):
        """D-227: 태그 분석기가 노출 없는 미성년·아이 캐릭터를 못 거르므로(D-226) 사용자 확인 없이는 시작하지 않는다(화면·API 공통)."""
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "char.png"
            Image.new("RGB", (400, 600)).save(source)
            calls = []
            for value in (False, None, "yes", 1):
                out = run_style_redraw(source, root / f"run-{value}", RUNTIME, run=fake_run(calls), adult_confirmed=value)
                self.assertEqual((out["state"], out["code"]), ("refused", "adult_unconfirmed"))
            self.assertEqual(calls, [])                                     # 관절·준비·생성 아무것도 안 돎
            self.assertFalse((root / "run-False" / "character.png").exists())
            run_style_redraw(source, root / "ok", RUNTIME, run=fake_run(calls), adult_confirmed=True)
            self.assertTrue(json.loads((root / "ok" / "request.json").read_text(encoding="utf-8"))["adult_confirmed"])


class TestStyleRedrawPipeline(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "char.png"
        Image.new("RGB", (400, 600), (90, 60, 40)).save(self.source)

    def test_review_pending_after_three_stages_on_fitted_canvas(self):
        calls = []
        out = run_style_redraw(self.source, self.root / "run", RUNTIME, run=fake_run(calls), adult_confirmed=True)
        self.assertEqual(calls, ["pose", "prepare", "generate"])
        self.assertEqual(out["state"], "review_pending")
        self.assertEqual([Path(c).name for c in out["candidates"]], ["result-seed42.png", "result-seed43.png", "result-seed44.png"])
        self.assertEqual(out["warnings"][0]["code"], "upper_body_only")
        with Image.open(self.root / "run" / "character.png") as canvas:
            self.assertEqual(canvas.size, (768, 1024))
        saved = json.loads((self.root / "run" / "outcome.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["state"], "review_pending")
        request = json.loads((self.root / "run" / "request.json").read_text(encoding="utf-8"))
        self.assertEqual(request["model_cache"], "cache")
        self.assertTrue({"flux_model", "hand_cache"} <= set(request))
        self.assertFalse({"style_models", "style_reference", "style_reference_pose", "style_lora"} & set(request))   # D-216
        self.assertEqual(request["engine"], "flux")

    def test_refused_stops_before_generation(self):
        calls = []
        out = run_style_redraw(self.source, self.root / "run", RUNTIME, run=fake_run(calls, "refused"), adult_confirmed=True)
        self.assertEqual(calls, ["pose", "prepare"])
        self.assertEqual((out["state"], out["code"]), ("refused", "safety"))
        self.assertFalse((self.root / "run" / "result.png").exists())

    def test_failed_stage_reported_with_worker_message(self):
        for stage in ("pose", "prepare", "generate"):
            with self.subTest(stage=stage):
                calls = []
                out = run_style_redraw(self.source, self.root / stage, RUNTIME, run=fake_run(calls, fail=stage), adult_confirmed=True)
                self.assertEqual((out["state"], out["stage"]), ("failed", stage))
                self.assertIn("sentinel", out["message"])
                self.assertEqual(calls[-1], stage)

    def test_existing_directory_rejected_and_unreadable_image_fails(self):
        (self.root / "used").mkdir()
        with self.assertRaises(FileExistsError):
            run_style_redraw(self.source, self.root / "used", RUNTIME, run=fake_run([]), adult_confirmed=True)
        bad = self.root / "bad.png"
        bad.write_bytes(b"not an image")
        out = run_style_redraw(bad, self.root / "bad-run", RUNTIME, run=fake_run([]), adult_confirmed=True)
        self.assertEqual((out["state"], out["stage"]), ("failed", "input"))

    def test_only_confirmed_gender_is_passed_on(self):
        run_style_redraw(self.source, self.root / "none", RUNTIME, run=fake_run([]), adult_confirmed=True)
        self.assertIsNone(json.loads((self.root / "none" / "request.json").read_text(encoding="utf-8"))["confirmed_gender"])
        run_style_redraw(self.source, self.root / "male", RUNTIME, confirmed_gender="male", run=fake_run([]), adult_confirmed=True)
        self.assertEqual(json.loads((self.root / "male" / "request.json").read_text(encoding="utf-8"))["confirmed_gender"], "male")
        with self.assertRaises(ValueError):
            run_style_redraw(self.source, self.root / "bad", RUNTIME, confirmed_gender="unspecified", run=fake_run([]), adult_confirmed=True)

    def test_path_overrides_win(self):
        run_style_redraw(self.source, self.root / "run", RUNTIME, paths={"style_models": self.root / "m"}, run=fake_run([]), adult_confirmed=True)
        request = json.loads((self.root / "run" / "request.json").read_text(encoding="utf-8"))
        self.assertEqual(request["style_models"], str(self.root / "m"))


class TestWorkerHelpers(unittest.TestCase):
    def test_joints_keep_confident_detected_only(self):
        record = {"joints": [dict(joint_name="neck", x=1, y=2, confidence_score=0.9, detected=True),
                             dict(joint_name="nose", x=3, y=4, confidence_score=0.2, detected=True),
                             dict(joint_name="left_knee", x=5, y=6, confidence_score=0.9, detected=False)]}
        self.assertEqual(joints_from_pose(record), {"neck": [1.0, 2.0]})

    def test_support_warnings(self):
        legs = {"left_knee": [0, 0]}
        self.assertEqual(support_warnings({}, legs), [])
        codes = [w["code"] for w in support_warnings({"undercut": 0.7, "heterochromia": 0.6, "sitting": 0.4}, {})]
        self.assertEqual(codes, ["upper_body_only", "rare_hairstyle", "heterochromia"])


if __name__ == "__main__":
    unittest.main()
