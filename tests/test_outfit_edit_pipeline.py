"""원본 편집형 의상 교체 실행 흐름(GPU 없이): 분할 작업 프로세스 계약과 합성·거부 판정."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from genai_lab import outfit_edit_pipeline as pipeline
from genai_lab import outfit_edit_worker as worker
from genai_lab.head_paste_rules import TAGS


def scene(seed=0):
    """특징점이 많은 얼굴·머리(위쪽)와 몸(아래쪽). parts는 TAGS 전체."""
    rng = np.random.default_rng(seed)
    image = np.full((1024, 768, 3), 255, np.uint8)
    image[140:420, 250:520] = (230, 150, 200)          # 머리(단색) 위에 특징점용 무늬
    image[240:360, 330:440] = (250, 220, 200)          # 얼굴
    for _ in range(140):
        x, y = int(rng.integers(260, 510)), int(rng.integers(140, 420))
        cv2.circle(image, (x, y), int(rng.integers(3, 14)), tuple(int(c) for c in rng.integers(0, 200, 3)), -1)
    image[450:980, 300:470] = (60, 40, 30)
    parts = {name: np.zeros((1024, 768), bool) for name in TAGS}
    parts["hair"][140:420, 250:520] = True
    parts["face"][240:360, 330:440] = True
    parts["hair"][240:360, 330:440] = False
    parts["topwear"][450:980, 300:470] = True
    return image, parts


def save_case(tmp_path, image, parts, raw):
    Image.fromarray(image).save(tmp_path / "canvas.png")
    np.savez_compressed(tmp_path / "parts.npz", **parts)
    Image.fromarray(raw).resize((896, 1184), Image.LANCZOS).save(tmp_path / "raw.png")
    return tmp_path / "canvas.png", tmp_path / "parts.npz", tmp_path / "raw.png"


def moved(image, scale=1.0, dy=0.0):
    matrix = cv2.getRotationMatrix2D((384, 280), 0, scale)
    matrix[1, 2] += dy
    return cv2.warpAffine(image, matrix, (768, 1024), borderValue=(255, 255, 255))


def test_compose_realigns_and_pastes_head_when_checks_pass(tmp_path):
    image, parts = scene()
    paths = save_case(tmp_path, image, parts, moved(image, 1.1, -60))      # Qwen이 머리를 키우고 위로 옮긴 경우
    outcome = pipeline.compose(*paths, "keep", tmp_path / "out")
    assert outcome["composited"] and outcome["alignment"]["decision"] == "aligned"
    assert outcome["hair_shape"]["status"] == "ok"
    assert (tmp_path / "out" / "result.png").is_file() and json.loads((tmp_path / "out" / "outcome.json").read_text(encoding="utf-8"))["composited"]


def test_compose_does_not_paste_hair_island_that_qwen_result_does_not_continue(tmp_path):
    # 요청 14 G19/S04: 꼬리 끝이 머리카락으로 분할돼, Qwen이 꼬리를 지운 자리에 보라색 조각으로 붙었다
    image, parts = scene()
    raw = moved(image, 1.1, -60)
    image[600:660, 600:680] = (230, 150, 200)      # 얼굴과 이어지지 않은 "머리카락" 섬(실제로는 꼬리)
    parts["hair"][600:660, 600:680] = True
    outcome = pipeline.compose(*save_case(tmp_path, image, parts, raw), "keep", tmp_path / "out")
    assert outcome["composited"] and outcome["hair_shape"]["detached_px"] == 60 * 80
    result = np.asarray(Image.open(tmp_path / "out" / "result.png").convert("RGB"))
    assert (result[600:660, 600:680] == 255).all()                    # 섬 자리는 Qwen 결과(흰 배경) 그대로
    assert (result[150:230, 260:330] == image[150:230, 260:330]).all()  # 머리 본체는 원본 그대로


def test_compose_refuses_unrelated_result_and_explains(tmp_path):
    image, parts = scene(1)
    other, _ = scene(2)
    outcome = pipeline.compose(*save_case(tmp_path, image, parts, other), "keep", tmp_path / "out")
    assert not outcome["composited"] and outcome["reason"]
    assert not (tmp_path / "out" / "result.png").exists() and (tmp_path / "out" / "review.png").is_file()


class FakeSegmenter:
    def __init__(self, repo, checkpoint, parts):
        self.parts, self.closed = parts, False

    def parse(self, image):
        return {name: np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((1024, 1024))) > 127
                for name, mask in self.parts.items()}

    def metrics(self):
        return {"max_reserved_gib": 0.0}

    def close(self):
        self.closed = True


def test_worker_segments_canvas_and_always_closes_model(tmp_path):
    image, _ = scene()
    prepared = pipeline.prepare_canvas(_write(tmp_path, image), tmp_path / "inputs")
    square = {name: np.zeros((1024, 1024), bool) for name in TAGS}
    square["face"][300:400, 450:560] = True
    square["hair"][220:300, 420:600] = True
    made = []
    def factory(repo, checkpoint):
        made.append(FakeSegmenter(repo, checkpoint, {k: v for k, v in square.items()}))
        made[-1].parse = lambda img, s=square: s
        return made[-1]
    result = worker.segment(dict(prepared, semantic_repo="r", semantic_checkpoint="c"), tmp_path, factory)
    assert made[0].closed and result["face_px"] > 0 and (tmp_path / "parts.npz").is_file()
    empty = {name: np.zeros((1024, 1024), bool) for name in TAGS}
    def blank(repo, checkpoint):
        seg = FakeSegmenter(repo, checkpoint, {}); seg.parse = lambda img: empty; made.append(seg); return seg
    with pytest.raises(ValueError, match="얼굴·머리"):
        worker.segment(dict(prepared, semantic_repo="r", semantic_checkpoint="c"), tmp_path, blank)
    assert made[-1].closed


def _write(tmp_path, image):
    path = tmp_path / "character.png"
    Image.fromarray(image).save(path)
    return path


def test_compose_refuses_when_arms_vanished_even_if_head_aligns(tmp_path):
    image, parts = scene()
    limbs = {"status": "rejected", "missing": ["left"], "checked": ["left", "right"]}
    outcome = pipeline.compose(*save_case(tmp_path, image, parts, moved(image, 1.1, -60)), "keep", tmp_path / "out", limbs=limbs)
    assert not outcome["composited"] and outcome["reason"] == pipeline.REASONS["limbs"]
    assert outcome["alignment"]["decision"] == "aligned"            # 머리는 맞았지만 팔 때문에 거부


def test_pipeline_retries_once_with_second_seed_when_checks_fail(tmp_path, monkeypatch):
    # D-105: 검사에 걸리면 다른 seed로 한 번만 자동 재시도한다
    from genai_lab import qwen_outfit_edit
    image, parts = scene()
    character, garment = tmp_path / "c.png", tmp_path / "g.png"
    Image.fromarray(image).save(character)
    Image.new("RGB", (300, 500), (40, 40, 40)).save(garment)
    np.savez_compressed(tmp_path / "parts.npz", **parts)
    monkeypatch.setattr(pipeline, "segment_canvas", lambda prepared, directory, runtime, **k: {"parts": str(tmp_path / "parts.npz")})
    monkeypatch.setattr(qwen_outfit_edit.OutfitEditWorkflow, "__init__", lambda self, spec, **k: None)
    seeds = []
    def fake_run(request, directory, workflow, **kw):
        seeds.append(request["seed"])
        Path(directory).mkdir(parents=True)
        Image.fromarray(image).resize((896, 1184)).save(Path(directory) / "raw.png")
    monkeypatch.setattr(qwen_outfit_edit, "run_outfit_edit", fake_run)
    monkeypatch.setattr(pipeline, "check_limbs", lambda *a, **k: {"status": "ok"})
    results = iter([dict(composited=False, reason="머리 위치", alignment={}), dict(composited=True, alignment={})])
    monkeypatch.setattr(pipeline, "compose", lambda *a, **k: dict(next(results)))
    settings = qwen_outfit_edit.QwenPoseSettings(sys.executable, str(tmp_path), str(tmp_path / "m.gguf"), str(tmp_path / "m.json"), "a" * 64)
    outcome = pipeline.run_outfit_pipeline(character, garment, settings=settings, head_accessories="keep", runtime=None,
                                           directory=tmp_path / "run")
    assert seeds == [42, 43] and outcome["composited"]
    assert [a["composited"] for a in outcome["attempts"]] == [False, True]
    assert (tmp_path / "run" / "qwen-2" / "raw.png").is_file()


def test_pipeline_does_not_retry_when_first_result_passes(tmp_path, monkeypatch):
    from genai_lab import qwen_outfit_edit
    image, parts = scene()
    character, garment = tmp_path / "c.png", tmp_path / "g.png"
    Image.fromarray(image).save(character)
    Image.new("RGB", (300, 500), (40, 40, 40)).save(garment)
    monkeypatch.setattr(pipeline, "segment_canvas", lambda *a, **k: {"parts": "p"})
    monkeypatch.setattr(qwen_outfit_edit.OutfitEditWorkflow, "__init__", lambda self, spec, **k: None)
    seeds = []
    monkeypatch.setattr(qwen_outfit_edit, "run_outfit_edit", lambda request, *a, **k: seeds.append(request["seed"]))
    monkeypatch.setattr(pipeline, "check_limbs", lambda *a, **k: {"status": "ok"})
    monkeypatch.setattr(pipeline, "compose", lambda *a, **k: dict(composited=True, alignment={}))
    settings = qwen_outfit_edit.QwenPoseSettings(sys.executable, str(tmp_path), str(tmp_path / "m.gguf"), str(tmp_path / "m.json"), "a" * 64)
    outcome = pipeline.run_outfit_pipeline(character, garment, settings=settings, head_accessories="keep", runtime=None,
                                           directory=tmp_path / "run")
    assert seeds == [42] and len(outcome["attempts"]) == 1
