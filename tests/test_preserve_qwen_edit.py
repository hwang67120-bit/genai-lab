"""Qwen 비교 스크립트의 순수 함수: 평행 이동 측정·보정과 마스크 안 합성을 GPU 없이 확인한다."""
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("preserve_qwen_edit", ROOT / "scripts/preserve_qwen_edit.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def textured(seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (1024 // 16, 768 // 16, 3)).astype(np.uint8)
    return np.kron(base, np.ones((16, 16, 1), np.uint8))


def test_shift_is_measured_and_undone():
    original = textured()
    edited = np.roll(original, (6, -4), axis=(0, 1))   # 아래로 6, 왼쪽으로 4
    area = np.ones(original.shape[:2], bool)
    dx, dy, _ = script.estimate_shift(original, edited, area)
    assert abs(dx + 4) < 1 and abs(dy - 6) < 1
    back = script.align(edited, dx, dy)
    inner = (slice(40, -40), slice(40, -40))
    assert np.abs(back[inner].astype(int) - original[inner]).mean() < 3


def test_composite_changes_only_inside_mask():
    person = textured(1)
    edited = textured(2)
    change = np.zeros(person.shape[:2], bool); change[300:700, 200:500] = True
    result = script.composite(person, edited, change)
    assert np.array_equal(result[~change], person[~change])
    assert not np.array_equal(result[change], person[change])


def test_head_composite_keeps_head_core_and_drops_original_background():
    person = textured(3)
    edited = textured(4)
    protect = np.zeros(person.shape[:2], bool); protect[100:400, 250:550] = True
    result, core = script.head_composite(person, edited, protect)
    assert core.any() and np.array_equal(result[core], person[core])
    assert np.array_equal(result[~protect], edited[~protect])          # 머리 밖은 원본 배경·소품이 남지 않는다
    edge = protect & ~(np.roll(protect, 3, 0) & np.roll(protect, -3, 0))
    assert not np.array_equal(result[edge], person[edge])               # 가장자리는 Qwen 쪽과 섞인다


def test_border_white_ratio():
    image = np.full((1024, 768, 3), 255, np.uint8)
    assert script.border_white_ratio(image) == 1.0
    image[:, :20] = (200, 40, 90)
    assert 0 < script.border_white_ratio(image) < 1


class FakeCuda:
    def __init__(self, reserved):
        self.reserved = reserved

    def max_memory_reserved(self):
        return self.reserved * 2 ** 30

    def memory_allocated(self):
        return 2 ** 30


def test_memory_guard_records_and_stops_over_limit(tmp_path):
    guard = script.MemoryGuard(FakeCuda(6.0), 6.5, tmp_path / "memory.json")
    callback = guard.step_callback("W3")
    assert callback(None, 0, None, {"latents": 1}) == {"latents": 1}
    guard.cuda.reserved = 7.0
    try:
        callback(None, 1, None, {})
        raise AssertionError("한도 초과인데 멈추지 않았습니다")
    except MemoryError as error:
        assert "W3/step1" in str(error)
    import json
    log = json.loads((tmp_path / "memory.json").read_text(encoding="utf-8"))
    assert [s["stage"] for s in log["stages"]] == ["W3/step0", "W3/step1"] and log["limit_gib"] == 6.5
