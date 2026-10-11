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
