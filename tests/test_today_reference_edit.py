import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_today_reference_edit.py"
SPEC = importlib.util.spec_from_file_location("today_reference_edit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_soft_mask_never_crosses_hard_mask():
    raw = Image.new("L", (64, 64), 0)
    protection = Image.new("L", (64, 64), 0)
    foreground = Image.new("L", (64, 64), 255)
    for y in range(16, 48):
        for x in range(16, 48):
            raw.putpixel((x, y), 255)
    hard, soft, _ = MODULE.create_masks(
        raw, protection, foreground, 0.0, 1.0
    )
    try:
        hard_array = np.asarray(hard)
        soft_array = np.asarray(soft)
        assert np.count_nonzero(soft_array[hard_array == 0]) == 0
    finally:
        for image in (raw, protection, foreground, hard, soft):
            image.close()


def test_composite_preserves_every_outside_pixel():
    source = Image.new("RGB", (32, 32), (10, 20, 30))
    proposed = Image.new("RGB", (32, 32), (200, 100, 50))
    hard = Image.new("L", (32, 32), 0)
    soft = Image.new("L", (32, 32), 0)
    for y in range(8, 24):
        for x in range(8, 24):
            hard.putpixel((x, y), 255)
            soft.putpixel((x, y), 255)
    result, outside_changes = MODULE.composite(source, proposed, hard, soft)
    try:
        assert outside_changes == 0
        assert result.getpixel((0, 0)) == source.getpixel((0, 0))
        assert result.getpixel((16, 16)) == proposed.getpixel((16, 16))
    finally:
        for image in (source, proposed, hard, soft, result):
            image.close()
