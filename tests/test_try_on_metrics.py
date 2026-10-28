import pytest
from PIL import Image, ImageDraw

from genai_lab.try_on_metrics import (
    create_try_on_difference_image,
    measure_try_on_effect_metrics,
)


def create_mask(
    size: tuple[int, int],
    rectangle: tuple[int, int, int, int],
) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(rectangle, fill=255)
    return mask


def test_effect_metrics_prove_raw_and_final_changes() -> None:
    base = Image.new("RGB", (4, 4), "black")
    raw = Image.new("RGB", (4, 4), "red")
    approved_mask = create_mask((4, 4), (1, 1, 2, 2))
    final = Image.composite(raw, base, approved_mask)
    model_mask = Image.new("L", (2, 2), 255)

    metrics = measure_try_on_effect_metrics(
        base,
        raw,
        final,
        approved_mask,
        model_mask,
    )

    assert metrics.raw_changed_inside_model_mask == 16
    assert metrics.final_changed_inside_approved_mask == 4
    assert metrics.discarded_by_protection_pixels == 0
    assert metrics.mean_rgb_l1_inside == 255.0
    assert metrics.mask_leakage_pixels == 0
    assert metrics.no_effect is False


def test_effect_metrics_mark_exact_no_effect() -> None:
    base = Image.new("RGB", (4, 4), "black")
    raw = Image.new("RGB", (4, 4), "red")
    approved_mask = create_mask((4, 4), (1, 1, 2, 2))

    metrics = measure_try_on_effect_metrics(
        base,
        raw,
        base.copy(),
        approved_mask,
        Image.new("L", (2, 2), 255),
    )

    assert metrics.raw_changed_inside_model_mask == 16
    assert metrics.final_changed_inside_approved_mask == 0
    assert metrics.discarded_by_protection_pixels == 4
    assert metrics.mean_rgb_l1_inside == 0.0
    assert metrics.mask_leakage_pixels == 0
    assert metrics.no_effect is True


def test_effect_metrics_count_exact_leakage_outside_approved_mask() -> None:
    base = Image.new("RGB", (4, 4), "black")
    final = base.copy()
    final.putpixel((0, 0), (1, 0, 0))
    approved_mask = create_mask((4, 4), (1, 1, 2, 2))

    metrics = measure_try_on_effect_metrics(
        base,
        final,
        final,
        approved_mask,
        Image.new("L", (4, 4), 255),
    )

    assert metrics.mask_leakage_pixels == 1


def test_effect_metrics_reject_coordinate_mismatch() -> None:
    with pytest.raises(ValueError, match="크기가 다릅니다"):
        measure_try_on_effect_metrics(
            Image.new("RGB", (4, 4), "black"),
            Image.new("RGB", (3, 4), "black"),
            Image.new("RGB", (4, 4), "black"),
            Image.new("L", (4, 4), 255),
            Image.new("L", (2, 2), 255),
        )


def test_difference_image_amplifies_visible_change_four_times() -> None:
    base = Image.new("RGB", (1, 1), (10, 20, 30))
    final = Image.new("RGB", (1, 1), (11, 22, 34))

    difference = create_try_on_difference_image(base, final)

    assert difference.getpixel((0, 0)) == (4, 8, 16)
