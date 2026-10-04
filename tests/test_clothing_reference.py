from pathlib import Path

import numpy as np

from PIL import Image
import pytest

from genai_lab.clothing_reference import (
    ClothingCombinedMaskCandidate,
    ClothingDetectionSettings,
    ClothingPixelExtractionSettings,
    ClothingRegionCandidate,
    NormalizedClothingSource,
    build_clothing_mask_review_candidates,
    combine_clothing_mask_candidates,
    convert_sam2_mask_to_alpha,
    create_sam2_image_config,
    _build_valid_clothing_region_candidates,
    create_manual_clothing_region,
    extract_clothing_pixels,
    fill_enclosed_clothing_mask_holes,
    measure_clothing_region,
    ClothingSourceInput,
    ClothingSourceValidationCode,
    ClothingSourceValidationError,
    load_and_normalize_clothing_source,
)


def test_jpeg_source_is_normalized_to_rgb(tmp_path: Path) -> None:
    source_path = tmp_path / "clothing.jpg"
    Image.new("RGB", (6, 4), (20, 40, 60)).save(
        source_path,
        format="JPEG",
    )

    normalized_source = load_and_normalize_clothing_source(
        ClothingSourceInput(image_path=source_path)
    )

    try:
        assert normalized_source.source_format == "JPEG"
        assert normalized_source.source_size_bytes > 0
        assert normalized_source.image.mode == "RGB"
        assert normalized_source.image.size == (6, 4)
        assert normalized_source.original_width == 6
        assert normalized_source.original_height == 4
    finally:
        normalized_source.image.close()


def test_transparent_png_uses_white_background(tmp_path: Path) -> None:
    source_path = tmp_path / "transparent-clothing.png"
    source_image = Image.new("RGBA", (2, 1), (0, 0, 255, 255))
    source_image.putpixel((0, 0), (255, 0, 0, 0))
    source_image.save(source_path, format="PNG")
    source_image.close()

    normalized_source = load_and_normalize_clothing_source(
        ClothingSourceInput(image_path=source_path)
    )

    try:
        assert normalized_source.source_format == "PNG"
        assert normalized_source.image.getpixel((0, 0)) == (255, 255, 255)
        assert normalized_source.image.getpixel((1, 0)) == (0, 0, 255)
    finally:
        normalized_source.image.close()


def test_exif_orientation_is_applied(tmp_path: Path) -> None:
    source_path = tmp_path / "rotated-clothing.jpg"
    source_image = Image.new("RGB", (4, 2), (80, 100, 120))
    exif_data = Image.Exif()
    exif_data[274] = 6
    source_image.save(
        source_path,
        format="JPEG",
        exif=exif_data,
    )
    source_image.close()

    normalized_source = load_and_normalize_clothing_source(
        ClothingSourceInput(image_path=source_path)
    )

    try:
        assert (
            normalized_source.original_width,
            normalized_source.original_height,
        ) == (4, 2)
        assert normalized_source.image.size == (2, 4)
        assert (
            normalized_source.normalized_width,
            normalized_source.normalized_height,
        ) == (2, 4)
    finally:
        normalized_source.image.close()


def test_damaged_image_returns_korean_recovery_action(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "damaged.png"
    source_path.write_bytes(b"not-an-image")

    with pytest.raises(ClothingSourceValidationError) as captured_error:
        load_and_normalize_clothing_source(
            ClothingSourceInput(image_path=source_path)
        )

    assert (
        captured_error.value.code
        is ClothingSourceValidationCode.SOURCE_DECODE_FAILED
    )
    assert "읽을 수 없습니다" in captured_error.value.message_ko
    assert "다시 선택하세요" in captured_error.value.recovery_action_ko



def test_manual_region_records_area_without_fake_confidence() -> None:
    manual_candidate = create_manual_clothing_region(
        image_size=(100, 100),
        box_xyxy=(10, 20, 60, 80),
    )

    measurement = measure_clothing_region(
        manual_candidate,
        image_size=(100, 100),
    )

    assert measurement.detection_method == "manual"
    assert measurement.confidence_percent is None
    assert measurement.area_ratio_percent == pytest.approx(30.0)
    assert measurement.width_pixels == 50
    assert measurement.height_pixels == 60


def test_manual_region_outside_image_is_rejected() -> None:
    with pytest.raises(ValueError, match="이미지 안"):
        create_manual_clothing_region(
            image_size=(100, 100),
            box_xyxy=(-1, 0, 50, 50),
        )


def test_automatic_candidates_are_filtered_by_area_and_sorted() -> None:
    settings = ClothingDetectionSettings(
        minimum_area_ratio=0.02,
        maximum_area_ratio=0.95,
    )

    candidates = _build_valid_clothing_region_candidates(
        boxes=[
            [0.0, 0.0, 10.0, 10.0],
            [10.0, 10.0, 60.0, 60.0],
            [20.0, 20.0, 80.0, 80.0],
            [0.0, 0.0, 100.0, 100.0],
        ],
        scores=[0.95, 0.70, 0.85, 0.99],
        labels=("small", "lower", "higher", "too_large"),
        image_size=(100, 100),
        settings=settings,
    )

    assert [candidate.label for candidate in candidates] == [
        "higher",
        "lower",
    ]
    assert candidates[0].confidence == pytest.approx(0.85)


def test_sam2_mask_candidates_are_ranked_and_measured() -> None:
    restored_masks = np.zeros((3, 4, 5), dtype=bool)
    restored_masks[0, 1:3, 1:3] = True
    restored_masks[1, 0, 0] = True
    restored_masks[1, 3, 4] = True

    candidates = build_clothing_mask_review_candidates(
        restored_masks=restored_masks,
        model_scores=np.array([0.80, 0.90, 0.70], dtype=np.float32),
        approved_region=ClothingRegionCandidate(
            box_xyxy=(1, 1, 4, 3),
            label="clothing",
            confidence=0.75,
        ),
        image_size=(5, 4),
        maximum_candidate_count=3,
    )

    try:
        assert len(candidates) == 2
        assert candidates[0].model_score == pytest.approx(0.90)
        assert candidates[0].selected_pixel_count == 2
        assert candidates[0].connected_region_count == 2
        assert candidates[0].boundary_touch_pixel_count == 2
        assert candidates[1].model_score == pytest.approx(0.80)
        assert candidates[1].selected_pixel_count == 4
        assert candidates[1].region_coverage_percent == pytest.approx(
            4 / 6 * 100.0
        )
        assert candidates[1].connected_region_count == 1
        assert candidates[1].boundary_touch_pixel_count == 0
    finally:
        for candidate in candidates:
            candidate.mask_image.close()


def test_separated_clothing_masks_are_combined_without_losing_islands() -> None:
    upper_masks = np.zeros((1, 6, 6), dtype=bool)
    upper_masks[0, 0:2, 1:5] = True
    shoe_masks = np.zeros((1, 6, 6), dtype=bool)
    shoe_masks[0, 4:6, 0:2] = True
    upper_candidates = build_clothing_mask_review_candidates(
        restored_masks=upper_masks,
        model_scores=np.array([0.90], dtype=np.float32),
        approved_region=ClothingRegionCandidate(
            box_xyxy=(1, 0, 5, 2),
            label="upper",
            confidence=0.90,
        ),
        image_size=(6, 6),
        maximum_candidate_count=3,
    )
    shoe_candidates = build_clothing_mask_review_candidates(
        restored_masks=shoe_masks,
        model_scores=np.array([0.85], dtype=np.float32),
        approved_region=ClothingRegionCandidate(
            box_xyxy=(0, 4, 2, 6),
            label="shoes",
            confidence=0.85,
        ),
        image_size=(6, 6),
        maximum_candidate_count=3,
    )

    combined_mask = combine_clothing_mask_candidates(
        selected_candidates=(upper_candidates[0], shoe_candidates[0]),
        image_size=(6, 6),
    )

    try:
        assert combined_mask.source_region_count == 2
        assert combined_mask.selected_pixel_count == 12
        assert combined_mask.connected_region_count == 2
        assert combined_mask.boundary_touch_pixel_count == 7
        combined_pixels = np.asarray(combined_mask.mask_image)
        assert np.all(combined_pixels[0:2, 1:5] == 255)
        assert np.all(combined_pixels[4:6, 0:2] == 255)
        assert combined_pixels[3, 3] == 0
    finally:
        combined_mask.mask_image.close()
        upper_candidates[0].mask_image.close()
        shoe_candidates[0].mask_image.close()


def test_sam2_video_checkpoint_config_is_converted_to_image_config() -> None:
    from transformers import Sam2VideoConfig

    source_config = Sam2VideoConfig()

    image_config = create_sam2_image_config(source_config)

    assert source_config.model_type == "sam2_video"
    assert image_config.model_type == "sam2"
    assert image_config.architectures == ["Sam2Model"]
    assert (
        image_config.vision_config.backbone_config.blocks_per_stage
        == source_config.vision_config.backbone_config.blocks_per_stage
    )


def test_sam2_logits_keep_soft_alpha_boundary() -> None:
    predicted_mask = np.array(
        [[-20.0, 0.0, 20.0]],
        dtype=np.float32,
    )

    mask_alpha = convert_sam2_mask_to_alpha(predicted_mask)

    assert mask_alpha.tolist() == [[0, 128, 255]]


def test_pixel_extraction_preserves_rgb_and_soft_alpha() -> None:
    source_image = Image.new("RGB", (5, 5), (20, 40, 60))
    mask_alpha = np.zeros((5, 5), dtype=np.uint8)
    mask_alpha[1:4, 1:4] = 255
    mask_alpha[0, 2] = 64
    mask_image = Image.fromarray(mask_alpha, mode="L")
    normalized_source = NormalizedClothingSource(
        image=source_image,
        source_name="clothing.png",
        source_format="PNG",
        source_mode="RGB",
        source_size_bytes=75,
        original_width=5,
        original_height=5,
        normalized_width=5,
        normalized_height=5,
    )
    combined_mask = ClothingCombinedMaskCandidate(
        mask_image=mask_image,
        source_region_count=1,
        selected_pixel_count=9,
        connected_region_count=1,
        boundary_touch_pixel_count=0,
    )

    extraction_candidate = extract_clothing_pixels(
        normalized_source,
        combined_mask,
    )

    try:
        extracted_rgba = np.asarray(
            extraction_candidate.extracted_image,
            dtype=np.uint8,
        )
        assert extraction_candidate.preview_crop_box == (1, 0, 4, 4)
        assert extraction_candidate.soft_edge_pixel_count == 1
        assert extraction_candidate.changed_rgb_pixel_count == 0
        assert (
            extraction_candidate.original_pixel_preservation_percent
            == pytest.approx(100.0)
        )
        assert extracted_rgba[0, 2].tolist() == [20, 40, 60, 64]
    finally:
        extraction_candidate.extracted_image.close()
        extraction_candidate.clothing_mask.close()
        mask_image.close()
        source_image.close()


def test_small_enclosed_hole_is_filled_only_when_color_matches() -> None:
    mask_alpha = np.zeros((7, 7), dtype=np.uint8)
    mask_alpha[1:6, 1:6] = 255
    mask_alpha[3, 3] = 0
    matching_source = np.full((7, 7, 3), (80, 90, 100), dtype=np.uint8)
    contrasting_source = matching_source.copy()
    contrasting_source[3, 3] = (255, 0, 0)
    settings = ClothingPixelExtractionSettings(
        maximum_hole_area_pixels=4,
        maximum_hole_area_ratio=0.10,
        maximum_rgb_distance=36.0,
        white_clothing_luminance=200.0,
        maximum_white_luminance_difference=48.0,
    )

    (
        repaired_matching_alpha,
        matching_holes,
        matching_filled,
        matching_filled_pixels,
        matching_skipped,
    ) = fill_enclosed_clothing_mask_holes(
        matching_source,
        mask_alpha,
        settings,
    )
    (
        repaired_contrasting_alpha,
        contrasting_holes,
        contrasting_filled,
        contrasting_filled_pixels,
        contrasting_skipped,
    ) = fill_enclosed_clothing_mask_holes(
        contrasting_source,
        mask_alpha,
        settings,
    )

    assert repaired_matching_alpha[3, 3] == 255
    assert (matching_holes, matching_filled, matching_filled_pixels) == (
        1,
        1,
        1,
    )
    assert matching_skipped == 0
    assert repaired_contrasting_alpha[3, 3] == 0
    assert (
        contrasting_holes,
        contrasting_filled,
        contrasting_filled_pixels,
        contrasting_skipped,
    ) == (1, 0, 0, 1)
