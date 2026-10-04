import numpy as np
import pytest
from PIL import Image

from genai_lab.garment_warp import (
    GarmentTpsWarpError,
    GarmentTpsWarpRequest,
    warp_garment_rgba_tps,
)


SOURCE_POINTS = np.array(
    [[5, 5], [15, 5], [5, 15], [15, 15], [10, 10]],
    dtype=np.float32,
)


def create_rgba_garment(size: tuple[int, int] = (40, 40)) -> Image.Image:
    pixels = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    pixels[5:16, 5:16, :3] = (20, 100, 220)
    pixels[5:16, 5:16, 3] = 255
    return Image.fromarray(pixels)


def test_identity_tps_preserves_rgba_pixels_exactly() -> None:
    garment = create_rgba_garment()
    result = warp_garment_rgba_tps(
        GarmentTpsWarpRequest(
            garment, SOURCE_POINTS, SOURCE_POINTS.copy(), (40, 40)
        )
    )
    assert np.array_equal(np.asarray(garment), np.asarray(result.warped_rgba))
    assert result.source_point_count == 5
    assert result.source_alpha_pixels == 121
    assert result.warped_alpha_pixels == 121
    assert result.alpha_pixel_change == 0
    result.close()
    garment.close()


def test_translation_moves_rgba_and_alpha_to_target_coordinates() -> None:
    garment = create_rgba_garment()
    target_points = SOURCE_POINTS.copy()
    target_points[:, 0] += 15
    result = warp_garment_rgba_tps(
        GarmentTpsWarpRequest(
            garment, SOURCE_POINTS, target_points, (40, 40)
        )
    )
    assert result.warped_rgba.getchannel("A").getbbox() == (20, 5, 31, 16)
    assert result.warped_rgba.getpixel((25, 10)) == (20, 100, 220, 255)
    assert result.warped_rgba.getpixel((10, 10))[3] == 0
    result.close()
    garment.close()


@pytest.mark.parametrize(
    ("source_points", "target_points", "message"),
    [
        (SOURCE_POINTS[:4], SOURCE_POINTS[:4], "최소 5개"),
        (
            np.vstack((SOURCE_POINTS[:4], SOURCE_POINTS[0])),
            SOURCE_POINTS,
            "중복 좌표",
        ),
        (
            SOURCE_POINTS,
            SOURCE_POINTS + np.array([30, 0], dtype=np.float32),
            "캔버스 밖",
        ),
        (
            np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], np.float32),
            SOURCE_POINTS,
            "한 직선",
        ),
    ],
)
def test_tps_rejects_invalid_control_points(
    source_points: np.ndarray,
    target_points: np.ndarray,
    message: str,
) -> None:
    garment = create_rgba_garment()
    with pytest.raises(GarmentTpsWarpError, match=message):
        warp_garment_rgba_tps(
            GarmentTpsWarpRequest(
                garment, source_points, target_points, (40, 40)
            )
        )
    garment.close()


def test_tps_rejects_coordinate_count_mismatch() -> None:
    garment = create_rgba_garment()
    with pytest.raises(GarmentTpsWarpError, match="대응점 수"):
        warp_garment_rgba_tps(
            GarmentTpsWarpRequest(
                garment,
                SOURCE_POINTS,
                np.vstack((SOURCE_POINTS, [[20, 20]])),
                (40, 40),
            )
        )
    garment.close()


def test_tps_rejects_canvas_size_mismatch() -> None:
    garment = create_rgba_garment()
    with pytest.raises(GarmentTpsWarpError, match="캔버스 크기가 다릅니다"):
        warp_garment_rgba_tps(
            GarmentTpsWarpRequest(
                garment, SOURCE_POINTS, SOURCE_POINTS, (32, 32)
            )
        )
    garment.close()


def test_tps_rejects_zero_alpha_garment() -> None:
    garment = Image.new("RGBA", (40, 40), (255, 255, 255, 0))
    with pytest.raises(GarmentTpsWarpError, match="알파 픽셀이 0개"):
        warp_garment_rgba_tps(
            GarmentTpsWarpRequest(
                garment, SOURCE_POINTS, SOURCE_POINTS, (40, 40)
            )
        )
    garment.close()
