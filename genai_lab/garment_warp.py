"""승인된 의상 RGBA를 캐릭터 공통 좌표로 TPS 워핑한다."""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class GarmentTpsWarpRequest:
    """TPS에 전달할 공통 좌표 의상과 대응점."""

    garment_rgba_canvas: Image.Image
    source_points_xy: np.ndarray
    target_points_xy: np.ndarray
    canvas_size: tuple[int, int]
    regularization: float = 0.0


@dataclass(frozen=True)
class GarmentTpsWarpResult:
    """사용자 검토 전 TPS 워핑 결과와 픽셀 수치."""

    warped_rgba: Image.Image
    warped_alpha_mask: Image.Image
    source_point_count: int
    source_alpha_pixels: int
    warped_alpha_pixels: int
    alpha_pixel_change: int

    def close(self) -> None:
        """워핑 결과 이미지 2개를 해제한다."""
        self.warped_rgba.close()
        self.warped_alpha_mask.close()


class GarmentTpsWarpError(ValueError):
    """TPS 입력 계약을 만족하지 못하거나 워핑을 실행할 수 없는 오류."""


def warp_garment_rgba_tps(
    request: GarmentTpsWarpRequest,
) -> GarmentTpsWarpResult:
    """의상 RGB와 소프트 알파를 같은 TPS 변환으로 워핑한다.

    OpenCV `warpImage`는 출력 픽셀에서 입력 픽셀을 찾는 역방향 샘플링을
    사용하므로 이미지 워핑용 변환은 목표점에서 원본점 순서로 추정한다.
    """
    width, height = _validate_canvas_size(request.canvas_size)
    _validate_regularization(request.regularization)
    if not hasattr(cv2, "createThinPlateSplineShapeTransformer"):
        raise GarmentTpsWarpError(
            "현재 OpenCV에 TPS 함수가 없습니다. "
            "opencv-contrib-python 설치가 필요합니다."
        )

    garment_rgba = request.garment_rgba_canvas.convert("RGBA")
    try:
        if garment_rgba.size != (width, height):
            raise GarmentTpsWarpError(
                "의상과 목표 캔버스 크기가 다릅니다: "
                f"의상={garment_rgba.size}, 캔버스={(width, height)}"
            )

        source_points = _validate_control_points(
            request.source_points_xy, "의상 원본", width, height
        )
        target_points = _validate_control_points(
            request.target_points_xy, "캐릭터 목표", width, height
        )
        if source_points.shape != target_points.shape:
            raise GarmentTpsWarpError(
                "TPS 대응점 수가 다릅니다: "
                f"의상={source_points.shape[0]}개, "
                f"캐릭터={target_points.shape[0]}개"
            )

        rgba_array = np.asarray(garment_rgba, dtype=np.uint8)
        alpha = rgba_array[:, :, 3].astype(np.float32)
        source_alpha_pixels = int(np.count_nonzero(alpha > 0))
        if source_alpha_pixels == 0:
            raise GarmentTpsWarpError("의상 알파 픽셀이 0개입니다.")

        transformer = cv2.createThinPlateSplineShapeTransformer(
            float(request.regularization)
        )
        matches = [
            cv2.DMatch(index, index, 0)
            for index in range(source_points.shape[0])
        ]
        transformer.estimateTransformation(
            target_points.reshape(1, -1, 2),
            source_points.reshape(1, -1, 2),
            matches,
        )

        rgb = rgba_array[:, :, :3].astype(np.float32)
        premultiplied_rgb = rgb * (alpha[:, :, None] / 255.0)
        warped_premultiplied_rgb = transformer.warpImage(
            premultiplied_rgb,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        warped_alpha = transformer.warpImage(
            alpha,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    except cv2.error as error:
        raise GarmentTpsWarpError(
            f"OpenCV TPS 워핑에 실패했습니다: {error}"
        ) from error
    finally:
        garment_rgba.close()

    warped_alpha = np.clip(warped_alpha, 0.0, 255.0)
    output_alpha = np.rint(warped_alpha).astype(np.uint8)
    restored_rgb = np.zeros_like(warped_premultiplied_rgb)
    visible_pixels = output_alpha > 0
    restored_rgb[visible_pixels] = (
        warped_premultiplied_rgb[visible_pixels]
        * 255.0
        / warped_alpha[visible_pixels, None]
    )
    output_rgba = np.dstack(
        (
            np.rint(np.clip(restored_rgb, 0.0, 255.0)),
            output_alpha,
        )
    ).astype(np.uint8)
    warped_alpha_pixels = int(np.count_nonzero(output_alpha > 0))

    return GarmentTpsWarpResult(
        warped_rgba=Image.fromarray(output_rgba),
        warped_alpha_mask=Image.fromarray(output_alpha),
        source_point_count=int(source_points.shape[0]),
        source_alpha_pixels=source_alpha_pixels,
        warped_alpha_pixels=warped_alpha_pixels,
        alpha_pixel_change=warped_alpha_pixels - source_alpha_pixels,
    )


def _validate_canvas_size(canvas_size: tuple[int, int]) -> tuple[int, int]:
    if len(canvas_size) != 2:
        raise GarmentTpsWarpError("TPS 캔버스 크기는 가로·세로 2개 값이어야 합니다.")
    width, height = canvas_size
    if width < 1 or height < 1:
        raise GarmentTpsWarpError(
            f"TPS 캔버스 크기는 1px 이상이어야 합니다: {canvas_size}"
        )
    return int(width), int(height)


def _validate_regularization(regularization: float) -> None:
    if not np.isfinite(regularization) or regularization < 0:
        raise GarmentTpsWarpError(
            "TPS 정규화 값은 0 이상의 유한한 수여야 합니다."
        )


def _validate_control_points(
    points_xy: np.ndarray,
    point_name: str,
    width: int,
    height: int,
) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[1:] != (2,):
        raise GarmentTpsWarpError(
            f"{point_name} TPS 대응점은 N×2 XY 좌표여야 합니다: "
            f"shape={points.shape}"
        )
    if points.shape[0] < 5:
        raise GarmentTpsWarpError(
            f"{point_name} TPS 대응점은 최소 5개가 필요합니다: "
            f"{points.shape[0]}개"
        )
    if not np.isfinite(points).all():
        raise GarmentTpsWarpError(
            f"{point_name} TPS 대응점에 NaN 또는 무한대가 있습니다."
        )
    if np.unique(points, axis=0).shape[0] != points.shape[0]:
        raise GarmentTpsWarpError(
            f"{point_name} TPS 대응점에 중복 좌표가 있습니다."
        )
    outside = (
        (points[:, 0] < 0)
        | (points[:, 0] > width - 1)
        | (points[:, 1] < 0)
        | (points[:, 1] > height - 1)
    )
    outside_count = int(np.count_nonzero(outside))
    if outside_count:
        raise GarmentTpsWarpError(
            f"{point_name} TPS 대응점 {outside_count}개가 "
            f"{width}×{height}px 캔버스 밖에 있습니다."
        )
    hull_area = float(cv2.contourArea(cv2.convexHull(points)))
    if hull_area <= 0.0:
        raise GarmentTpsWarpError(
            f"{point_name} TPS 대응점이 한 직선 위에 있어 변형할 수 없습니다."
        )
    return points
