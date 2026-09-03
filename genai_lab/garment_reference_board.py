"""승인 의상 RGBA를 IP-Adapter용 구성요소 참조 보드로 정규화한다."""

from dataclasses import dataclass
import math

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class GarmentReferenceBoardSettings:
    """연결요소 필터와 흰 배경 참조 보드 배치 계약."""

    board_size: int = 1024
    outer_padding: int = 32
    cell_padding: int = 16
    alpha_threshold: int = 128
    minimum_component_pixels: int = 16
    maximum_components: int = 8
    primary_height_ratio: float = 0.62


@dataclass(frozen=True)
class GarmentReferenceBoardComponent:
    """원본 의상 조각과 참조 보드 배치 좌표의 대응 기록."""

    component_index: int
    source_bbox_xywh: tuple[int, int, int, int]
    board_bbox_xywh: tuple[int, int, int, int]
    foreground_pixel_count: int


@dataclass(frozen=True)
class GarmentReferenceBoard:
    """IP-Adapter에 전달할 흰 배경 RGB 보드와 감사 수치."""

    image: Image.Image
    components: tuple[GarmentReferenceBoardComponent, ...]
    source_component_count: int
    retained_component_count: int
    discarded_component_count: int
    source_foreground_pixel_count: int
    retained_foreground_pixel_count: int
    board_occupied_pixel_count: int
    layout_method: str = "largest_primary_grid_v1"

    def close(self) -> None:
        self.image.close()


class GarmentReferenceBoardError(ValueError):
    """승인 알파에서 신뢰할 수 있는 의상 참조 보드를 만들 수 없는 오류."""


def create_garment_reference_board(
    garment_image: Image.Image,
    settings: GarmentReferenceBoardSettings | None = None,
) -> GarmentReferenceBoard:
    """OpenCV 연결요소를 분리해 여백이 작은 정사각형 참조 보드를 만든다."""
    resolved = settings or GarmentReferenceBoardSettings()
    validate_garment_reference_board_settings(resolved)
    if "A" not in garment_image.getbands():
        raise GarmentReferenceBoardError(
            "의상 참조 보드는 승인 추출본의 알파 채널이 필요합니다."
        )

    rgba = garment_image.convert("RGBA")
    try:
        rgba_array = np.asarray(rgba, dtype=np.uint8).copy()
    finally:
        rgba.close()
    alpha = rgba_array[:, :, 3]
    binary = (alpha >= resolved.alpha_threshold).astype(np.uint8)
    source_foreground = int(np.count_nonzero(binary))
    if source_foreground == 0:
        raise GarmentReferenceBoardError(
            "알파 임계값 이상인 승인 의상 픽셀이 0개입니다."
        )

    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    source_component_count = label_count - 1
    retained_labels = [
        label
        for label in range(1, label_count)
        if int(stats[label, cv2.CC_STAT_AREA])
        >= resolved.minimum_component_pixels
    ]
    retained_labels.sort(
        key=lambda label: (
            -int(stats[label, cv2.CC_STAT_AREA]),
            int(stats[label, cv2.CC_STAT_TOP]),
            int(stats[label, cv2.CC_STAT_LEFT]),
        )
    )
    retained_labels = retained_labels[:resolved.maximum_components]
    if not retained_labels:
        raise GarmentReferenceBoardError(
            "최소 면적을 만족한 승인 의상 조각이 0개입니다."
        )

    cells = _create_layout_cells(len(retained_labels), resolved)
    board = Image.new(
        "RGB", (resolved.board_size, resolved.board_size), (255, 255, 255)
    )
    occupied = np.zeros(
        (resolved.board_size, resolved.board_size), dtype=np.uint8
    )
    component_records: list[GarmentReferenceBoardComponent] = []
    retained_foreground = 0
    try:
        for component_index, (label, cell) in enumerate(
            zip(retained_labels, cells, strict=True)
        ):
            left = int(stats[label, cv2.CC_STAT_LEFT])
            top = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            component_rgba = rgba_array[
                top:top + height, left:left + width
            ].copy()
            component_labels = labels[top:top + height, left:left + width]
            component_rgba[:, :, 3] = np.where(
                component_labels == label,
                component_rgba[:, :, 3],
                0,
            ).astype(np.uint8)
            cell_left, cell_top, cell_right, cell_bottom = cell
            available_width = cell_right - cell_left
            available_height = cell_bottom - cell_top
            scale = min(
                available_width / max(1, width),
                available_height / max(1, height),
            )
            target_width = max(1, int(round(width * scale)))
            target_height = max(1, int(round(height * scale)))
            resized_rgba = _resize_premultiplied_rgba(
                component_rgba, (target_width, target_height)
            )
            paste_left = cell_left + (available_width - target_width) // 2
            paste_top = cell_top + (available_height - target_height) // 2
            overlay = Image.fromarray(resized_rgba, mode="RGBA")
            overlay_rgb = overlay.convert("RGB")
            overlay_alpha = overlay.getchannel("A")
            try:
                board.paste(
                    overlay_rgb,
                    (paste_left, paste_top),
                    mask=overlay_alpha,
                )
            finally:
                overlay_rgb.close()
                overlay_alpha.close()
                overlay.close()
            resized_alpha = resized_rgba[:, :, 3]
            occupied[
                paste_top:paste_top + target_height,
                paste_left:paste_left + target_width,
            ] = np.maximum(
                occupied[
                    paste_top:paste_top + target_height,
                    paste_left:paste_left + target_width,
                ],
                resized_alpha,
            )
            retained_foreground += area
            component_records.append(
                GarmentReferenceBoardComponent(
                    component_index=component_index,
                    source_bbox_xywh=(left, top, width, height),
                    board_bbox_xywh=(
                        paste_left, paste_top, target_width, target_height
                    ),
                    foreground_pixel_count=area,
                )
            )
        return GarmentReferenceBoard(
            image=board,
            components=tuple(component_records),
            source_component_count=source_component_count,
            retained_component_count=len(component_records),
            discarded_component_count=(
                source_component_count - len(component_records)
            ),
            source_foreground_pixel_count=source_foreground,
            retained_foreground_pixel_count=retained_foreground,
            board_occupied_pixel_count=int(np.count_nonzero(occupied)),
        )
    except Exception:
        board.close()
        raise


def _create_layout_cells(
    component_count: int,
    settings: GarmentReferenceBoardSettings,
) -> tuple[tuple[int, int, int, int], ...]:
    padding = settings.outer_padding
    inner_left = padding
    inner_top = padding
    inner_right = settings.board_size - padding
    inner_bottom = settings.board_size - padding
    cell_padding = settings.cell_padding
    if component_count == 1:
        return ((
            inner_left + cell_padding,
            inner_top + cell_padding,
            inner_right - cell_padding,
            inner_bottom - cell_padding,
        ),)

    split_y = int(round(settings.board_size * settings.primary_height_ratio))
    cells = [(
        inner_left + cell_padding,
        inner_top + cell_padding,
        inner_right - cell_padding,
        split_y - cell_padding,
    )]
    remaining = component_count - 1
    columns = min(3, remaining)
    rows = int(math.ceil(remaining / columns))
    grid_top = split_y
    grid_width = inner_right - inner_left
    grid_height = inner_bottom - grid_top
    for index in range(remaining):
        column = index % columns
        row = index // columns
        left = inner_left + (grid_width * column) // columns
        right = inner_left + (grid_width * (column + 1)) // columns
        top = grid_top + (grid_height * row) // rows
        bottom = grid_top + (grid_height * (row + 1)) // rows
        cells.append((
            left + cell_padding,
            top + cell_padding,
            right - cell_padding,
            bottom - cell_padding,
        ))
    return tuple(cells)


def _resize_premultiplied_rgba(
    rgba: np.ndarray,
    target_size: tuple[int, int],
) -> np.ndarray:
    target_width, target_height = target_size
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    premultiplied = rgba[:, :, :3].astype(np.float32) * alpha[:, :, None]
    resized_alpha = cv2.resize(
        alpha,
        (target_width, target_height),
        interpolation=cv2.INTER_LANCZOS4,
    )
    resized_premultiplied = cv2.resize(
        premultiplied,
        (target_width, target_height),
        interpolation=cv2.INTER_LANCZOS4,
    )
    resized_alpha = np.clip(resized_alpha, 0.0, 1.0)
    rgb = np.full_like(resized_premultiplied, 255.0)
    visible = resized_alpha > 1e-6
    rgb[visible] = (
        resized_premultiplied[visible] / resized_alpha[visible, None]
    )
    result = np.empty((target_height, target_width, 4), dtype=np.uint8)
    result[:, :, :3] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    result[:, :, 3] = np.clip(
        np.rint(resized_alpha * 255.0), 0, 255
    ).astype(np.uint8)
    return result


def validate_garment_reference_board_settings(
    settings: GarmentReferenceBoardSettings,
) -> None:
    if settings.board_size < 64 or settings.board_size % 8 != 0:
        raise GarmentReferenceBoardError(
            "참조 보드 크기는 64 이상인 8의 배수여야 합니다."
        )
    if settings.outer_padding < 0 or settings.cell_padding < 0:
        raise GarmentReferenceBoardError("참조 보드 여백은 0 이상이어야 합니다.")
    if (
        settings.outer_padding * 2 + settings.cell_padding * 2
        >= settings.board_size
    ):
        raise GarmentReferenceBoardError("참조 보드 여백이 캔버스보다 큽니다.")
    if not 1 <= settings.alpha_threshold <= 255:
        raise GarmentReferenceBoardError("알파 임계값은 1~255여야 합니다.")
    if settings.minimum_component_pixels < 1:
        raise GarmentReferenceBoardError("최소 조각 면적은 1px 이상이어야 합니다.")
    if not 1 <= settings.maximum_components <= 8:
        raise GarmentReferenceBoardError("최대 의상 조각 수는 1~8개여야 합니다.")
    if not 0.40 <= settings.primary_height_ratio <= 0.80:
        raise GarmentReferenceBoardError(
            "주 의상 영역 높이 비율은 0.40~0.80이어야 합니다."
        )
