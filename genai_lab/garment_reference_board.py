"""Pack the approved garment condition for the active reference pipeline."""

import math
import cv2
import numpy as np
from PIL import Image, ImageOps

def prepare_garment_board(image, size=512):
    rgba = image.convert('RGBA')
    alpha = np.asarray(rgba.getchannel('A'))
    support = (alpha > 0).astype(np.uint8)
    if not support.any():
        rgba.close()
        raise ValueError('승인 의상 참조의 알파 영역이 비어 있습니다.')
    count, labels, stats, _ = cv2.connectedComponentsWithStats(support, connectivity=8)
    ids = sorted(range(1, count), key=lambda i: int(stats[i, 4]), reverse=True)
    # Small detached details belong to the nearest major region; never drop pixels.
    anchors = [i for i in ids if stats[i, 4] >= stats[ids[0], 4] * .02][:8]
    groups = {i: [i] for i in anchors}
    def distance(i, j):
        x, y, w, h = stats[i, :4]
        a, b, c, d = stats[j, :4]
        return max(int(a-x-w), int(x-a-c), 0)**2 + max(int(b-y-h), int(y-b-d), 0)**2
    for i in ids:
        if i not in groups:
            groups[min(anchors, key=lambda j: distance(i, j))].append(i)
    regions = []
    for members in groups.values():
        boxes = stats[members, :4]
        x, y = boxes[:, :2].min(axis=0)
        right, bottom = (boxes[:, :2] + boxes[:, 2:4]).max(axis=0)
        regions.append(((int(x), int(y), int(right), int(bottom)), members))
    regions.sort(key=lambda item: (item[0][1], item[0][0]))
    board = Image.new('RGB', (size, size), 'white')
    columns = math.ceil(math.sqrt(len(regions)))
    rows = math.ceil(len(regions) / columns)
    cell_w, cell_h = size // columns, size // rows
    gutter = 8
    for index, (box, members) in enumerate(regions):
        x, y, right, bottom = box
        pixels = np.array(rgba.crop(box))
        selected = np.isin(labels[y:bottom, x:right], members)
        pixels[:, :, 3] = np.where(selected, pixels[:, :, 3], 0)
        crop = Image.fromarray(pixels)
        white = Image.new('RGBA', crop.size, 'white')
        white.alpha_composite(crop)
        fitted = ImageOps.contain(white.convert('RGB'),
            (cell_w - 2*gutter, cell_h - 2*gutter), Image.Resampling.LANCZOS)
        board.paste(fitted, ((index % columns)*cell_w + (cell_w-fitted.width)//2,
                            (index // columns)*cell_h + (cell_h-fitted.height)//2))
        crop.close()
        white.close()
        fitted.close()
    rgba.close()
    return board, {'source_size': image.size, 'source_alpha_pixels': int(support.sum()),
                   'regions': [box for box, _ in regions], 'board_size': board.size,
                   'opaque_input': bool((alpha == 255).all()), 'components': count - 1}
