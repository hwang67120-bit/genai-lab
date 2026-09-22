"""부위 색 분포 측정. 명암·오드아이·배색 패턴을 자동 확정하지 않는다."""
from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image
from genai_lab.reference_regions import read_binary_mask


@dataclass
class ColorEvidence:
    palette_rgb: list
    area_ratios: list
    label_map: np.ndarray
    pixel_count: int
    pattern: str = "unresolved"

    def record(self):
        return {
            "palette_rgb": self.palette_rgb, "area_ratios": self.area_ratios,
            "pixel_count": self.pixel_count, "pattern": self.pattern,
            "method": "bounded Lab clustering; not semantic colors",
            "warnings": [
                "색 군집은 디자인 색·그림자·반사광을 구분하지 않습니다.",
                "희소한 포인트 색은 표본 학습에서 누락될 수 있습니다.",
                "이 측정값은 색상 이름·성별·오드아이를 자동 확정하지 않습니다.",
            ],
        }


def analyze_part_colors(source, mask, *, max_clusters=5, max_fit_pixels=20000,
                        excluded_mask=None, cancelled=lambda: False):
    if not isinstance(max_clusters, int) or not 1 <= max_clusters <= 8:
        raise ValueError("색 군집 수는 1~8 정수여야 합니다.")
    if not isinstance(max_fit_pixels, int) or not max_clusters <= max_fit_pixels <= 20000:
        raise ValueError("색 분석 표본 수 설정이 잘못됐습니다.")
    selected = read_binary_mask(mask, source.size)
    if excluded_mask is not None:
        selected &= ~read_binary_mask(excluded_mask, source.size)
    with source.convert("RGBA") as rgba:
        # 투명 픽셀의 숨겨진 RGB를 색상으로 세지 않는다.
        selected &= np.asarray(rgba)[..., 3] > 0
    labels_map = np.full(selected.shape, -1, dtype=np.int16)
    if not selected.any():
        return ColorEvidence([], [], labels_map, 0)
    with source.convert("RGB") as rgb:
        pixels = np.asarray(rgb, dtype=np.float32) / 255.
    samples = cv2.cvtColor(pixels, cv2.COLOR_RGB2Lab)[selected]
    rng = np.random.default_rng(0)
    fit = samples[rng.choice(len(samples), min(len(samples), max_fit_pixels), replace=False)]
    unique = np.unique(fit, axis=0)
    k = min(max_clusters, len(unique))
    centers = unique[rng.choice(len(unique), k, replace=False)].copy()
    # 결정론적인 유한 반복. 전역 난수 상태/새 모델 의존성을 추가하지 않는다.
    for _ in range(20):
        if cancelled():
            raise InterruptedError("색 분포 분석을 취소했습니다.")
        labels = ((fit[:, None] - centers[None]) ** 2).sum(axis=2).argmin(axis=1)
        updated = np.array([
            fit[labels == i].mean(axis=0) if np.any(labels == i) else centers[i]
            for i in range(k)
        ], dtype=np.float32)
        if np.allclose(updated, centers, atol=1e-3):
            centers = updated
            break
        centers = updated
    labels = np.empty(len(samples), dtype=np.int16)
    # 큰 원본에도 거리 행렬은 최대 20,000 픽셀 단위로 제한한다.
    for start in range(0, len(samples), 20000):
        if cancelled():
            raise InterruptedError("색 분포 분석을 취소했습니다.")
        chunk = samples[start:start + 20000]
        labels[start:start + len(chunk)] = (
            (chunk[:, None] - centers[None]) ** 2).sum(axis=2).argmin(axis=1)
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts)
    order = order[counts[order] > 0]
    remap = np.full(k, -1, dtype=np.int16)
    remap[order] = np.arange(len(order))
    labels_map[selected] = remap[labels]
    rgb_centers = cv2.cvtColor(centers[None], cv2.COLOR_Lab2RGB)[0]
    palette = np.rint(np.clip(rgb_centers, 0, 1) * 255).astype(np.uint8)
    return ColorEvidence(
        [palette[i].tolist() for i in order],
        [float(counts[i] / len(samples)) for i in order],
        labels_map, int(selected.sum()))


def color_distribution_distance(source_evidence, candidate_evidence):
    """두 Lab 팔레트의 대칭 가중 최근접 거리. 색 위치나 무늬 유사도는 아니다."""
    if source_evidence is None or candidate_evidence is None:
        return None

    def palette_and_weights(evidence):
        palette = np.asarray(evidence.palette_rgb, dtype=np.float32)
        weights = np.asarray(evidence.area_ratios, dtype=np.float64)
        if not evidence.pixel_count or not len(palette):
            return None
        if (palette.ndim != 2 or palette.shape[1:] != (3,)
                or weights.shape != (len(palette),)
                or not np.isfinite(palette).all()
                or np.any((palette < 0) | (palette > 255))
                or not np.isfinite(weights).all() or np.any(weights < 0)
                or not np.isclose(weights.sum(), 1.0, atol=1e-5)):
            raise ValueError("색상 분포 비교 입력 형식이 잘못됐습니다.")
        lab = cv2.cvtColor(palette[None] / 255.0, cv2.COLOR_RGB2Lab)[0]
        return lab, weights

    source = palette_and_weights(source_evidence)
    candidate = palette_and_weights(candidate_evidence)
    if source is None or candidate is None:
        return None
    source_lab, source_weights = source
    candidate_lab, candidate_weights = candidate
    distances = np.linalg.norm(
        source_lab[:, None, :] - candidate_lab[None, :, :], axis=2)
    source_to_candidate = float(np.dot(source_weights, distances.min(axis=1)))
    candidate_to_source = float(np.dot(candidate_weights, distances.min(axis=0)))
    return (source_to_candidate + candidate_to_source) / 2.0


def save_color_evidence(evidence, directory, name):
    """분석 지도/RGB 수치는 확인용이다. 승인할 텍스트 설명은 별도 모듈에서 만든다."""
    import json
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(
        json.dumps(evidence.record(), ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(directory / f"{name}_labels.npy", evidence.label_map, allow_pickle=False)
    preview = np.full((*evidence.label_map.shape, 3), 255, dtype=np.uint8)
    for index, rgb in enumerate(evidence.palette_rgb):
        preview[evidence.label_map == index] = rgb
    with Image.fromarray(preview) as image:
        image.save(directory / f"{name}_color_map.png")
