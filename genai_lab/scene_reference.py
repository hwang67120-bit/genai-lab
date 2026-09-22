"""Validated scene references; no RGB initialization, inpainting or hidden-part reconstruction.

The analyzer is a project integration boundary, not an implemented semantic model.
All masks and target landmarks use the final output canvas coordinates.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
import logging
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


class AnalysisRejected(ValueError):
    """Bounded reanalysis may resolve this; do not silently invent coordinates."""


@dataclass
class PartReference:
    name: str
    rgb: Image.Image
    region: Image.Image
    scale: float

    def close(self):
        self.rgb.close()
        self.region.close()


@dataclass
class SceneAnalysis:
    source: Image.Image
    garment_rgba: Image.Image
    keep_mask: Image.Image
    old_garment_mask: Image.Image
    front_mask: Image.Image
    placement_mask: Image.Image
    src_points: np.ndarray
    dst_points: np.ndarray
    parts: list[PartReference]
    unresolved: tuple[str, ...] = ()
    point_labels: tuple[str, ...] = ()
    # Producer must record source/target coordinate conversion and evidence.
    evidence: dict = field(default_factory=dict)

    def close(self):
        for image in (self.source, self.garment_rgba, self.keep_mask,
                      self.old_garment_mask, self.front_mask, self.placement_mask):
            image.close()
        for part in self.parts:
            part.close()


class AutomaticSceneAnalyzer(Protocol):
    def analyze(self, source: Image.Image, garment: Image.Image, *,
                attempt: int) -> SceneAnalysis:
        """Return OWNED copies. Distinguish garment-only from worn images.
        Keep per-part evidence; no-detection is not proof of absence.
        Depth alone must not establish occlusion; report unresolved cases.
        """
        ...

    def close(self) -> None:
        """Release models/tensors owned by this analyzer before diffusion."""
        ...


def binary_mask(image, size, name):
    if not isinstance(image, Image.Image) or image.mode != "L" or image.size != size:
        raise AnalysisRejected(f"{name}: expected output-sized L mask")
    values = np.asarray(image)
    if not np.isin(values, (0, 255)).all():
        raise AnalysisRejected(f"{name}: structural masks must be 0/255")
    return values == 255


def validate_points(points, size, name):
    points = np.asarray(points, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 2 or not 3 <= len(points) <= 64
            or not np.isfinite(points).all()):
        raise AnalysisRejected(f"{name}: invalid landmarks")
    width, height = size
    if ((points < 0).any() or (points[:, 0] >= width).any()
            or (points[:, 1] >= height).any()):
        raise AnalysisRejected(f"{name}: landmarks outside image")
    if len(np.unique(points, axis=0)) != len(points):
        raise AnalysisRejected(f"{name}: duplicate landmarks")
    normalized = points / np.array([width, height])
    if np.linalg.matrix_rank(normalized - normalized.mean(axis=0)) < 2:
        raise AnalysisRejected(f"{name}: collinear landmarks")
    return points


def validate_scene(scene, maximum_parts=8):
    if not isinstance(scene, SceneAnalysis):
        raise AnalysisRejected("analyzer must return SceneAnalysis")
    if scene.unresolved:
        raise AnalysisRejected("; ".join(scene.unresolved))
    if scene.source.mode != "RGB" or min(scene.source.size) < 8:
        raise AnalysisRejected("source must be a valid RGB canvas")
    if scene.garment_rgba.mode != "RGBA" or scene.garment_rgba.getchannel("A").getbbox() is None:
        raise AnalysisRejected("garment must contain visible RGBA pixels")
    if not 1 <= len(scene.parts) <= maximum_parts:
        raise AnalysisRejected("character reference count exceeds analysis budget")
    size = scene.source.size
    keep = binary_mask(scene.keep_mask, size, "keep")
    old = binary_mask(scene.old_garment_mask, size, "old_garment")
    front = binary_mask(scene.front_mask, size, "front")
    placement = binary_mask(scene.placement_mask, size, "placement")
    if not keep.any() or not placement.any():
        raise AnalysisRejected("empty keep/placement region")
    if np.any(keep & old) or np.any(front & ~keep):
        raise AnalysisRejected("contradictory keep/old/front masks")
    occupied = np.zeros_like(keep)
    names = set()
    for part in scene.parts:
        region = binary_mask(part.region, size, part.name)
        if (not part.name or part.name in names or part.name == "garment"
                or part.rgb.mode != "RGB" or min(part.rgb.size) <= 0):
            raise AnalysisRejected("invalid or duplicate part reference")
        if not region.any() or np.any(region & ~keep) or np.any(occupied & region):
            raise AnalysisRejected(f"{part.name}: invalid/overlapping influence region")
        if not np.isfinite(part.scale) or not 0 <= part.scale <= 1:
            raise AnalysisRejected(f"{part.name}: invalid influence scale")
        names.add(part.name)
        occupied |= region
    if "identity" not in names:
        raise AnalysisRejected("face/hair identity reference is required")
    src = validate_points(scene.src_points, scene.garment_rgba.size, "source")
    dst = validate_points(scene.dst_points, size, "target")
    if src.shape != dst.shape:
        raise AnalysisRejected("landmark counts differ")
    if (len(scene.point_labels) != len(src)
            or any(not name for name in scene.point_labels)
            or len(set(scene.point_labels)) != len(src)):
        raise AnalysisRejected("matching semantic landmark labels are required")
    if not isinstance(scene.evidence, dict) or not scene.evidence:
        raise AnalysisRejected("analysis evidence is required")


def analyze_with_retry(analyzer, source, garment, attempts=2, cancelled=lambda: False,
                       status=lambda message: None):
    if not 1 <= attempts <= 3:
        raise ValueError("analysis attempts must be 1..3")
    last_error = None
    for attempt in range(attempts):
        if cancelled():
            raise InterruptedError("scene analysis cancelled")
        scene = None
        accepted = False
        try:
            status(f"장면 자동 분석 {attempt + 1}/{attempts}")
            scene = analyzer.analyze(source, garment, attempt=attempt)
            validate_scene(scene)
            if scene.source.size != source.size or scene.source.tobytes() != source.tobytes():
                raise AnalysisRejected("analyzer changed source pixels/coordinates")
            if cancelled():
                raise InterruptedError("scene analysis cancelled")
            accepted = True
            return scene
        except AnalysisRejected as exc:
            last_error = exc
            logger.warning("scene analysis attempt %d: %s", attempt + 1, exc)
        finally:
            if isinstance(scene, SceneAnalysis) and not accepted:
                scene.close()
    raise AnalysisRejected(f"automatic scene analysis unresolved: {last_error}")


def warp_garment(garment, src_points, dst_points, canvas_size, *,
                 chunk_rows=64, maximum_pixels=4_194_304, cancelled=lambda: False):
    import cv2
    from scipy.interpolate import RBFInterpolator

    if garment.mode != "RGBA":
        raise AnalysisRejected("garment must be RGBA")
    width, height = canvas_size
    if min(width, height) < 2 or width * height > maximum_pixels or chunk_rows < 1:
        raise AnalysisRejected("warp canvas exceeds resource budget")
    src = validate_points(src_points, garment.size, "source")
    dst = validate_points(dst_points, canvas_size, "target")
    if src.shape != dst.shape:
        raise AnalysisRejected("landmark counts differ")
    # Normalization improves conditioning without mixing the two coordinate frames.
    target_scale = np.array([width, height], dtype=np.float64)
    try:
        inverse = RBFInterpolator(dst / target_scale, src,
                                  kernel="thin_plate_spline", smoothing=0.0)
    except (np.linalg.LinAlgError, ValueError) as exc:
        raise AnalysisRejected("unstable TPS landmarks") from exc
    map_x = np.empty((height, width), dtype=np.float32)
    map_y = np.empty_like(map_x)
    for top in range(0, height, chunk_rows):
        if cancelled():
            raise InterruptedError("garment alignment cancelled")
        bottom = min(top + chunk_rows, height)
        yy, xx = np.mgrid[top:bottom, :width]
        mapped = inverse(np.column_stack([xx.ravel(), yy.ravel()]) / target_scale)
        if not np.isfinite(mapped).all():
            raise AnalysisRejected("non-finite TPS mapping")
        map_x[top:bottom] = mapped[:, 0].reshape(bottom - top, width)
        map_y[top:bottom] = mapped[:, 1].reshape(bottom - top, width)
    rgba = np.asarray(garment, dtype=np.float32) / 255
    rgba[..., :3] *= rgba[..., 3:4]
    warped = cv2.remap(rgba, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    visible = warped[..., 3] > .01
    if not visible.any():
        raise AnalysisRejected("aligned garment is empty")
    dx_dy, dx_dx = np.gradient(map_x)
    dy_dy, dy_dx = np.gradient(map_y)
    if np.any((dx_dx * dy_dy - dx_dy * dy_dx)[visible] <= 0):
        raise AnalysisRejected("TPS fold or inverted coordinates")
    alpha = warped[..., 3:4]
    rgb = np.divide(warped[..., :3], alpha, out=np.zeros_like(warped[..., :3]),
                    where=alpha > 1e-6)
    output = np.concatenate([rgb, alpha], axis=2)
    return Image.fromarray(np.round(np.clip(output, 0, 1) * 255).astype(np.uint8))


class LineartExtractor:
    def __init__(self, cache_dir, *, output_white_lines):
        if not isinstance(output_white_lines, bool):
            raise ValueError("extractor polarity must be explicitly configured")
        from controlnet_aux import LineartDetector
        self.output_white_lines = output_white_lines
        self.detector = LineartDetector.from_pretrained(
            "lllyasviel/Annotators", cache_dir=str(cache_dir)).to("cpu")

    def extract(self, image):
        import torch
        with torch.inference_mode():
            result = self.detector(image.convert("RGB"), detect_resolution=512,
                                   image_resolution=512).convert("L")
        result = result.resize(image.size, Image.Resampling.BILINEAR)
        if not self.output_white_lines:
            result = ImageOps.invert(result)
        return np.array(result)

    def close(self):
        self.detector = None


def flatten_white(rgba):
    return Image.alpha_composite(
        Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba).convert("RGB")


def as_mask(values):
    return Image.fromarray(np.asarray(values, dtype=np.uint8) * 255)


@dataclass
class SceneCondition:
    # Only black-line-on-white structural hints; never original RGB initialization.
    hint: Image.Image
    references: list[PartReference]
    debug_dir: Path

    def close(self):
        self.hint.close()
        for ref in self.references:
            ref.close()


def compose_scene(scene, warped, extractor, debug_dir, garment_scale=.45):
    validate_scene(scene)
    if warped.mode != "RGBA" or warped.size != scene.source.size:
        raise AnalysisRejected("aligned garment frame mismatch")
    if not np.isfinite(garment_scale) or not 0 <= garment_scale <= 1:
        raise AnalysisRejected("invalid garment influence scale")
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    size = scene.source.size
    keep = binary_mask(scene.keep_mask, size, "keep")
    old = binary_mask(scene.old_garment_mask, size, "old")
    front = binary_mask(scene.front_mask, size, "front")
    placement = binary_mask(scene.placement_mask, size, "placement")
    coverage = np.asarray(warped.getchannel("A")) >= 128
    if np.any(coverage & ~placement):
        raise AnalysisRejected("aligned garment outside permitted placement")
    visible = coverage & ~front
    if not visible.any():
        raise AnalysisRejected("new garment has no visible region")
    character_lines = np.asarray(extractor.extract(scene.source))
    garment_lines = np.asarray(extractor.extract(flatten_white(warped)))
    for lines in (character_lines, garment_lines):
        if lines.shape != keep.shape or lines.dtype != np.uint8:
            raise AnalysisRejected("extractor must return output-sized uint8 white lines")
    # Do not silently accept a blank garment hint while character lines remain valid.
    if not np.any(garment_lines[visible] > 16):
        raise AnalysisRejected("garment lineart is empty")
    merged = np.where(keep & ~old, character_lines, 0).astype(np.uint8)
    merged[coverage] = 0
    merged[visible] = garment_lines[visible]
    merged[front] = character_lines[front]
    for part in scene.parts:
        if part.name == "identity" and np.any(binary_mask(part.region, size, part.name) & visible):
            raise AnalysisRejected("garment would cover face/hair identity region")
    refs = []
    for part in scene.parts:
        region = binary_mask(part.region, size, part.name) & ~visible
        if region.any():
            refs.append(PartReference(part.name, part.rgb.copy(), as_mask(region), part.scale))
    box = scene.garment_rgba.getchannel("A").getbbox()
    refs.append(PartReference("garment", flatten_white(scene.garment_rgba.crop(box)),
                              as_mask(visible), garment_scale))
    hint = ImageOps.invert(Image.fromarray(merged)).convert("RGB")
    result = SceneCondition(hint, refs, debug_dir)
    try:
        hint.save(debug_dir / "scene_lineart.png")
        warped.save(debug_dir / "aligned_garment.png")
        as_mask(visible).save(debug_dir / "garment_region.png")
        for index, part in enumerate(refs):
            part.rgb.save(debug_dir / f"reference_{index:02d}.png")
            part.region.save(debug_dir / f"region_{index:02d}.png")
        import json
        (debug_dir / "analysis.json").write_text(json.dumps({
            "point_labels": scene.point_labels, "src_points": np.asarray(scene.src_points).tolist(),
            "dst_points": np.asarray(scene.dst_points).tolist(), "evidence": scene.evidence,
            "references": [{"name": p.name, "scale": p.scale} for p in refs],
            "note": "structural validation is not image-quality or species proof",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        result.close()
        raise
    return result
