"""Advisory eye-color analysis: raw WD scores + two square head crops.

No softmax, no forced gold/yellow mapping, no automatic heterochromia diagnosis.
Two agreeing views of the same image are correlated evidence, not independent proof.
"""
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import json
import math
import tempfile
import numpy as np
from PIL import Image

from genai_lab.reference_tag_policy import normalize_tag

EYE_COLORS = frozenset(
    f"{color} eyes" for color in (
        "black", "blue", "brown", "gray", "green", "orange", "pink", "purple",
        "red", "white", "yellow", "gold", "golden", "amber", "aqua", "silver")
)
MULTICOLOR = frozenset(("heterochromia", "multicolored eyes"))


def is_eye_color_tag(tag):
    return normalize_tag(tag) in EYE_COLORS


def eye_scores(result):
    return sorted(((tag, float(score)) for tag, score in result.raw_general_scores
                   if is_eye_color_tag(tag)), key=lambda pair: (-pair[1], pair[0]))


def multicolor_scores(result):
    return [(tag, float(score)) for tag, score in result.raw_general_scores
            if normalize_tag(tag) in MULTICOLOR]


def crop_square_head(source, mask, *, pad_ratio, min_short_side=32):
    if not np.isfinite(pad_ratio) or not 0 <= pad_ratio <= .25:
        raise ValueError("얼굴 크롭 여백은 0~0.25 범위여야 합니다.")
    if min_short_side < 1:
        raise ValueError("최소 얼굴 영역 크기가 올바르지 않습니다.")
    if mask is None:
        return None, {"reason": "missing_head_mask"}
    if mask.mode != "L" or mask.size != source.size:
        raise ValueError("눈색 분석 마스크의 좌표/형식이 다릅니다.")
    values = np.asarray(mask)
    if not np.isin(values, (0, 255)).all():
        raise ValueError("눈색 분석 구조 마스크는 0/255여야 합니다.")
    box = mask.getbbox()
    if box is None:
        return None, {"reason": "empty_head_mask"}
    left, top, right, bottom = box
    if min(right - left, bottom - top) < min_short_side:
        return None, {"reason": "head_roi_too_small", "mask_bbox": box}
    side = math.ceil(max(right - left, bottom - top) * (1 + 2 * pad_ratio))
    # Ceil side and floor origin preserve odd-width/height ROI endpoints.
    x = math.floor((left + right - side) / 2)
    y = math.floor((top + bottom - side) / 2)
    clipped = (max(0, x), max(0, y),
               min(source.width, x + side), min(source.height, y + side))
    canvas = Image.new("RGB", (side, side), "white")
    with source.convert("RGBA") as rgba:
        with rgba.crop(clipped) as patch:
            with Image.new("RGBA", patch.size, "white") as backing:
                backing.alpha_composite(patch)
                with backing.convert("RGB") as rgb:
                    canvas.paste(rgb, (clipped[0] - x, clipped[1] - y))
    return canvas, {
        "reason": None, "mask_bbox": box, "requested_box": (x, y, x + side, y + side),
        "source_intersection": clipped, "side": side, "pad_ratio": pad_ratio,
        "region_type": "face_hair_bbox_not_iris",
    }


def decide_eye_color(results, *, threshold=.35, minimum_margin=.10):
    if len(results) != 2:
        raise ValueError("서로 다른 얼굴 크롭 결과 2개가 필요합니다.")
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in (threshold, minimum_margin)):
        raise ValueError("눈색 후보 기준은 유한한 0~1 값이어야 합니다.")
    evidence, reasons, choices = [], [], []
    for index, result in enumerate(results):
        ranked = eye_scores(result)
        multi = multicolor_scores(result)
        if any(not np.isfinite(score) or not 0 <= score <= 1
               for _, score in ranked + multi):
            raise ValueError("눈색 원본 점수 형식 오류")
        entry = {"view": index, "eye_scores": ranked, "multicolor_scores": multi}
        evidence.append(entry)
        if any(score >= threshold for _, score in multi):
            reasons.append(f"view_{index}:multicolor_signal")
        elif not ranked:
            reasons.append(f"view_{index}:no_supported_eye_labels")
        else:
            tag, top = ranked[0]
            gap = top - (ranked[1][1] if len(ranked) > 1 else 0.)
            entry.update(top_tag=tag, gap=gap)
            if top < threshold:
                reasons.append(f"view_{index}:low_score")
            elif gap < minimum_margin:
                reasons.append(f"view_{index}:ambiguous_colors")
            else:
                choices.append(tag)
    if reasons:
        return {"status": "unresolved", "suggested_tag": None, "reasons": reasons,
                "views": evidence}
    if choices[0] != choices[1]:
        return {"status": "unresolved", "suggested_tag": None,
                "reasons": ["crop_disagreement"], "views": evidence}
    return {"status": "suggested", "suggested_tag": choices[0],
            "reasons": ["consistent_across_crops"], "views": evidence}


def analyze_with_eye_review(session, source, mask, *, debug_dir=None,
                            threshold=.35, minimum_margin=.10,
                            cancelled=lambda: False):
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in (threshold, minimum_margin)):
        raise ValueError("눈색 후보 기준은 유한한 0~1 값이어야 합니다.")
    directory = Path(debug_dir) if debug_dir is not None else Path(
        tempfile.mkdtemp(prefix="genai-eye-review-"))
    directory.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    report = {
        "status": "failed", "suggested_tag": None, "reasons": [], "views": [],
        "threshold": threshold, "minimum_margin": minimum_margin,
        "thresholds_calibrated": False, "debug_dir": str(directory),
        "note": "WD raw scores; not eye-color accuracy. Same-image crops are correlated.",
    }
    try:
        if cancelled():
            raise InterruptedError("캐릭터 눈색 분석을 취소했습니다.")
        full = session.analyze(source)
        report.update(model_id=full.model_id, execution_provider=full.execution_provider,
                      model_input_size=full.model_input_size,
                      source_size=source.size, inference_count=1)
        report["full_eye_scores"] = eye_scores(full)
        report["full_multicolor_scores"] = multicolor_scores(full)
        report["displayed_full_eye_tags"] = [
            tag.tag_name for tag in full.tag_candidates if is_eye_color_tag(tag.tag_name)]
        report["full_eye_display_filter"] = [
            {"tag": tag, "raw_score": score,
             "state": ("visible" if tag in report["displayed_full_eye_tags"] else
                       "below_display_threshold" if score < full.score_threshold else
                       "outside_display_top_n")}
            for tag, score in report["full_eye_scores"]]
        results = []
        for index, padding in enumerate((0., .15)):
            if cancelled():
                raise InterruptedError("캐릭터 눈색 분석을 취소했습니다.")
            crop, geometry = crop_square_head(source, mask, pad_ratio=padding)
            if crop is None:
                report.update(status="unresolved", reasons=[geometry["reason"]])
                report["crop_geometry"] = [geometry]
                break
            try:
                crop.save(directory / f"head_input_{index}.png")
                results.append(session.analyze(crop))
                report["inference_count"] += 1
                # Keep evidence even if a later inference is cancelled or fails.
                report["views"].append({
                    "view": index, "eye_scores": eye_scores(results[-1]),
                    "multicolor_scores": multicolor_scores(results[-1])})
                report.setdefault("crop_geometry", []).append(geometry)
            finally:
                crop.close()
        if len(results) == 2:
            report.update(decide_eye_color(results, threshold=threshold,
                                          minimum_margin=minimum_margin))
            if any(score >= threshold for _, score in multicolor_scores(full)):
                report.update(status="unresolved", suggested_tag=None,
                              reasons=["full_view_multicolor_signal"])
        if cancelled():
            raise InterruptedError("캐릭터 눈색 분석을 취소했습니다.")
        report["elapsed_seconds"] = perf_counter() - started
        return replace(full, eye_color_report=report,
                       elapsed_seconds=report["elapsed_seconds"])
    except Exception as exc:
        report.update(status="failed", reasons=[f"{type(exc).__name__}: {exc}"])
        raise
    finally:
        report["elapsed_seconds"] = perf_counter() - started
        (directory / "eye_color_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def review_eye_candidates(report):
    """Separate eye choices from global top-N. Low scores are visible, not auto selected."""
    scores = {}
    for tag, score in report.get("full_eye_scores", []):
        scores[tag] = max(scores.get(tag, 0.), score)
    for view in report.get("views", []):
        for tag, score in view["eye_scores"]:
            scores[tag] = max(scores.get(tag, 0.), score)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
