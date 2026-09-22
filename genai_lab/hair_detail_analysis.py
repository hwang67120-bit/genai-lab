"""Refined-hair multi-view analysis with isolated texture crops."""

from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from time import perf_counter
import tempfile

import numpy as np
from PIL import Image

from genai_lab.clothing_reference import ClothingDesignTagCandidate
from genai_lab.part_color_analysis import analyze_part_colors
from genai_lab.reference_regions import read_binary_mask
from genai_lab.reference_tag_policy import normalize_tag


HAIR_LENGTH_TAGS = (
    "very short hair", "short hair", "medium hair", "long hair",
    "very long hair", "absurdly long hair",
)
HAIR_TEXTURE_TAGS = frozenset((
    "straight hair", "wavy hair", "curly hair", "messy hair",
    "spiked hair", "flipped hair", "floating hair", "wet hair",
))
HAIR_COLOR_TAGS = frozenset((
    "multicolored hair", "gradient hair", "two-tone hair",
    "split-color hair", "streaked hair", "colored inner hair",
    "rainbow hair", "alternate hair color",
))
HAIR_COLOR_WORDS = frozenset((
    "aqua", "black", "blonde", "blue", "brown", "dark blue", "green",
    "grey", "light blue", "light brown", "orange", "pink", "purple",
    "red", "white",
))
HAIR_ARRANGEMENT_MARKERS = (
    "ponytail", "twintail", "braid", "hair bun", "drill", "dreadlock",
)
HAIR_STRUCTURE_TAGS = frozenset((
    "ahoge", "antenna hair", "asymmetrical hair", "big hair", "bob cut",
    "hair down", "hair intakes", "hair pulled back", "hair slicked back",
    "hair up", "inverted bob", "mohawk", "pointy hair",
))


@dataclass(frozen=True)
class HairDetailAnalysisSettings:
    enabled: bool = True
    minimum_region_pixels: int = 128
    maximum_detail_tags: int = 6
    maximum_tags_per_group: int = 2
    score_threshold: float = 0.35
    timeout_seconds: float = 120.0
    front_width_ratio: float = 1.0
    front_bottom_ratio: float = 0.55
    side_inner_ratio: float = 0.35
    side_bottom_ratio: float = 0.35
    rear_start_ratio: float = 0.75
    length_boundary_margin_pixels: int = 2
    very_short_max_face_heights: float = 0.0
    short_max_face_heights: float = 0.4
    medium_max_face_heights: float = 1.0
    long_max_face_heights: float = 2.5
    very_long_max_face_heights: float = 4.0


def resolve_hair_detail_settings(config):
    raw = config.get("reference_analysis", {}).get("hair_detail_analysis", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("헤어 상세 분석 설정은 객체여야 합니다.")
    allowed = HairDetailAnalysisSettings.__dataclass_fields__
    unknown = set(raw) - set(allowed)
    if unknown:
        raise ValueError(f"알 수 없는 헤어 상세 분석 설정: {sorted(unknown)}")
    settings = HairDetailAnalysisSettings(**raw)
    if type(settings.minimum_region_pixels) is not int or settings.minimum_region_pixels < 16:
        raise ValueError("헤어 관찰 영역 최소 픽셀 수는 16 이상의 정수여야 합니다.")
    if type(settings.maximum_detail_tags) is not int or not 1 <= settings.maximum_detail_tags <= 30:
        raise ValueError("헤어 상세 태그 수는 1~30 정수여야 합니다.")
    if (type(settings.maximum_tags_per_group) is not int
            or not 1 <= settings.maximum_tags_per_group <= 5):
        raise ValueError("헤어 그룹별 태그 수는 1~5 정수여야 합니다.")
    if (type(settings.length_boundary_margin_pixels) is not int
            or settings.length_boundary_margin_pixels < 0):
        raise ValueError("헤어 길이 경계 여백은 0 이상의 정수여야 합니다.")
    ratios = (
        settings.score_threshold, settings.timeout_seconds,
        settings.front_width_ratio,
        settings.front_bottom_ratio, settings.side_inner_ratio,
        settings.side_bottom_ratio, settings.rear_start_ratio,
    )
    if not all(isfinite(value) and value > 0 for value in ratios):
        raise ValueError("헤어 상세 분석 비율과 점수 기준은 양의 유한 값이어야 합니다.")
    if not 0 < settings.score_threshold <= 1:
        raise ValueError("헤어 상세 분석 점수 기준은 0 초과 1 이하여야 합니다.")
    if not 0 < settings.side_inner_ratio <= .5:
        raise ValueError("옆머리 안쪽 비율은 0 초과 0.5 이하여야 합니다.")
    if not 0 < settings.rear_start_ratio <= 2:
        raise ValueError("후면 실루엣 시작 비율은 0 초과 2 이하여야 합니다.")
    length_limits = (
        settings.very_short_max_face_heights,
        settings.short_max_face_heights,
        settings.medium_max_face_heights,
        settings.long_max_face_heights,
        settings.very_long_max_face_heights,
    )
    if (not all(isfinite(value) for value in length_limits)
            or tuple(sorted(length_limits)) != length_limits
            or len(set(length_limits)) != len(length_limits)):
        raise ValueError("헤어 길이 상대 기준은 오름차순 유한 값이어야 합니다.")
    return settings


def hair_detail_group(tag):
    """Bounded routing only; this does not prove that a hairstyle exists."""
    name = normalize_tag(tag)
    if name in HAIR_LENGTH_TAGS:
        return "length"
    if name == "bangs" or name.endswith(" bangs") or name in {
            "hair between eyes", "hair over one eye", "hair over eyes"}:
        return "front"
    if ("sidelock" in name or name in {
            "sideburns", "short hair with long locks"}):
        return "side"
    if name in HAIR_TEXTURE_TAGS:
        return "texture"
    if (name in HAIR_COLOR_TAGS or
            (name.endswith(" hair") and name[:-5] in HAIR_COLOR_WORDS)):
        return "color"
    if any(marker in name for marker in HAIR_ARRANGEMENT_MARKERS):
        return "arrangement"
    if name in HAIR_STRUCTURE_TAGS:
        return "structure"
    return "unresolved"


def hair_length_geometry(hair_mask, face_mask, settings):
    """Measure continuous hair extent relative to the detected face height."""
    hair = read_binary_mask(hair_mask, hair_mask.size)
    face = read_binary_mask(face_mask, hair_mask.size)
    hair_rows, _ = np.nonzero(hair)
    face_rows, _ = np.nonzero(face)
    if not len(hair_rows) or not len(face_rows):
        return {"status": "unresolved", "reason": "missing_hair_or_face"}
    face_top, face_bottom = int(face_rows.min()), int(face_rows.max()) + 1
    face_height = max(1, face_bottom - face_top)
    hair_bottom = int(hair_rows.max()) + 1
    margin = settings.length_boundary_margin_pixels
    touches_bottom = hair_bottom >= hair.shape[0] - margin
    extension = (hair_bottom - face_bottom) / face_height
    limits = (
        (settings.very_short_max_face_heights, "very short hair"),
        (settings.short_max_face_heights, "short hair"),
        (settings.medium_max_face_heights, "medium hair"),
        (settings.long_max_face_heights, "long hair"),
        (settings.very_long_max_face_heights, "very long hair"),
    )
    band = next((name for limit, name in limits if extension <= limit),
                "absurdly long hair")
    ordered = tuple(name for _, name in limits) + ("absurdly long hair",)
    index = ordered.index(band)
    compatible = ordered[max(0, index - 1):min(len(ordered), index + 2)]
    return {
        "status": "unresolved" if touches_bottom else "measured",
        "reason": "hair_touches_bottom_boundary" if touches_bottom else None,
        "bottom_extension_face_heights": extension,
        "suggested_length_band": None if touches_bottom else band,
        "compatible_length_tags": () if touches_bottom else compatible,
        "hair_bottom": hair_bottom,
        "face_bottom": face_bottom,
        "face_height": face_height,
        "touches_bottom_boundary": touches_bottom,
    }


def evidence_views(group, available_views):
    """Keep local observations from deciding unrelated global attributes."""
    available = set(available_views)
    if group == "front":
        return ("front",) if "front" in available else ()
    if group == "side":
        return tuple(name for name in ("left_side", "right_side")
                     if name in available)
    if group in {"length", "color", "texture", "arrangement", "structure"}:
        return ("whole",) if "whole" in available else ()
    return ()


def hair_prompt_delivery_report(approved_tags, prompt, analysis_report=None):
    """Record which approved hair tags reached the exact generation prompt.

    Analysis candidates and delivered prompt terms are kept separate so a
    diagnostic suggestion cannot be mistaken for a generation condition.
    """
    approved = tuple(dict.fromkeys(
        normalize_tag(tag) for tag in approved_tags
        if hair_detail_group(tag) != "unresolved"
    ))
    prompt_terms = {
        normalize_tag(term) for term in str(prompt).split(",") if term.strip()
    }
    delivered = tuple(tag for tag in approved if tag in prompt_terms)
    missing = tuple(tag for tag in approved if tag not in prompt_terms)

    report = analysis_report if isinstance(analysis_report, dict) else {}
    evidence = report.get("evidence", ())
    analyzed = tuple(dict.fromkeys(
        normalize_tag(item.get("tag", ""))
        for item in evidence if isinstance(item, dict)
        and hair_detail_group(item.get("tag", "")) != "unresolved"
    ))
    optional = tuple(dict.fromkeys(
        normalize_tag(tag) for tag in report.get("optional_detail_tags", ())
        if hair_detail_group(tag) != "unresolved"
    ))
    not_delivered = tuple(tag for tag in analyzed if tag not in delivered)
    return {
        "approved_hair_tags": approved,
        "delivered_hair_tags": delivered,
        "missing_approved_hair_tags": missing,
        "analyzed_hair_candidates": analyzed,
        "analyzed_but_not_delivered": not_delivered,
        "optional_hair_candidates": optional,
        "visual_reference_scope": "identity(face_hair)",
        "separate_hair_adapter": False,
    }


def hair_analysis_log_detail(report):
    """Format the complete hair-analysis decision without implying delivery."""
    if not isinstance(report, dict):
        return "분석 보고서 없음"
    geometry = report.get("length_geometry", {})
    evidence = []
    for item in report.get("evidence", ()):
        if not isinstance(item, dict):
            continue
        evidence.append({
            "tag": normalize_tag(item.get("tag", "")),
            "group": item.get("group", "unresolved"),
            "score": round(float(item.get("max_score", 0.0)), 3),
            "status": item.get("status", "unknown"),
            "views": list(item.get("evidence_views", ())),
            "eligible": bool(item.get("eligible_for_prompt", False)),
        })
    return (
        f"길이 측정={geometry.get('status', 'unavailable')}, "
        f"길이 비율={geometry.get('bottom_extension_face_heights')}, "
        f"호환 길이={list(geometry.get('compatible_length_tags', ()))}, "
        f"분석 후보={evidence}, "
        f"승인 화면 추가 후보={list(report.get('optional_detail_tags', ()))}, "
        "현재 단계 프롬프트 전달=아님"
    )


def hair_observation_masks(hair_mask, face_mask, settings):
    """Build face-anchored observation regions, not semantic segmentations."""
    size = hair_mask.size
    hair = read_binary_mask(hair_mask, size)
    face = read_binary_mask(face_mask, size)
    if not hair.any():
        return {}, "empty_hair_mask"
    rows, columns = np.nonzero(face)
    if not len(rows):
        return {"whole": hair}, "missing_face_anchor"
    left, right = int(columns.min()), int(columns.max()) + 1
    top, bottom = int(rows.min()), int(rows.max()) + 1
    width, height = right - left, bottom - top
    center_x = (left + right) / 2
    front_half = width * settings.front_width_ratio / 2
    yy, xx = np.indices(hair.shape)
    front_window = (
        (xx >= center_x - front_half) & (xx < center_x + front_half)
        & (yy <= top + height * settings.front_bottom_ratio)
    )
    side_bottom = bottom + height * settings.side_bottom_ratio
    left_window = (
        (xx < left + width * settings.side_inner_ratio)
        & (yy >= top) & (yy < side_bottom)
    )
    right_window = (
        (xx >= right - width * settings.side_inner_ratio)
        & (yy >= top) & (yy < side_bottom)
    )
    # A frontal image cannot prove semantic rear hair. This lower outer
    # silhouette is retained only as a zoomed observation candidate.
    rear_window = yy >= top + height * settings.rear_start_ratio
    masks = {"whole": hair}
    for name, selected in (
            ("front", hair & front_window),
            ("left_side", hair & left_window),
            ("right_side", hair & right_window),
            ("rear_silhouette_candidate", hair & rear_window)):
        if int(selected.sum()) >= settings.minimum_region_pixels:
            masks[name] = selected
    return masks, None


def masked_texture(source, selected):
    alpha = Image.fromarray(selected.astype(np.uint8) * 255)
    try:
        box = alpha.getbbox()
        if box is None:
            raise ValueError("헤어 텍스처 영역이 비었습니다.")
        with source.convert("RGBA") as rgba:
            rgba.putalpha(alpha)
            return rgba.crop(box)
    finally:
        alpha.close()


def analyze_hair_details(session, image, hair_mask, face_mask, base_result,
                         settings, *, cancelled=lambda: False):
    """Reuse one WD session and add only bounded hair-detail candidates."""
    started = perf_counter()
    def check():
        if cancelled():
            raise InterruptedError("헤어 상세 분석을 취소했습니다.")
        if perf_counter() - started > settings.timeout_seconds:
            raise TimeoutError("헤어 상세 분석 제한 시간을 초과했습니다.")
    if not settings.enabled:
        return replace(base_result, hair_detail_report={
            "version": "hair_multiaxis_v3", "status": "disabled"})
    masks, reason = hair_observation_masks(hair_mask, face_mask, settings)
    length_geometry = hair_length_geometry(hair_mask, face_mask, settings)
    directory = Path(tempfile.mkdtemp(prefix="genai-hair-detail-"))
    view_records, scores_by_tag, colors = [], {}, {}
    for index, (name, selected) in enumerate(masks.items()):
        check()
        texture = masked_texture(image, selected)
        mask_image = Image.fromarray(selected.astype(np.uint8) * 255)
        try:
            texture_path = directory / f"{name}_texture.png"
            mask_path = directory / f"{name}_mask.png"
            texture.save(texture_path)
            mask_image.save(mask_path)
            tick = perf_counter()
            observation = session.analyze(texture)
            check()
            elapsed = perf_counter() - tick
            colors[name] = analyze_part_colors(image, mask_image, cancelled=cancelled).record()
        finally:
            texture.close()
            mask_image.close()
        raw = observation.raw_general_scores or tuple(
            (tag.tag_name, tag.score) for tag in observation.tag_candidates)
        for tag, score in raw:
            if not isfinite(score) or not 0 <= score <= 1:
                raise ValueError("헤어 상세 태그 점수가 0~1 범위를 벗어났습니다.")
            scores_by_tag.setdefault(tag, {})[name] = float(score)
        view_records.append({
            "id": index, "name": name,
            "pixel_count": int(selected.sum()),
            "texture_path": str(texture_path), "mask_path": str(mask_path),
            "seconds": elapsed,
        })
    evidence = []
    for tag, view_scores in scores_by_tag.items():
        group = hair_detail_group(tag)
        relevant_views = evidence_views(group, view_scores)
        relevant_scores = {
            view: view_scores[view] for view in relevant_views
            if view in view_scores
        }
        if (group == "unresolved" or not relevant_scores
                or max(relevant_scores.values()) < settings.score_threshold):
            continue
        normalized = normalize_tag(tag)
        geometry_supported = (
            group != "length" or
            (length_geometry.get("status") == "measured"
             and normalized in length_geometry.get("compatible_length_tags", ()))
        )
        evidence.append({
            "tag": tag, "group": group, "view_scores": view_scores,
            "evidence_views": relevant_views,
            "max_score": max(relevant_scores.values()),
            "status": ("model_candidate" if geometry_supported
                       else "geometry_conflict"),
            "semantic_location": group,
            "eligible_for_prompt": geometry_supported,
        })
    geometry_label = length_geometry.get("suggested_length_band")
    geometry_tag = (
        geometry_label.replace(" ", "_") if geometry_label else None
    )
    if (length_geometry.get("status") == "measured" and geometry_tag
            and not any(normalize_tag(item["tag"]) == geometry_tag
                        for item in evidence)):
        evidence.append({
            "tag": geometry_tag,
            "group": "length",
            "view_scores": {},
            "evidence_views": ("mask_geometry",),
            "max_score": 1.0,
            "status": "mask_geometry_candidate",
            "semantic_location": "length",
            # Mask geometry narrows the range but cannot prove the semantic
            # length label by itself. A WD observation must agree before the
            # tag can become a user-approval candidate.
            "eligible_for_prompt": False,
            "score_is_probability": False,
        })
    evidence.sort(key=lambda item: (-item["max_score"], item["tag"]))
    # Full-image length tags are replaced by mask-verified length evidence.
    # Other full-image character attributes remain available for review.
    base_candidates = tuple(
        tag for tag in base_result.tag_candidates
        if hair_detail_group(tag.tag_name) != "length"
    )
    existing = {normalize_tag(tag.tag_name) for tag in base_candidates}
    additions, group_counts = [], {}
    for item in evidence:
        if not item["eligible_for_prompt"]:
            continue
        if normalize_tag(item["tag"]) in existing:
            continue
        count = group_counts.get(item["group"], 0)
        if count >= settings.maximum_tags_per_group:
            continue
        additions.append(item)
        group_counts[item["group"]] = count + 1
        if len(additions) >= settings.maximum_detail_tags:
            break
    candidates = base_candidates + tuple(
        ClothingDesignTagCandidate(item["tag"], normalize_tag(item["tag"]), item["max_score"])
        for item in additions)
    report = {
        "version": "hair_multiaxis_v3",
        "status": "analyzed" if masks else "unresolved",
        "reason": reason,
        "views": view_records,
        "view_count": len(view_records),
        "evidence": evidence,
        "length_geometry": length_geometry,
        "optional_detail_tags": [item["tag"] for item in additions],
        "region_colors": colors,
        "rear_silhouette_observation": {
            "status": (
                "observed_candidate"
                if "rear_silhouette_candidate" in masks else "unresolved"
            ),
            "semantic_identity_confirmed": False,
        },
        "debug_dir": str(directory),
        "model_load_count": 0,
        "session_reused": True,
        "semantic_accuracy_verified": False,
        "warnings": [
            "앞머리·옆머리·후면 실루엣 후보 마스크는 얼굴 좌표 기반 관찰 영역이며 의미 분할의 정답이 아닙니다.",
            "정면 이미지의 후면 실루엣 후보는 뒤쪽 머리로 확정하거나 생성 조건에 자동 주입하지 않습니다.",
            "마스크 길이만으로 길이 태그를 자동 제안하지 않으며, whole 뷰의 AI 태그와 좌표 범위가 일치할 때만 승인 후보가 됩니다.",
            "WD 점수는 태그별 독립 점수이며 정확도 확률이 아닙니다.",
            "확대 영역에서 나온 태그는 사용자 승인 전 생성 조건이 아닙니다.",
        ],
        "seconds": perf_counter() - started,
    }
    return replace(base_result, tag_candidates=candidates, hair_detail_report=report)


def hair_detail_review_text(report):
    if not report:
        return ""
    if report.get("status") != "analyzed":
        return f"헤어 상세 분석: {report.get('status')} ({report.get('reason')})"
    groups = {
        "length": "전체 길이 후보",
        "front": "앞머리 후보",
        "side": "옆머리 후보",
        "arrangement": "묶음배열 후보",
        "structure": "전체 형태 후보",
        "texture": "질감 후보",
        "color": "색상 구조 후보",
    }
    lines = [f"헤어 상세 분석: 정밀 마스크 기반 {report['view_count']}개 영역 (확정 아님)"]
    rear = report.get("rear_silhouette_observation", {})
    lines.append(
        "후면 실루엣 후보: "
        + ("관찰됨 (의미 확정 아님)"
           if rear.get("status") == "observed_candidate" else "판별 보류")
    )
    for group, label in groups.items():
        tags = [
            normalize_tag(item["tag"])
            + (" (마스크 길이 불일치)"
               if item.get("status") == "geometry_conflict" else "")
            for item in report["evidence"] if item["group"] == group
        ]
        if tags:
            lines.append(label + ": " + ", ".join(tags))
    lines.append("좌표는 관찰 범위이며, 선택한 태그만 생성 조건에 들어갑니다.")
    return "\n".join(lines)
