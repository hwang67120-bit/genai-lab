"""의상 전용 다중 크롭 분석. 좌표는 관찰 영역이며 부품 위치/겹침의 정답이 아니다."""
from dataclasses import replace
from math import ceil, isfinite
from time import perf_counter

import numpy as np
from PIL import Image

from genai_lab.clothing_reference import ClothingDesignTagCandidate
from genai_lab.reference_features import tag_scope
from genai_lab.reference_tag_policy import normalize_tag, excluded_garment_tag

# 표시/전달 분류용 어휘다. 옷의 존재/부재, 성별별 의상 규칙으로 사용하지 않는다.
CONSTRUCTION = frozenset(("buttons", "zipper", "pockets", "collar", "cuffs",
                          "frills", "lace", "ribbon", "bow", "embroidery"))
SHAPE = frozenset(("sleeves", "sleeveless"))
SURFACE = frozenset(("pinstripes", "plaid", "polka dot", "striped",
                     "denim", "leather", "knit", "satin", "silk"))
STYLE = frozenset(("formal", "suit", "uniform"))
LOCAL_DETAIL_GROUPS = frozenset(("construction", "shape", "surface"))


def detail_group(tag):
    """모르는 태그를 구성품으로 추정하지 않는다."""
    name = normalize_tag(tag)
    if excluded_garment_tag(name):
        return "excluded"
    if name in STYLE:
        return "style_context"
    for group, vocabulary in (("construction", CONSTRUCTION), ("shape", SHAPE),
                              ("surface", SURFACE)):
        if any(name == word or name.endswith(" " + word) for word in vocabulary):
            return group
    return "component_candidate" if tag_scope(name) == "garment" else "unresolved"


def detail_view_boxes(image, maximum_views=3, minimum_side=64):
    """알파 외곽 + 긴 축의 겹치는 확대 창. 상체/하체라는 의미를 부여하지 않는다."""
    if type(maximum_views) is not int or not 1 <= maximum_views <= 5:
        raise ValueError("의상 분석 뷰 수는 전체 포함 1~5 정수여야 합니다.")
    if type(minimum_side) is not int or minimum_side < 16:
        raise ValueError("확대 영역 최소 변 길이는 16 이상의 정수여야 합니다.")
    with image.convert("RGBA") as rgba, rgba.getchannel("A") as alpha:
        bbox = alpha.getbbox()
    if bbox is None:
        raise ValueError("분석할 의상 픽셀이 없습니다.")
    left, top, right, bottom = bbox
    width, height = right-left, bottom-top
    boxes = [bbox]
    if maximum_views == 1 or min(width, height) < minimum_side:
        return boxes
    vertical = height >= width
    length = height if vertical else width
    span = max(minimum_side, ceil(length * .65))
    if span >= length:
        return boxes
    for offset in np.linspace(0, length-span, maximum_views-1).round().astype(int):
        box = ((left, top+int(offset), right, top+int(offset)+span) if vertical
               else (left+int(offset), top, left+int(offset)+span, bottom))
        if box not in boxes:
            with image.crop(box) as crop, crop.convert('RGBA') as rgba, rgba.getchannel('A') as alpha:
                if alpha.getbbox() is not None:
                    boxes.append(box)
    return boxes


def measured_palette(image):
    """최대 512변 표본의 알파 가중 RGB 구간. 색 이름/부품 색으로 추론하지 않는다."""
    with image.convert("RGBA") as rgba:
        rgba.thumbnail((512, 512), Image.Resampling.NEAREST)
        pixels = np.asarray(rgba).reshape(-1, 4).astype(np.float64)
    visible = pixels[:, 3] > 0
    pixels = pixels[visible]
    if not len(pixels):
        return {"colors": [], "scope": "visible_pixels"}
    weights = pixels[:, 3] / 255.
    bins = (pixels[:, :3] // 32).astype(int)
    keys = bins[:, 0]*64 + bins[:, 1]*8 + bins[:, 2]
    counts = np.bincount(keys, weights=weights, minlength=512)
    order = np.argsort(-counts, kind="stable")[:5]
    colors = []
    for key in order:
        if counts[key] <= 0:
            continue
        selected = keys == key
        rgb = np.average(pixels[selected, :3], weights=weights[selected], axis=0)
        colors.append({"rgb": rgb.round().astype(int).tolist(),
                       "sample_fraction": float(counts[key]/weights.sum())})
    return {"colors": colors, "scope": "alpha_visible_pixels_including_opaque_background",
            "method": "alpha_weighted_RGB_bins_32_on_nearest_512_sample",
            "semantic_color_assignment": "unresolved",
            "unreported_fraction": float(1-sum(c["sample_fraction"] for c in colors))}


def analyze_garment_details(image, settings, session_factory, *, check_running=None):
    """한 WD 세션을 재사용한다. 확대 관찰은 보조 증거이며 독립적인 확률이 아니다."""
    started = perf_counter()
    timeout = settings.detail_timeout_seconds
    if not isfinite(timeout) or timeout <= 0:
        raise ValueError("의상 분석 시간 제한은 양의 유한 값이어야 합니다.")
    def check():
        if check_running is not None:
            check_running()
        if perf_counter()-started > timeout:
            raise TimeoutError("의상 세부 분석 시간 제한을 초과했습니다.")
    check()
    boxes = detail_view_boxes(image, settings.detail_maximum_views)
    observations, view_records, scores_by_tag = [], [], {}
    with session_factory(settings) as session:
        check()
        for index, box in enumerate(boxes):
            check()
            tick = perf_counter()
            with image.crop(box) as crop:
                result = session.analyze(crop)
            check()
            observations.append(result)
            view_records.append({"id": index, "kind": "whole" if index == 0 else "local",
                                 "source_bbox": list(box), "seconds": perf_counter()-tick})
            raw = result.raw_general_scores or tuple((t.tag_name, t.score) for t in result.tag_candidates)
            for name, score in raw:
                if not isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("세부 분석 점수가 0~1 범위를 벗어났습니다.")
                values = scores_by_tag.setdefault(name, [0.] * len(boxes))
                values[index] = float(score)
    base = observations[0]
    evidence = []
    for name, scores in scores_by_tag.items():
        if max(scores) < settings.score_threshold:
            continue
        group = detail_group(name)
        if group in ("excluded", "unresolved"):
            continue
        evidence.append({"tag": name, "group": group, "view_scores": scores,
                         "whole_score": scores[0], "max_score": max(scores),
                         "status": "model_candidate", "location": "unresolved"})
    evidence.sort(key=lambda entry: (-entry["max_score"], entry["tag"]))
    base_names = {tag.tag_name for tag in base.tag_candidates}
    additions = [e for e in evidence if e["tag"] not in base_names
                 and e["group"] in LOCAL_DETAIL_GROUPS][:12]
    optional_names = [e["tag"] for e in additions]
    candidates = base.tag_candidates + tuple(ClothingDesignTagCandidate(
        e["tag"], normalize_tag(e["tag"]), e["max_score"]) for e in additions)
    check()
    with image.crop(boxes[0]) as crop:
        palette = measured_palette(crop)
    check()
    report = {
        "version": "garment_multiview_v1", "views": view_records,
        "evidence": evidence, "optional_detail_tags": optional_names, "palette": palette,
        "view_count": len(boxes), "model_load_count": 1,
        "absence_status": "unresolved", "layer_order": "unresolved",
        "gender_inference": False, "semantic_accuracy_verified": False,
        "warnings": [
            "점수는 모델의 태그 점수이며 정확도나 부품 존재 확률이 아닙니다.",
            "확대 창 좌표는 부품 위치/분할 마스크가 아닙니다.",
            "미검출은 부재가 아닙니다. 치마와 바지의 겹침 여부는 자동 확정하지 않습니다.",
            "추가 세부 태그는 승인 후 토큰 여유가 있을 때만 전달합니다.",
            "시간 제한은 호출 사이에서 확인하며 진행 중인 ONNX 호출을 강제 종료하지 않습니다.",
        ]}
    return replace(base, input_width=image.width, input_height=image.height,
                   tag_candidates=candidates, garment_detail_report=report,
                   elapsed_seconds=perf_counter()-started)


def detail_review_text(report):
    if not report:
        return ""
    names = {"component_candidate": "구성품 후보", "construction": "깃·여밈·장식",
             "shape": "소매·형태", "surface": "소재·무늬", "style_context": "스타일 맥락"}
    lines = [f"세부 분석: 전체 및 확대 {report['view_count']}개 영역 (확정 아님)"]
    for group, title in names.items():
        tags = [normalize_tag(e["tag"]) for e in report["evidence"] if e["group"] == group]
        if tags:
            lines.append(title + ": " + ", ".join(tags))
    colors = report.get("palette", {}).get("colors", [])
    if colors:
        lines.append("관찰 RGB 색 표본 (그림자·배경 포함 가능): " + ", ".join(
            "#" + "".join(f"{value:02X}" for value in color["rgb"]) for color in colors))
    lines.append("겹침 순서·미검출 의상 부재: 판단 보류. 확대에서만 나온 구성품은 자동 추가하지 않습니다.")
    return "\n".join(lines)
