"""원본 귀·꼬리 자동 검출. 생성 좌표를 임의로 만들지 않는다."""
from pathlib import Path
from time import perf_counter
import inspect
import json
import tempfile
from types import SimpleNamespace
import numpy as np
from PIL import Image
from genai_lab.regional_reference import DetectedPart, crop_reference, valid_part_name
from genai_lab.part_candidate_policy import EXPERIMENTAL_OVERLAP_POLICY, OverlapProposalPolicy
from genai_lab.part_detection_candidates import detect_part_candidates
from genai_lab.part_cross_review import PartCrossReviewer
from genai_lab.accessory_analysis import (
    AccessoryObservationPolicy,
    analyze_accessory_observations,
)
from genai_lab.part_spatial_diagnostics import (
    PartSpatialDiagnosticPolicy,
    locate_head_roi,
)


class TransformersPartsBackend:
    """기존 HF 캐시만 사용하며 CPU에서 분석한다. 새 다운로드는 하지 않는다."""

    def __init__(self, detector_id, sam_id, cache_dir):
        self.detector_id, self.sam_id, self.cache_dir = detector_id, sam_id, cache_dir
        self.detector = self.processor = self.sam = self.sam_processor = None

    def _load(self):
        if self.detector is not None:
            return
        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Sam2Processor
        from genai_lab.clothing_reference import load_sam2_image_model
        try:
            dino_path = snapshot_download(self.detector_id, cache_dir=self.cache_dir, local_files_only=True)
            sam_path = snapshot_download(self.sam_id, cache_dir=self.cache_dir, local_files_only=True)
            self.processor = AutoProcessor.from_pretrained(dino_path, local_files_only=True)
            self.sam_processor = Sam2Processor.from_pretrained(sam_path, local_files_only=True)
            self.sam, _, _ = load_sam2_image_model(
                SimpleNamespace(model_id=sam_path, cache_dir=self.cache_dir), "cpu")
            self.sam.eval()
            self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
                dino_path, local_files_only=True).to("cpu").eval()
        except Exception as error:
            self.close()
            raise RuntimeError("귀·꼬리 검출 모델 로드 실패. 기존 DINO/SAM2 캐시와 의존성을 확인하세요.") from error

    def detect(self, image, query, threshold):
        import torch
        self._load()
        with torch.inference_mode():
            inputs = self.processor(images=image, text=query, return_tensors="pt")
            output = self.detector(**inputs)
            post = self.processor.post_process_grounded_object_detection
            key = "box_threshold" if "box_threshold" in inspect.signature(post).parameters else "threshold"
            result = post(output, inputs.input_ids, text_threshold=.25,
                          target_sizes=[(image.height, image.width)], **{key: threshold})[0]
        return result["boxes"].detach().cpu().numpy(), result["scores"].detach().cpu().numpy()

    def detect_labeled(self, image, query_groups, threshold):
        """Detect all accessory phrases in one GroundingDINO forward pass."""
        import torch
        self._load()
        phrases = []
        phrase_groups = {}
        for group, query in query_groups:
            for phrase in (item.strip() for item in query.split(".")):
                if phrase:
                    normalized = phrase.casefold()
                    phrases.append(phrase)
                    phrase_groups[normalized] = group
        with torch.inference_mode():
            inputs = self.processor(
                images=image,
                text=phrases,
                return_tensors="pt",
            )
            output = self.detector(**inputs)
            post = self.processor.post_process_grounded_object_detection
            key = (
                "box_threshold"
                if "box_threshold" in inspect.signature(post).parameters
                else "threshold"
            )
            result = post(
                output,
                inputs.input_ids,
                text_threshold=.25,
                target_sizes=[(image.height, image.width)],
                **{key: threshold},
            )[0]
        mapped = []
        for value in result.get("text_labels", result.get("labels", ())):
            normalized = str(value).strip().casefold().rstrip(".")
            group = phrase_groups.get(normalized)
            if group is None:
                matches = [
                    (phrase, name)
                    for phrase, name in phrase_groups.items()
                    if phrase in normalized or normalized in phrase
                ]
                group = max(matches, default=(normalized, "unmapped"))[1]
            mapped.append(group)
        return (
            result["boxes"].detach().cpu().numpy(),
            result["scores"].detach().cpu().numpy(),
            tuple(mapped),
        )

    def segment(self, image, boxes):
        import torch
        with torch.inference_mode():
            inputs = self.sam_processor(images=image, input_boxes=[boxes.tolist()], return_tensors="pt")
            output = self.sam(**inputs, multimask_output=False)
            masks = self.sam_processor.post_process_masks(
                output.pred_masks.detach().cpu(), inputs["original_sizes"].detach().cpu(),
                binarize=True)[0]
        return masks.numpy(), output.iou_scores.detach().cpu().numpy()[0]

    def close(self):
        self.detector = self.processor = self.sam = self.sam_processor = None


DEFAULT_PART_QUERIES = (
    {"name": "human_ears", "query": "human ears. person ears.", "hair_context": False},
    {"name": "animal_ears", "query": "animal ears.", "hair_context": True},
    {"name": "tail", "query": "tail.", "hair_context": False},
)

DEFAULT_MAXIMUM_FOREGROUND_AREA_RATIOS = {
    "human_ears": 0.12,
    "animal_ears": 0.25,
    "tail": 0.60,
    "hair_accessory": 0.20,
}


def validate_maximum_foreground_area_ratios(value):
    raw = DEFAULT_MAXIMUM_FOREGROUND_AREA_RATIOS if value is None else value
    if not isinstance(raw, dict):
        raise ValueError("part maximum foreground area ratios must be a mapping")
    result = dict(DEFAULT_MAXIMUM_FOREGROUND_AREA_RATIOS)
    for name, ratio in raw.items():
        if name not in result:
            raise ValueError(f"unknown part area-ratio limit: {name}")
        if (
            not isinstance(ratio, (int, float))
            or isinstance(ratio, bool)
            or not np.isfinite(float(ratio))
            or not 0 < float(ratio) <= 1
        ):
            raise ValueError(f"invalid part area-ratio limit: {name}")
        result[name] = float(ratio)
    return result



def _box_area(box):
    x1, y1, x2, y2 = (float(value) for value in box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_overlap(first, second):
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1))
    first_area, second_area = _box_area(first), _box_area(second)
    union = first_area + second_area - intersection
    smaller = min(first_area, second_area)
    return {
        "iou": intersection / union if union else 0.0,
        "smaller_coverage": intersection / smaller if smaller else 0.0,
    }


def deduplicate_ear_candidate_indices(boxes, selected_indices, scores,
                                      *, duplicate_iou=.70,
                                      enclosure_coverage=.90):
    """Keep anatomical ear boxes, not detector duplicates or a two-ear envelope."""
    selected = [int(index) for index in selected_indices]
    if len(selected) < 2:
        return np.asarray(selected, dtype=int), []
    removed = {}
    for index in selected:
        contained = [
            other for other in selected
            if other != index
            and _box_area(boxes[other]) < _box_area(boxes[index])
            and _box_overlap(boxes[index], boxes[other])["smaller_coverage"]
            >= enclosure_coverage
        ]
        if len(contained) >= 2:
            removed[index] = "enclosing_multiple_ears"
    remaining = [index for index in selected if index not in removed]
    ranked = sorted(
        remaining,
        key=lambda index: (float(scores[index]), -_box_area(boxes[index])),
        reverse=True,
    )
    kept = []
    for index in ranked:
        duplicate = next((
            other for other in kept
            if _box_overlap(boxes[index], boxes[other])["iou"] >= duplicate_iou
        ), None)
        if duplicate is None:
            kept.append(index)
        else:
            removed[index] = f"duplicate_of:{duplicate}"
    kept.sort(key=selected.index)
    decisions = [
        {"candidate_index": index, "reason": reason}
        for index, reason in sorted(removed.items())
    ]
    return np.asarray(kept, dtype=int), decisions


def filter_tail_garment_overlaps(selected_indices, decisions,
                                  maximum_garment_overlap):
    """Fail closed for tail candidates that are mostly approved garment."""
    if (not np.isfinite(maximum_garment_overlap)
            or not 0 < maximum_garment_overlap <= 1):
        raise ValueError("꼬리 후보 의상 중첩 한도는 0 초과 1 이하여야 합니다.")
    selected = {int(index) for index in selected_indices}
    rejected = []
    for decision in decisions:
        index = int(decision["candidate_index"])
        overlap = float(decision.get("garment_overlap", 0.0))
        if (index in selected and decision.get("accepted")
                and overlap >= maximum_garment_overlap):
            selected.remove(index)
            rejected.append(index)
            decision.update(
                accepted=False,
                reason="tail_garment_overlap_exceeded",
                selection_status="not_proposed",
                removed_pixels=int(decision.get("area_pixels", 0)),
            )
    ordered = np.asarray([
        int(index) for index in selected_indices if int(index) in selected
    ], dtype=int)
    return ordered, decisions, rejected


def validate_part_queries(queries):
    """검출 검색 대상은 해당 부위가 실제로 존재한다는 뜻이 아니다."""
    if not isinstance(queries, (list, tuple)) or not 1 <= len(queries) <= 7:
        raise ValueError("부위 검출 대상은 1~7개여야 합니다.")
    result, names = [], set()
    for spec in queries:
        if not isinstance(spec, dict) or set(spec) - {"name", "query", "hair_context"}:
            raise ValueError("부위 검출 대상 설정 형식 오류")
        name, query = spec.get("name"), spec.get("query")
        context = spec.get("hair_context", False)
        if not valid_part_name(name) or name in names:
            raise ValueError("부위 검출 이름 오류 또는 중복")
        if not isinstance(query, str) or not query.strip() or len(query) > 160:
            raise ValueError("부위 검출 검색어 오류")
        if not isinstance(context, bool):
            raise ValueError("헤어 문맥 사용 여부는 bool이어야 합니다.")
        names.add(name)
        result.append((name, query.strip(), context))
    return tuple(result)


class AdditionalPartsAnalyzer:
    """호환 조정자: 독립 검출 -> 교차 검토 -> 미리보기/배치 요청 순으로 실행한다."""
    def __init__(self, backend, *, box_threshold=.35, mask_threshold=.8,
                 scale=.35, debug_dir=None, target_locator=None, source_masks=None,
                 garment_dominance=.90, broad_hair_coverage=.50,
                 minimum_hair_overlap=.50, tail_maximum_garment_overlap=.75,
                 part_queries=None, proposal_policy=EXPERIMENTAL_OVERLAP_POLICY,
                 accessory_observation_settings=None,
                 spatial_diagnostic_settings=None,
                 maximum_foreground_area_ratios=None):
        self.backend = backend
        self.part_queries = validate_part_queries(DEFAULT_PART_QUERIES if part_queries is None else part_queries)
        if proposal_policy != EXPERIMENTAL_OVERLAP_POLICY:
            raise ValueError("등록되지 않은 부위 후보 제안 정책입니다.")
        self.proposal_policy = proposal_policy
        self.minimum_hair_overlap = minimum_hair_overlap
        OverlapProposalPolicy(garment_dominance, broad_hair_coverage, minimum_hair_overlap)
        self.box_threshold, self.mask_threshold, self.scale = box_threshold, mask_threshold, scale
        if not all(np.isfinite(v) and 0 <= v <= 1 for v in (box_threshold, mask_threshold, scale)):
            raise ValueError("귀·꼬리 분석 임계값/강도는 0~1이어야 합니다.")
        if not all(np.isfinite(v) and 0 < v <= 1 for v in (
                garment_dominance, broad_hair_coverage,
                tail_maximum_garment_overlap)):
            raise ValueError("부위 후보 제외 임계값은 0 초과 1 이하여야 합니다.")
        self.debug_dir = debug_dir
        self.target_locator = target_locator
        self.preview_only = target_locator is None
        self.report = {}
        self.reviewer = (PartCrossReviewer(source_masks,
            OverlapProposalPolicy(garment_dominance, broad_hair_coverage, minimum_hair_overlap))
            if source_masks is not None else None)
        self.garment_dominance = garment_dominance
        self.broad_hair_coverage = broad_hair_coverage
        self.tail_maximum_garment_overlap = tail_maximum_garment_overlap
        self.source_masks = source_masks
        self.maximum_foreground_area_ratios = (
            validate_maximum_foreground_area_ratios(
                maximum_foreground_area_ratios
            )
        )
        self.accessory_observation_policy = (
            AccessoryObservationPolicy.from_mapping(
                accessory_observation_settings
            )
        )
        self.spatial_diagnostic_policy = (
            PartSpatialDiagnosticPolicy.from_mapping(
                spatial_diagnostic_settings
            )
            if self.accessory_observation_policy is not None
            else None
        )

    def analyze(self, source, output_size, *, cancelled, deadline):
        def check():
            if cancelled():
                raise InterruptedError("귀·꼬리 분석 취소")
            if perf_counter() >= deadline:
                raise TimeoutError("귀·꼬리 분석 시간 제한 초과")
        check()
        directory = Path(self.debug_dir) if self.debug_dir else Path(tempfile.mkdtemp(prefix="genai-parts-"))
        directory.mkdir(parents=True, exist_ok=True)
        self.report = {"mode": "preview_only" if self.preview_only else "regional_condition",
                       "parts": {}, "debug_dir": str(directory),
                       "image_size": list(source.size)}
        self.report["filter_policy"] = {
            "version": self.proposal_policy, "enabled": self.reviewer is not None,
            "semantic_accuracy_verified": False, "requires_input_review": True,
            "minimum_hair_overlap": self.minimum_hair_overlap,
            "part_queries": [{"name": name, "query": query, "hair_context": context}
                             for name, query, context in self.part_queries],
            "garment_dominance": self.garment_dominance,
            "broad_hair_coverage": self.broad_hair_coverage,
            "tail_maximum_garment_overlap": (
                self.tail_maximum_garment_overlap),
            "maximum_foreground_area_ratios": dict(
                self.maximum_foreground_area_ratios
            ),
            "oversized_part_policy": (
                "unresolved_without_automatic_conditioning"
            ),
            "accepted_mask_pixels_preserved": True,
        }
        parts = []
        try:
            if self.reviewer is not None:
                self.reviewer.save_sources(directory)
            with source.convert("RGB") as rgb:
                for name, query, hair_context in self.part_queries:
                    raw_result = detect_part_candidates(
                        self.backend, rgb, name, query, self.box_threshold, check)
                    entry = {"status": "unresolved", "boxes": [list(b) for b in raw_result.boxes],
                             "detection_scores": list(raw_result.detection_scores), "accepted_masks": 0,
                             "target_status": "unresolved", "semantic_status": "unresolved",
                             "presence": "unresolved", "query": query,
                             "raw_candidate_fingerprint": raw_result.fingerprint(),
                             "analysis_stage": "independent_detection"}
                    self.report["parts"][name] = entry
                    if not raw_result.boxes:
                        entry.update(
                            status="uncertain",
                            presence="not_observed",
                            outcome_reason="no_candidate_above_threshold",
                            automatic_conditioning=False,
                        )
                        continue
                    entry["mask_quality_scores"] = list(raw_result.mask_scores)
                    entry["candidate_mask_paths"] = []
                    raw_masks = raw_result.array()
                    for index, raw in enumerate(raw_masks):
                        path = directory / f"{name}_candidate_{index}_mask.png"
                        with Image.fromarray(raw.astype(np.uint8) * 255) as diagnostic:
                            diagnostic.save(path)
                        entry["candidate_mask_paths"].append(str(path))
                    if self.reviewer is None:
                        selected_indices = np.flatnonzero(
                            np.asarray(raw_result.mask_scores) >= self.mask_threshold)
                        candidates = raw_masks[selected_indices]
                    else:
                        check()
                        review = self.reviewer.review(
                            raw_result, self.mask_threshold,
                            hair_context=hair_context)
                        selected_indices = np.asarray(
                            review.selected_indices, dtype=int)
                        decisions = review.decisions()
                        if name == "tail":
                            selected_indices, decisions, rejected = (
                                filter_tail_garment_overlaps(
                                    selected_indices,
                                    decisions,
                                    self.tail_maximum_garment_overlap,
                                )
                            )
                            entry["source_preflight"] = {
                                "version": "tail_source_preflight_v1",
                                "status": (
                                    "passed" if len(selected_indices)
                                    else "rejected"
                                ),
                                "maximum_garment_overlap": (
                                    self.tail_maximum_garment_overlap),
                                "rejected_candidate_indices": rejected,
                                "safe_candidate_count": int(
                                    len(selected_indices)),
                            }
                        candidates = raw_masks[selected_indices]
                        entry["source_filter"] = decisions
                        entry["review_stage"] = "cross_part_overlap_review"
                        for decision in entry["source_filter"]:
                            decision["mask_path"] = entry[
                                "candidate_mask_paths"][
                                    decision["candidate_index"]]
                        check()
                    if name in {"human_ears", "animal_ears"}:
                        input_count = int(len(selected_indices))
                        selected_indices, duplicate_decisions = (
                            deduplicate_ear_candidate_indices(
                                raw_result.boxes,
                                selected_indices,
                                raw_result.detection_scores,
                            )
                        )
                        candidates = raw_masks[selected_indices]
                        entry["duplicate_box_resolution"] = {
                            "version": "ear_candidate_deduplication_v1",
                            "input_count": input_count,
                            "output_count": int(len(candidates)),
                            "removed": duplicate_decisions,
                            "policy": (
                                "drop_two_ear_envelope_then_score_ordered_nms"
                            ),
                        }
                    combined = candidates.any(axis=0)
                    if not combined.any():
                        entry.update(
                            status="uncertain",
                            presence="unresolved",
                            outcome_reason=(
                                "all_candidates_rejected_or_low_mask_quality"
                            ),
                            automatic_conditioning=False,
                        )
                        continue
                    foreground = None
                    if self.source_masks is not None:
                        candidate_foreground = self.source_masks.get(
                            "foreground"
                        )
                        if isinstance(candidate_foreground, Image.Image):
                            foreground = (
                                np.asarray(
                                    candidate_foreground.convert("L"),
                                    dtype=np.uint8,
                                )
                                >= 128
                            )
                    denominator_source = (
                        "foreground_mask"
                        if foreground is not None and foreground.any()
                        else "canvas"
                    )
                    denominator_pixels = (
                        int(foreground.sum())
                        if denominator_source == "foreground_mask"
                        else int(combined.size)
                    )
                    mask_pixels = int(combined.sum())
                    area_ratio = mask_pixels / denominator_pixels
                    area_limit_name = name.removeprefix("output_")
                    maximum_area_ratio = (
                        self.maximum_foreground_area_ratios.get(
                            area_limit_name
                        )
                    )
                    entry["area_validity"] = {
                        "version": "small_part_area_validity_v1",
                        "mask_pixels": mask_pixels,
                        "denominator_pixels": denominator_pixels,
                        "denominator_source": denominator_source,
                        "foreground_area_ratio": float(area_ratio),
                        "maximum_foreground_area_ratio": maximum_area_ratio,
                        "valid": (
                            True
                            if maximum_area_ratio is None
                            else area_ratio <= maximum_area_ratio
                        ),
                        "status": (
                            "not_applicable"
                            if maximum_area_ratio is None
                            else "evaluated"
                        ),
                        "reason": (
                            "not_small_part_class"
                            if maximum_area_ratio is None
                            else None
                        ),
                    }
                    if (
                        maximum_area_ratio is not None
                        and area_ratio > maximum_area_ratio
                    ):
                        entry.update(
                            status="uncertain",
                            presence="unresolved",
                            outcome_reason=(
                                "small_part_area_upper_bound_exceeded"
                            ),
                            automatic_conditioning=False,
                            accepted_masks=0,
                            measured_mask_pixels=mask_pixels,
                            rejected_candidate_indices=[
                                int(index) for index in selected_indices
                            ],
                        )
                        continue
                    mask = Image.fromarray(combined.astype(np.uint8) * 255)
                    part = DetectedPart(name, mask, None, self.scale)
                    parts.append(part)
                    mask.save(directory / f"{name}_mask.png")
                    with crop_reference(rgb, mask) as crop:
                        crop.save(directory / f"{name}_reference.png")
                    entry.update(status="detected", presence="model_candidate",
                                 automatic_conditioning=True,
                                 outcome_reason="accepted_mask_available",
                                 accepted_masks=len(candidates),
                                 accepted_boxes=[list(raw_result.boxes[index])
                                                 for index in selected_indices],
                                 mask_pixels=int(combined.sum()),
                                 mask_path=str(directory / f"{name}_mask.png"),
                                 crop_path=str(directory / f"{name}_reference.png"))
                    if self.target_locator is not None:
                        part.target_region = self.target_locator(
                            name=name, source_mask=mask, output_size=output_size,
                            cancelled=cancelled, deadline=deadline)
                        entry["target_status"] = "resolved" if part.target_region is not None else "unresolved"
                        check()
            if self.accessory_observation_policy is not None:
                try:
                    hair_context = None
                    if self.source_masks and "hair" in self.source_masks:
                        hair_context = (
                            np.asarray(
                                self.source_masks["hair"].convert("L"),
                                dtype=np.uint8,
                            )
                            >= 128
                        )
                    _, head_source, head_box = locate_head_roi(
                        rgb.size,
                        hair_context,
                        self.spatial_diagnostic_policy,
                    )
                    observation = analyze_accessory_observations(
                        self.backend,
                        rgb,
                        head_box=head_box,
                        box_threshold=self.box_threshold,
                        policy=self.accessory_observation_policy,
                        check=check,
                        debug_directory=directory,
                    )
                    observation["head_roi_source"] = head_source
                    self.report["accessory_observations"] = observation
                except Exception as error:
                    self.report["accessory_observations"] = {
                        "version": "accessory_observation_v1",
                        "mode": "observe_only",
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "predictions_modified": False,
                        "masks_created": False,
                        "masks_modified": False,
                        "sam2_invoked": False,
                        "automatic_conditioning_applied": False,
                        "generation_policy_changed": False,
                    }
            by_name = {
                part.name: np.asarray(part.source_mask) >= 128
                for part in parts
            }
            overlaps = {}
            names = tuple(by_name)
            for first_index, first in enumerate(names):
                for second in names[first_index + 1:]:
                    shared = int((by_name[first] & by_name[second]).sum())
                    smaller = min(
                        int(by_name[first].sum()), int(by_name[second].sum()))
                    overlaps[f"{first}:{second}"] = {
                        "pixels": shared,
                        "smaller_part_ratio": (
                            float(shared / smaller) if smaller else None),
                    }
            self.report["part_overlap"] = overlaps
            self.report["status"] = "detected" if parts else "uncertain"
            return parts
        except BaseException as error:
            self.report["status"] = "failed"
            self.report["error"] = str(error)
            for part in parts:
                part.close()
            raise
        finally:
            (directory / "parts.json").write_text(
                json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf-8")

    def close(self):
        self.backend.close()


EarTailAnalyzer = AdditionalPartsAnalyzer  # 기존 호출자 호환 별칭


def create_extra_parts_analyzer(config, *, source_masks=None):
    settings = config.get("reference_analysis", {}).get("part_detection", {})
    if not settings.get("enabled", False):
        return None
    detection = config["clothing_preparation"]
    segmentation = config["clothing_mask_extraction"]
    return AdditionalPartsAnalyzer(
        TransformersPartsBackend(detection["detector_model_id"],
                                 segmentation["model_id"], detection["cache_dir"]),
        box_threshold=float(settings.get("box_threshold", .35)),
        mask_threshold=float(settings.get("mask_threshold", .8)),
        scale=float(settings.get("scale", .35)), source_masks=source_masks,
        garment_dominance=float(settings.get("garment_dominance", .90)),
        broad_hair_coverage=float(settings.get("broad_hair_coverage", .50)),
        minimum_hair_overlap=float(settings.get("minimum_hair_overlap", .50)),
        tail_maximum_garment_overlap=float(
            settings.get("tail_maximum_garment_overlap", .75)),
        part_queries=settings.get("part_queries"),
        proposal_policy=settings.get("proposal_policy", EXPERIMENTAL_OVERLAP_POLICY),
        maximum_foreground_area_ratios=settings.get(
            "maximum_foreground_area_ratios"
        ))
