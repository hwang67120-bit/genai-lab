"""승인된 제품 생성 입력을 GUI와 Codex 실행기 사이에서 손실 없이 전달한다."""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from genai_lab.approved_reference_run import ApprovedReferenceRun, canonical
from genai_lab.part_color_analysis import ColorEvidence
from genai_lab.part_color_descriptions import PartColorDescription
from genai_lab.request import CharacterFramingType, CharacterGenerationRequest
from genai_lab.body_morphology import BodyMorphologyVector
from genai_lab.scene_reference import PartReference
from genai_lab.visual_reference import VisualInputs


IMAGE_FIELDS = (
    "source",
    "identity",
    "garment",
    "identity_mask",
    "garment_mask",
    "vibe_reference",
    "full_character_mask",
    "hair_mask",
    "face_mask",
    "hair_reference",
    "hair_source_mask",
    "source_identity_mask",
)
CONFIG_OBJECT_KEYS = {
    "garment_image",
    "part_color_descriptions",
    "approved_part_color_descriptions",
}


class GenerationReplayError(ValueError):
    """승인 재생 번들이 없거나 변조됐을 때 발생한다."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any, *, omit_config_objects: bool = False) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {
            str(key): _json_safe(
                item, omit_config_objects=omit_config_objects
            )
            for key, item in value.items()
            if not (
                omit_config_objects and str(key) in CONFIG_OBJECT_KEYS
            )
        }
    if isinstance(value, (tuple, list)):
        return [
            _json_safe(item, omit_config_objects=omit_config_objects)
            for item in value
        ]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, BodyMorphologyVector):
        return value.record()
    raise GenerationReplayError(
        f"재생 번들에 저장할 수 없는 설정 타입입니다: {type(value).__name__}"
    )


def _description_payload(description: PartColorDescription) -> dict[str, Any]:
    return {
        "part_name": description.part_name,
        "colors": list(description.colors),
        "status": description.status,
        "reasons": list(description.reasons),
        "evidence_json": description.evidence_json.decode("utf-8"),
        "policy_fingerprint": description.policy_fingerprint,
    }


def _load_description(record: dict[str, Any]) -> PartColorDescription:
    return PartColorDescription(
        part_name=str(record["part_name"]),
        colors=tuple(record["colors"]),
        status=str(record["status"]),
        reasons=tuple(record["reasons"]),
        evidence_json=str(record["evidence_json"]).encode("utf-8"),
        policy_fingerprint=str(record["policy_fingerprint"]),
    )


def _request_payload(request: CharacterGenerationRequest) -> dict[str, Any]:
    result = {}
    for item in fields(request):
        if item.name == "reference_image":
            continue
        result[item.name] = _json_safe(getattr(request, item.name))
    return result


def _load_request(directory: Path, record: dict[str, Any]) -> CharacterGenerationRequest:
    reference_path = directory / "request_reference_image.png"
    if not reference_path.is_file():
        raise GenerationReplayError("요청 참조 이미지가 없습니다.")
    with Image.open(reference_path) as opened:
        reference_image = opened.convert("RGB")
    return CharacterGenerationRequest(
        reference_image=reference_image,
        reference_image_name=str(record["reference_image_name"]),
        reference_enhancement_applied=bool(
            record["reference_enhancement_applied"]),
        reference_enhancement_model_id=record[
            "reference_enhancement_model_id"],
        reference_quality_status=str(record["reference_quality_status"]),
        framing_type=CharacterFramingType(record["framing_type"]),
        width=int(record["width"]),
        height=int(record["height"]),
        prompt=str(record["prompt"]),
        negative_prompt=str(record["negative_prompt"]),
        seed=int(record["seed"]),
        candidate_number=int(record["candidate_number"]),
        inference_steps=int(record["inference_steps"]),
        guidance_scale=float(record["guidance_scale"]),
        original_image_change_strength=float(
            record["original_image_change_strength"]),
        reference_image_strength=float(record["reference_image_strength"]),
        model_id=str(record["model_id"]),
        reference_adapter_id=str(record["reference_adapter_id"]),
        body_morphology=(
            BodyMorphologyVector.from_record(record["body_morphology"])
            if record.get("body_morphology") is not None
            else None
        ),
    )


def save_generation_replay_bundle(
    inputs: VisualInputs,
    config: dict[str, Any],
    request: CharacterGenerationRequest,
    project_root: Path,
) -> Path:
    approval = getattr(inputs, "approved_generation", None)
    if not isinstance(approval, ApprovedReferenceRun):
        raise GenerationReplayError("승인 계약이 없는 입력은 재생 번들로 저장할 수 없습니다.")
    if inputs.scene_condition is not None:
        raise GenerationReplayError(
            "장면 선화 조건 재생은 아직 지원하지 않습니다."
        )

    root = Path(project_root)
    output_name = str(config.get("paths", {}).get("output_dir", "outputs"))
    replay_root = root / output_name / "product-run-specs"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    directory = replay_root / f"{stamp}-{approval.fingerprint[:12]}"
    directory.mkdir(parents=True, exist_ok=False)

    image_files = {}
    for name in IMAGE_FIELDS:
        image = getattr(inputs, name, None)
        if image is None:
            continue
        path = directory / f"{name}.png"
        image.save(path)
        image_files[name] = path.name
    request.reference_image.save(directory / "request_reference_image.png")

    extra_records = []
    extra_dir = directory / "extra_references"
    for index, reference in enumerate(inputs.extra_references):
        extra_dir.mkdir(parents=True, exist_ok=True)
        rgb_name = f"{index:02d}_{reference.name}_rgb.png"
        region_name = f"{index:02d}_{reference.name}_region.png"
        reference.rgb.save(extra_dir / rgb_name)
        reference.region.save(extra_dir / region_name)
        extra_records.append({
            "name": reference.name,
            "rgb": rgb_name,
            "region": region_name,
            "scale": float(reference.scale),
        })

    color_records = {}
    color_dir = directory / "color_evidence"
    for name, evidence in (inputs.color_evidence or {}).items():
        color_dir.mkdir(parents=True, exist_ok=True)
        labels_name = f"{name}_labels.npy"
        np.save(color_dir / labels_name, evidence.label_map, allow_pickle=False)
        color_records[name] = {
            "palette_rgb": _json_safe(evidence.palette_rgb),
            "area_ratios": _json_safe(evidence.area_ratios),
            "label_map": labels_name,
            "pixel_count": int(evidence.pixel_count),
            "pattern": str(evidence.pattern),
        }

    condition_record = None
    if inputs.reference_conditions is not None:
        condition_dir = directory / "independent_conditions"
        inputs.reference_conditions.save(condition_dir)
        condition_record = json.loads(
            (condition_dir / "conditions.json").read_text(encoding="utf-8")
        )

    section = config.get("clothing_reference_generation", {})
    descriptions = tuple(inputs.part_color_descriptions)
    approved_descriptions = tuple(
        section.get("approved_part_color_descriptions", ()))
    metadata = {
        "version": "generation_replay_bundle_v1",
        "approval_fingerprint": approval.fingerprint,
        "image_files": image_files,
        "extra_references": extra_records,
        "analysis_record": _json_safe(inputs.analysis_record),
        "color_evidence": color_records,
        "part_color_descriptions": [
            _description_payload(item) for item in descriptions
        ],
        "approved_part_color_names": [
            item.part_name for item in approved_descriptions
        ],
        "hair_reference_scale": float(inputs.hair_reference_scale),
        "reference_conditions": condition_record,
        "request": _request_payload(request),
    }
    (directory / "config.json").write_text(
        json.dumps(
            _json_safe(config, omit_config_objects=True),
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (directory / "approval.json").write_text(
        json.dumps({
            "fingerprint": approval.fingerprint,
            "record": approval.record(),
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata["artifact_sha256"] = {
        path.relative_to(directory).as_posix(): _file_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "bundle.json"
    }
    (directory / "bundle.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return directory


def _load_image(directory: Path, filename: str) -> Image.Image:
    path = directory / filename
    if not path.is_file():
        raise GenerationReplayError(f"재생 이미지가 없습니다: {filename}")
    with Image.open(path) as opened:
        opened.load()
        return opened.copy()


def _load_reference_conditions(directory: Path, record: dict[str, Any] | None):
    if record is None:
        return None
    from genai_lab.reference_conditions import (
        ReferenceCondition,
        ReferenceConditions,
    )
    condition_dir = directory / "independent_conditions"
    conditions = []
    for item in record.get("conditions", ()):
        name = str(item["name"])
        with Image.open(condition_dir / f"{name}_reference.png") as opened:
            rgb = opened.convert("RGB")
        with Image.open(condition_dir / f"{name}_source_region.png") as opened:
            region = opened.convert("L")
        try:
            conditions.append(ReferenceCondition.capture(
                name, rgb, region, item.get("scale")))
        finally:
            rgb.close()
            region.close()
    return ReferenceConditions(
        tuple(conditions), str(record["layout_policy"]))


def load_generation_replay_bundle(
    directory: Path,
    *,
    verify_approved_run: bool = True,
) -> tuple[VisualInputs, dict[str, Any], CharacterGenerationRequest, ApprovedReferenceRun]:
    directory = Path(directory)
    try:
        metadata = json.loads(
            (directory / "bundle.json").read_text(encoding="utf-8"))
        config = json.loads(
            (directory / "config.json").read_text(encoding="utf-8"))
        approval_data = json.loads(
            (directory / "approval.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GenerationReplayError(
            f"재생 번들을 읽을 수 없습니다: {directory}") from error
    if metadata.get("version") != "generation_replay_bundle_v1":
        raise GenerationReplayError("지원하지 않는 재생 번들 버전입니다.")
    artifact_hashes = metadata.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        raise GenerationReplayError("재생 번들의 파일 SHA-256 목록이 없습니다.")
    for relative, expected in artifact_hashes.items():
        artifact = directory / relative
        if (not artifact.is_file()
                or _file_sha256(artifact) != expected):
            raise GenerationReplayError(
                f"재생 번들 파일이 승인 저장 이후 변경됐습니다: {relative}"
            )

    approval_record = approval_data.get("record")
    fingerprint = approval_data.get("fingerprint")
    approval = ApprovedReferenceRun(canonical(approval_record), fingerprint)
    if fingerprint != metadata.get("approval_fingerprint"):
        raise GenerationReplayError("재생 번들의 승인 fingerprint가 서로 다릅니다.")

    # Early v1 writers applied config-only key filtering recursively and could
    # remove analysis fields such as part_color_descriptions. The complete
    # source analysis is already sealed inside the approved condition snapshot.
    analysis_record = metadata.get("analysis_record")
    approved_analysis = (
        approval.record()
        .get("condition_snapshot", {})
        .get("observations", {})
        .get("source_analysis_full")
    )
    if isinstance(approved_analysis, dict):
        analysis_record = approved_analysis

    images = {
        name: _load_image(directory, filename)
        for name, filename in metadata.get("image_files", {}).items()
    }
    required = {"source", "identity", "garment", "identity_mask", "garment_mask"}
    if not required.issubset(images):
        for image in images.values():
            image.close()
        raise GenerationReplayError("재생 번들의 필수 이미지가 누락됐습니다.")

    extra_references = []
    try:
        for item in metadata.get("extra_references", ()):
            extra_references.append(PartReference(
                str(item["name"]),
                _load_image(directory / "extra_references", item["rgb"]),
                _load_image(directory / "extra_references", item["region"]),
                float(item["scale"]),
            ))
        color_evidence = {}
        for name, item in metadata.get("color_evidence", {}).items():
            label_map = np.load(
                directory / "color_evidence" / item["label_map"],
                allow_pickle=False,
            )
            color_evidence[name] = ColorEvidence(
                palette_rgb=item["palette_rgb"],
                area_ratios=item["area_ratios"],
                label_map=label_map,
                pixel_count=int(item["pixel_count"]),
                pattern=str(item.get("pattern", "unresolved")),
            )
        descriptions = tuple(
            _load_description(item)
            for item in metadata.get("part_color_descriptions", ())
        )
        approved_names = set(metadata.get("approved_part_color_names", ()))
        section = config.setdefault("clothing_reference_generation", {})
        section["part_color_descriptions"] = descriptions
        section["approved_part_color_descriptions"] = tuple(
            item for item in descriptions if item.part_name in approved_names)
        conditions = _load_reference_conditions(
            directory, metadata.get("reference_conditions"))
        inputs = VisualInputs(
            source=images.pop("source"),
            identity=images.pop("identity"),
            garment=images.pop("garment"),
            identity_mask=images.pop("identity_mask"),
            garment_mask=images.pop("garment_mask"),
            vibe_reference=images.pop("vibe_reference", None),
            full_character_mask=images.pop("full_character_mask", None),
            extra_references=tuple(extra_references),
            analysis_record=analysis_record,
            color_evidence=color_evidence,
            reference_conditions=conditions,
            part_color_descriptions=descriptions,
            hair_mask=images.pop("hair_mask", None),
            face_mask=images.pop("face_mask", None),
            hair_reference=images.pop("hair_reference", None),
            hair_reference_scale=float(
                metadata.get("hair_reference_scale", 0.60)),
            hair_source_mask=images.pop("hair_source_mask", None),
            source_identity_mask=images.pop("source_identity_mask", None),
        )
    except BaseException:
        for image in images.values():
            image.close()
        for reference in extra_references:
            reference.close()
        raise

    request = _load_request(directory, metadata["request"])
    if verify_approved_run:
        inputs.approved_generation = approval
        from genai_lab.approved_reference_run import require_reference_run
        try:
            require_reference_run(inputs, config, request)
        except BaseException:
            request.reference_image.close()
            inputs.close()
            raise
    return inputs, config, request, approval



