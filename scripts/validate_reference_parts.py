"""Validate animal-ear, hair-accessory, and tail observations offline."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from genai_lab.extra_parts_analysis import (  # noqa: E402
    AdditionalPartsAnalyzer,
    TransformersPartsBackend,
)
from genai_lab.reference_analysis_validation import (  # noqa: E402
    PARTS,
    ReferencePartValidationError,
    run_reference_part_validation,
)
from run import configure_console_encoding, load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "생성과 분리된 정답 자료로 동물귀·헤어 장신구·꼬리를 검증합니다."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "animagine.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


def _validation_queries(settings: dict) -> list[dict]:
    configured = settings.get("part_queries", ())
    queries = [
        dict(item)
        for item in configured
        if isinstance(item, dict) and item.get("name") in PARTS
    ]
    names = {item.get("name") for item in queries}
    if names != set(PARTS):
        raise ReferencePartValidationError(
            "현재 설정에 animal_ears, hair_accessory, tail 질의가 모두 필요합니다."
        )
    return queries


def main() -> int:
    configure_console_encoding()
    args = parse_args()
    config = load_yaml(args.config)
    detection = config.get("clothing_preparation", {})
    segmentation = config.get("clothing_mask_extraction", {})
    settings = config.get("reference_analysis", {}).get("part_detection", {})
    if not settings.get("enabled", False):
        raise ReferencePartValidationError(
            "reference_analysis.part_detection이 비활성화되어 있습니다."
        )
    output = args.output_dir or (
        PROJECT_ROOT
        / "outputs"
        / "reference-analysis-validation"
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    backend = TransformersPartsBackend(
        detection["detector_model_id"],
        segmentation["model_id"],
        detection["cache_dir"],
    )
    queries = _validation_queries(settings)

    def analyzer_factory(*, debug_dir, source_masks):
        return AdditionalPartsAnalyzer(
            backend,
            box_threshold=float(settings.get("box_threshold", 0.35)),
            mask_threshold=float(settings.get("mask_threshold", 0.80)),
            scale=float(settings.get("scale", 0.35)),
            debug_dir=debug_dir,
            target_locator=None,
            source_masks=source_masks,
            garment_dominance=float(
                settings.get("garment_dominance", 0.90)
            ),
            broad_hair_coverage=float(
                settings.get("broad_hair_coverage", 0.50)
            ),
            minimum_hair_overlap=float(
                settings.get("minimum_hair_overlap", 0.50)
            ),
            tail_maximum_garment_overlap=float(
                settings.get("tail_maximum_garment_overlap", 0.75)
            ),
            part_queries=queries,
            proposal_policy=settings.get(
                "proposal_policy",
                "experimental_overlap_v2",
            ),
            accessory_observation_settings=settings.get(
                "accessory_observations",
            ),
            spatial_diagnostic_settings=settings.get(
                "diagnostics",
            ),
        )

    snapshot = {
        "detector_model_id": detection.get("detector_model_id"),
        "segmentation_model_id": segmentation.get("model_id"),
        "part_detection": {
            key: value
            for key, value in settings.items()
            if key != "generation_region_strengthening"
        },
        "validation_scope": list(PARTS),
        "generation_disabled": True,
    }
    try:
        report = run_reference_part_validation(
            args.manifest,
            output,
            analyzer_factory=analyzer_factory,
            timeout_seconds=args.timeout_seconds,
            configuration=snapshot,
            diagnostic_policy=settings.get("diagnostics", {}),
        )
    finally:
        backend.close()

    metrics_path = output / "metrics.json"
    print(json.dumps({
        "status": report["status"],
        "case_count": report["case_count"],
        "failed_case_count": report["failed_case_count"],
        "metrics": str(metrics_path.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0 if report["failed_case_count"] == 0 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        print(f"검증 실행 실패: {error}", file=sys.stderr)
        raise SystemExit(1)
