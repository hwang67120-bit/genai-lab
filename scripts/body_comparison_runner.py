"""CatVTON의 SCHP·DensePose로 기준 후보의 의상·신체 영역을 분석한다."""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from time import perf_counter

from PIL import Image

logger = logging.getLogger(__name__)


def read_class_labels(mask, name):
    """P-mode palette indices are class IDs, not grayscale intensities."""
    import numpy as np
    labels = np.asarray(mask)
    logger.info('[분류 마스크] 이름=%s mode=%s size=%s shape=%s',
                name, mask.mode, mask.size, labels.shape)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f'{name}: {mask.mode} 시각화/비정수 배열이 아닌 원시 클래스 ID가 필요합니다.')
    values, counts = np.unique(labels, return_counts=True)
    logger.info('[분류 마스크] 이름=%s 클래스별 픽셀=%s', name,
                dict(zip(values.tolist(), counts.tolist())))
    return labels.copy()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-path", required=True)
    parser.add_argument("--person-image", required=True)
    parser.add_argument(
        "--clothing-type",
        choices=("upper", "lower", "overall", "inner", "outer"),
        required=True,
    )
    parser.add_argument("--output-raw-mask", required=True)
    parser.add_argument("--output-protection-mask", required=True)
    parser.add_argument("--output-foreground-mask", required=True)
    parser.add_argument("--output-densepose", required=True)
    parser.add_argument("--output-metadata-json", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--foreground-model-id", default="isnet-anime")
    parser.add_argument("--explicit-target-masks", action="store_true")
    parser.add_argument("--layered-target-masks", action="store_true")
    parser.add_argument("--output-face-hair-mask")
    parser.add_argument("--output-face-reference-mask")
    parser.add_argument("--reference-output-dir")
    parser.add_argument("--approved-garment-tags-json")
    parser.add_argument(
        "--allow-parser-foreground-fallback",
        action="store_true",
        help=("생성 후보 출력 좌표 재검출에서 isnet-anime이 0px일 때만 "
              "SCHP LIP/ATR 전경을 사용합니다."),
    )
    return parser.parse_args()


GARMENT_SEMANTIC_TAG_RULES = (
    (("shirt", "blouse", "sweater", "cardigan", "hoodie", "vest"),
     ("Upper-clothes",)),
    (("jacket", "blazer", "uniform", "suit"),
     ("Upper-clothes", "Coat")),
    (("coat",), ("Coat",)),
    (("dress", "gown"), ("Dress",)),
    (("skirt",), ("Skirt",)),
    (("pants", "trousers", "jeans", "slacks", "shorts", "leggings"),
     ("Pants",)),
    (("jumpsuit", "romper", "overalls"), ("Jumpsuits",)),
    (("socks", "stockings", "pantyhose", "tights", "legwear"),
     ("Socks",)),
    (("shoe", "shoes", "boots", "heels", "loafers", "sneakers", "footwear"),
     ("Left-shoe", "Right-shoe")),
    (("scarf",), ("Scarf",)),
    (("glove", "gloves"), ("Glove",)),
)
GARMENT_SEMANTIC_PARTS = (
    "Upper-clothes", "Coat", "Dress", "Skirt", "Pants", "Jumpsuits",
    "Socks", "Left-shoe", "Right-shoe", "Scarf", "Glove",
)
GARMENT_ANATOMY_VETO_PARTS = (
    "Left-leg", "Right-leg", "Left-shoe", "Right-shoe",
)


def approved_garment_semantic_parts(approved_tags):
    """승인 태그를 SCHP 의상 클래스로만 변환한다; 유사 의미를 새로 추론하지 않는다."""
    normalized = tuple(
        str(tag).strip().lower().replace("_", " ").replace("-", " ")
        for tag in (approved_tags or ()) if str(tag).strip()
    )
    selected = []
    for terms, parts in GARMENT_SEMANTIC_TAG_RULES:
        if any(term in tag.split() for tag in normalized for term in terms):
            selected.extend(parts)
    return tuple(dict.fromkeys(selected))


def filter_small_disconnected_garment_components(selected, approved_parts):
    """승인된 분리형 의상이 없을 때 본체 1% 미만의 고립 조각을 제거한다."""
    import cv2
    import numpy as np
    detached_parts = {"Socks", "Left-shoe", "Right-shoe", "Glove", "Scarf"}
    selected = np.asarray(selected, dtype=bool)
    if detached_parts.intersection(approved_parts) or not selected.any():
        return selected.copy(), 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        selected.astype(np.uint8), connectivity=8)
    if count <= 1:
        return selected.copy(), 0
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    minimum_pixels = max(1, int(np.ceil(largest * .01)))
    keep_ids = [
        component_id for component_id in range(1, count)
        if int(stats[component_id, cv2.CC_STAT_AREA]) >= minimum_pixels
    ]
    filtered = np.isin(labels, keep_ids)
    return filtered, int((selected & ~filtered).sum())


def create_layered_semantic_masks(auto_mask, lip_mask, atr_mask, lip_mapping,
                                  atr_mapping, approved_garment_tags=()):
    """기존 분석 출력만 재사용해 얼굴·머리 보호와 전체 의상 후보를 분리한다."""
    import numpy as np
    if len({auto_mask.size, lip_mask.size, atr_mask.size}) != 1:
        raise ValueError("자동 마스크와 의상 분류 좌표 크기가 다릅니다.")
    with auto_mask.convert("L") as auto:
        lip_array = read_class_labels(lip_mask, 'SCHP-LIP')
        atr_array = read_class_labels(atr_mask, 'SCHP-ATR')
        for name, labels, mapping in (('LIP', lip_array, lip_mapping), ('ATR', atr_array, atr_mapping)):
            for part in ('Face', 'Hair'):
                class_id = mapping.get(part)
                if not isinstance(class_id, (int, np.integer)):
                    raise ValueError(f'{name}: {part} 정수 클래스 매핑이 없습니다.')
                logger.info('[얼굴·헤어 분리] 모델=%s 부위=%s ID=%s 픽셀=%d',
                            name, part, class_id, np.count_nonzero(labels == class_id))

        def select(parts):
            return (np.isin(lip_array, [lip_mapping[p] for p in parts if p in lip_mapping])
                    | np.isin(atr_array, [atr_mapping[p] for p in parts if p in atr_mapping]))

        face_hair = select(["Face", "Hair"])
        if not face_hair.any():
            raise ValueError('생성용 얼굴·헤어 분류 영역이 0px입니다. 클래스 ID 로그를 확인하세요. 생성/유사도 비교 전 중단입니다.')
        all_clothing_parts = GARMENT_SEMANTIC_PARTS
        approved_parts = approved_garment_semantic_parts(approved_garment_tags)
        clothes = select(all_clothing_parts)
        auto_array = np.asarray(auto) >= 128
        if approved_parts:
            rejected_parts = [part for part in all_clothing_parts
                              if part not in approved_parts]
            approved = select(approved_parts)
            # 두 파서 중 하나라도 하반신 해부 영역 또는 승인되지 않은 의상으로
            # 판정한 픽셀은 다른 파서의 의상 판정으로 되살리지 않는다.
            rejected = select(rejected_parts)
            anatomy_veto = select(GARMENT_ANATOMY_VETO_PARTS)
            # overall 자동 마스크는 클래스가 불명확한 다리 픽셀까지 포함할 수
            # 있으므로 승인 태그를 해석한 경우에는 생성 조건으로 재사용하지 않는다.
            union = approved & ~rejected & ~anatomy_veto
            union, component_pixels_removed = (
                filter_small_disconnected_garment_components(
                    union, approved_parts))
            logger.info(
                '[승인 의상 기반 마스크] 태그=%s 포함=%s 제외=%s '
                '해부영역차단=%s 충돌제거=%dpx 고립제거=%dpx 픽셀=%d',
                tuple(approved_garment_tags), approved_parts,
                tuple(rejected_parts), GARMENT_ANATOMY_VETO_PARTS,
                np.count_nonzero(approved & (rejected | anatomy_veto)),
                component_pixels_removed,
                np.count_nonzero(union),
            )
        else:
            union = auto_array | clothes
            logger.info('[승인 의상 기반 마스크] 상태=legacy_overall 사유=구조 태그 없음')
        return (Image.fromarray(face_hair.astype(np.uint8) * 255),
                Image.fromarray(union.astype(np.uint8) * 255))


def create_face_reference_mask(lip_mask, atr_mask, lip_mapping, atr_mapping):
    """Reuse parser labels; exclude any pixels classified as hair by either parser."""
    import numpy as np
    if lip_mask.size != atr_mask.size:
        raise ValueError('얼굴 참조 분류 좌표 크기가 다릅니다.')
    face = np.zeros((lip_mask.height, lip_mask.width), dtype=bool)
    hair = np.zeros_like(face)
    for name, mask, mapping in (('LIP', lip_mask, lip_mapping), ('ATR', atr_mask, atr_mapping)):
        labels = read_class_labels(mask, name)
        for part, selected in (('Face', face), ('Hair', hair)):
            class_id = mapping.get(part)
            if not isinstance(class_id, (int, np.integer)):
                raise ValueError(f'{name}: {part} 정수 클래스 매핑이 없습니다.')
            selected |= labels == class_id
    selected = face & ~hair
    logger.info('[얼굴 참조 분리] 얼굴=%dpx 헤어 충돌 제외=%dpx 최종=%dpx',
                np.count_nonzero(face), np.count_nonzero(face & hair), np.count_nonzero(selected))
    if not selected.any():
        raise ValueError('생성용 얼굴 단독 분류 영역이 0px입니다. 머리 전체로 대체하지 않습니다.')
    return Image.fromarray(selected.astype(np.uint8) * 255)


def create_parser_foreground_mask(lip_mask, atr_mask, lip_mapping, atr_mapping):
    """두 SCHP 파서 중 하나라도 배경이 아닌 생성 후보 픽셀을 전경으로 사용한다."""
    import numpy as np
    if lip_mask.size != atr_mask.size:
        raise ValueError("SCHP 전경 분류 좌표 크기가 다릅니다.")
    lip = read_class_labels(lip_mask, "SCHP-LIP")
    atr = read_class_labels(atr_mask, "SCHP-ATR")
    background_ids = []
    for name, mapping in (("LIP", lip_mapping), ("ATR", atr_mapping)):
        class_id = mapping.get("Background")
        if not isinstance(class_id, (int, np.integer)):
            raise ValueError(f"{name}: Background 정수 클래스 매핑이 없습니다.")
        background_ids.append(class_id)
    foreground = ~((lip == background_ids[0]) & (atr == background_ids[1]))
    if not foreground.any():
        raise RuntimeError("SCHP-LIP/ATR도 캐릭터 외곽 픽셀을 찾지 못했습니다.")
    logger.warning(
        "[출력 좌표 전경 폴백] source=schp_lip_atr_union_fallback pixels=%d",
        int(np.count_nonzero(foreground)),
    )
    return Image.fromarray(foreground.astype(np.uint8) * 255).convert("L")


def load_rgb_image(image_path: str) -> Image.Image:
    """파일 핸들을 즉시 닫고 RGB 픽셀만 소유한 이미지로 반환한다."""
    with Image.open(image_path) as opened_image:
        return opened_image.convert("RGB")


def extract_anime_character_foreground_mask(
    person_image: Image.Image,
    model_id: str,
    model_cache_dir: Path,
) -> Image.Image:
    """공식 isnetis ONNX로 캐릭터 전체 외곽의 알파 마스크를 반환한다."""
    if model_id != "isnet-anime":
        raise RuntimeError(
            "지원하지 않는 캐릭터 외곽 모델입니다: " f"{model_id}"
        )

    import numpy as np
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(
        repo_id="skytnt/anime-seg",
        filename="isnetis.onnx",
        cache_dir=model_cache_dir,
    )
    foreground_session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )
    input_shape = foreground_session.get_inputs()[0].shape
    model_height = int(input_shape[2])
    model_width = int(input_shape[3])
    scale = min(
        model_width / person_image.width,
        model_height / person_image.height,
    )
    content_width = max(1, round(person_image.width * scale))
    content_height = max(1, round(person_image.height * scale))
    content_left = (model_width - content_width) // 2
    content_top = (model_height - content_height) // 2
    resized_image = person_image.resize(
        (content_width, content_height),
        Image.Resampling.LANCZOS,
    )
    model_canvas = Image.new("RGB", (model_width, model_height), (0, 0, 0))
    try:
        model_canvas.paste(resized_image, (content_left, content_top))
        model_input = np.asarray(model_canvas, dtype=np.float32)
        model_input = (
            model_input[:, :, ::-1]
            .transpose((2, 0, 1))[None, :, :, :]
            / 255.0
        )
        predicted_mask = foreground_session.run(
            [foreground_session.get_outputs()[0].name],
            {foreground_session.get_inputs()[0].name: model_input},
        )[0]
        predicted_mask = np.squeeze(predicted_mask)
        predicted_mask = np.clip(predicted_mask, 0.0, 1.0)
        model_mask = Image.fromarray(
            (predicted_mask * 255.0).round().astype(np.uint8),
            mode="L",
        )
        try:
            content_mask = model_mask.crop(
                (
                    content_left,
                    content_top,
                    content_left + content_width,
                    content_top + content_height,
                )
            )
            try:
                return content_mask.resize(
                    person_image.size,
                    Image.Resampling.LANCZOS,
                )
            finally:
                content_mask.close()
        finally:
            model_mask.close()
    finally:
        resized_image.close()
        model_canvas.close()


def prepare_reference_canvas(source, size):
    """가로세로 비율과 전체 내용을 보존하고 여백 좌표를 함께 반환한다."""
    from PIL import ImageOps
    content = ImageOps.contain(source, size, Image.Resampling.LANCZOS)
    left, top = (size[0] - content.width) // 2, (size[1] - content.height) // 2
    canvas = Image.new("RGB", size, "white")
    canvas.paste(content, (left, top))
    box = (left, top, left + content.width, top + content.height)
    content.close()
    return canvas, box


def save_reference_regions(result, foreground, original_size, output_dir,
                           lip_mapping, atr_mapping, content_box=None,
                           approved_garment_tags=(),
                           foreground_source="isnet-anime"):
    """합성용 출력 없이 참조 추출 영역만 저장한다."""
    import numpy as np
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    head, clothes = create_layered_semantic_masks(
        result["mask"], result["schp_lip"], result["schp_atr"],
        lip_mapping, atr_mapping, approved_garment_tags,
    )
    hair = face = None
    try:
        lip = read_class_labels(result["schp_lip"], "SCHP-LIP")
        atr = read_class_labels(result["schp_atr"], "SCHP-ATR")
        hair = Image.fromarray(
            ((lip == lip_mapping["Hair"]) | (atr == atr_mapping["Hair"]))
            .astype(np.uint8) * 255)
        # Hair와 독립적으로 계산한 Face 후보를 보존한다. 두 파서가 같은
        # 좌표를 서로 다르게 분류한 충돌은 후단 정밀화가 직접 판단한다.
        face = Image.fromarray(
            ((lip == lip_mapping["Face"]) | (atr == atr_mapping["Face"]))
            .astype(np.uint8) * 255)
        arrays = {}
        for name, mask in (("identity", head), ("hair", hair), ("face", face),
                           ("garment", clothes), ("foreground", foreground)):
            with mask.crop(content_box or (0, 0, *mask.size)) as content:
                with content.resize(original_size, Image.Resampling.NEAREST) as resized:
                    arrays[name] = np.asarray(resized) >= 128
        # 참조 영향 영역의 충돌만 제거한다. 팽창·중립화·제거 검증은 없다.
        arrays["garment"] &= arrays["foreground"] & ~arrays["identity"]
        approved_parts = approved_garment_semantic_parts(approved_garment_tags)
        final_component_pixels_removed = 0
        if approved_parts:
            arrays["garment"], final_component_pixels_removed = (
                filter_small_disconnected_garment_components(
                    arrays["garment"], approved_parts))
        for name, values in arrays.items():
            with Image.fromarray(values.astype(np.uint8) * 255) as mask:
                mask.save(directory / f"{name}.png")
        payload = {
            "mode": "reference_regions_v1",
            "size": list(original_size),
            "pixel_counts": {name: int(a.sum()) for name, a in arrays.items()},
            "unresolved": ["iris_left", "iris_right", "ears", "tail", "wings", "horns"],
            "note": "미검출/미지원은 부재의 증거가 아닙니다. 원본 좌표는 포즈 고정이 아닙니다.",
            "neutralization_used": False,
            "removal_verification_used": False,
            "backend": "SCHP-LIP/ATR + DensePose AutoMasker + foreground",
            "foreground_detection": {
                "source": foreground_source,
                "fallback_used": foreground_source != "isnet-anime",
            },
            "analysis_size": list(foreground.size),
            "content_box": list(content_box) if content_box else None,
            "garment_mask_policy": {
                "mode": ("approved_semantic_classes"
                         if approved_garment_semantic_parts(approved_garment_tags)
                         else "legacy_overall"),
                "approved_tags": list(approved_garment_tags),
                "included_classes": list(
                    approved_garment_semantic_parts(approved_garment_tags)),
                "excluded_classes": [
                    part for part in GARMENT_SEMANTIC_PARTS
                    if part not in approved_garment_semantic_parts(
                        approved_garment_tags)
                ],
                "cross_parser_veto_classes": list(
                    GARMENT_ANATOMY_VETO_PARTS),
                "detached_component_policy": (
                    "keep_components_at_least_one_percent_of_largest"
                    if not {"Socks", "Left-shoe", "Right-shoe", "Glove", "Scarf"}
                    .intersection(approved_garment_semantic_parts(
                        approved_garment_tags))
                    else "disabled_for_approved_detached_garment_parts"
                ),
                "final_disconnected_pixels_removed": (
                    final_component_pixels_removed),
            },
        }
        (directory / "regions.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        head.close()
        clothes.close()
        if hair is not None:
            hair.close()
        if face is not None:
            face.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    """기준 후보에서 의상 마스크·신체 보호 영역·DensePose를 반환한다."""
    arguments = parse_arguments()
    started_at = perf_counter()
    repository_path = Path(arguments.repository_path).resolve()
    cache_dir = Path(arguments.cache_dir).resolve()
    os.environ["HF_HOME"] = str(cache_dir)
    sys.path.insert(0, str(repository_path))

    import truststore

    truststore.inject_into_ssl()
    import numpy as np
    from huggingface_hub import snapshot_download
    from model.cloth_masker import (
        ATR_MAPPING, LIP_MAPPING, AutoMasker, part_mask_of,
    )
    from utils import resize_and_crop

    original_person_image = None
    resized_person_image = None
    raw_mask = None
    identity_protection_mask = None
    character_foreground_mask = None
    densepose_preview = None
    automatic_mask_result = None
    foreground_fallback_required = False
    foreground_source = arguments.foreground_model_id
    try:
        original_person_image = load_rgb_image(arguments.person_image)
        original_size = original_person_image.size
        reference_box = None
        if arguments.reference_output_dir:
            resized_person_image, reference_box = prepare_reference_canvas(
                original_person_image, (arguments.width, arguments.height))
        else:
            resized_person_image = resize_and_crop(
                original_person_image, (arguments.width, arguments.height))

        foreground_started_at = perf_counter()
        character_foreground_mask = extract_anime_character_foreground_mask(
            resized_person_image,
            arguments.foreground_model_id,
            cache_dir,
        )
        foreground_elapsed_seconds = perf_counter() - foreground_started_at
        foreground_mask_array = np.asarray(
            character_foreground_mask,
            dtype=np.uint8,
        )
        foreground_pixel_count = int(
            np.count_nonzero(foreground_mask_array >= 128)
        )
        if foreground_pixel_count == 0:
            if not (arguments.reference_output_dir
                    and arguments.allow_parser_foreground_fallback):
                raise RuntimeError(
                    "isnet-anime이 캐릭터 외곽 픽셀을 찾지 못했습니다."
                )
            foreground_fallback_required = True
            logger.warning(
                "[출력 좌표 전경 폴백 대기] isnet-anime=0px, "
                "SCHP LIP/ATR 분석을 계속합니다."
            )
        foreground_percent = (
            foreground_pixel_count / foreground_mask_array.size * 100.0
        )

        checkpoint_path = snapshot_download(
            repo_id="zhengchong/CatVTON",
            cache_dir=cache_dir,
        )
        automatic_masker = AutoMasker(
            densepose_ckpt=str(Path(checkpoint_path) / "DensePose"),
            schp_ckpt=str(Path(checkpoint_path) / "SCHP"),
            device="cuda",
        )
        automatic_mask_result = automatic_masker(
            resized_person_image,
            "overall" if arguments.layered_target_masks else arguments.clothing_type,
        )
        if foreground_fallback_required:
            parser_foreground = create_parser_foreground_mask(
                automatic_mask_result["schp_lip"],
                automatic_mask_result["schp_atr"],
                LIP_MAPPING,
                ATR_MAPPING,
            )
            character_foreground_mask.close()
            character_foreground_mask = parser_foreground
            foreground_mask_array = np.asarray(
                character_foreground_mask, dtype=np.uint8)
            foreground_pixel_count = int(
                np.count_nonzero(foreground_mask_array >= 128))
            foreground_percent = (
                foreground_pixel_count / foreground_mask_array.size * 100.0)
            foreground_source = "schp_lip_atr_union_fallback"
        if arguments.reference_output_dir:
            approved_garment_tags = ()
            if arguments.approved_garment_tags_json:
                decoded_tags = json.loads(arguments.approved_garment_tags_json)
                if (not isinstance(decoded_tags, list)
                        or not all(isinstance(tag, str) for tag in decoded_tags)):
                    raise ValueError("승인 의상 태그 JSON은 문자열 배열이어야 합니다.")
                approved_garment_tags = tuple(decoded_tags)
            save_reference_regions(
                automatic_mask_result, character_foreground_mask, original_size,
                arguments.reference_output_dir, LIP_MAPPING, ATR_MAPPING,
                reference_box, approved_garment_tags,
                foreground_source=foreground_source)
            return
        raw_mask = automatic_mask_result["mask"].convert("L")
        densepose_preview = automatic_mask_result["densepose"].convert("RGB")

        protected_parts = [
            "Hat", "Hair", "Sunglasses", "Face", "Left-arm", "Right-arm",
            "Left-leg", "Right-leg", "Left-shoe", "Right-shoe", "Glove",
            "Bag", "Scarf",
        ]
        if arguments.explicit_target_masks:
            # 교체 범위는 별도 사용자 승인 SAM2 마스크가 제한한다.
            # 신발을 무조건 보호하지 않되 얼굴·머리는 계속 보호한다.
            protected_parts = ["Hat", "Hair", "Sunglasses", "Face"]
        schp_lip_mask = np.asarray(automatic_mask_result["schp_lip"])
        schp_atr_mask = np.asarray(automatic_mask_result["schp_atr"])
        identity_protection_array = (
            part_mask_of(protected_parts, schp_lip_mask, LIP_MAPPING)
            | part_mask_of(protected_parts, schp_atr_mask, ATR_MAPPING)
        )
        identity_protection_mask = Image.fromarray(
            (identity_protection_array > 0).astype(np.uint8) * 255
        ).convert("L")
        if arguments.layered_target_masks:
            if not arguments.output_face_hair_mask:
                raise ValueError("레이어 마스크에는 얼굴·머리 보호 출력 경로가 필요합니다.")
            face_hair, full_clothing = create_layered_semantic_masks(
                raw_mask, automatic_mask_result["schp_lip"], automatic_mask_result["schp_atr"],
                LIP_MAPPING, ATR_MAPPING,
            )
            raw_mask.close()
            raw_mask = full_clothing
            with face_hair:
                with face_hair.resize(original_size, Image.Resampling.NEAREST) as original_mask:
                    original_mask.save(arguments.output_face_hair_mask)

        if arguments.output_face_reference_mask:
            with create_face_reference_mask(automatic_mask_result['schp_lip'],
                    automatic_mask_result['schp_atr'], LIP_MAPPING, ATR_MAPPING) as face_reference:
                with face_reference.resize(original_size, Image.Resampling.NEAREST) as original_mask:
                    original_mask.save(arguments.output_face_reference_mask)

        resized_output = raw_mask.resize(
            original_size,
            Image.Resampling.NEAREST,
        )
        try:
            resized_output.save(arguments.output_raw_mask)
        finally:
            resized_output.close()
        resized_output = identity_protection_mask.resize(
            original_size,
            Image.Resampling.NEAREST,
        )
        try:
            resized_output.save(arguments.output_protection_mask)
        finally:
            resized_output.close()
        resized_output = character_foreground_mask.resize(
            original_size,
            Image.Resampling.LANCZOS,
        )
        try:
            resized_foreground_array = np.asarray(
                resized_output,
                dtype=np.uint8,
            )
            foreground_pixel_count = int(
                np.count_nonzero(resized_foreground_array >= 128)
            )
            foreground_percent = (
                foreground_pixel_count
                / resized_foreground_array.size
                * 100.0
            )
            resized_output.save(arguments.output_foreground_mask)
        finally:
            resized_output.close()
        resized_output = densepose_preview.resize(
            original_size,
            Image.Resampling.NEAREST,
        )
        try:
            resized_output.save(arguments.output_densepose)
        finally:
            resized_output.close()
        Path(arguments.output_metadata_json).write_text(
            json.dumps(
                {
                    "model_ids": [
                        "zhengchong/CatVTON:SCHP+DensePose",
                        arguments.foreground_model_id,
                    ],
                    "foreground_model_id": arguments.foreground_model_id,
                    "explicit_target_masks": arguments.explicit_target_masks,
                    "foreground_pixel_count": foreground_pixel_count,
                    "foreground_percent": foreground_percent,
                    "foreground_elapsed_seconds": foreground_elapsed_seconds,
                    "elapsed_seconds": perf_counter() - started_at,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        for image in (
            original_person_image, resized_person_image, raw_mask,
            identity_protection_mask, character_foreground_mask,
            densepose_preview,
        ):
            if image is not None:
                image.close()
        if automatic_mask_result is not None:
            for result_name in ("mask", "densepose", "schp_lip", "schp_atr"):
                result_image = automatic_mask_result.get(result_name)
                if result_image is not None:
                    result_image.close()


if __name__ == "__main__":
    main()
