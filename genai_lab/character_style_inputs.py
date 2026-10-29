"""캐릭터 특징과 목표 그림체를 분리한 CPU 입력 묶음을 만든다.

prepare_character_style_bundle → validate_annotations → write_input_assets 순서다.
특징·관절은 출처가 있는 주석을 받는다. 자동 인식 결과로 가장하지 않는다.
"""
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import re

from PIL import Image, ImageDraw
from scripts.pose_reference_runner import BODY_JOINT_NAMES

REQUIRED_TRAITS = {"eyes", "skin", "hair", "clothes", "build"}
BASE_ROLES = {"identity", "pose", "identity_mask", "style_mask", "style", "audit_source"}
MAX_OUTFITS = 3


def outfit_roles(profile):
    """의상 참조마다 그림·마스크 역할 이름(주석 순서 = 어댑터 순서)."""
    return [(f"outfit_{row['name']}", f"outfit_{row['name']}_mask") for row in profile.get("outfits", [])]


class CharacterInputError(ValueError):
    """원본 보존 상태로 중단해야 하는 입력 오류."""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_rgb_snapshot(path, expected_sha):
    """검증한 동일 바이트를 디코딩해 검사와 사용 사이 파일 변경을 피한다."""
    data = Path(path).read_bytes()
    if sha256_bytes(data) != expected_sha:
        raise CharacterInputError(f"source_sha_mismatch: {path}")
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_box(box, canvas, name="face_box"):
    if (not isinstance(box, list) or len(box) != 4
            or any(type(v) is not int for v in box)):
        raise CharacterInputError(f"{name} must contain four integer pixel coordinates")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= canvas[0] and 0 <= y0 < y1 <= canvas[1]):
        raise CharacterInputError(f"{name}_outside_source")


def validate_annotations(profile, canvas):
    """좌우는 화면 기준이다. 좌표 출처와 미확인 항목을 유지하며 검사한다."""
    if profile.get("schema_version") != 1 or profile.get("canvas") != list(canvas):
        raise CharacterInputError("annotation_canvas_or_version_mismatch")
    if profile.get("coordinate_system") != "image_xy":
        raise CharacterInputError("coordinate_system must be image_xy")
    if profile.get("height_cm") is not None:
        raise CharacterInputError("physical_height_not_measured")
    validate_box(profile.get("face_box"), canvas)
    if "identity_mask_box" in profile:              # 생성 자세의 얼굴 자리(원본 얼굴 상자와 다를 때, D-201)
        validate_box(profile["identity_mask_box"], canvas, "identity_mask_box")
    outfits = profile.get("outfits", [])
    if not isinstance(outfits, list) or len(outfits) > MAX_OUTFITS:
        raise CharacterInputError("outfits_invalid")
    for row in outfits:
        # scale(선택): 이 참조만의 강도. 없으면 생성 옵션의 outfit_scale(머리 참조는 0.7, D-200).
        if (not isinstance(row, dict) or not {"name", "sha256", "box"} <= set(row) <= {"name", "sha256", "box", "scale"}
                or not re.fullmatch(r"[a-z]{1,12}", str(row["name"]))):
            raise CharacterInputError("outfits_invalid")
        if "scale" in row and (type(row["scale"]) not in (int, float) or not math.isfinite(row["scale"]) or not 0 <= row["scale"] <= 1):
            raise CharacterInputError("outfits_invalid")
        validate_box(row["box"], canvas, "outfit_box")
    if len({row["name"] for row in outfits}) != len(outfits):
        raise CharacterInputError("outfit_names_duplicated")
    traits = profile.get("traits", [])
    names = [row.get("name") for row in traits]
    if len(names) != len(set(names)) or not REQUIRED_TRAITS.issubset(names):
        raise CharacterInputError("required_traits_missing_or_duplicated")
    for row in traits:
        # 재지 못한 항목(unmeasured)만 문장이 비어도 된다 — 문장에서 빠지고 참조 그림에 맡긴다.
        keys = ("name", "description_ko", "evidence") if row.get("status") == "unmeasured" else ("name", "text", "description_ko", "evidence")
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in keys) or not isinstance(row.get("text"), str):
            raise CharacterInputError("trait_evidence_missing")
        if row.get("status") not in ("operator_observation", "user_confirmed", "automatic_measurement", "unmeasured"):
            raise CharacterInputError("trait_status_missing")
        if row["status"] == "unmeasured" and row["text"].strip():
            raise CharacterInputError("unmeasured_trait_has_text")
    if not isinstance(profile.get("scene_prompt"), str) or not profile["scene_prompt"].strip():
        raise CharacterInputError("scene_prompt_missing")
    if not isinstance(profile.get("style_prompt"), str) or not profile["style_prompt"].strip():
        raise CharacterInputError("style_prompt_missing")
    if not isinstance(profile.get("negative_terms", ""), str) or len(profile.get("negative_terms", "")) > 120:
        raise CharacterInputError("negative_terms_invalid")
    layout = profile.get("layout", {})
    if layout.get("origin") not in ("manual_image_annotation", "detector"):
        raise CharacterInputError("layout_origin_missing")
    if not layout.get("evidence"):
        raise CharacterInputError("layout_evidence_missing")
    points = layout.get("joints", {})
    if len(points) < 12 or not set(points).issubset(BODY_JOINT_NAMES):
        raise CharacterInputError("layout_joints_missing_or_unknown")
    for xy in points.values():
        if (not isinstance(xy, list) or len(xy) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in xy)
                or not 0 <= xy[0] < canvas[0] or not 0 <= xy[1] < canvas[1]):
            raise CharacterInputError("layout_joint_outside_source")


def render_pose_map(layout, canvas):
    """주석 좌표를 기존 OpenPose 렌더러로 그린다. 원본 RGB·명암은 읽지 않는다."""
    from genai_lab.pose_guidance import draw_skeleton

    points = layout["joints"]
    records = [
        {"joint_name": name, "x": points.get(name, [0, 0])[0],
         "y": points.get(name, [0, 0])[1], "detected": name in points}
        for name in BODY_JOINT_NAMES
    ]
    return Image.fromarray(draw_skeleton(records, canvas)).convert("RGB")


def write_input_assets(source, style, profile, output, outfits=()):
    """참조/마스크/구조 지도를 저장한다. 색 복원 또는 생성 후 합성은 하지 않는다.
    의상 참조(D-195)는 몸 영역에만 적용하는 시각 참조다. 윗옷·아래옷을 나누면 각자 영역에만 적용한다(D-200)."""
    face = source.crop(tuple(profile["face_box"]))
    face.save(output / "identity-face.png")
    pose = render_pose_map(profile["layout"], source.size)
    pose.save(output / "structure-pose.png")
    face_mask = Image.new("L", source.size, 0)
    # 얼굴 참조는 원본의 얼굴 상자로 자르고, 생성 자세의 얼굴 자리(identity_mask_box)에 적용한다. 없으면 원본 얼굴 상자 자리.
    x0, y0, x1, y1 = profile.get("identity_mask_box", profile["face_box"])
    ImageDraw.Draw(face_mask).rectangle((x0, y0, x1-1, y1-1), fill=255)
    face_mask.save(output / "identity-mask.png")
    Image.new("L", source.size, 255).save(output / "style-mask.png")
    style.save(output / "style-reference.png")
    source.save(output / "source-audit-only.png")
    names = {
        "identity": "identity-face.png", "pose": "structure-pose.png",
        "identity_mask": "identity-mask.png", "style_mask": "style-mask.png",
        "style": "style-reference.png", "audit_source": "source-audit-only.png",
    }
    for row, image, (role, mask_role) in zip(profile.get("outfits", []), outfits, outfit_roles(profile)):
        image.save(output / f"outfit-{row['name']}.png")
        outfit_mask = Image.new("L", source.size, 0)
        x0, y0, x1, y1 = row["box"]
        ImageDraw.Draw(outfit_mask).rectangle((x0, y0, x1-1, y1-1), fill=255)
        outfit_mask.save(output / f"outfit-{row['name']}-mask.png")
        names.update({role: f"outfit-{row['name']}.png", mask_role: f"outfit-{row['name']}-mask.png"})
    return names


def prepare_character_style_bundle(source_path, style_path, profile, output, outfit_paths=None):
    """검증 성공 시 새 폴더만 만든다. 기존 결과·원본을 덮어쓰지 않는다.
    outfit_paths: {의상 참조 이름: 그림 경로}. 주석의 outfits와 이름이 같아야 한다."""
    profile = deepcopy(profile)
    outfit_paths = dict(outfit_paths or {})
    rows = profile.get("outfits", []) if isinstance(profile.get("outfits", []), list) else []
    if set(outfit_paths) != {row.get("name") for row in rows if isinstance(row, dict)}:
        raise CharacterInputError("outfit_reference_and_annotation_must_match")
    source = read_rgb_snapshot(source_path, profile["source_sha256"])
    style = read_rgb_snapshot(style_path, profile["style_sha256"])
    outfits = []
    try:
        validate_annotations(profile, source.size)
        for row in profile.get("outfits", []):
            outfits.append(read_rgb_snapshot(outfit_paths[row["name"]], row["sha256"]))
        if profile["source_sha256"] == profile["style_sha256"]:
            raise CharacterInputError("source_and_style_must_be_separate")
        if any(size % 8 for size in source.size):
            raise CharacterInputError("output_size_must_be_multiple_of_8")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        names = write_input_assets(source, style, profile, output, outfits)
        assets = {
            role: {"file": name, "sha256": sha256_bytes((output/name).read_bytes())}
            for role, name in names.items()
        }
        manifest = {
            "schema_version": 1, "state": "cpu_inputs_prepared",
            "source_sha256": profile["source_sha256"], "style_sha256": profile["style_sha256"],
            "canvas": list(source.size), "profile": profile, "assets": assets,
            "routing": {
                "original_rgb_as_initial_latent": False,
                "identity_visual": "face_crop_via_region_masked_ip_adapter",
                "structure": "pose_controlnet_input", "style": "separate_style_ip_adapter",
                "adapter_order": ["style", "identity"] + [role for role, _ in outfit_roles(profile)],
                "source_color_restoration": False, "original_background_composite": False,
                "audit_source_sent_to_model": False,
            },
            "limits": [
                "Face crop still contains source style; style-free identity is NOT claimed.",
                "Sparse pose is not a facial identity or exact stature representation.",
                "Manual annotations are not detector measurements or user quality approval.",
                "Appearance and style conditioning may interfere inside the model.",
                "Loaded model compatibility and GPU image quality remain unverified.",
            ],
            "review": {
                "character_preservation": "not_generated",
                "style_application": "not_generated", "user_quality_approval": None,
            },
        }
        write_json(output/"manifest.json", manifest)
        return manifest
    finally:
        source.close()
        style.close()
        for image in outfits:
            image.close()


def read_verified_bundle(folder):
    """생성 입력 파일의 경로·SHA·좌표를 다시 검사하고 manifest와 이미지를 반환한다."""
    folder = Path(folder).resolve()
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise CharacterInputError("bundle_version_unsupported")
    validate_annotations(manifest["profile"], tuple(manifest["canvas"]))
    images = {}
    try:
        for role, asset in manifest["assets"].items():
            path = (folder/asset["file"]).resolve()
            if path.parent != folder or path.is_symlink():
                raise CharacterInputError("asset_path_escapes_bundle")
            images[role] = read_rgb_snapshot(path, asset["sha256"])
        roles = outfit_roles(manifest["profile"])
        if set(images) != BASE_ROLES | {r for pair in roles for r in pair}:
            raise CharacterInputError("asset_roles_mismatch")
        canvas = tuple(manifest["canvas"])
        sized = ("pose", "identity_mask", "style_mask", "audit_source") + tuple(mask for _, mask in roles)
        if any(images[r].size != canvas for r in sized):
            raise CharacterInputError("asset_dimensions_mismatch")
        return manifest, images
    except BaseException:
        for im in images.values():
            im.close()
        raise
