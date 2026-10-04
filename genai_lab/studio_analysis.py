"""Offline CPU preparation executable; no model provisioning or global chdir.

DWPose uses its already installed Wholebody with explicit local ONNX paths;
C6, WD feature routing and isnet use the same production functions as tests.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from dataclasses import asdict
import hashlib
from PIL import Image


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def pose(directory, models):
    from easy_dwpose.body_estimation import Wholebody
    from genai_lab.onepass_input_backends import DwPoseInputBackend
    for name in ("yolox_l.onnx", "dw-ll_ucoco_384.onnx"):
        if not (models / name).is_file():
            raise FileNotFoundError(f"설치된 자세 분석 모델이 없습니다: {models / name}")
    class Detector:
        pose_estimation = Wholebody(str(models / "yolox_l.onnx"),
                                    str(models / "dw-ll_ucoco_384.onnx"), device="cpu")
    with Image.open(directory / "character.png") as image:
        observation = DwPoseInputBackend(Detector())(image)
    try:
        save(directory / "pose.json", {
            "person_count": observation.person_count,
            "joints": [asdict(j) for j in observation.joints],
            "face_points": observation.face_points, "face_scores": observation.face_scores})
    finally:
        observation.close()


BODY = frozenset(("small_breasts", "medium_breasts", "large_breasts", "flat_chest",
    "muscular", "muscular_male", "muscular_female", "abs", "tall", "thick_thighs",
    "pectorals", "broad_shoulders"))


def classify_character(tags):
    groups = {"appearance": [], "body": [], "fixed": []}
    for tag, _score in tags:
        key = "body" if tag in BODY else ("fixed" if tag == "tail" or tag.endswith(("_ears", "_tail")) else "appearance")
        groups[key].append(tag)
    return groups


def features(directory, cache):
    # Import onnxruntime first: imgutils must not attempt its optional auto-install.
    import onnxruntime
    from imgutils.detect import detect_heads
    from genai_lab.onepass_input_backends import HeadInputBackend, ForegroundInputBackend
    from genai_lab.onepass_character import prepare_c6_crop, select_character_tags, measure_slimness
    from genai_lab.onepass_pose import joints_from_records
    from scripts.body_comparison_runner import extract_anime_character_foreground_mask
    from genai_lab.clothing_analysis import WdTagSession, ClothingDesignAnalysisSettings
    from genai_lab.garment_detail_analysis import detail_group
    from genai_lab.onepass_garment_vocabulary import garment_nouns
    from genai_lab.reference_tag_policy import normalize_tag, excluded_garment_tag
    observed = json.loads((directory / "pose.json").read_text(encoding="utf-8"))
    if observed["person_count"] != 1:
        raise ValueError("캐릭터가 한 명 보이는 이미지를 선택해 주세요.")
    joints = joints_from_records(observed["joints"])
    with Image.open(directory / "character.png") as source:
        heads = HeadInputBackend(detect_heads)(source)
        foreground = ForegroundInputBackend(extract_anime_character_foreground_mask, cache)(source)
        try:
            crop = prepare_c6_crop(source, heads, joints, foreground_mask=foreground,
                face_points=observed["face_points"], face_scores=observed["face_scores"])
        finally:
            foreground.close()
        try:
            if crop.status != "review":
                raise ValueError("캐릭터 얼굴을 준비하지 못했습니다. " + str(crop.reason))
            crop.image.save(directory / "face.png")
            crop.mask.save(directory / "face-mask.png")
            crop_details = crop.details
        finally:
            crop.close()
        settings = ClothingDesignAnalysisSettings(cache_dir=cache, local_files_only=True)
        with WdTagSession(settings) as tagger:
            character_scores = dict(tagger.analyze(source).raw_general_scores)
            with Image.open(directory / "garment.png") as garment:
                garment_scores = dict(tagger.analyze(garment).raw_general_scores)
    selected = select_character_tags(character_scores)
    groups = classify_character(selected.appearance)
    garments = [normalize_tag(tag) for tag, score in sorted(garment_scores.items(), key=lambda x: -x[1])
        if score >= settings.score_threshold and not excluded_garment_tag(tag)
        and (garment_nouns((tag,)) or detail_group(tag) in
             ("component_candidate", "construction", "shape", "surface", "style_context"))]
    if not any(groups.values()):
        raise ValueError("キャ릭터 외형을 읽지 못했습니다. 얼굴과 머리가 선명한 다른 이미지를 선택해 주세요.")
    if not garments:
        raise ValueError("옷을 인식하지 못했습니다. 옷이 잘 보이는 다른 이미지를 선택해 주세요.")
    save(directory / "analysis.json", {"groups": groups, "garment_tags": garments,
        "slim": measure_slimness(joints)["slim"], "crop_details": crop_details,
        "face_sha256": hashlib.sha256((directory / "face.png").read_bytes()).hexdigest(),
        "character_scores": character_scores, "garment_scores": garment_scores,
        "analysis_device": "cpu", "models_downloaded": False})


if __name__ == "__main__":
    mode, directory, models, cache, head_cache = sys.argv[1:]
    import os
    os.environ["HF_HUB_CACHE"] = head_cache
    try:
        if mode == "pose":
            pose(Path(directory), Path(models))
        elif mode == "features":
            features(Path(directory), Path(cache))
        else:
            raise ValueError("알 수 없는 분석 단계")
    except Exception as error:
        save(Path(directory) / "analysis-error.json", {"stage": mode,
             "message": str(error), "error_type": type(error).__name__})
        raise
