"""오프라인 CPU에서 자세·SAM 머리 초안을 만든다. 생성 전 각 작업을 종료한다."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from dataclasses import asdict, replace
from types import SimpleNamespace
import numpy as np
from PIL import Image
from genai_lab.proportion_inputs import prepare_head_outline, validate_head, neck_gap_only, sha, require, json_value
from genai_lab.qwen_record_io import write_json


def save_pose(request, directory, estimate):
    from genai_lab.onepass_pose import prepare_onepass_pose, PoseInputError
    require(sha(request["source_file"]) == request["source_sha256"], "원본 사본이 변경됐습니다.")
    try:
        with Image.open(request["source_file"]) as source:
            prepared = prepare_onepass_pose(source, estimate=estimate, tag=lambda _: request["tag_scores"])
    except PoseInputError as error:
        try:
            write_json(directory / "pose-rejected.json", json_value({"decision": asdict(error.decision),
                       "assessment": asdict(error.assessment), "error": str(error)}))
        finally:
            error.close()
        raise
    try:
        for name, image in (("normalized", prepared.normalized_image), ("control", prepared.control_image),
                            ("overlay", prepared.overlay), ("original-overlay", prepared.original_overlay)):
            image.save(directory / (name + ".png"))
        joints = {j.joint_name: j for j in prepared.observation_joints}
        require(all(name in joints and joints[name].detected and joints[name].confidence_score >= .30
                    for name in ("nose", "neck")), "코·목 위치를 확인할 수 없습니다. 비율 참고를 끄거나 다른 이미지를 선택해 주세요.")
        result = {"source_file": request["source_file"], "source_sha256": request["source_sha256"],
            "decision": asdict(prepared.decision), "assessment": asdict(prepared.assessment),
            "face_points": prepared.face_points, "face_scores": prepared.face_scores,
            "crop_box": prepared.crop_box, "joints": [asdict(j) for j in prepared.observation_joints],
            "ip_early": .5,  # 정면 입력을 포함해 검증된 머리 윤곽 시험의 얼굴 보호 규칙을 유지한다.
            "initial_ip_policy": "head_contour_trial_0.5",
            "nose": [joints["nose"].x, joints["nose"].y], "neck": [joints["neck"].x, joints["neck"].y],
            "normalized_file": str(directory / "normalized.png"), "normalized_sha256": sha(directory / "normalized.png"),
            "control_file": str(directory / "control.png"), "control_sha256": sha(directory / "control.png"),
            "overlay_file": str(directory / "overlay.png"), "device": "cpu", "confirmed": False}
        write_json(directory / "result.json", json_value(result))
        from genai_lab.studio_proportion import verify_pose
        verify_pose(result)
        return result
    finally:
        prepared.close()


def point_mask(segmenter, image, point):
    import torch
    with torch.inference_mode():
        inputs = segmenter.processor(images=image, input_points=[[[list(point)]]],
            input_labels=[[[1]]], return_tensors="pt")
        output = segmenter.model(**inputs, multimask_output=False)
        masks = segmenter.processor.post_process_masks(output.pred_masks.cpu(),
            inputs["original_sizes"].cpu(), binarize=True)[0]
    return np.asarray(masks).reshape(-1, image.height, image.width)[0].astype(bool)


def merge_head_parts(hair, face, box):
    """시험 초안 규칙: 상자 내부 얼굴과 아래 15px를 추가한 뒤 가장 큰 성분만 유지한다."""
    import cv2
    x0,y0,x1,y1 = box
    face = np.asarray(face, dtype=np.uint8).copy()
    face[:, :x0] = 0; face[:, x1:] = 0; face[:y0] = 0; face[y1+15:] = 0
    mask = np.maximum(np.asarray(hair, dtype=np.uint8), face)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    require(count > 1, "머리 영역을 찾지 못했습니다. 범위를 다시 지정해 주세요.")
    return (labels == 1 + int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))).astype(np.uint8) * 255


def save_head(request, directory, segmenter, point_segment=point_mask):
    from genai_lab.studio_proportion import verify_pose
    from genai_lab.qwen_tail_edit import validate_box
    pose = request["pose"]
    verify_pose(pose)
    box, point = tuple(request["box"]), tuple(request["face_point"])
    validate_box(box, (736,1232))
    require(len(point) == 2 and np.isfinite(point).all() and box[0] <= point[0] < box[2]
            and box[1] <= point[1] < box[3], "얼굴 중심점이 선택한 머리 범위 안에 있어야 합니다.")
    with Image.open(pose["normalized_file"]) as image:
        hair, score = segmenter.segment(image, box)
        face = point_segment(segmenter, image, point)
    mask = merge_head_parts(hair, face, box)
    mask_file = directory / "head-mask.png"
    Image.fromarray(mask).save(mask_file)
    head = prepare_head_outline(pose["normalized_file"], pose["control_file"], mask_file,
        directory / "preview", nose=pose["nose"], neck=pose["neck"])
    error = None
    try:
        # 좌표 유효성만 검사한다. 저장한 초안은 미확인 상태다.
        validate_head(replace(head, confirmed=True), SimpleNamespace(control_sha256=pose["control_sha256"]), (736,1232))
    except ValueError as failure:
        error = str(failure)
    result = {"head": asdict(head), "box": box, "face_point": point, "geometry_error": error,
              "geometry_overridable": neck_gap_only(error),
              "preview_file": str(directory / "preview/preview.png"), "device": "cpu",
              "method": "sam_box_plus_face_point_largest_component", "sam_iou": score,
              "model_snapshot": str(getattr(segmenter,"snapshot","test")), "status": "needs_user_review"}
    write_json(directory / "result.json", json_value(result))
    return json_value(result)


def main():
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", ONNX_MODE="cpu")
    mode, filename = sys.argv[1:]
    directory = Path(filename).parent
    request = json.loads(Path(filename).read_text(encoding="utf-8"))
    try:
        if mode == "pose":
            from easy_dwpose.body_estimation import Wholebody
            from genai_lab.onepass_input_backends import DwPoseInputBackend
            models = Path(request["pose_models"])
            for name in ("yolox_l.onnx","dw-ll_ucoco_384.onnx"):
                require((models/name).is_file(), "로컬 골격 모델이 없습니다.")
            detector = SimpleNamespace(pose_estimation=Wholebody(str(models/"yolox_l.onnx"),
                str(models/"dw-ll_ucoco_384.onnx"), device="cpu"))
            save_pose(request, directory, DwPoseInputBackend(detector))
        elif mode == "head":
            from genai_lab.tail_complexity_worker import CpuTailSegmenter
            save_head(request, directory, CpuTailSegmenter(request["model_cache"]))
        else:
            raise ValueError("알 수 없는 비율 준비 단계입니다.")
    except Exception as error:
        write_json(directory / "analysis-error.json", {"stage": mode, "message":str(error), "error_type":type(error).__name__})
        raise


if __name__ == "__main__":
    main()
