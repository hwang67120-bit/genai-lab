"""CPU에서 머리 선을 준비하고 검토한 스케치만 변경 없이 조립한다."""
from dataclasses import asdict
from pathlib import Path
import json
import time
import cv2
import numpy as np
from PIL import Image
from genai_lab.proportion_inputs import sha, require, checked_image, json_value
from genai_lab.qwen_record_io import write_json

RULES = {"gray":"RGB2GRAY","gaussian_sigma":1,"gaussian_kernel":[0,0],
         "canny":[100,200],"connectivity":8,"min_component_pixels":20,
         "line_dilation_kernel":[3,3],"line_dilation_iterations":1,
         "face_dilation_pixels":2,"face_area_max":.7,"nose_confidence_min":.3}


SKIN_RULES = {"method":"skin_color_v1", "sample_size":[7,7], "color_space":"Lab",
              "conversion":"RGB8_to_OpenCV_Lab8_then_L100_ab_centered",
              "distance":"delta_e76", "threshold":12, "comparison":"strict_less",
              "connectivity":8, "fill":"RETR_EXTERNAL"}


CHEEK_RULES = {**SKIN_RULES, "method": "skin_color_v2", "sample_location": "bilateral_cheeks_2_31_14_35",
               "component_seed": "both_cheeks", "missing_landmarks": "stop"}


def cheek_skin_candidates(rgb, head_mask, face_points, face_scores):
    from genai_lab.skin_tone import measure_cheeks
    measurement = measure_cheeks(rgb, face_points, face_scores)
    centers = measurement["centers"]
    require(all(head_mask[y, x] > 0 for x, y in centers), "cheek_outside_head: 볼 표본이 확인한 머리 밖입니다.")
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255
    lab[..., 1:] -= 128
    candidates = (np.linalg.norm(lab - np.asarray(measurement["lab"]), axis=2) < 12) & (head_mask > 0)
    return candidates, {**CHEEK_RULES, "sample_lab": measurement["lab"], "cheek_pixels": centers,
                        "candidate_pixels": int(candidates.sum())}


def connected_cheeks(candidates, centers):
    require(all(candidates[y, x] for x, y in centers), "cheek_not_candidate: 볼 픽셀이 피부색 후보에 포함되지 않습니다.")
    _, labels = cv2.connectedComponents(candidates.astype(np.uint8), connectivity=8)
    selected = {int(labels[y, x]) for x, y in centers}
    face = np.isin(labels, list(selected)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(face, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(face)
    cv2.drawContours(filled, contours, -1, 255, -1)
    return filled > 0


def skin_candidates(rgb, head_mask, point):
    """고정된 코 주변 7×7 표본을 측정하고 후보와 측정 근거를 반환한다."""
    x,y=(int(round(v)) for v in point)
    h,w=head_mask.shape
    require(3<=x<w-3 and 3<=y<h-3,"skin_sample_outside: 코 주변 7×7 표본이 이미지 밖입니다.")
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[...,0]*=100/255
    lab[...,1:]-=128
    sample=np.median(lab[y-3:y+4,x-3:x+4].reshape(-1,3),axis=0)
    candidates=(np.linalg.norm(lab-sample,axis=2)<SKIN_RULES["threshold"]) & (head_mask>0)
    details={**SKIN_RULES,"sample_lab":sample.tolist(),"nose_pixel":[x,y],
             "candidate_pixels":int(candidates.sum()),"nose_is_candidate":bool(candidates[y,x])}
    return candidates,details


def connected_skin(candidates, point):
    """코와 연결된 성분만 유지하고 눈·입의 빈 곳을 채운다. 떨어진 성분은 합치지 않는다."""
    x,y=(int(round(v)) for v in point)
    require(bool(candidates[y,x]),"nose_not_candidate: 코 픽셀이 피부색 후보에 포함되지 않습니다.")
    _,labels=cv2.connectedComponents(candidates.astype(np.uint8),connectivity=8)
    face=(labels==labels[y,x]).astype(np.uint8)*255
    contours,_=cv2.findContours(face,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    filled=np.zeros_like(face)
    cv2.drawContours(filled,contours,-1,255,-1)
    return filled>0


def extract_lines(rgb, head_mask):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray,(0,0),1),100,200)
    edges[head_mask==0] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(edges,connectivity=8)
    keep = np.zeros(count,dtype=bool)
    keep[1:] = stats[1:,cv2.CC_STAT_AREA] >= 20
    filtered = keep[labels].astype(np.uint8)*255
    thick = cv2.dilate(filtered,np.ones((3,3),np.uint8),iterations=1)
    thick[head_mask==0] = 0
    return thick


def checked_nose(joint, head):
    require(joint is not None and joint.get("detected") is True and
            np.isfinite(joint.get("confidence_score",float("nan"))) and
            joint["confidence_score"]>=.3,"nose_unavailable: 신뢰도 0.3 이상 코 관절이 없습니다.")
    point = (joint["x"],joint["y"])
    require(np.isfinite(point).all() and tuple(point)==tuple(head.nose),"nose_mismatch: 원본 머리 좌표와 코 관절이 다릅니다.")
    return point


def face_regions(face, head_mask):
    require(np.asarray(face).shape==head_mask.shape,"face_shape: 얼굴 마스크 크기가 다릅니다.")
    clipped = np.asarray(face,dtype=bool) & (head_mask>0)
    ratio = float(clipped.sum()/np.count_nonzero(head_mask))
    require(clipped.any(),"face_empty: 얼굴 영역이 비었습니다.")
    require(ratio<=.7,"face_too_large: 얼굴 영역이 머리의 70%를 초과했습니다.")
    expanded = cv2.dilate(clipped.astype(np.uint8),np.ones((5,5),np.uint8),iterations=1)>0
    return clipped.astype(np.uint8)*255, ((head_mask>0)&~expanded).astype(np.uint8)*255, ratio


def segment_face(segmenter, image, point, box):
    import torch
    require(all(p.device.type=="cpu" for p in segmenter.model.parameters()),"SAM2는 CPU만 사용합니다.")
    with torch.inference_mode():
        inputs=segmenter.processor(images=image,input_points=[[[list(point)]]],input_labels=[[[1]]],
                                   input_boxes=[[list(box)]],return_tensors="pt")
        output=segmenter.model(**inputs,multimask_output=False)
        masks=segmenter.processor.post_process_masks(output.pred_masks.cpu(),inputs["original_sizes"].cpu(),binarize=True)[0]
    return np.asarray(masks).reshape(-1,image.height,image.width)[0].astype(bool), float(output.iou_scores.cpu().numpy().ravel()[0])


def sam_record(segmenter):
    snapshot=Path(segmenter.snapshot)
    local_files=[p for p in sorted(snapshot.iterdir()) if p.is_file()]
    require(any(p.suffix==".safetensors" for p in local_files),"로컬 SAM2 가중치가 없습니다.")
    files={str(p.absolute()):sha(p) for p in local_files}
    return {"id":"facebook/sam2.1-hiera-tiny","revision":snapshot.name,"files":files,"device":"cpu"}


def save_overlay(rgb, hair, face, lines, path):
    overlay=rgb.copy()
    for mask,color in ((hair,(30,200,80)),(face,(255,120,20))):
        region=mask>0
        overlay[region]=(overlay[region].astype(float)*.7+np.asarray(color)*.3).astype(np.uint8)
    overlay[lines>0]=(255,0,255)
    Image.fromarray(overlay).save(path)


def prepare_head_lines(head, nose_joint, directory, segmenter_factory=None, *, face_method="skin_color_v2"):
    """두 마스크와 선 종류를 모두 검토용으로 저장한다. 여기서 승인하지 않는다."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    record={"status":"started","rules":RULES,"head":asdict(head),"gpu_generation":False,"files":{},"face_method":face_method}
    started=time.perf_counter()
    try:
        require(face_method in ("skin_color_v1","skin_color_v2","sam2"),"알 수 없는 얼굴 분리 방식입니다.")
        point=checked_nose(nose_joint,head)
        size=(736,1232)
        rgb=checked_image(head.normalized_file,head.normalized_sha256,size,"RGB")
        mask=checked_image(head.mask_file,head.mask_sha256,size,"L")
        require(np.isin(mask,[0,255]).all() and mask.any(),"확인된 머리 마스크가 비었습니다.")
        ys,xs=np.where(mask>0);box=(int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1))
        input_record={"point":point,"label":1,"box":box,"nose_confidence":nose_joint["confidence_score"]}
        record["face_input"]=input_record
        if face_method=="skin_color_v2":
            candidates, details = cheek_skin_candidates(rgb, mask, nose_joint.get("face_points", []), nose_joint.get("face_scores", []))
            record["face_segmentation"] = details
            face = connected_cheeks(candidates, details["cheek_pixels"])
        elif face_method=="skin_color_v1":
            candidates,details=skin_candidates(rgb,mask,point)
            record["face_segmentation"]=details
            face=connected_skin(candidates,point)
        else:
            require(segmenter_factory is not None,"SAM2 실행기가 없습니다.")
            record["sam_input"]=input_record
            segmenter=segmenter_factory()
            try:
                record["sam_model"]=sam_record(segmenter)
                with Image.fromarray(rgb) as image: face,score=segment_face(segmenter,image,point,box)
            finally:
                del segmenter
            record["sam_iou"]=score
        raw_face=np.asarray(face,dtype=bool)&(mask>0)
        record["face_area_ratio"]=float(raw_face.sum()/np.count_nonzero(mask))
        Image.fromarray(raw_face.astype(np.uint8)*255).save(directory/"face-raw.png")
        # 거부된 마스크도 진단을 위해 표시하며 생성에는 쓰지 않는다.
        save_overlay(rgb,mask,raw_face.astype(np.uint8)*255,np.zeros_like(mask),directory/"mask-preview.png")
        skin,hair,ratio=face_regions(face,mask)
        all_lines=extract_lines(rgb,mask)
        hair_lines=all_lines.copy();hair_lines[hair==0]=0
        for name,values in (("face",skin),("hair-mask",hair),("all",all_lines),("hair",hair_lines)):
            path=directory/(name+".png");Image.fromarray(values).save(path)
            record["files"][name]={"path":str(path.resolve()),"sha256":sha(path)}
        for mode,lines in (("hair",hair_lines),("all",all_lines)):
            path=directory/(mode+"-preview.png");save_overlay(rgb,hair,skin,lines,path)
            record["files"][mode+"-preview"]={"path":str(path.resolve()),"sha256":sha(path)}
        record["line_pixels"]={"hair":int(np.count_nonzero(hair_lines)),"all":int(np.count_nonzero(all_lines))}
        record.update(status="needs_user_review",face_area_ratio=ratio)
        return record
    except BaseException as error:
        record.update(status="failed",error_type=type(error).__name__,error=str(error))
        raise
    finally:
        record["seconds"]=time.perf_counter()-started
        write_json(directory/"head-lines.json",json_value(record))


def checked_lines(options, size, *, require_review=True):
    """검토한 선 픽셀을 반환한다. 선택이 none이면 파일을 열거나 인식하지 않는다."""
    if options.head_lines=="none":return None,None
    require(options.outline_source=="base","머리 안쪽 선은 기존 1단계 몸 윤곽에서만 시험합니다.")
    require(not require_review or options.head_lines_confirmed is True,"머리 안쪽 선 미리보기 사용자 확인이 필요합니다.")
    require(options.head_lines_file is not None and options.head_lines_sha256 is not None,"머리 선 잠금 기록이 없습니다.")
    path=Path(options.head_lines_file)
    require(sha(path)==options.head_lines_sha256,"머리 선 기록 SHA 불일치")
    record=json.loads(path.read_text(encoding="utf-8"))
    require(record["status"]=="needs_user_review" and record["rules"]==RULES,"머리 선 규칙·준비 상태가 다릅니다.")
    require(record["head"]==json_value(asdict(options.head)),"다른 머리의 안쪽 선입니다.")
    method=record.get("face_method","sam2")
    require(method in ("skin_color_v1","skin_color_v2","sam2"),"얼굴 분리 방식 기록이 다릅니다.")
    if method=="sam2":
        for path,digest in record["sam_model"]["files"].items():require(sha(path)==digest,"SAM2 파일이 변경됐습니다.")
    else:
        details=record["face_segmentation"]
        expected_rules = CHEEK_RULES if method == "skin_color_v2" else SKIN_RULES
        require(all(details.get(k)==v for k,v in expected_rules.items()),"피부색 분리 규칙이 변경됐습니다.")
    for item in record["files"].values():require(sha(item["path"])==item["sha256"],"머리 선·미리보기 입력이 변경됐습니다.")
    item=record["files"][options.head_lines]
    pixels=checked_image(item["path"],item["sha256"],size,"L")
    mask=checked_image(options.head.mask_file,options.head.mask_sha256,size,"L")
    require(np.isin(pixels,[0,255]).all() and not np.any(pixels[mask==0]),"머리 선이 확인된 영역 밖에 있습니다.")
    return pixels,{"mode":options.head_lines,"record_file":str(options.head_lines_file),
                   "record_sha256":options.head_lines_sha256,"line_sha256":item["sha256"],
                   "line_pixels":record["line_pixels"][options.head_lines],"rules":RULES,
                   "face_area_ratio":record["face_area_ratio"], "face_method":method,
                   **({"sam_input":record["sam_input"],"sam_model":record["sam_model"]} if method=="sam2" else
                      {"face_input":record["face_input"],"face_segmentation":record["face_segmentation"]})}


def compose_head_lines(sketch, options, *, require_review=True):
    lines,record=checked_lines(options,(sketch.shape[1],sketch.shape[0]),require_review=require_review)
    if lines is None:return sketch,None
    return np.maximum(sketch,lines[...,None]),record
