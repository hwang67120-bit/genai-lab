"""Raw -> hires -> face detail, preserving raw records and explicit failure states."""
from dataclasses import asdict
from pathlib import Path
import io
import json
import math
import time
from PIL import Image, ImageFilter
from genai_lab.onepass_generation import OnePassBatch, OnePassCancelled, validate_prompt
from genai_lab.proportion_inputs import sha, require, json_value
from genai_lab.qwen_record_io import write_json
from genai_lab.finishing_backend import FinishingBackend
from genai_lab.finishing_reference import checked_reference

FINAL_SIZE = (1104,1848)
CONTRACT = {"scale":1.5,"strength":.35,"steps":28,"cfg":5.5,"face_size":[1024,1024],
            "head_model":"head_detect_v2.0_s","head_score_min":.5,"crop_scale":1.6,
            "ip_adapter":"not_loaded","max_reserved_gib":6.5,"vae_tiling":True}


def face_region(heads, size):
    candidates = [h for h in heads if len(h)==3 and math.isfinite(h[2]) and h[2]>=.5]
    if not candidates:
        return None
    box, _, score = max(candidates,key=lambda h:h[2])
    require(len(box)==4 and all(math.isfinite(v) for v in box), "머리 검출 좌표가 유효하지 않습니다.")
    x0,y0,x1,y1=box
    require(x1>x0 and y1>y0, "머리 검출 상자가 비어 있습니다.")
    cx,cy,side=(x0+x1)/2,(y0+y1)/2,max(x1-x0,y1-y0)*1.6
    crop=(int(max(0,cx-side/2)),int(max(0,cy-side/2)),int(min(size[0],cx+side/2)),int(min(size[1],cy+side/2)))
    require(crop[2]>crop[0] and crop[3]>crop[1], "머리 보정 영역이 이미지 밖입니다.")
    return {"head_box":list(box),"head_score":score,"face_box":list(crop)}


def blend_face(image, crop_box, fixed):
    x0,y0,x1,y1=crop_box
    size=(x1-x0,y1-y0)
    margin=max(2,min(size)//8)
    require(size[0]>margin*2 and size[1]>margin*2, "머리 보정 영역이 너무 작습니다.")
    with fixed.resize(size,Image.Resampling.LANCZOS) as patch, Image.new("L",size,0) as mask:
        mask.paste(255,(margin,margin,size[0]-margin,size[1]-margin))
        with mask.filter(ImageFilter.GaussianBlur(margin/2)) as soft:
            result=image.copy()
            result.paste(patch,(x0,y0),soft)
    return result


def detect_head_file(path, runtime, directory, cancelled):
    from genai_lab.studio_generation import cpu_process
    output=directory/"heads.json"
    worker=Path(__file__).with_name("finishing_head_worker.py")
    cpu_process([str(runtime.analysis_python),str(worker),str(path),str(output),str(runtime.head_cache)],
                directory/"heads.log",cancelled,runtime.analysis_timeout)
    return json.loads(output.read_text(encoding="utf-8"))["heads"]


def check_cancel(cancelled):
    if cancelled():
        raise OnePassCancelled("마무리를 취소했습니다. 완료된 파일은 보관됩니다.")


def validate_candidate(candidate, prompt):
    record=json.loads(candidate.record_path.read_text(encoding="utf-8"))
    require(record.get("valid") and record.get("completed") and sha(candidate.path)==record["raw_sha256"],
            "마무리 전 생성 원본이 변경됐습니다.")
    require(record["prompt"]["positive"]==prompt.positive and record["prompt"]["negative"]==prompt.negative,
            "생성에 쓴 문장과 마무리 문장이 다릅니다.")
    with Image.open(candidate.path) as image:
        require(image.size==(736,1232) and image.mode=="RGB", "마무리는 736×1232 RGB 원본만 받습니다.")


def finish_candidate(candidate, prompt, backend, detect, cancelled, *, reference=None):
    directory=candidate.path.parent
    record_path=directory/"finishing.json"
    require(not record_path.exists() and not (directory/"finished_hires.png").exists()
            and not (directory/"finished.png").exists(), "기존 마무리 결과를 덮어쓰지 않습니다.")
    validate_candidate(candidate,prompt)
    digest=sha(candidate.path)
    record={"status":"started","phase":"hires","raw_sha256":digest,"seed":candidate.seed,
            "prompt":asdict(prompt),"contract":dict(CONTRACT),"final_size":list(FINAL_SIZE),
            "face_detail":"pending","stages":{}}
    record.update(reference.record() if reference is not None else
                  {"face_reference":"off", "ip_scale":0.0, "face_sha256":None})
    if reference is not None:
        record["contract"]["ip_adapter"] = "loaded"
        record["face_embedding_preparation"] = getattr(backend,"face_preparation",None)
        record["offload"] = getattr(backend,"offload_record",None)
    started=time.perf_counter()
    try:
        check_cancel(cancelled)
        with Image.open(io.BytesIO(candidate.path.read_bytes())) as raw:
            require(raw.size==(736,1232) and raw.mode=="RGB", "마무리는 736×1232 RGB 원본만 받습니다.")
            with raw.resize(FINAL_SIZE,Image.Resampling.LANCZOS) as enlarged:
                hires,metrics=backend.refine(enlarged,candidate.seed,cancelled)
        try:
            require(hires.size==FINAL_SIZE,"고해상도 마무리 출력 크기 불일치")
            hires_file=directory/"finished_hires.png";hires.save(hires_file)
            record["stages"]["hires"]={**metrics,"sha256":sha(hires_file),"file":str(hires_file)}
            record["phase"]="head_detection"
            write_json(record_path,json_value(record))
            detect_start=time.perf_counter()
            region=face_region(detect(hires_file,directory,cancelled),hires.size)
            record["head_detection_seconds"]=time.perf_counter()-detect_start
            check_cancel(cancelled)
            if region:
                record.update(region);record["phase"]="face_detail"
                with hires.crop(tuple(region["face_box"])) as crop, crop.resize((1024,1024),Image.Resampling.LANCZOS) as enlarged:
                    fixed,metrics=backend.refine(enlarged,candidate.seed,cancelled)
                try:
                    require(fixed.size==(1024,1024),"얼굴 보정 출력 크기 불일치")
                    finished=blend_face(hires,region["face_box"],fixed)
                finally: fixed.close()
                record["face_detail"]="completed"
                record["stages"]["face"]=metrics
            else:
                finished=hires.copy()
                record.update(face_detail="skipped",face_skip_reason="no_head_at_threshold")
            with finished:
                finished_file=directory/"finished.png";finished.save(finished_file)
            record["finished_sha256"]=sha(finished_file)
            record["finished_file"]=str(finished_file)
            require(sha(candidate.path)==digest,"마무리 중 생성 원본이 변경됐습니다.")
            check_cancel(cancelled)
            record.update(status="completed",phase="done")
            return record
        finally: hires.close()
    except BaseException as error:
        if record["phase"] in ("hires", "face_detail") and getattr(backend, "last_refine", None) is not None:
            record["failed_stage_metrics"] = backend.last_refine
        record.update(status="cancelled" if isinstance(error,OnePassCancelled) else "failed",
                      error_type=type(error).__name__,error=str(error))
        raise
    finally:
        record["seconds"]=time.perf_counter()-started
        write_json(record_path,json_value(record))


def finish_batch(batch, prompt, runtime, *, cancelled=lambda:False, progress=lambda _:None,
                 backend_factory=FinishingBackend, detector=None, face_file=None):
    require(isinstance(batch,OnePassBatch),"마무리는 1회 생성 경로에서만 사용합니다.")
    validate_prompt(prompt)
    require((runtime.generation.width,runtime.generation.height,runtime.generation.steps,
             runtime.generation.guidance_scale,runtime.generation.max_reserved_gib)==(736,1232,28,5.5,6.5),
            "마무리 시험과 다른 생성 설정입니다.")
    status_path=batch.directory/"finishing-status.json"
    require(not status_path.exists(),"기존 마무리 실행을 덮어쓰지 않습니다.")
    state={"status":"started","completed_seeds":[],"contract":dict(CONTRACT),
           "model_root":str(runtime.generation.model_root),"quality_tags":runtime.quality_tags}
    backend=None
    try:
        check_cancel(cancelled)
        for candidate in batch.candidates:
            validate_candidate(candidate,prompt)
        reference = checked_reference(batch,runtime.generation,face_file=face_file) if runtime.finishing_face_reference else None
        state.update(reference.record() if reference else {"face_reference":"off"})
        if reference:
            state["contract"]["ip_adapter"] = "loaded"
        if detector is None:
            require(runtime.head_cache.is_dir() and runtime.analysis_python.is_file(), "오프라인 머리 검출 환경이 없습니다.")
        backend=backend_factory(runtime.generation,face_reference=True) if reference else backend_factory(runtime.generation)
        if reference:
            backend.set_face_reference(reference)
        backend.prepare(prompt,cancelled)
        if reference:
            state["face_embedding_preparation"] = getattr(backend,"face_preparation",None)
            state["offload"] = getattr(backend,"offload_record",None)
        detect=detector or (lambda path,directory,cancel:detect_head_file(path,runtime,directory,cancel))
        for index,candidate in enumerate(batch.candidates):
            progress(f"상태: 마무리 중 · {index+1}/{len(batch.candidates)} · 고해상도·얼굴 보정")
            finish_candidate(candidate,prompt,backend,detect,cancelled,reference=reference)
            state["completed_seeds"].append(candidate.seed)
            write_json(status_path,state)
        state["status"]="completed"
        return batch
    except BaseException as error:
        offload_record = getattr(backend,"offload_record",None) or getattr(error,"finishing_offload",None)
        if offload_record is not None:
            state["offload"] = offload_record
        if backend is not None and getattr(backend,"face_preparation",None) is not None:
            state["face_embedding_preparation"] = backend.face_preparation
        state.update(status="cancelled" if isinstance(error,OnePassCancelled) else "failed",error=str(error))
        raise
    finally:
        try:
            if backend is not None: backend.close()
        finally: write_json(status_path,state)


def read_finishing(candidate):
    path=candidate.path.parent/"finishing.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def finishing_source(candidate):
    record=read_finishing(candidate)
    require(record is not None and record["status"]=="completed","마무리가 정상 완료되지 않았습니다.")
    require(sha(candidate.path)==record["raw_sha256"],"마무리 기준 원본이 변경됐습니다.")
    for name,digest in (("finished_hires.png",record["stages"]["hires"]["sha256"]),("finished.png",record["finished_sha256"])):
        require(sha(candidate.path.parent/name)==digest,"마무리 이미지가 변경됐습니다.")
    return candidate.path.parent/"finished.png",record
