"""Local offline recognition worker; imports models only in main's inference path."""
import json
from pathlib import Path
import sys
import time
import traceback

from genai_lab.qwen_preservation import file_sha, json_sha
from genai_lab.qwen_record_io import write_json
from genai_lab.tail_recognition import (RecognitionSettings, PROMPTS, validate_fields, validate_model,
                                       check_versions, observation_key, filter_observation)


def parse_observation(text, role):
    stripped=text.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        stripped=stripped[7:-3].strip()
    fields=json.loads(stripped)
    return validate_fields(role,fields)


def inspect_image(model, processor, torch, job, settings):
    """One image at a time bounds peak image tokens; raw response is kept even if invalid."""
    from PIL import Image
    started=time.monotonic()
    if file_sha(job["path"])!=job["image_sha256"]: raise ValueError("인식 입력 SHA 불일치")
    with Image.open(job["path"]) as source:
        image=source.convert("RGB")
    try:
        messages=[{"role":"user","content":[{"type":"image"},{"type":"text","text":PROMPTS[job["role"]]}]}]
        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=processor(text=[text],images=[image],return_tensors="pt").to(model.device)
        input_count=int(inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            generated=model.generate(**inputs,max_new_tokens=settings.max_new_tokens,do_sample=False)
        raw=processor.batch_decode(generated[:,input_count:],skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)[0]
        metrics={"seconds":time.monotonic()-started,"input_tokens":input_count,
                 "output_tokens":int(generated.shape[-1])-input_count,"source_size":list(image.size),
                 "image_grid_thw":inputs["image_grid_thw"].tolist()}
        if metrics["output_tokens"]>=settings.max_new_tokens:
            return {"raw_text":raw,"error":"인식 응답 길이 한도 도달", "reason_code":"run_error", "metrics":metrics}
        try: fields=parse_observation(raw,job["role"])
        except (ValueError,TypeError) as error:
            return {"raw_text":raw,"error":str(error),"reason_code":"run_error","metrics":metrics}
        return filter_observation(job["role"], {"fields":fields,"raw_text":raw,"metrics":metrics})
    finally:
        image.close()


def infer(request, directory):
    settings=RecognitionSettings(**request["settings"])
    versions=check_versions()
    start=time.monotonic()
    validate_model(settings)
    verified=time.monotonic()
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    if settings.device=="cuda" and not torch.cuda.is_available(): raise RuntimeError("인식 CUDA 장치 없음")
    for job in request["jobs"]:
        if job["key"]!=observation_key(job["role"],job["image_sha256"],settings):
            raise ValueError("인식 요청 키 불일치")
    if settings.device=="cuda": torch.cuda.reset_peak_memory_stats()
    dtype=torch.bfloat16 if settings.device=="cuda" else torch.float32
    model=Qwen3VLForConditionalGeneration.from_pretrained(settings.model_root,local_files_only=True,
        trust_remote_code=False,dtype=dtype,device_map=settings.device,attn_implementation="sdpa")
    model.eval()
    processor=AutoProcessor.from_pretrained(settings.model_root,local_files_only=True,trust_remote_code=False,
        min_pixels=4096,max_pixels=settings.max_pixels)
    loaded=time.monotonic()
    observations={}
    for job in request["jobs"]:
        observations[job["role"]]=inspect_image(model,processor,torch,job,settings)
        write_json(directory/(job["role"]+"-response.json"),observations[job["role"]])
        if "error" in observations[job["role"]]: raise ValueError(observations[job["role"]]["error"])
    metrics={"versions":versions,"verify_seconds":verified-start,"load_seconds":loaded-verified,
             "total_seconds":time.monotonic()-start,"device":settings.device,"dtype":str(dtype),
             "max_pixels":settings.max_pixels,"allocated_peak":None,"reserved_peak":None}
    if settings.device=="cuda":
        torch.cuda.synchronize()
        metrics.update(allocated_peak=torch.cuda.max_memory_allocated(),reserved_peak=torch.cuda.max_memory_reserved())
    return {"observations":observations,"metrics":metrics}


def main():
    request_path=Path(sys.argv[1])
    directory=request_path.parent
    request=json.loads(request_path.read_text(encoding="utf-8"))
    record={"request_sha256":json_sha(request),"status":"failed"}
    try:
        result=infer(request,directory)
        record.update(result,status="completed")
    except Exception as error:
        record.update(error=str(error), reason_code="run_error")
        traceback.print_exc()
        raise
    finally:
        write_json(directory/"result.json",record)
    # Parent waits for interpreter exit, so even library-owned GPU tensors are released.

if __name__=="__main__": main()
