"""원본 편집형 의상 교체 실행 흐름(D-094 3단계): 캔버스 준비(CPU) → 부위 분할(별도 프로세스, GPU) → Qwen 편집(별도 프로세스, GPU)
→ 머리 정렬·형태 검사·머리 합성(CPU). 메인 GUI 프로세스는 CUDA를 쓰지 않는다. 자동 승인·재시도는 없다.
검사를 통과하지 못하면 머리를 붙이지 않고 그 이유와 Qwen 결과만 돌려준다.
"""
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image

from genai_lab.outfit_edit import (HEAD_ACCESSORIES, HEAD_KEEP, SIZE, canvas, fit_white, hair_shape, head_composite,
                                   head_protect, head_window, pad_image, realign, require, sha, write_json)

FACE_PARTS = ("face", "eyes", "nose", "mouth")
SEGMENT_TIMEOUT = 600
REASONS = {"unmeasured": "머리 위치를 확인할 근거(얼굴 특징점)가 부족합니다.",
           "rejected": "Qwen 결과의 머리 위치·크기가 원본과 맞지 않습니다.",
           "shape": "Qwen이 머리 모양을 바꿔서(예: 양갈래가 짧은 머리로) 원본 머리를 붙이면 어색해집니다."}


def prepare_canvas(character_path, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with Image.open(character_path) as image:
        canvas(image).save(directory / "canvas.png")
    return dict(canvas=str(directory / "canvas.png"), canvas_sha256=sha(directory / "canvas.png"),
                source=str(character_path), source_sha256=sha(character_path))


def segment_canvas(prepared, directory, runtime, *, cancelled=lambda: False, popen=subprocess.Popen):
    """See-through 분할을 별도 프로세스로 실행한다(모델은 프로세스와 함께 내려간다)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "request.json", dict(canvas=prepared["canvas"], canvas_sha256=prepared["canvas_sha256"],
               semantic_repo=str(runtime.head_semantic_repo), semantic_checkpoint=str(runtime.head_semantic_checkpoint)))
    worker = Path(__file__).with_name("outfit_edit_worker.py")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    start, process = time.perf_counter(), None
    try:
        with (directory / "worker.log").open("wb") as log:
            process = popen([str(runtime.analysis_python), "-B", str(worker), "parts", str(directory / "request.json")],
                            stdout=log, stderr=subprocess.STDOUT, env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            while process.poll() is None:
                if cancelled():
                    raise RuntimeError("사용자가 취소했습니다.")
                require(time.perf_counter() - start <= SEGMENT_TIMEOUT, "부위 분할 시간이 초과됐습니다.")
                time.sleep(.1)
        if process.returncode:
            error = directory / "worker-error.json"
            import json
            reason = json.loads(error.read_text(encoding="utf-8")).get("error") if error.is_file() else "부위 분할 실패"
            raise ValueError(f"{reason}\n기록: {directory / 'worker.log'}")
        import json
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        require(result["canvas_sha256"] == prepared["canvas_sha256"] and sha(result["parts"]) == result["parts_sha256"],
                "부위 분할 결과가 준비한 그림과 다릅니다.")
        return result
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def compose(canvas_path, parts_path, raw_path, head_accessories, directory):
    """Qwen 결과를 원본 좌표로 되돌리고 검사를 통과하면 머리만 원본 픽셀로 붙인다."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with Image.open(canvas_path) as image:
        person = np.asarray(image.convert("RGB"))
    with np.load(parts_path, allow_pickle=False) as data:
        parts = {name: data[name].copy() for name in data.files}
    with Image.open(raw_path) as raw:
        edited = np.asarray(raw.convert("RGB").resize(SIZE, Image.LANCZOS))
    union = np.logical_or.reduce([parts[k] for k in HEAD_KEEP + HEAD_ACCESSORIES])
    protect = head_protect(parts, union, head_accessories)
    face = np.logical_or.reduce([parts[k] for k in FACE_PARTS])
    aligned, alignment = realign(person, edited, head_window(protect), expand=True, face=face)
    outcome = dict(head_accessories=head_accessories, alignment=alignment, composited=False,
                   raw_sha256=sha(raw_path), canvas_sha256=sha(canvas_path), parts_sha256=sha(parts_path))
    Image.fromarray(edited).save(directory / "qwen_view.png")
    if aligned is None:
        outcome.update(reason=REASONS[alignment["decision"]])
    else:
        pad = alignment["canvas_pad"]
        person_c, protect_c = pad_image(person, pad, 255), pad_image(protect, pad, False)
        shape = hair_shape(person_c, aligned, pad_image(parts["hair"], pad, False), pad_image(face, pad, False), protect_c)
        outcome["hair_shape"] = shape
        if shape["status"] != "ok":
            outcome.update(reason=REASONS["shape"])
        else:
            result, core = head_composite(person_c, aligned, protect_c)
            require(np.array_equal(result[core], person_c[core]), "원본 머리 픽셀이 바뀌었습니다.")
            Image.fromarray(aligned).save(directory / "aligned.png")
            Image.fromarray(result).save(directory / "result.png")
            outcome.update(composited=True, result=str(directory / "result.png"), result_sha256=sha(directory / "result.png"),
                           output_size=[int(result.shape[1]), int(result.shape[0])], head_core_px=int(core.sum()))
    shown = np.asarray(Image.open(directory / "result.png").convert("RGB")) if outcome["composited"] else np.full_like(person, 255)
    Image.fromarray(np.concatenate([person, edited, fit_white(shown, SIZE)], 1)).save(directory / "review.png")
    write_json(directory / "outcome.json", outcome)
    return outcome


def run_outfit_pipeline(character_path, garment_path, *, settings, head_accessories, runtime, directory,
                        cancelled=lambda: False, progress=lambda _: None, popen=subprocess.Popen):
    """GUI·확인 스크립트 공통 진입점. 반환값은 compose()의 결과에 실행 폴더를 더한 것이다."""
    from genai_lab.qwen_outfit_edit import (OutfitEditWorkflow, make_outfit_request, prepare_outfit_spec,
                                            run_outfit_edit)
    directory = Path(directory)
    progress("상태: 캐릭터 그림 준비 중")
    prepared = prepare_canvas(character_path, directory / "inputs")
    progress("상태: 얼굴·머리 영역 분석 중")
    segmented = segment_canvas(prepared, directory / "parts", runtime, cancelled=cancelled, popen=popen)
    spec = prepare_outfit_spec(prepared["canvas"], garment_path, head_accessories=head_accessories, confirmed=True,
                               directory=directory / "qwen-inputs")
    request = make_outfit_request(spec, settings=settings)
    workflow = OutfitEditWorkflow(spec)
    def report(data):
        progress(f"상태: 옷 바꾸는 중 · {len(data.get('steps', []))}/40 · 약 30분")
    run_outfit_edit(request, directory / "qwen", workflow, cancelled=cancelled, on_progress=report, popen=popen)
    progress("상태: 원본 머리 맞추는 중")
    outcome = compose(prepared["canvas"], segmented["parts"], directory / "qwen" / "raw.png", head_accessories,
                      directory / "compose")
    return dict(outcome, directory=str(directory), source=prepared["source"], source_sha256=prepared["source_sha256"])
