"""원본 편집형 의상 교체 실행 흐름(D-094 3단계): 캔버스 준비(CPU) → 부위 분할(별도 프로세스, GPU) → Qwen 편집(별도 프로세스, GPU)
→ 머리 정렬·형태 검사·머리 합성(CPU). 메인 GUI 프로세스는 CUDA를 쓰지 않는다. 자동 승인·재시도는 없다.
검사를 통과하지 못하면 머리를 붙이지 않고 그 이유와 Qwen 결과만 돌려준다.
"""
import hashlib
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image

from genai_lab.character_features import extract_features, write_features
from genai_lab.character_verification import verify_result
from genai_lab.garment_reference import prepare_garment_reference
from genai_lab.outfit_edit import (EDIT_INTENT, HEAD_ACCESSORIES, HEAD_KEEP, SIZE, canvas, fit_white, hair_shape,
                                   head_composite, head_protect, head_window, input_facts, limb_check, original_background,
                                   outfit_conditions, pad_image, realign, require, sha, write_json)

FACE_PARTS = ("face", "eyes", "nose", "mouth")
SEGMENT_TIMEOUT = 600
REASONS = {"unmeasured": "머리 위치를 확인할 근거(얼굴 특징점)가 부족합니다.",
           "rejected": "Qwen 결과의 머리 위치·크기가 원본과 맞지 않습니다.",
           "shape": "Qwen이 머리 모양을 바꿔서(예: 양갈래가 짧은 머리로) 원본 머리를 붙이면 어색해집니다.",
           "limbs": "Qwen 결과에서 원본에 있던 팔·손이 사라졌습니다."}
POSE_TIMEOUT = 300


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


def detect_joints(image, directory, runtime, *, cancelled=lambda: False):
    """기존 입력 분석의 관절 인식(DWPose, CPU, 별도 환경)을 그림 한 장에 돌린다."""
    import json
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    Image.fromarray(image).save(directory / "character.png")
    script = Path(__file__).with_name("studio_analysis.py")
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    with (directory / "pose.log").open("wb") as log:
        process = subprocess.Popen([str(runtime.pose_python), str(script), "pose", str(directory), str(runtime.pose_models),
                                    str(runtime.model_cache), str(runtime.head_cache)], stdout=log, stderr=subprocess.STDOUT,
                                   env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        start = time.perf_counter()
        try:
            while process.poll() is None:
                if cancelled():
                    raise RuntimeError("사용자가 취소했습니다.")
                require(time.perf_counter() - start <= POSE_TIMEOUT, "관절 인식 시간이 초과됐습니다.")
                time.sleep(.1)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    require(process.returncode == 0, f"관절 인식 실패. 기록: {directory / 'pose.log'}")
    return json.loads((directory / "pose.json").read_text(encoding="utf-8"))["joints"]


def check_limbs(canvas_path, raw_path, directory, runtime, *, cancelled=lambda: False):
    with Image.open(canvas_path) as image:
        person = np.asarray(image.convert("RGB"))
    with Image.open(raw_path) as raw:
        edited = np.asarray(raw.convert("RGB").resize(SIZE, Image.LANCZOS))
    directory = Path(directory)
    original = detect_joints(person, directory / "original", runtime, cancelled=cancelled)
    result = detect_joints(edited, directory / "qwen", runtime, cancelled=cancelled)
    report = limb_check(original, person, result, edited)
    write_json(directory / "limbs.json", report)
    return report


def compose(canvas_path, parts_path, raw_path, head_accessories, directory, limbs=None):
    """Qwen 결과를 원본 좌표로 되돌리고 검사를 통과하면 머리만 원본 픽셀로 붙인다.
    limbs(팔 검사)가 rejected면 머리 검사와 상관없이 붙이지 않는다."""
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
    outcome = dict(head_accessories=head_accessories, alignment=alignment, composited=False, limbs=limbs,
                   raw=str(raw_path), qwen_view=str(directory / "qwen_view.png"),
                   raw_sha256=sha(raw_path), canvas_sha256=sha(canvas_path), parts_sha256=sha(parts_path))
    Image.fromarray(edited).save(directory / "qwen_view.png")
    if limbs is not None and limbs["status"] == "rejected":
        outcome.update(reason=REASONS["limbs"])
    elif aligned is None:
        outcome.update(reason=REASONS[alignment["decision"]])
    else:
        pad = alignment["canvas_pad"]
        person_c, protect_c = pad_image(person, pad, 255), pad_image(protect, pad, False)
        shape, detached = hair_shape(person_c, aligned, pad_image(parts["hair"], pad, False), pad_image(face, pad, False),
                                     protect_c)
        outcome["hair_shape"] = shape
        if shape["status"] != "ok":
            outcome.update(reason=REASONS["shape"])
        else:
            background = original_background(person_c) & protect_c   # 머리로 분할된 원본 배경은 붙이지 않는다(D-115)
            outcome["hair_shape"]["background_px"] = int(background.sum())
            result, core = head_composite(person_c, aligned, protect_c & ~detached & ~background)   # 떠 있는 섬도 안 붙임(D-107)
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
                        garment_terms=(), remove_props=True, gender="unspecified", body_parts=None, provenance=None,
                        clean_garment=True, character_features=None, character_scores=None,
                        cancelled=lambda: False, progress=lambda _: None, popen=subprocess.Popen):
    """GUI·확인 스크립트 공통 진입점. 반환값은 compose()의 결과에 실행 폴더를 더한 것이다.
    provenance는 성별·부위 값이 어디서 왔는지(사용자 확인·저장값·명령 인자·기본값)의 기록이며 실행에는 영향을 주지 않는다(D-112).
    clean_garment: 옷 참조의 배경·머리를 공통 규칙으로 뺀 정리본을 쓴다(D-117). 아무것도 적용되지 않으면 원래 파일 그대로.
    character_features: 확인 창에서 사람이 확정한 특징표 태그 항목({항목: [값]}), character_scores: 확정값이 없을 때 쓸 태그 점수.
    둘 다 없으면 태그 항목은 unknown이고 픽셀 항목(머리색·눈 색·피부색)만 잰다(D-120)."""
    from genai_lab.qwen_outfit_edit import (OutfitEditWorkflow, make_outfit_request, prepare_outfit_spec,
                                            run_outfit_edit)
    directory = Path(directory)
    progress("상태: 캐릭터 그림 준비 중")
    prepared = prepare_canvas(character_path, directory / "inputs")
    if clean_garment:
        progress("상태: 옷 참조 정리 중(배경·머리)")
        reference = prepare_garment_reference(garment_path, directory / "garment-reference", runtime, cancelled=cancelled)
        garment_path, garment_report = reference["path"], reference["report"]
    else:
        garment_report = dict(status="disabled", changed=False, source=str(garment_path), path=str(garment_path))
    progress("상태: 얼굴·머리 영역 분석 중")
    segmented = segment_canvas(prepared, directory / "parts", runtime, cancelled=cancelled, popen=popen)
    features = character_sheet(prepared["canvas"], segmented["parts"], directory / "character-features", gender, body_parts,
                               provenance, character_features, character_scores)
    spec = prepare_outfit_spec(prepared["canvas"], garment_path, head_accessories=head_accessories, confirmed=True,
                               directory=directory / "qwen-inputs", garment_terms=garment_terms,
                               remove_props=remove_props, gender=gender, body_parts=body_parts)
    assembled = outfit_conditions(head_accessories, spec.garment_terms, remove_props, gender, body_parts)
    facts = input_facts(gender, body_parts, provenance)
    facts["garment_reference"] = garment_report          # 옷 입력에 무엇을 했는지(입력 근거, D-117)
    facts["character_features"] = features                # 기준 캐릭터 특징표(입력 근거, D-120)
    write_json(directory / "qwen-inputs" / "conditions.json", dict(intent=EDIT_INTENT, input_facts=facts, **assembled))
    from genai_lab.qwen_outfit_edit import OUTFIT_SEEDS
    attempts = []
    for number, seed in enumerate(OUTFIT_SEEDS, 1):  # D-105: 검사에 걸리면 다른 seed로 한 번만 더 그린다
        suffix = "" if number == 1 else f"-{number}"
        label = "" if number == 1 else f" · 다시 그리기 {number}/{len(OUTFIT_SEEDS)}"
        request = make_outfit_request(spec, settings=settings, seed=seed)
        workflow = OutfitEditWorkflow(spec)
        def report(data, label=label):
            progress(f"상태: 옷 바꾸는 중{label} · {len(data.get('steps', []))}/40 · 약 30분")
        run_outfit_edit(request, directory / f"qwen{suffix}", workflow, cancelled=cancelled, on_progress=report, popen=popen)
        raw = directory / f"qwen{suffix}" / "raw.png"
        progress("상태: 팔·손 확인 중" + label)
        limbs = check_limbs(prepared["canvas"], raw, directory / f"limbs{suffix}", runtime, cancelled=cancelled)
        progress("상태: 원본 머리 맞추는 중" + label)
        outcome = compose(prepared["canvas"], segmented["parts"], raw, head_accessories, directory / f"compose{suffix}",
                          limbs=limbs)
        progress("상태: 캐릭터 특징 확인 중" + label)
        verification = verify_attempt(features, outcome, raw, segmented["parts"], directory / f"character-features{suffix}", runtime)
        attempts.append(dict(seed=seed, composited=outcome["composited"], reason=outcome.get("reason"),
                             directory=str(directory / f"compose{suffix}"), raw=str(raw),
                             checks=dict(limbs=limbs, alignment=outcome.get("alignment", {}).get("decision"),
                                         hair_shape=outcome.get("hair_shape"),
                                         character_features=verification.get("summary"))))
        if outcome["composited"]:
            break
    outcome["attempts"] = attempts
    outcome["character_verification"] = verification
    trace = write_trace(directory, facts, assembled, attempts, outcome)
    return dict(outcome, directory=str(directory), source=prepared["source"], source_sha256=prepared["source_sha256"],
                conditions=str(directory / "qwen-inputs" / "conditions.json"), trace=str(trace),
                support=facts["support"], garment_reference=garment_report,
                character_features=str(directory / "character-features" / "features.json"))


def character_sheet(canvas_path, parts_path, directory, gender, body_parts, provenance, confirmed, scores):
    """분할 결과와 태그로 특징표를 뽑아 실행 폴더에 쓴다(D-120). 분할 파일을 읽지 못하면 unavailable로 남기고 편집은 막지 않는다."""
    sources = dict(gender=(provenance or {}).get("gender", {}).get("source", "user") if isinstance((provenance or {}).get("gender"), dict) else "user",
                   body_parts=(provenance or {}).get("body_parts", {}).get("source", "user") if isinstance((provenance or {}).get("body_parts"), dict) else "user")
    try:
        with Image.open(canvas_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        parts = {k: v.astype(bool) for k, v in np.load(parts_path).items()}
    except (OSError, ValueError) as error:
        features = dict(schema="character_features/1", status="unavailable", error=str(error)[:200], items={}, unknown=[])
    else:
        features = extract_features(rgb, parts, scores or {}, gender=gender, body_parts=body_parts, provenance=sources, confirmed=confirmed)
    write_features(directory, features)
    return features


def verify_attempt(features, outcome, raw, parts_path, directory, runtime):
    """시도 하나의 결과에서 기준 특징표 유지 여부(D-121). Qwen 결과를 원본 좌표로 되돌린 그림(aligned.png)이 있으면 그것,
    없으면 raw를 캔버스 크기로. 실패해도 편집은 막지 않는다(unmeasured)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        aligned = Path(outcome.get("directory", "")) / "aligned.png" if outcome.get("directory") else None
        image_path = aligned if aligned and aligned.exists() else Path(raw)
        if image_path == Path(raw):
            with Image.open(raw) as source:
                resized = directory / "result-canvas.png"
                source.convert("RGB").resize(SIZE, Image.LANCZOS).save(resized)
            image_path = resized
        report = verify_result(features, image_path, parts_path, runtime)
        report["result_image"] = str(image_path)
    except Exception as error:                       # 검증은 보조 정보: 실패는 기록만
        report = dict(schema="character_verification/1", status="unmeasured", reason=str(error)[:200], items={}, summary={})
    write_json(directory / "verification.json", report)
    return report


def write_trace(directory, facts, assembled, attempts, outcome):
    """입력 근거 → 편집 의도 → 전달 조건 → 출력 관찰 → 판정을 한 기록으로 잇는다(D-112).
    출력 관찰 중 자동 검사(팔·머리 위치·머리 형태)만 확정이고, 귀·꼬리·날개·의상 잔존은 자동 검출이 없어 사람 확인 항목으로 남긴다.
    생성 모델이 왜 그렇게 그렸는지는 기록하지 않는다 — 가설은 보고서에서 관찰 근거와 함께 적는다."""
    prompt = "".join(c["text"] for c in assembled["conditions"])
    trace = dict(schema="outfit_edit_trace/1", intent=EDIT_INTENT, input_facts=facts,
                 conditions=assembled["conditions"], excluded=assembled["excluded"],
                 prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(), attempts=attempts,
                 output_observation=dict(
                     automatic=["limbs", "alignment", "hair_shape", "character_features(특징표 유지: kept/changed/missing/added/unknown)"],
                     manual=["body_parts(귀·꼬리·날개 개수·모양)", "original_garment_remaining(기존 의상 잔존)", "pose", "same_character(최종)"],
                     character_verification=outcome.get("character_verification"),
                     note="manual 항목은 자동 검출이 없거나 자동 결과가 근거일 뿐이다. 사용자 승인(appendages·garment 확인) 또는 검토 보고서가 판단 주체다."),
                 judgement=dict(automatic="composited" if outcome["composited"] else outcome.get("reason"),
                                composited=bool(outcome["composited"]),
                                certainty="자동 검사 결과는 확정. 생성 원인은 미확정(관찰 근거로 가설만)."))
    path = directory / "trace.json"
    write_json(path, trace)
    return path
