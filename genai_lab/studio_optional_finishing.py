"""Finish one immutable candidate in a separate folder; adoption is a later user decision."""
from dataclasses import asdict, replace
from pathlib import Path
import json
import shutil
import time
from genai_lab.onepass_generation import OnePassBatch, OnePassCandidate, OnePassCancelled, validate_prompt
from genai_lab.onepass_prompt import OnePassPrompt, EncoderChunkPlan, PromptChunk
from genai_lab.proportion_inputs import sha, require, json_value
from genai_lab.qwen_record_io import write_json
from genai_lab.finishing_reference import checked_reference
from genai_lab.studio_finishing import finish_batch, finishing_source, check_cancel
from genai_lab.studio_background import prepare_backgrounds, read_background


def restore_prompt(data):
    """Restore saved token IDs exactly; never re-analyze or add quality tags."""
    encoders = []
    for item in data["encoders"]:
        chunks = {kind: tuple(PromptChunk(c["kind"], c["text"], tuple(c["token_ids"]))
                             for c in item[kind]) for kind in ("positive", "negative")}
        encoders.append(EncoderChunkPlan(item["positive_tokens"], item["negative_tokens"], **chunks))
    prompt = OnePassPrompt(data["positive"], data["negative"], tuple(encoders), data["rules"])
    validate_prompt(prompt)
    require(json_value(asdict(prompt)) == data, "저장된 생성 문장을 복원하지 못했습니다.")
    return prompt


def copy_selected(candidate, destination, runtime):
    data = json.loads(candidate.record_path.read_text(encoding="utf-8"))
    require(data.get("valid") and data.get("completed") and data.get("seed") == candidate.seed
            and sha(candidate.path) == data.get("raw_sha256"), "선택한 생성 원본이 변경됐습니다.")
    expected = json_value(asdict(runtime.generation))
    require(data.get("settings") == expected, "생성 당시 설정과 현재 마무리 설정이 다릅니다.")
    require(data.get("models", {}).get("base") == runtime.generation.model_record()["base"],
            "생성 당시 기본 모델과 현재 마무리 모델이 다릅니다.")
    prompt = restore_prompt(data["prompt"])
    original = OnePassBatch(candidate.path.parent, (candidate,))
    reference = checked_reference(original, runtime.generation)
    folder = destination / f"seed-{candidate.seed}"
    folder.mkdir()
    shutil.copyfile(candidate.path, folder / "raw.png")
    shutil.copyfile(candidate.record_path, folder / "run.json")
    shutil.copyfile(reference.path, destination / "face.png")
    require(sha(folder / "raw.png") == data["raw_sha256"]
            and sha(destination / "face.png") == reference.sha256, "원본 복사 중 파일이 변경됐습니다.")
    copied = OnePassCandidate(candidate.seed, folder / "raw.png", folder / "run.json", data)
    return OnePassBatch(destination, (copied,)), prompt


def finish_selected(candidate, before, before_sha, destination, runtime, *,
                    cancelled=lambda: False, progress=lambda _: None):
    """Only this fresh folder can receive generated files, including failed partial outputs."""
    destination, before = Path(destination), Path(before)
    destination.mkdir(parents=True, exist_ok=False)
    state = {"status": "started", "before_file": str(before), "before_sha256": before_sha,
             "source_run": str(candidate.record_path), "seed": candidate.seed,
             "face_reference": "on", "selection": "original"}
    started = time.perf_counter()
    try:
        check_cancel(cancelled)
        require(sha(before) == before_sha, "마무리 전 결과가 변경됐습니다.")
        batch, prompt = copy_selected(candidate, destination, runtime)
        finish_batch(batch, prompt, replace(runtime, finishing=True, finishing_face_reference=True),
                     cancelled=cancelled, progress=progress, face_file=destination / "face.png")
        prepare_backgrounds(batch, runtime.model_cache, cancelled=cancelled, progress=progress, finishing=True)
        check_cancel(cancelled)
        copied = batch.candidates[0]
        _, finishing = finishing_source(copied)
        background = read_background(copied)
        require(background and background["status"] == "completed", "흰 배경 정리를 완료하지 못했습니다. 원본을 유지합니다.")
        state.update(status="awaiting_user_review", raw_sha256=sha(candidate.path),
                     candidate_file=str(copied.path), candidate_record=str(copied.record_path),
                     product_file=str(copied.path.parent / "product.png"),
                     product_sha256=background["product_sha256"], finishing=finishing, background=background)
        require(sha(before) == before_sha, "마무리 중 기준 결과가 변경됐습니다.")
    except BaseException as error:
        state.update(status="cancelled" if isinstance(error, OnePassCancelled) else "failed", error=str(error))
        raise
    finally:
        state["seconds"] = time.perf_counter() - started
        write_json(destination / "optional-finishing.json", json_value(state))
    return finishing_info(destination, state["raw_sha256"], before_sha)


def finishing_info(directory, raw_sha, before_sha):
    """Re-read the whole output chain before comparison, adoption, tail edit and export."""
    directory = Path(directory)
    state = json.loads((directory / "optional-finishing.json").read_text(encoding="utf-8"))
    require(state["status"] == "awaiting_user_review" and state["raw_sha256"] == raw_sha
            and state["before_sha256"] == before_sha and sha(Path(state["before_file"])) == before_sha,
            "마무리 완료 기록 또는 기준 이미지가 일치하지 않습니다.")
    path, record_path = Path(state["candidate_file"]), Path(state["candidate_record"])
    record = json.loads(record_path.read_text(encoding="utf-8"))
    require(record.get("valid") and record.get("completed") and record["raw_sha256"] == raw_sha
            and record["seed"] == state["seed"], "복사한 생성 기록이 다릅니다.")
    candidate = OnePassCandidate(state["seed"], path, record_path, record)
    finished_path, finishing = finishing_source(candidate)
    background = read_background(candidate)
    require(finishing == state["finishing"] and finishing["face_reference"] == "on"
            and finishing["ip_scale"] == .9 and finishing["prompt"] == record["prompt"]
            and sha(directory / "face.png") == finishing["face_sha256"] == record["inputs"]["face_sha256"],
            "얼굴 참조 마무리 기록이 변경됐습니다.")
    require(background == state["background"] and background["status"] == "completed"
            and background["raw_sha256"] == raw_sha
            and background["source_sha256"] == finishing["finished_sha256"]
            and Path(background["source_file"]).resolve() == finished_path.resolve()
            and Path(state["product_file"]).resolve() == (path.parent / "product.png").resolve()
            and sha(Path(state["product_file"])) == state["product_sha256"] == background["product_sha256"],
            "마무리와 흰 배경 결과의 연결이 다릅니다.")
    return {"directory": str(directory), **state}
