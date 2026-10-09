"""생성 실행기를 해제한 뒤 CPU에서 제품 이미지를 준비한다. 원본은 변경하지 않는다."""
from dataclasses import dataclass
from pathlib import Path
import json
import logging
import time
import numpy as np
from PIL import Image
from genai_lab.onepass_generation import OnePassCancelled
from genai_lab.proportion_inputs import FOREGROUND_SHA, sha, white_background
from genai_lab.proportion_foreground import AnimeForeground

MODEL_RELATIVE = Path("models--skytnt--anime-seg/snapshots/493cb60893f47441b26ec4fb9a306bce9e342982/isnetis.onnx")
BACKGROUND_LIMIT = "몸에 붙은 배경 장식·줄·덩어리는 남을 수 있습니다. 결과를 직접 확인해 주세요."


@dataclass(frozen=True)
class BackgroundOptions:
    foreground_model: Path
    foreground_sha256: str = FOREGROUND_SHA


def prepare_candidate(candidate, foreground, options, load_error=None, *, source_file=None):
    """검증된 제품 또는 명시적인 원본 대체 결과를 별도 해시와 함께 저장한다."""
    destination = candidate.path.parent
    record_path = destination / "background.json"
    product = destination / "product.png"
    if record_path.exists() or product.exists():
        raise FileExistsError("배경 정리 기록이 이미 있습니다. 기존 결과를 덮어쓰지 않습니다.")
    raw_sha = sha(candidate.path)
    raw_record = json.loads(candidate.record_path.read_text(encoding="utf-8"))
    if not raw_record.get("valid") or not raw_record.get("completed") or raw_sha != raw_record.get("raw_sha256"):
        raise ValueError("배경 정리 전 원본 생성 기록이 일치하지 않습니다.")
    result = {"status": "failed", "raw_sha256": raw_sha, "product_sha256": None,
              "model": str(options.foreground_model), "model_sha256_expected": options.foreground_sha256,
              "provider": "CPUExecutionProvider", "composition": "soft_alpha_white_no_component_filter",
              "automatic_gates_executed": False, "limitation": BACKGROUND_LIMIT}
    source = Path(source_file) if source_file is not None else candidate.path
    source_sha = sha(source)
    if source_file is not None:
        result.update(source_file=str(source), source_sha256=source_sha)
    started = time.monotonic()
    try:
        if load_error is not None:
            raise load_error
        with Image.open(source) as image:
            rgb = image.convert("RGB")
        with rgb:
            alpha = foreground.alpha(rgb)
            output = white_background(np.asarray(rgb), alpha)
        with Image.fromarray(output) as image:
            with product.open("xb") as stream:
                image.save(stream, format="PNG")
        result.update(status="completed", product_sha256=sha(product), message="흰 배경 정리 완료")
    except OnePassCancelled:
        raise
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error), message="배경 정리 안 됨 · 마무리 결과 표시" if source_file is not None else "배경 정리 안 됨 · 원본 표시")
    if sha(candidate.path) != raw_sha or sha(source) != source_sha:
        raise ValueError("배경 정리 중 생성 원본이 변경됐습니다.")
    result["seconds"] = time.monotonic() - started
    with record_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return result


def prepare_backgrounds(batch, model_cache, *, cancelled=lambda: False, progress=lambda _: None,
                        foreground_factory=AnimeForeground, finishing=False):
    """고정된 로컬 CPU 모델을 묶음마다 한 번 로드한다. 실패 대체 시 다운로드하지 않는다."""
    options = BackgroundOptions(Path(model_cache) / MODEL_RELATIVE)
    foreground, load_error = None, None
    try:
        if cancelled():
            raise OnePassCancelled("배경 정리를 취소했습니다. 생성 원본은 보관됩니다.")
        try:
            # 주입한 시험 구현도 포함해 어떤 제공자를 호출하기 전에 검증한다.
            if sha(options.foreground_model) != options.foreground_sha256:
                raise ValueError("isnet-anime SHA가 일치하지 않습니다.")
            foreground = foreground_factory(options)
        except Exception as error:
            load_error = error
        for i, candidate in enumerate(batch.candidates):
            if cancelled():
                raise OnePassCancelled("배경 정리를 취소했습니다. 생성 원본은 보관됩니다.")
            progress(f"상태: 배경 정리 중 · {i+1}/{len(batch.candidates)} · CPU 처리")
            source = None
            if finishing:
                from genai_lab.studio_finishing import finishing_source
                source, _ = finishing_source(candidate)
            prepare_candidate(candidate, foreground, options, load_error, source_file=source)
        if cancelled():
            raise OnePassCancelled("배경 정리를 취소했습니다. 생성 원본은 보관됩니다.")
        return batch
    finally:
        if foreground is not None:
            try:
                foreground.close()
            except Exception:
                logging.getLogger(__name__).warning("배경 모델 해제 실패", exc_info=True)


def read_background(candidate):
    path = candidate.path.parent / "background.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
