"""속도 계획 M1·M2 CPU 측정(MEMORY_OPTIMIZATION_PLAN 11.4, D-180). GPU를 쓰지 않는다(CUDA_VISIBLE_DEVICES=-1로 실행).

  sha <Qwen 설정 json> <출력 json>
      작업 프로세스가 시계 시작 전에 하는 일을 같은 순서·같은 함수로 따로 잰다:
      버전 확인 → 모델 SHA 검사(명세 27개 + GGUF) → torch → diffusers 클래스 → transformers 클래스 가져오기.
      CUDA 준비(torch.cuda)·메모리 상한 설정은 GPU라 재지 않는다. 디스크 캐시 상태는 조절하지 않는다(기록만).
  preview <요청 37 run.json> <쓰기 폴더(새 폴더)> <출력 json> [<반복 수>]
      단계마다 콜백이 하는 CPU 일: 잠재값 → 미리보기 RGB 변환, 미리보기 PNG 쓰기(+오래된 것 지우기), 진행 기록 JSON 쓰기.
      잠재값은 요청 37과 같은 크기(896×1184 → 1×4144×64)의 무작위 값. GPU→CPU 복사(.cpu())는 재지 않는다.
기준 빌드 런타임에서 `python.exe -B`로 실행한다(기준 빌드 폴더에 캐시 파일을 쓰지 않게).
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path


def timed(fn):
    start = time.perf_counter()
    value = fn()
    return time.perf_counter() - start, value


def measure_sha(settings_path, output):
    from genai_lab.qwen_pose_settings import QwenPoseSettings, check_runtime_versions, validate_model_files
    report = dict(kind="sha", python=sys.executable, started=time.strftime("%Y-%m-%d %H:%M:%S"),
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    settings = QwenPoseSettings.load(settings_path)
    manifest = json.loads(Path(settings.artifact_manifest).read_text(encoding="utf-8"))
    sizes = [(Path(settings.model_root) / rel).stat().st_size for rel in manifest["files"]]
    gguf = Path(settings.gguf_file).stat().st_size
    report["bytes"] = dict(manifest_files=len(sizes), manifest_bytes=sum(sizes), gguf_bytes=gguf, total=sum(sizes) + gguf)
    seconds = {}
    seconds["check_runtime_versions"], _ = timed(check_runtime_versions)
    seconds["validate_model_files"], _ = timed(lambda: validate_model_files(settings))
    seconds["import_torch"], _ = timed(lambda: __import__("torch"))
    seconds["import_diffusers"], _ = timed(lambda: exec("from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel, "
                                                         "GGUFQuantizationConfig\nfrom diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus "
                                                         "import calculate_dimensions, CONDITION_IMAGE_SIZE", {}))
    seconds["import_transformers"], _ = timed(lambda: exec("from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig", {}))
    report["seconds"] = seconds
    report["sha_gib_per_second"] = round(report["bytes"]["total"] / 2**30 / seconds["validate_model_files"], 3)
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(dict(seconds={k: round(v, 2) for k, v in seconds.items()}, gib=round(report["bytes"]["total"] / 2**30, 2),
                          gib_per_s=report["sha_gib_per_second"]), ensure_ascii=False))


def measure_preview(run_json, folder, output, repeats=40):
    import numpy as np
    from genai_lab.generation_preview import load_factors, preview_rgb, write_preview
    from genai_lab.qwen_record_io import publish_progress
    folder = Path(folder)
    folder.mkdir(parents=True)                      # 새 폴더만(기존 결과 폴더에 쓰지 않는다)
    record = json.loads(Path(run_json).read_text(encoding="utf-8"))
    height, width = record["call_arguments"]["height"], record["call_arguments"]["width"]
    factors = load_factors()
    rng = np.random.default_rng(0)
    patches = (height // 16) * (width // 16)
    convert, write, progress = [], [], []
    for index in range(repeats):
        latents = rng.standard_normal((1, patches, 64), dtype=np.float32)
        t, rgb = timed(lambda: preview_rgb(latents, height, width, 8, factors))
        convert.append(t)
        t, _ = timed(lambda: write_preview(folder, rgb, index))
        write.append(t)
        t, _ = timed(lambda: publish_progress(folder, record))
        progress.append(t)

    def summary(values):
        return dict(mean=statistics.mean(values), median=statistics.median(values), max=max(values), first=values[0])
    report = dict(kind="preview", python=sys.executable, repeats=repeats, latent_shape=[1, patches, 64], output_size=[width, height],
                  record_bytes=len(json.dumps(record, ensure_ascii=False, indent=2)), folder=str(folder),
                  seconds=dict(convert=summary(convert), write_preview=summary(write), publish_progress=summary(progress)),
                  per_step_total_mean=statistics.mean(c + w + p for c, w, p in zip(convert, write, progress)),
                  not_measured="GPU→CPU 잠재값 복사(.detach().float().cpu())")
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(dict(per_step_mean=round(report["per_step_total_mean"], 4),
                          convert=round(report["seconds"]["convert"]["mean"], 4), write=round(report["seconds"]["write_preview"]["mean"], 4),
                          progress=round(report["seconds"]["publish_progress"]["mean"], 4)), ensure_ascii=False))


if __name__ == "__main__":
    if sys.argv[1] == "sha":
        measure_sha(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "preview":
        measure_preview(sys.argv[2], sys.argv[3], sys.argv[4], *(int(a) for a in sys.argv[5:6]))
    else:
        raise SystemExit(__doc__)
