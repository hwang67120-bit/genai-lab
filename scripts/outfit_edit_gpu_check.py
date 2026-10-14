"""원본 편집형 의상 교체 제품 경로 GPU 확인(메인 환경에서 실행, D-094 2~3단계).
제품과 같은 진입점(genai_lab.outfit_edit_pipeline.run_outfit_pipeline)을 캐릭터마다 한 번씩 부른다:
캔버스 준비 → See-through 분할(별도 프로세스) → Qwen 편집(별도 프로세스, 6.5GiB 상한·VAE 타일·lm_head CPU) → 머리 합성.
자동 재시도는 없다. 한 명이 실패하면 기록을 남기고 다음 명으로 간다.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image

from genai_lab.outfit_edit import write_json
from genai_lab.outfit_edit_pipeline import run_outfit_pipeline
from genai_lab.qwen_pose_settings import QwenPoseSettings
from genai_lab.studio_generation import StudioRuntime


def measure_support(image, body_parts, runtime):
    """시작 전 지원 범위 경고(D-113). 태그 분석기를 CPU로 돌린다. 모델이 없으면 unmeasured로 남긴다."""
    from genai_lab.outfit_support import support_warnings
    try:
        from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, WdTagSession
        settings = ClothingDesignAnalysisSettings(cache_dir=str(runtime.model_cache), local_files_only=True)
        with WdTagSession(settings) as tagger, Image.open(image) as source:
            scores = dict(tagger.analyze(source.convert("RGB")).raw_general_scores)
    except Exception as error:
        return dict(status="unmeasured", source="tags", error=str(error)[:200], warnings=[])
    return dict(status="measured", source="tags", warnings=support_warnings(scores, body_parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=2, action="append", metavar=("NAME", "IMAGE"), required=True)
    parser.add_argument("--garment", required=True)
    parser.add_argument("--settings", required=True, help="Qwen 실행 설정 JSON(꼬리 편집과 같은 파일)")
    parser.add_argument("--head-accessories", choices=("keep", "remove"), default="keep")
    parser.add_argument("--garment-terms", default="", help="쉼표로 나눈 옷 이름(제품의 의상 자동 분석 결과). 비우면 목록 없이")
    parser.add_argument("--no-prop-removal", action="store_true", help="원인 확인 전용: 소품 제거 문장을 뺀다(D-096)")
    parser.add_argument("--no-clean-garment", action="store_true",
                        help="원인 확인 전용: 옷 참조의 배경·머리 공통 전처리(D-117, 제품 기본)를 끈다")
    parser.add_argument("--gender", nargs=2, action="append", default=[], metavar=("NAME", "GENDER"),
                        help="캐릭터 성별(male/female/unspecified). 없으면 작업실에 저장된 값, 그것도 없으면 unspecified")
    parser.add_argument("--parts", nargs=2, action="append", default=[], metavar=("NAME", "PARTS"),
                        help="캐릭터에 있는 동물 귀·꼬리·날개(쉼표: ears,tail,wings / 없으면 none). 없으면 저장값, 그것도 없으면 확인 안 함")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    settings = QwenPoseSettings.load(args.settings)
    runtime = StudioRuntime.from_environment()
    summary, rows = [], []
    given, given_parts = dict(args.gender), dict(args.parts)
    from genai_lab.character_preferences import load_character_body_parts, load_character_gender
    for name, image in args.case:
        saved_gender, saved_parts = load_character_gender(str(Path(image))), load_character_body_parts(str(Path(image)))
        gender = given.get(name) or saved_gender or "unspecified"
        if name in given_parts:
            body_parts = tuple(p for p in given_parts[name].split(",") if p and p != "none")
        else:
            body_parts = saved_parts
        provenance = dict(gender=dict(source="flag" if name in given else ("saved" if saved_gender else "default")),
                          body_parts=dict(source="flag" if name in given_parts else ("saved" if saved_parts is not None else "default")),
                          support=measure_support(image, body_parts, runtime))
        start = time.perf_counter()
        try:
            outcome = run_outfit_pipeline(image, args.garment, settings=settings, head_accessories=args.head_accessories,
                                          runtime=runtime, directory=args.output / name,
                                          garment_terms=[t for t in args.garment_terms.split(",") if t.strip()],
                                          remove_props=not args.no_prop_removal, gender=gender, body_parts=body_parts,
                                          provenance=provenance, clean_garment=not args.no_clean_garment,
                                          progress=lambda text: print(f"[{name}] {text}", flush=True))
            memory = json.loads((Path(outcome["raw"]).parent / "memory.json").read_text(encoding="utf-8"))
            record = dict(case=name, gender=gender, body_parts=None if body_parts is None else list(body_parts), status="completed", composited=outcome["composited"], reason=outcome.get("reason"),
                          decision=outcome["alignment"]["decision"], hair_shape=outcome.get("hair_shape"),
                          limbs=outcome.get("limbs"), attempts=outcome.get("attempts"), trace=outcome.get("trace"),
                          support=outcome.get("support"), garment_reference=outcome.get("garment_reference"),
                          canvas_pad=outcome["alignment"].get("canvas_pad"), memory_stopped=memory.get("stopped"),
                          max_stage_peak_reserved_gib=max(s.get("stage_peak_reserved_gib", 0) for s in memory["stages"]))
            rows.append(np.asarray(Image.open(Path(outcome["qwen_view"]).parent / "review.png").convert("RGB").resize((1152, 512))))
        except Exception as error:
            record = dict(case=name, status="failed", error=str(error), traceback=traceback.format_exc())
        record["seconds"] = round(time.perf_counter() - start, 1)
        summary.append(record)
        print(json.dumps(record, ensure_ascii=False)[:600], flush=True)
        write_json(args.output / "summary.json", dict(head_accessories=args.head_accessories, garment=args.garment,
                   clean_garment=not args.no_clean_garment,
                   garment_terms=args.garment_terms, prop_removal=not args.no_prop_removal, results=summary))
    if rows:
        Image.fromarray(np.concatenate(rows, 0)).save(args.output / "review-sheet.png")


if __name__ == "__main__":
    main()
