"""머리 안쪽 선 시험 결과에서 생성 얼굴 위치가 원본과 얼마나 다른지 잰다(GPU 분할 1회 로드).
원본 머리 붙이기 확인 폴더의 원본 분할과 머리 상자를 그대로 쓰고, 생성 분할은 다음 CPU 붙이기 미리보기용으로 저장한다."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from genai_lab.head_paste_rules import crop_head, face_offset, FACE_OFFSET_LIMIT
from genai_lab.head_paste_semantics import HeadSegmenter, load_parts, save_parts
from genai_lab.proportion_inputs import require, sha
from genai_lab.qwen_record_io import write_json


def raws(trial, comparison):
    """조건별 CONTOUR raw를 모은다. comparison은 같은 seed의 기존(C0) 실행 generation 폴더다."""
    found = {}
    for condition in ("hair", "all"):
        for folder in sorted((trial / condition).glob("CONTOUR_*")):
            found[(condition, int(folder.name.split("_")[1]))] = folder / "raw.png"
    seeds = sorted({seed for _, seed in found})
    for seed in seeds:
        found[("C0", seed)] = comparison / f"CONTOUR_{seed}" / "raw.png"
    require(found and all(path.is_file() for path in found.values()), "비교할 raw가 없습니다.")
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=Path, required=True, help="head_lines_trial.py --run 출력 폴더")
    parser.add_argument("--comparison", type=Path, required=True, help="기존 실행의 generation 폴더(C0)")
    parser.add_argument("--hair-confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--semantic-repo", type=Path, help="기본: 제품 실행 설정(StudioRuntime)")
    parser.add_argument("--semantic-checkpoint", type=Path, help="기본: 제품 실행 설정(StudioRuntime)")
    args = parser.parse_args()
    if args.semantic_repo is None or args.semantic_checkpoint is None:
        from genai_lab.studio_generation import StudioRuntime
        runtime = StudioRuntime()
        args.semantic_repo = args.semantic_repo or runtime.head_semantic_repo
        args.semantic_checkpoint = args.semantic_checkpoint or runtime.head_semantic_checkpoint
    args.output.mkdir(parents=True, exist_ok=False)
    info = json.loads((args.hair_confirmation / "result.json").read_text(encoding="utf-8"))
    original_parts = load_parts(args.hair_confirmation / "original-parts.npz")
    box = tuple(info["box"])
    items = raws(args.trial, args.comparison)
    rows = []
    segmenter = HeadSegmenter(args.semantic_repo, args.semantic_checkpoint)
    try:
        for (condition, seed), path in items.items():
            with Image.open(path) as image:
                crop = crop_head(np.asarray(image.convert("RGB")), box)
            parts = segmenter.parse(crop)
            parts_file = args.output / f"{condition}_{seed}_generated-parts.npz"
            save_parts(parts_file, parts)
            Image.fromarray(crop).save(args.output / f"{condition}_{seed}_crop.png")
            try:
                dx, dy = face_offset(original_parts, parts)
                row = dict(condition=condition, seed=seed, face_offset_px=[dx, dy],
                           within_limit=max(abs(dx), abs(dy)) <= FACE_OFFSET_LIMIT)
            except ValueError as error:
                row = dict(condition=condition, seed=seed, error=str(error))
            rows.append(dict(row, raw=str(path), raw_sha256=sha(path), parts_file=str(parts_file)))
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
            write_json(args.output / "face-offset.json", dict(limit_px=FACE_OFFSET_LIMIT, box=box, rows=rows))
        metrics = segmenter.metrics()
    finally:
        segmenter.close()
    write_json(args.output / "face-offset.json", dict(limit_px=FACE_OFFSET_LIMIT, box=box, rows=rows, segmenter=metrics))


if __name__ == "__main__":
    main()
