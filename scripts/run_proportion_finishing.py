"""원본 비율 참고(2단계) 결과에 기존 고화질 마무리(1.5배 + 얼굴 참조 0.9)를 적용하고 눈을 전후 비교한다.
finish: 장마다 기존 finish_selected로 새 폴더에만 마무리한다(원래 실행 폴더는 읽기만).
measure: 마무리 전(2단계 raw)·후(마무리 raw)의 머리 자른 그림을 분할해 눈 간격·크기·눈색을 잰다(분할 모델 1회 로드).
--gpu가 없으면 입력 계약만 검사한다."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from genai_lab.proportion_inputs import require, sha
from genai_lab.qwen_record_io import write_json

BOX = (242, 42, 518, 318)  # 검은 머리 캐릭터 머리 상자(736×1232). --box로 바꾼다.


def candidates(run, seeds):
    from genai_lab.onepass_generation import OnePassCandidate
    generation = Path(run) / "generation"
    lock = json.loads((generation / "preflight.json").read_text(encoding="utf-8"))
    chosen = seeds or lock["seeds"]
    require(set(chosen) <= set(lock["seeds"]), "실행에 없는 seed입니다.")
    for seed in chosen:
        folder = generation / f"CONTOUR_{seed}"
        record = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        yield seed, OnePassCandidate(seed, folder / "raw.png", folder / "run.json", record), generation / f"seed-{seed}" / "product.png"


def finish(args):
    from genai_lab.studio_generation import StudioRuntime
    from genai_lab.studio_optional_finishing import copy_selected, finish_selected
    runtime = StudioRuntime.from_environment()
    rows = []
    for seed, candidate, before in candidates(args.run, args.seeds):
        target = args.output / "finish" / f"seed-{seed}"
        if not args.gpu:
            probe = args.output / "contract" / f"seed-{seed}"
            probe.mkdir(parents=True)
            copy_selected(candidate, probe, runtime)  # 복사·프롬프트·얼굴 참조 SHA만 확인
            rows.append(dict(seed=seed, status="contract_ok"))
        else:
            info = finish_selected(candidate, before, sha(before), target, runtime, progress=print)
            rows.append(dict(seed=seed, status=info["status"], finished=info["finishing"].get("finished_file"),
                             finished_sha256=info["finishing"].get("finished_sha256"), product=info["product_file"],
                             seconds=info.get("seconds"), face_detail=info["finishing"].get("face_detail"),
                             face_skip_reason=info["finishing"].get("face_skip_reason")))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        write_json(args.output / "finish.json", dict(gpu_executed=args.gpu, rows=rows))


def eye_stats(parts, crop):
    """두 큰 눈알 덩어리(왼쪽부터): 면적, 무게중심, 채도 높은 홍채 픽셀의 Lab 평균."""
    import cv2
    from genai_lab.head_paste_rules import lab
    n, labels, st, cen = cv2.connectedComponentsWithStats(parts["eyes"].astype(np.uint8), 8)
    balls = sorted(sorted((i for i in range(1, n) if st[i, 4] >= 1500), key=lambda i: -st[i, 4])[:2], key=lambda i: cen[i][0])
    lab_img = lab(crop)
    chroma = np.hypot(lab_img[..., 1], lab_img[..., 2])
    eyes = []
    for i in balls:
        mask = labels == i
        iris = mask & (chroma >= np.quantile(chroma[mask], 0.5))  # 색이 고르면 전체를 쓴다
        eyes.append(dict(area=int(st[i, 4]), center=[round(float(cen[i][0]), 1), round(float(cen[i][1]), 1)],
                         iris_lab=lab_img[iris].mean(0).round(2).tolist()))
    return eyes


def measure(args):
    from genai_lab.head_paste_rules import crop_head
    from genai_lab.head_paste_semantics import HeadSegmenter
    from genai_lab.studio_generation import StudioRuntime
    runtime = StudioRuntime()
    finished = json.loads((args.output / "finish.json").read_text(encoding="utf-8"))
    require(finished["gpu_executed"], "마무리 결과가 없습니다.")
    box = tuple(args.box)
    rows = []
    segmenter = HeadSegmenter(runtime.head_semantic_repo, runtime.head_semantic_checkpoint)
    try:
        for row, (seed, candidate, _) in zip(finished["rows"], candidates(args.run, args.seeds)):
            require(row["seed"] == seed, "seed 순서가 다릅니다.")
            with Image.open(candidate.path) as image:
                before = crop_head(np.asarray(image.convert("RGB")), box)
            with Image.open(row["finished"]) as image:
                big = np.asarray(image.convert("RGB"))
                scale = big.shape[1] / 736
                after = crop_head(big, tuple(int(round(v * scale)) for v in box))
            stats = {}
            for name, crop in (("before", before), ("after", after)):
                Image.fromarray(crop).save(args.output / f"{name}_{seed}_crop.png")
                stats[name] = eye_stats(segmenter.parse(crop), crop)
            b, a = stats["before"], stats["after"]
            result = dict(seed=seed, before=b, after=a)
            if len(b) == 2 and len(a) == 2:
                gap = lambda e: e[1]["center"][0] - e[0]["center"][0]
                result.update(gap_ratio=round(gap(a) / gap(b), 3),
                              area_ratio=[round(a[k]["area"] / b[k]["area"], 3) for k in (0, 1)],
                              iris_dE=[round(float(np.linalg.norm(np.subtract(a[k]["iris_lab"], b[k]["iris_lab"]))), 2) for k in (0, 1)])
            rows.append(result)
            print(json.dumps({k: v for k, v in result.items() if k not in ("before", "after")}, ensure_ascii=False), flush=True)
            write_json(args.output / "eyes.json", dict(box=box, rows=rows))
        metrics = segmenter.metrics()
    finally:
        segmenter.close()
    write_json(args.output / "eyes.json", dict(box=box, rows=rows, segmenter=metrics))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("finish", "measure"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--box", type=int, nargs=4, default=BOX)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true", help="명시적 GPU 실행 승인(finish). 없으면 입력 계약만 검사")
    args = parser.parse_args()
    if args.stage == "finish":
        args.output.mkdir(parents=True, exist_ok=False)
        finish(args)
    else:
        measure(args)


if __name__ == "__main__":
    main()
