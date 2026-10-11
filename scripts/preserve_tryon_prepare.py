"""보존형 의상 교체 시험 1단계(메인 환경, GPU 분할 1회 로드): 원본 캐릭터를 768×1024 흰 여백 캔버스로 맞추고,
See-through로 보존 영역(머리카락·얼굴·눈·코·입·귀·장식·꼬리·날개)을 찾는다. 이미지 생성은 하지 않는다.
다음 단계는 CatVTON 환경의 scripts/preserve_tryon.py mask/run이다."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from genai_lab.proportion_inputs import require, sha
from genai_lab.qwen_record_io import write_json

SIZE = (768, 1024)
PROTECT = ("hair", "face", "eyes", "nose", "mouth", "ears", "earwear", "headwear", "eyewear", "tail", "wings")


def canvas(image):
    """CatVTON의 resize_and_padding과 같은 규칙: 비율 유지, 흰 여백. 머리·발이 잘리지 않게 자르지 않는다."""
    w, h = image.size
    tw, th = SIZE
    if w / h < tw / th:
        nh, nw = th, w * th // h
    else:
        nw, nh = tw, h * tw // w
    resized = image.convert("RGB").resize((nw, nh), Image.LANCZOS)
    out = Image.new("RGB", SIZE, (255, 255, 255))
    out.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", nargs=2, action="append", metavar=("NAME", "IMAGE"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true", help="명시적 GPU 실행 승인. 없으면 캔버스만 만들고 분할은 하지 않는다")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for name, image in args.case:
        folder = args.output / name
        folder.mkdir()
        source = Path(image)
        with Image.open(source) as opened:
            canvas(opened).save(folder / "person.png")
        cases.append(dict(name=name, source=str(source), source_sha256=sha(source), person=str(folder / "person.png"),
                          person_sha256=sha(folder / "person.png")))
    if args.gpu:
        from genai_lab.head_paste_semantics import HeadSegmenter
        from genai_lab.head_paste_worker import figure_parts
        from genai_lab.studio_generation import StudioRuntime
        runtime = StudioRuntime()
        segmenter = HeadSegmenter(runtime.head_semantic_repo, runtime.head_semantic_checkpoint)
        try:
            for case in cases:
                folder = args.output / case["name"]
                rgb = np.asarray(Image.open(folder / "person.png").convert("RGB"))
                parts = figure_parts(segmenter, rgb)
                np.savez_compressed(folder / "parts.npz", **parts)  # 768×1024 원래 좌표(정사각 분할 결과를 되돌림)
                protect = np.logical_or.reduce([parts[name] for name in PROTECT])
                require(protect.any(), f"{case['name']}: 보존 영역을 찾지 못했습니다.")
                Image.fromarray(protect.astype(np.uint8) * 255).save(folder / "protect.png")
                case.update(protect=str(folder / "protect.png"), protect_sha256=sha(folder / "protect.png"),
                            protect_px=int(protect.sum()), parts_present=[k for k, v in parts.items() if v.any()])
                print(json.dumps({k: case[k] for k in ("name", "protect_px")}, ensure_ascii=False), flush=True)
            metrics = segmenter.metrics()
        finally:
            segmenter.close()
    else:
        metrics = None
    write_json(args.output / "prepare.json", dict(size=SIZE, protect_parts=PROTECT, cases=cases, gpu_executed=args.gpu,
                                                 segmenter=metrics))


if __name__ == "__main__":
    main()
