"""원본 편집형 의상 교체의 부위 분할 작업 프로세스(메인 분석 환경). GUI 프로세스가 CUDA를 잡지 않도록 따로 실행하고 끝나면 내린다.
python -B outfit_edit_worker.py parts REQUEST.json → parts.npz, result.json (실패 시 worker-error.json)
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image

from genai_lab.outfit_edit import HEAD_ACCESSORIES, HEAD_KEEP, SIZE, sha, write_json

FACE_PARTS = ("face", "eyes", "nose", "mouth")


def segment(request, directory, segmenter_factory=None):
    from genai_lab.head_paste_worker import figure_parts
    canvas = Path(request["canvas"])
    if sha(canvas) != request["canvas_sha256"]:
        raise ValueError("준비한 캐릭터 그림이 변경됐습니다.")
    with Image.open(canvas) as image:
        rgb = np.asarray(image.convert("RGB"))
    if rgb.shape[:2] != (SIZE[1], SIZE[0]):
        raise ValueError("캐릭터 그림은 768×1024여야 합니다.")
    if segmenter_factory is None:
        from genai_lab.head_paste_semantics import HeadSegmenter as segmenter_factory
    segmenter = segmenter_factory(request["semantic_repo"], request["semantic_checkpoint"])
    try:
        parts = figure_parts(segmenter, rgb)
        metrics = segmenter.metrics()
    finally:
        segmenter.close()
    protect = np.logical_or.reduce([parts[k] for k in HEAD_KEEP + HEAD_ACCESSORIES])
    face = np.logical_or.reduce([parts[k] for k in FACE_PARTS])
    if not protect.any() or not face.any():
        raise ValueError("얼굴·머리를 찾지 못했습니다. 이 그림은 원본 편집형 의상 교체를 쓸 수 없습니다.")
    np.savez_compressed(directory / "parts.npz", **parts)
    result = dict(parts=str(directory / "parts.npz"), parts_sha256=sha(directory / "parts.npz"),
                  canvas_sha256=request["canvas_sha256"], protect_px=int(protect.sum()), face_px=int(face.sum()),
                  accessory_px=int(np.logical_or.reduce([parts[k] for k in HEAD_ACCESSORIES]).sum()), segmenter=metrics)
    write_json(directory / "result.json", result)
    return result


def main():
    mode, filename = sys.argv[1:]
    directory = Path(filename).parent
    try:
        if mode != "parts":
            raise ValueError("알 수 없는 단계입니다.")
        segment(json.loads(Path(filename).read_text(encoding="utf-8")), directory)
    except BaseException as error:
        write_json(directory / "worker-error.json", {"status": "failed", "stage": mode, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
