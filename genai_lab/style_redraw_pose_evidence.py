"""기존 관절 전용 Python에서 CPU 관절 근거를 일괄 수집한다. 설치/다운로드/생성은 하지 않는다."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HF_HUB_OFFLINE"] = "1"
import json
import sys
from pathlib import Path


def observe(request):
    import numpy as np
    from PIL import Image
    import onnxruntime as ort
    from easy_dwpose.body_estimation import Wholebody
    models = Path(request["models"])
    paths = [models / name for name in ("yolox_l.onnx", "dw-ll_ucoco_384.onnx")]
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError("로컬 관절 모델이 없습니다. 자동 다운로드하지 않습니다.")
    detector = Wholebody(*(str(p) for p in paths), device="cpu")
    providers = [detector.session_det.get_providers(), detector.session_pose.get_providers()]
    if any(p != ["CPUExecutionProvider"] for p in providers):
        raise RuntimeError("관절 검사는 CPU 전용이어야 합니다.")
    records = []
    for path in request["images"]:
        try:
            with Image.open(path) as image:
                points, scores = detector(np.asarray(image.convert("RGB")).copy())
            records.append(dict(state="observed", points=np.asarray(points).tolist(), scores=np.asarray(scores).tolist()))
        except Exception as error:
            records.append(dict(state="unassessed", error=f"{type(error).__name__}: {error}"))
    return dict(records=records, providers=providers, models=list(map(str, paths)))


def main():
    request_path, output_path = map(Path, sys.argv[1:3])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    try:
        result = observe(request)
    except Exception as error:
        result = dict(records=[dict(state="unassessed", error=f"{type(error).__name__}: {error}") for _ in request["images"]])
    output_path.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
