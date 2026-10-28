"""GPU 마무리 모델과 분리된 오프라인 CPU 머리 검출기다."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from genai_lab.qwen_record_io import write_json

if __name__ == "__main__":
    source, output, cache = sys.argv[1:]
    os.environ.update(CUDA_VISIBLE_DEVICES="", ONNX_MODE="cpu", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", HF_HUB_CACHE=cache)
    try:
        from PIL import Image
        from imgutils.detect import detect_heads
        with Image.open(source) as image:
            heads = detect_heads(image, model_name="head_detect_v2.0_s")
        write_json(Path(output), {"heads":[[list(map(float,box)),str(label),float(score)]
                                         for box,label,score in heads], "device":"cpu"})
    except Exception as error:
        write_json(Path(output).parent/"analysis-error.json", {"message":str(error),"error_type":type(error).__name__})
        raise
