"""분리된 오프라인 CPU SAM2 작업이다. 프로세스 종료로 취소하며 이미지 생성은 없다."""
import os
import sys
import json
import time
import threading
from pathlib import Path

MODEL_ID = "facebook/sam2.1-hiera-tiny"


def cpu_environment():
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="")


class PeakRAM:
    """현재 프로세스의 상주 메모리를 측정한다. Windows에서는 최대 작업 메모리도 읽는다."""
    def __enter__(self):
        import psutil
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        while not self.stopped.wait(.05):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __exit__(self, *args):
        self.stopped.set(); self.thread.join()
        info = self.process.memory_info()
        self.peak = max(self.peak, info.rss, getattr(info, 'peak_wset', 0))


class CpuTailSegmenter:
    def __init__(self, cache_dir=None):
        started = time.monotonic()
        cpu_environment()
        from huggingface_hub import snapshot_download
        from transformers import Sam2Processor, AutoConfig, Sam2Model
        from genai_lab.clothing_reference import ClothingMaskExtractionSettings, create_sam2_image_config
        cache = cache_dir or os.environ.get('GENAI_TAIL_SAM_CACHE') or str(ClothingMaskExtractionSettings().cache_dir)
        self.snapshot = snapshot_download(MODEL_ID, cache_dir=cache, local_files_only=True)
        config = AutoConfig.from_pretrained(self.snapshot, local_files_only=True)
        self.processor = Sam2Processor.from_pretrained(self.snapshot, local_files_only=True)
        self.model = Sam2Model.from_pretrained(self.snapshot, config=create_sam2_image_config(config),
                                               local_files_only=True).to('cpu')
        self.model.eval()
        self.load_seconds = time.monotonic() - started

    def segment(self, image, box):
        import torch
        import numpy as np
        with torch.inference_mode():
            inputs = self.processor(images=image, input_boxes=[[list(box)]], return_tensors='pt')
            output = self.model(**inputs, multimask_output=False)
            masks = self.processor.post_process_masks(output.pred_masks.cpu(), inputs['original_sizes'].cpu(),
                                                      binarize=True)[0]
        mask = np.asarray(masks).reshape(-1, image.height, image.width)[0].astype(bool)
        return mask, float(output.iou_scores.cpu().numpy().ravel()[0])


def analyze_image(segmenter, source, expected_sha, box, directory):
    """원본 자료를 검증하고 한 번 분할한 뒤 참고 측정값을 저장한다."""
    import io
    import hashlib
    import numpy as np
    from PIL import Image
    from genai_lab.tail_complexity import measure_tail, save_overlay
    from genai_lab.qwen_tail_edit import validate_box
    started = time.monotonic()
    data = Path(source).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("원본 이미지가 변경됐습니다.")
    with Image.open(io.BytesIO(data)) as original:
        # 승인된 측정과 기존 크롭의 좌표 방향을 일치시킨다.
        image = original.convert('RGB')
    try:
        validate_box(box, image.size)
        mask, iou = segmenter.segment(image, box)
        record, kept = measure_tail(np.asarray(image), mask, box)
        Path(directory).mkdir(parents=True, exist_ok=True)
        save_overlay(np.asarray(image), kept, box, directory)
        record.update(source_sha256=expected_sha, source_size=list(image.size), sam_iou=round(iou, 3),
                      model_id=MODEL_ID, model_revision=Path(segmenter.snapshot).name,
                      device='cpu', analysis_seconds=time.monotonic()-started,
                      mask_path=str(Path(directory)/'mask.png'), overlay_path=str(Path(directory)/'overlay.png'))
        return record
    finally:
        image.close()


def main():
    cpu_environment()
    request_path = Path(sys.argv[1])
    request = json.loads(request_path.read_text(encoding='utf-8'))
    from genai_lab.qwen_record_io import write_json
    from genai_lab.qwen_preservation import json_sha
    started = time.monotonic()
    record = dict(status='failed', advisory_only=True, request_sha256=json_sha(request))
    try:
        with PeakRAM() as ram:
            segmenter = CpuTailSegmenter(request.get('cache_dir'))
            record.update(analyze_image(segmenter, request['source'], request['source_sha256'],
                                        request['box'], request_path.parent))
            record['load_seconds'] = segmenter.load_seconds
        record['peak_ram_bytes'] = ram.peak
    except Exception as error:
        record.update(status='failed', error=str(error))
    record['total_seconds'] = time.monotonic() - started
    write_json(request_path.parent/'result.json', record)


if __name__ == '__main__':
    main()
