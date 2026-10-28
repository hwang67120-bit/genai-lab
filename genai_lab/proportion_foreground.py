"""로컬 CPU isnet-anime을 사용하며 시험에서 측정한 픽셀 전처리와 맞춘다."""
import gc
import numpy as np
from PIL import Image
from genai_lab.proportion_inputs import require, sha


class AnimeForeground:
    def __init__(self, options):
        require(sha(options.foreground_model) == options.foreground_sha256, "isnet-anime가 변경됐습니다.")
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(options.foreground_model), providers=["CPUExecutionProvider"])

    def alpha(self, image):
        session = self.session
        shape = session.get_inputs()[0].shape
        height, width = int(shape[2]), int(shape[3])
        scale = min(width / image.width, height / image.height)
        cw, ch = max(1, round(image.width * scale)), max(1, round(image.height * scale))
        left, top = (width - cw) // 2, (height - ch) // 2
        with image.resize((cw, ch), Image.Resampling.LANCZOS) as resized:
            with Image.new("RGB", (width, height), (0, 0, 0)) as canvas:
                canvas.paste(resized, (left, top))
                tensor = np.asarray(canvas, dtype=np.float32)[:, :, ::-1].transpose((2, 0, 1))[None] / 255.0
                prediction = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: tensor})[0]
        mask = (np.clip(np.squeeze(prediction), 0, 1) * 255.0).round().astype(np.uint8)
        with Image.fromarray(mask) as full:
            with full.crop((left, top, left + cw, top + ch)) as content:
                with content.resize(image.size, Image.Resampling.LANCZOS) as result:
                    return np.asarray(result).copy()

    def close(self):
        self.session = None
        gc.collect()
