"""Bind finishing to the face and IP model verified by the original generation."""
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from PIL import Image
from genai_lab.proportion_inputs import require

IP_SCALE = 0.9


@dataclass(frozen=True)
class FinishingFaceReference:
    path: Path
    sha256: str
    models: dict

    def open_image(self):
        data = self.path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == self.sha256, "마무리 얼굴 참조 SHA 불일치")
        with Image.open(io.BytesIO(data)) as image:
            require(image.mode == "RGB", "확인된 RGB 얼굴 참조가 필요합니다.")
            return image.copy()

    def record(self):
        return {"face_reference":"on", "ip_scale":IP_SCALE, "face_file":str(self.path),
                "face_sha256":self.sha256, "face_reference_models":self.models}


def checked_reference(batch, settings, *, face_file=None):
    """An optional copied file is allowed only when its bytes match every raw record."""
    require(bool(batch.candidates), "마무리 후보가 없습니다.")
    expected_models = settings.model_record()
    expected_face = None
    source = None
    for candidate in batch.candidates:
        record = json.loads(candidate.record_path.read_text(encoding="utf-8"))
        inputs = record["inputs"]
        require(record["settings"]["ip_scale"] == IP_SCALE, "생성 단계 IP 강도가 0.9가 아닙니다.")
        for name in ("ip_adapter", "image_encoder"):
            actual, expected = record["models"][name], expected_models[name]
            require(Path(actual["path"]).resolve() == Path(expected["path"]).resolve(), "생성 때와 IP 모델 경로가 다릅니다.")
            require({k:v for k,v in actual.items() if k != "path"} ==
                    {k:v for k,v in expected.items() if k != "path"}, "생성 때와 IP 모델 revision 또는 가중치가 다릅니다.")
        if expected_face is None:
            expected_face, source = inputs["face_sha256"], Path(inputs["face_file"])
        require(inputs["face_sha256"] == expected_face, "한 배치의 얼굴 참조가 서로 다릅니다.")
    reference = FinishingFaceReference(Path(face_file) if face_file is not None else source,
        expected_face, {name:expected_models[name] for name in ("ip_adapter", "image_encoder")})
    with reference.open_image():
        pass
    return reference


def observe_face_reference(module, args, kwargs, calls):
    """Read actual attention scales and image conditioning before each UNet call."""
    scales = []
    for processor in module.attn_processors.values():
        if hasattr(processor, "scale"):
            value = processor.scale
            scales.extend(float(v) for v in (value if isinstance(value, (list,tuple)) else [value]))
    observed = sorted(set(scales))
    require(observed == [IP_SCALE], "마무리 실제 IP 강도가 0.9가 아닙니다.")
    require(kwargs.get("added_cond_kwargs", {}).get("image_embeds") is not None,
            "마무리 UNet에 얼굴 이미지 임베딩이 전달되지 않았습니다.")
    calls.append({"index":len(calls), "ip_scales":observed, "image_embeds_present":True})
