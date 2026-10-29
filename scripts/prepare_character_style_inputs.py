"""실제 T014·목표 이미지로 CPU 조건 묶음과 전달 기록을 만든다. 생성 옵션은 없다."""
import argparse
import html
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genai_lab.character_style_inputs import prepare_character_style_bundle, write_json


def inspect_cpu_delivery(bundle, options):
    """실제 Diffusers 마스크 처리와 호출 인자를 검사한다. 모델은 불러오지 않는다."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from unittest.mock import patch
    import socket
    import inspect
    import torch
    import diffusers
    from diffusers import StableDiffusionXLControlNetPipeline, StableDiffusionXLControlNetImg2ImgPipeline
    from genai_lab.character_style_conditioning import build_conditioning_call

    def forbidden(*args, **kwargs):
        raise RuntimeError("CPU verification forbids network and CUDA initialization")

    checks = {}
    with patch.object(socket.socket, "connect", forbidden), patch.object(torch.cuda, "_lazy_init", forbidden):
        call = build_conditioning_call(bundle, options)
        from dataclasses import replace
        identity_off = build_conditioning_call(bundle, replace(options, identity_visual=False))
        structure_off = build_conditioning_call(bundle, replace(options, structure=False))
        check_names = inspect.signature(StableDiffusionXLControlNetPipeline.__call__).parameters
        checks["installed_text_to_image_accepts_arguments"] = set(call.kwargs).issubset(check_names)
        checks["img2img_is_distinguishable"] = "strength" in inspect.signature(StableDiffusionXLControlNetImg2ImgPipeline.__call__).parameters
        checks["all_masks_are_cpu"] = call.evidence["mask_devices"] == ["cpu", "cpu"]
        masks = call.kwargs["cross_attention_kwargs"]["ip_adapter_masks"]
        checks["identity_has_spatial_limit"] = 0 < int(torch.count_nonzero(masks[1])) < masks[1].numel()
        checks["style_covers_canvas"] = bool(torch.all(masks[0] == 1))
        checks["identity_off_has_no_source_crop"] = identity_off.kwargs["ip_adapter_image"][1].getbbox() is None
        checks["identity_off_zero_scale_and_mask"] = identity_off.adapter_scales[1] == 0 and not bool(torch.count_nonzero(identity_off.kwargs["cross_attention_kwargs"]["ip_adapter_masks"][1]))
        checks["identity_off_keeps_style_and_traits"] = (
            call.adapter_scales[0] == identity_off.adapter_scales[0]
            and call.kwargs["prompt"] == identity_off.kwargs["prompt"]
            and call.kwargs["ip_adapter_image"][0].tobytes() == identity_off.kwargs["ip_adapter_image"][0].tobytes())
        checks["structure_off_zero_image_and_scale"] = (
            structure_off.kwargs["image"].getbbox() is None
            and structure_off.kwargs["controlnet_conditioning_scale"] == 0)
        checks["structure_off_keeps_references"] = all(
            a.tobytes() == b.tobytes()
            for a, b in zip(call.kwargs["ip_adapter_image"], structure_off.kwargs["ip_adapter_image"]))
        checks["no_img2img_or_color_restore"] = (
            not {"strength", "latents", "init_image", "color_anchor"} & call.kwargs.keys()
            and call.evidence["postprocessing"] == [])
    report = {
        "state": "cpu_pass" if all(checks.values()) else "cpu_failed",
        "checks": checks, "evidence": call.evidence,
        "versions": {"torch": torch.__version__, "diffusers": diffusers.__version__},
        "gpu_generation_calls": 0, "model_weights_loaded": False,
        "probe_values_are_not_tuned_or_execution_approval": True,
        "unverified": [
            "Actual model weights, ordered adapters and pose checkpoint compatibility",
            "Actual tokenizer lengths: guarded at dispatch, unavailable without selected local model",
            "Identity preservation, style leakage and target style quality",
            "Manual pose annotations are approximate, not automatically detected",
        ],
    }
    write_json(Path(bundle)/"cpu-report.json", report)
    if not all(checks.values()):
        raise RuntimeError("CPU delivery failed: " + ", ".join(k for k, v in checks.items() if not v))
    return report


def write_review_gallery(output, manifest):
    """실제로 전달할 PNG와 각 경로의 한계를 나란히 표시한다."""
    cards = [
        ("원본 · 기록용", "source-audit-only.png", "전체 원본은 생성 시작 이미지로 전달하지 않습니다."),
        ("특징 참조 · 얼굴", "identity-face.png", "얼굴 참조에도 원본 채색이 남습니다. 독립적인 켜기/끄기로 영향을 확인할 수 있습니다."),
        ("얼굴 참조 적용 영역", "identity-mask.png", "흰 영역에만 얼굴 참조를 적용합니다. 목표 그림체 참조는 얼굴에도 적용됩니다."),
        ("형태 지도", "structure-pose.png", "원본 픽셀 없이 주석 좌표만 그렸습니다. 이번 좌표는 수동 관찰이며 자동 추출이나 실제 키 측정이 아닙니다."),
        ("목표 그림체 참조", "style-reference.png", "원본 캐릭터 참조와 별도의 IP-Adapter 입력입니다. 목표의 인물 내용이 섞일 가능성은 아직 검증하지 않았습니다."),
    ]
    card_html = "".join(
        '<article><h2>'+html.escape(title)+'</h2><a href="'+name+'"><img src="'+name+'" alt="'+html.escape(title)+'"></a><p>'+html.escape(note)+'</p></article>'
        for title,name,note in cards)
    rows = "".join("<tr><td>"+html.escape(t["name"])+"</td><td>"+html.escape(t["description_ko"])+"</td><td>"+html.escape(t["status"])+"</td></tr>"
                   for t in manifest["profile"]["traits"])
    page = """<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>캐릭터 특징·그림체 분리 입력 — CPU 검증</title><style>
body{background:#111c26;color:#e9f1f8;font:16px/1.6 'Malgun Gothic',sans-serif;margin:0;padding:28px}main{max-width:1600px;margin:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}article{background:#1b2b39;border:1px solid #405164;padding:15px;border-radius:10px}h1{font-size:26px}h2{font-size:19px}img{width:100%;height:380px;object-fit:contain;background:#0a1118}p{color:#c3d2de}a{color:#8be0d2}
table{border-collapse:collapse;width:100%;margin:20px 0}th,td{padding:12px;border:1px solid #405164;text-align:left}.note{background:#233b47;padding:18px;border-left:4px solid #8be0d2}
</style><main><h1>캐릭터 특징·그림체 분리 입력</h1>
<div class="note">CPU 입력 준비와 전달 경로 검사입니다. GPU 생성·모델 다운로드·새 이미지 품질 합격은 포함하지 않았습니다.</div>
<p>기존 원본 전체 입력과 색 복원 대신, 특징 문장·얼굴 참조·형태 지도·목표 그림체를 분리했습니다. 완벽한 그림체 제거 또는 동일성 보장은 아닙니다.</p>
<div class="grid">""" + card_html + """</div><h2>문장으로 전달하는 특징</h2><table><thead><tr><th>항목</th><th>보존 후보</th><th>근거 상태</th></tr></thead><tbody>""" + rows + """</tbody></table>
<p>화면 좌우를 기준으로 기록했습니다. 특징 주석은 기존 이미지의 관찰이며 사용자의 새 품질 승인으로 간주하지 않습니다.</p>
<p>형태 지도는 관절·자세 보조입니다. 얼굴의 개성, 머리 모양, 의상 디자인 전체를 보존하는 장치는 아닙니다. 원본과 다른 구도로 생성할 때 적용 위치는 다시 준비해야 합니다.</p>
<p><a href="manifest.json">입력 SHA·경로·한계</a> · <a href="cpu-report.json">CPU 검사 결과</a></p></main></html>"""
    (output/"gallery.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--style", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    profile = json.loads(args.annotations.read_text(encoding="utf-8"))
    manifest = prepare_character_style_bundle(args.source, args.style, profile, args.output)
    from genai_lab.character_style_conditioning import ConditioningOptions
    options = ConditioningOptions(**profile["cpu_probe_options"])
    report = inspect_cpu_delivery(args.output, options)
    write_review_gallery(args.output, manifest)
    print(json.dumps({"state": report["state"], "checks": len(report["checks"]),
                      "output": str(args.output), "gpu_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
