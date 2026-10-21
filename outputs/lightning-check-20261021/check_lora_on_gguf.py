"""CPU 확인(생성 없음): 가속 LoRA(8단계 bf16)가 지금 쓰는 GGUF 변환기(Q4_K_M)에 붙는지. 2026-10-21, 사용자 승인 범위.
확인: ① LoRA 상태 사전 변환 ② 변환기에 어댑터 주입(빠진 키·안 쓰인 키) ③ GGUF 층 하나에서 LoRA 켬/끔 출력 차이(실제로 더해지는지)."""
import json, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from diffusers import GGUFQuantizationConfig, QwenImageEditPlusPipeline, QwenImageTransformer2DModel
settings = json.loads((ROOT / "configs/qwen_edit_settings.json").read_text(encoding="utf-8"))
gguf = next(Path("D:/genai-cache/models/Qwen-Image-Edit-2511-GGUF").glob("*.gguf"))
lora = Path("D:/genai-cache/models/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors")
report = dict(gguf=str(gguf), lora=str(lora))
t = time.time()
tr = QwenImageTransformer2DModel.from_single_file(str(gguf), quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                                                  torch_dtype=torch.bfloat16, config=settings["model_root"], subfolder="transformer", local_files_only=True)
report["load_seconds"] = round(time.time() - t, 1)
state = QwenImageEditPlusPipeline.lora_state_dict(str(lora.parent), weight_name=lora.name)
state = state[0] if isinstance(state, tuple) else state
report["lora_tensors"] = len(state)
report["lora_key_examples"] = list(state)[:3]
before = {n for n, _ in tr.named_modules()}
try:
    tr.load_lora_adapter(state, adapter_name="lightning", prefix="transformer" if any(k.startswith("transformer.") for k in state) else None)
    report["injected"] = True
except Exception as error:
    report["injected"] = False
    report["error"] = f"{type(error).__name__}: {error}"[:800]
lora_layers = [n for n, m in tr.named_modules() if hasattr(m, "lora_A") and len(getattr(m, "lora_A", {})) > 0]
report["lora_layers"] = len(lora_layers)
report["lora_layer_examples"] = lora_layers[:3]
if lora_layers:
    name = lora_layers[0]
    layer = dict(tr.named_modules())[name]
    report["base_layer_type"] = type(getattr(layer, "base_layer", layer)).__name__
    x = torch.randn(1, 4, layer.in_features, dtype=torch.bfloat16)
    with torch.no_grad():
        on = layer(x)
        tr.disable_adapters()
        off = layer(x)
        tr.enable_adapters()
    report["lora_effect_mean_abs"] = float((on.float() - off.float()).abs().mean())
    report["output_finite"] = bool(torch.isfinite(on).all())
print(json.dumps(report, ensure_ascii=False, indent=1))
(Path(__file__).parent / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
