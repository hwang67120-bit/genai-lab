"""CPU 확인(생성 없음): 작업 프로세스와 같은 순서로 파이프라인을 만들고 가속 선택지를 적용(스케줄러 교체 → pipe.load_lora_weights).
GPU 오프로드·생성은 하지 않는다."""
import json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from diffusers import FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig, QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from genai_lab.qwen_accelerators import accelerator, lora_path
from genai_lab.qwen_pose_settings import QwenPoseSettings
from genai_lab.qwen_preservation import file_sha
settings = QwenPoseSettings.load(ROOT / "configs/qwen_edit_settings.json")
accel = accelerator("lightning8"); path = lora_path("lightning8", settings.model_root)
report = dict(lora=str(path), sha_ok=file_sha(str(path)) == accel["sha256"])
tr = QwenImageTransformer2DModel.from_single_file(settings.gguf_file, quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16, config=settings.model_root, subfolder="transformer", local_files_only=True)
pipe = QwenImageEditPlusPipeline.from_pretrained(settings.model_root, transformer=tr, text_encoder=None, tokenizer=None, processor=None,
    torch_dtype=torch.bfloat16, local_files_only=True)
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(accel["scheduler"])
pipe.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name="lightning8")
report["active_adapters"] = list(pipe.get_active_adapters())
report["lora_layers"] = sum(1 for _, m in tr.named_modules() if hasattr(m, "lora_A") and len(m.lora_A) > 0)
report["scheduler"] = {k: pipe.scheduler.config[k] for k in ("base_shift", "max_shift", "shift_terminal", "use_dynamic_shifting")}
pipe.scheduler.set_timesteps(accel["steps"], mu=float(accel["scheduler"]["base_shift"]))
report["sigmas"] = [round(float(s), 4) for s in pipe.scheduler.sigmas]
print(json.dumps(report, ensure_ascii=False, indent=1))
(Path(__file__).parent / "worker-sequence.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
