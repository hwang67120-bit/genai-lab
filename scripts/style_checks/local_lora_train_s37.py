"""S37 이 PC(8GB, 상한 6.5GiB)에서 캐릭터 LoRA 학습이 되는가(사용자 11/1 "자동 학습이 배포 이후 제품 흐름의 부품으로 작동할 수 있냐").
diffusers v0.38.0 공식 train_dreambooth_lora_flux2_klein.py를 그대로 쓰되 세 군데만 바꿔 실행한다(원본 파일은 고치지 않음):
① 본체 4비트(스크립트의 --bnb_quantization_config_path) ② 문장 인코더(bf16 8GB)는 GPU에 올리지 않고 CPU에서 설명문만 미리 계산
③ datasets 패키지 대신 같은 모양의 작은 대체물(이 가상환경에 새 패키지를 넣지 않으려고 — metadata.jsonl 읽기만).
메모리 상한 6.5GiB를 학습 프로세스에 건다.
사용: python local_lora_train_s37.py <train_dreambooth_lora_flux2_klein.py 경로> <자료 폴더> <출력 폴더> <트리거> [단계, 기본 400] [크기, 기본 512]
"""
import json, os, sys, types
from pathlib import Path

script, data, out, trigger = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
steps = sys.argv[5] if len(sys.argv) > 5 else "400"
size = sys.argv[6] if len(sys.argv) > 6 else "512"
sys.path.insert(0, r"D:\Code\genai-lab\genai_lab")
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")

import torch  # noqa: E402
from style_redraw_worker import MEMORY_CAP_GIB  # noqa: E402
torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)


class _Split(dict):
    @property
    def column_names(self):
        return list(self)


def load_dataset(path, *args, **kwargs):
    """imagefolder + metadata.jsonl만(학습 스크립트가 쓰는 것: ["train"].column_names, ["train"][열])."""
    from PIL import Image
    rows = [json.loads(line) for line in (Path(path) / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"train": _Split(image=[Image.open(Path(path) / r["file_name"]).convert("RGB") for r in rows], text=[r["text"] for r in rows])}


# transformers·diffusers는 가져올 때 datasets가 있는지 살핀다 → 대체물을 넣기 전에 먼저 가져와 '없음'으로 기억하게 한다
import transformers, diffusers, peft, accelerate  # noqa: E401,E402,F401
sys.modules["datasets"] = types.SimpleNamespace(load_dataset=load_dataset)

source = script.read_text(encoding="utf-8")
patches = [
    ("""                        with offload_models(text_encoding_pipeline, device=accelerator.device, offload=args.offload):
                            prompt_embeds, text_ids = compute_text_embeddings(batch["prompts"], text_encoding_pipeline)""",
     """                        with offload_models(text_encoding_pipeline, device="cpu", offload=args.offload):   # S37: 문장 인코더는 CPU에서
                            prompt_embeds, text_ids = compute_text_embeddings(batch["prompts"], text_encoding_pipeline)"""),
    ("""                    prompt_embeds = prompt_embeds_cache[step]
                    text_ids = text_ids_cache[step]""",
     """                    prompt_embeds = prompt_embeds_cache[step].to(accelerator.device)   # S37: CPU에서 계산한 것을 GPU로
                    text_ids = text_ids_cache[step].to(accelerator.device)"""),
]
for old, new in patches:
    assert source.count(old) == 1, old[:60]
    source = source.replace(old, new)
out.mkdir(parents=True, exist_ok=True)
bnb = out / "bnb-4bit.json"
bnb.write_text(json.dumps(dict(load_in_4bit=True, bnb_4bit_quant_type="nf4")), encoding="utf-8")
sys.argv = [str(script),
            "--pretrained_model_name_or_path=D:/genai-cache/models/FLUX.2-klein-base-4B",
            f"--dataset_name={data}", "--image_column=image", "--caption_column=text", f"--instance_prompt={trigger}",
            f"--output_dir={out}", "--mixed_precision=bf16",
            # 세로·정사각·가로 묶음(전신 세로 그림을 정사각으로 자르지 않게), 긴 변 = 크기 × 1.375
            f"--aspect_ratio_buckets={int(int(size) * 1.375) // 16 * 16},{size};{size},{size};{size},{int(int(size) * 1.375) // 16 * 16}",
            "--bnb_quantization_config_path=" + str(bnb), "--offload", "--gradient_checkpointing", "--cache_latents",
            "--train_batch_size=1", "--gradient_accumulation_steps=1", "--guidance_scale=1",
            "--rank=16", "--lora_alpha=16", "--optimizer=adamW", "--use_8bit_adam", "--learning_rate=1e-4",
            "--lr_scheduler=constant", "--lr_warmup_steps=50", f"--max_train_steps={steps}", "--checkpointing_steps=200", "--seed=0",
            "--max_sequence_length=256", "--dataloader_num_workers=0"]
code = compile(source, str(script), "exec")
globals_ = {"__name__": "__main__", "__file__": str(script)}
exec(code, globals_)
print("PEAK_GIB", round(torch.cuda.max_memory_reserved() / 2**30, 2), flush=True)
