"""S40 Pod 디스크가 작을 때(컨테이너 44GB, /dev/shm 64MB — Qwen-Image 57.7GB가 한 번에 안 들어감): 두 단계로 나눠 같은 그림을 그린다.
encode: 문장 인코더만 있는 폴더로 80개 문장 + 부정 문장을 미리 계산해 저장 → (쉘이 문장 인코더를 지우고 그림 본체를 받음) →
draw: 문장 인코더 없이 미리 계산한 값으로 그린다. 그림 seed·크기·단계·CFG는 S25·S40과 같다(qwen_image_dataset_s25.generate와 같은 기록).
사용: python qwen_image_split_pod_s40.py encode <모델 폴더> <값 폴더> | draw <모델 폴더> <값 폴더> <출력 폴더>
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import qwen_image_dataset_s40  # noqa: E402,F401 — s25.plan을 S40 계획으로 바꾼다
import qwen_image_dataset_s25 as s25  # noqa: E402


def encode(model, store):
    import torch
    from diffusers import QwenImagePipeline
    store.mkdir(parents=True, exist_ok=True)
    pipe = QwenImagePipeline.from_pretrained(str(model), transformer=None, torch_dtype=torch.bfloat16).to("cuda")
    with torch.no_grad():
        for item in s25.plan() + [dict(id="NEGATIVE", prompt=s25.NEGATIVE)]:
            embeds, mask = pipe.encode_prompt(item["prompt"], device="cuda")
            torch.save(dict(embeds=embeds.cpu(), mask=mask.cpu()), store / f"{item['id']}.pt")
    print("ENCODE_DONE", len(list(store.glob("*.pt"))), flush=True)


def draw(model, store, out):
    import torch
    from diffusers import QwenImagePipeline
    out.mkdir(parents=True, exist_ok=True)
    pipe = QwenImagePipeline.from_pretrained(str(model), text_encoder=None, tokenizer=None, torch_dtype=torch.bfloat16).to("cuda")
    neg = torch.load(store / "NEGATIVE.pt")
    log = out / "generate.jsonl"
    for item in s25.plan():
        if (out / f"{item['id']}.png").exists():
            continue
        e = torch.load(store / f"{item['id']}.pt")
        t0 = time.time()
        image = pipe(prompt_embeds=e["embeds"].to("cuda"), prompt_embeds_mask=e["mask"].to("cuda"),
                     negative_prompt_embeds=neg["embeds"].to("cuda"), negative_prompt_embeds_mask=neg["mask"].to("cuda"),
                     width=s25.SIZE[0], height=s25.SIZE[1], num_inference_steps=s25.STEPS, true_cfg_scale=s25.CFG,
                     generator=torch.Generator(device="cuda").manual_seed(item["seed"])).images[0]
        image.save(out / f"{item['id']}.png")
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(item, seconds=round(time.time() - t0, 1), gpu=torch.cuda.get_device_name(0))) + "\n")
        print(item["id"], round(time.time() - t0, 1), "s", flush=True)
    revision = (model.parent / "model-revision.txt").read_text().strip() if (model.parent / "model-revision.txt").exists() else ""
    (out / "model.json").write_text(json.dumps(dict(model="Qwen/Qwen-Image-2512", revision=revision, license="Apache-2.0",
                                                    note="문장 인코더로 미리 계산한 값으로 그림(디스크 부족, S40)"), indent=2))
    print("GENERATE_DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "encode":
        encode(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        draw(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
