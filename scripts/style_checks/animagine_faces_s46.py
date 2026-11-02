"""S46 2D 얼굴 자료(사용자 11/2: S44 3판은 옷감 질감만 목표에 가깝고 얼굴은 '전혀 다른 그림체', 사용자 허락으로 Animagine XL 4.0 사용).
cagliostrolab/animagine-xl-4.0(OpenRAIL++, 2b7c1b39)로 성인 원본 캐릭터의 가슴 위 근접 40장 — 그림체 LoRA 2판에서 얼굴을 가르칠 자료.
태그 문장(이 모델의 방식), 작가·작품 이름 없음, 성인(adult, mature)을 명시하고 아이·노출은 부정 문장에. 클라우드 전용.
사용: generate <모델 폴더> <출력 폴더>  /  check <출력 폴더>(이 PC: WD 안전 검사·고르기 판, S25 check와 같음)
"""
import json
import random
import sys
import time
from pathlib import Path

COUNT = 40
SIZE = (896, 1152)
STEPS, CFG = 28, 5.0
WHO = (("1girl", "adult, mature female, solo"), ("1boy", "adult, mature male, solo"))
HAIR = ("short black hair", "long brown hair, straight hair", "silver hair, messy hair", "blonde hair, high ponytail", "red hair, wavy hair",
        "dark blue hair, bob cut", "pink hair, side braid", "ash grey hair, short hair", "white hair, long hair", "green hair, undercut",
        "light brown hair, medium hair, swept bangs", "black hair, long hair, hair between eyes")
EYES = ("brown eyes", "green eyes", "blue eyes", "amber eyes", "grey eyes", "purple eyes", "red eyes")
OUTFIT = ("white shirt, collared shirt", "black jacket, turtleneck", "knit sweater", "denim jacket", "trench coat", "hoodie",
          "leather jacket, t-shirt", "cardigan, blouse", "suit, necktie", "scarf, coat")
PLACE = ("outdoors, garden, flowers, soft sunlight", "indoors, library, bookshelf, warm light", "outdoors, city street, evening, bokeh",
         "outdoors, seaside, blue sky, wind", "indoors, cafe, window, sunlight", "outdoors, forest, dappled sunlight")
QUALITY = "upper body, looking at viewer, detailed eyes, detailed hair, beautiful detailed face, soft lighting, masterpiece, high score, great score, absurdres"
NEGATIVE = ("child, loli, shota, young, nsfw, nude, cleavage, lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, "
            "fewer digits, cropped, worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry")


def plan():
    items = []
    for i in range(1, COUNT + 1):
        rng = random.Random(16000 + i)
        tag, adult = WHO[i % 2]
        prompt = (f"{tag}, original, {adult}, {rng.choice(HAIR)}, {rng.choice(EYES)}, {rng.choice(OUTFIT)}, {rng.choice(PLACE)}, "
                  f"{QUALITY}")
        items.append(dict(id=f"A{i:03d}", seed=16000 + i, prompt=prompt))
    return items


def generate(model, out):
    import torch
    from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline
    out.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(str(model), torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    log = out / "generate.jsonl"
    for item in plan():
        t0 = time.time()
        image = pipe(prompt=item["prompt"], negative_prompt=NEGATIVE, width=SIZE[0], height=SIZE[1], num_inference_steps=STEPS,
                     guidance_scale=CFG, generator=torch.Generator(device="cuda").manual_seed(item["seed"])).images[0]
        image.save(out / f"{item['id']}.png")
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(item, seconds=round(time.time() - t0, 1), gpu=torch.cuda.get_device_name(0))) + "\n")
        print(item["id"], round(time.time() - t0, 1), "s", flush=True)
    revision = (model / "revision.txt").read_text().strip() if (model / "revision.txt").exists() else ""
    (out / "model.json").write_text(json.dumps(dict(model="cagliostrolab/animagine-xl-4.0", revision=revision, license="openrail++",
                                                    negative=NEGATIVE, steps=STEPS, cfg=CFG, size=SIZE), indent=2))
    print("GENERATE_DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "generate":
        generate(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        sys.path.insert(0, str(Path(__file__).parent))
        import qwen_image_dataset_s25 as s25
        s25.plan = plan
        s25.SIZE, s25.STEPS, s25.CFG, s25.STYLE, s25.NEGATIVE = SIZE, STEPS, CFG, "Animagine XL 4.0 tags (S46)", NEGATIVE
        s25.check(Path(sys.argv[2]))
