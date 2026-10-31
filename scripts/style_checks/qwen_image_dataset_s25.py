"""S25 그림체 자료 2단계(D-234): 사용자가 고른 Q4 그림체(F-Q4-s43·M-Q4-s43)로 오리지널 성인 캐릭터 72장을 Qwen-Image-2512(Apache-2.0)로 그린다.
이 자료에서 사용자가 30~40장을 고르면 FLUX.2-klein-4B 그림체 LoRA 학습 자료가 된다. 자료 생성용이며 제품 앱에는 들어가지 않는다.
학습 자료 규칙(D-203): 직접 짠 문장, 'Original adult character'로 시작, 작가·작품 이름 없음. 어려 보이는 옷(교복 등)·노출 옷 없음.
장면은 D-232에서 사용자가 방향이 맞다고 본 쪽(서 있는 자세·부드러운 빛·밝은 색)만. 구도는 전신 48·허벅지 위 16·상반신 8(S24는 전신이 무릎에서 잘림).
사용:
  generate (클라우드 A100): python qwen_image_dataset_s25.py generate <모델 폴더> <출력 폴더> [시작 번호, 기본 1]
  check    (이 PC):        python qwen_image_dataset_s25.py check <출력 폴더>   — WD 태그 안전 검사·출처 SHA·고르기 판 6장
"""
import hashlib
import json
import random
import sys
import time
from pathlib import Path

COUNT = 72
SIZE = (1104, 1472)          # 3:4 세로, S24와 같음
STEPS, CFG = 50, 4.0
STYLE = ("Style: vibrant high-quality anime key visual illustration, crisp clean line art, glossy vivid colours, smooth cel shading with "
         "soft gradients, luminous detailed eyes with bright highlights, shiny silky hair with light streaks, glowing light with soft "
         "bloom and floating light particles, polished professional finish.")   # S24 Q4 그대로(사용자 선택)
TAIL = " No text, no lettering, no logo, no watermark, a single person."
NEGATIVE = "text, watermark, logo, multiple people, cropped head, cut off feet, extra limbs, deformed hands, child, childlike"

WHO = ("a woman in her twenties", "a man in his twenties", "a woman in her thirties", "a man in his thirties",
       "a tall woman in her late twenties", "a broad-shouldered man in his late twenties")
SKIN = ("fair", "light", "olive", "tan", "brown", "dark brown")
HAIR = ("short black hair", "long straight chestnut hair", "a high blonde ponytail", "short spiky silver hair", "wavy shoulder-length red hair",
        "a navy blue bob", "long braided brown hair", "short curly black hair", "an undercut with dark green hair on top", "long white hair tied low",
        "pink hair in a loose side braid", "messy ash-grey hair")
EYES = ("brown", "green", "blue", "amber", "grey", "violet", "hazel")
OUTFIT = ("a beige trench coat over a turtleneck sweater and slacks", "a knit cardigan, a blouse and a long pleated skirt",
          "a leather jacket, a plain t-shirt and jeans", "a tailored navy business suit", "an oversized hoodie and cargo pants",
          "a military-style long coat with boots", "a fantasy knight armour with a cloak", "a hooded mage robe with embroidered trim",
          "a denim jacket over a sundress that reaches the ankles", "a chef's white jacket and apron", "a mechanic's jumpsuit with tool belt",
          "a wool coat, scarf and gloves")
ACCESSORY = ("", "", "small earrings", "round glasses", "a hair clip", "a wristwatch", "a scarf")
PLACE = ("in a sunlit greenhouse full of plants", "on a quiet street in soft morning light", "in a bright library with tall windows",
         "on a seaside promenade under a pale blue sky", "in a flower garden in spring", "in a cosy cafe with warm light",
         "on a stone bridge over a calm river", "in a fantasy town square with pastel houses", "in a park with cherry blossoms",
         "on a rooftop garden at golden hour with soft light", "in a white art gallery", "at a train platform on a clear afternoon")
ACTION = ("standing and looking at the viewer", "walking slowly toward the viewer", "standing with one hand raised to brush the hair back",
          "standing with hands in the coat pockets", "holding a book at the side", "turning slightly and smiling over the shoulder",
          "standing with arms loosely crossed", "holding a cup of coffee")
FRAMING = (["Full body shot from head to shoes: the whole figure is visible with space above the head and below the feet, "
            "both feet standing on the ground."] * 48
           + ["Cowboy shot from the head to the mid thigh."] * 16
           + ["Upper body portrait from the head to the waist."] * 8)


def describe(i):
    rng = random.Random(7000 + i)
    accessory = rng.choice(ACCESSORY)
    return (f"Original adult character: {rng.choice(WHO)} with {rng.choice(SKIN)} skin, {rng.choice(HAIR)} and {rng.choice(EYES)} eyes, "
            f"wearing {rng.choice(OUTFIT)}" + (f" and {accessory}" if accessory else "") +
            f", {rng.choice(ACTION)} {rng.choice(PLACE)}.")


def plan():
    framing = list(FRAMING)
    random.Random(7).shuffle(framing)
    return [dict(id=f"Q{i:03d}", seed=9000 + i, prompt=f"{describe(i)} {framing[i - 1]} {STYLE}{TAIL}") for i in range(1, COUNT + 1)]


def generate(model, out, start):
    import torch
    from diffusers import DiffusionPipeline
    out.mkdir(parents=True, exist_ok=True)
    pipe = DiffusionPipeline.from_pretrained(str(model), torch_dtype=torch.bfloat16).to("cuda")
    log = out / "generate.jsonl"
    for item in plan()[start - 1:]:
        if (out / f"{item['id']}.png").exists():
            continue   # Pod가 끊겼다 다시 이을 때 이미 그린 것은 건너뛴다
        t0 = time.time()
        image = pipe(prompt=item["prompt"], negative_prompt=NEGATIVE, width=SIZE[0], height=SIZE[1], num_inference_steps=STEPS,
                     true_cfg_scale=CFG, generator=torch.Generator(device="cuda").manual_seed(item["seed"])).images[0]
        image.save(out / f"{item['id']}.png")
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(item, seconds=round(time.time() - t0, 1), gpu=torch.cuda.get_device_name(0))) + "\n")
        print(item["id"], round(time.time() - t0, 1), "s", flush=True)
    revision = (model.parent / "model-revision.txt").read_text().strip() if (model.parent / "model-revision.txt").exists() else ""
    (out / "model.json").write_text(json.dumps(dict(model="Qwen/Qwen-Image-2512", revision=revision, license="Apache-2.0"), indent=2))
    print("GENERATE_DONE", flush=True)


def check(out):
    import os
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    sys.path.insert(0, r"D:\Code\genai-lab")
    from PIL import Image, ImageDraw
    from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, WdTagSession
    from genai_lab.free_generation import safety_observation
    logged = {}
    for line in (out / "generate.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        logged[row["id"]] = row
    records = []
    with WdTagSession(ClothingDesignAnalysisSettings(cache_dir=Path(r"D:\genai-cache\huggingface"), local_files_only=True)) as tagger:
        for item in plan():
            path = out / f"{item['id']}.png"
            if not path.exists():
                continue
            safety = safety_observation(dict(tagger.analyze(Image.open(path).convert("RGB")).raw_general_scores))
            records.append(dict(item, seconds=logged.get(item["id"], {}).get("seconds"), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                safety=safety, excluded=bool(safety["blocked"] or safety["review"])))
    usable = [r for r in records if not r["excluded"]]
    W, H = 240, 320
    for n in range(0, len(usable), 12):
        group = usable[n:n + 12]
        sheet = Image.new("RGB", (W * 6, (H + 24) * 2), "white"); dr = ImageDraw.Draw(sheet)
        for k, r in enumerate(group):
            row, col = divmod(k, 6)
            sheet.paste(Image.open(out / f"{r['id']}.png").convert("RGB").resize((W, H)), (col * W, row * (H + 24) + 24))
            dr.rectangle([col * W, row * (H + 24), col * W + 70, row * (H + 24) + 22], fill="black")
            dr.text((col * W + 6, row * (H + 24) + 5), r["id"], fill="white")
        sheet.save(out / f"pick-sheet-{n // 12 + 1}.png")
    model = json.loads((out / "model.json").read_text()) if (out / "model.json").exists() else {}
    (out / "dataset.json").write_text(json.dumps(dict(model, size=list(SIZE), steps=STEPS, true_cfg_scale=CFG, style="S24 Q4 (사용자 선택)",
                                                      negative=NEGATIVE, rule="D-203 직접 짠 문장, 작가·작품 이름 없음", records=records,
                                                      excluded=[r["id"] for r in records if r["excluded"]]), ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print("CHECK_DONE images", len(records), "usable", len(usable), "excluded", [r["id"] for r in records if r["excluded"]], flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "generate":
        generate(Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    else:
        check(Path(sys.argv[2]))
