"""S24 그림체 자료 1단계(D-233): Qwen-Image-2512(Apache-2.0)가 문장만으로 목표 그림체(세이버 참고 4번 화려한 애니메이션 일러스트·
3번 매끈한 반실사)를 그릴 수 있는가. FLUX.2-klein은 못 했다(D-229). 자료 생성용이며 제품 앱에는 들어가지 않는다.
학습 자료 규칙(D-203): 직접 짠 문장, 'Original adult character'로 시작, 작가·작품 이름 없음. 클라우드 A100, bf16 전부 GPU.
사용: python qwen_image_drafts_s24.py <모델 폴더> <출력 폴더>
"""
import json
import sys
import time
from pathlib import Path

import torch
from diffusers import DiffusionPipeline

SUBJECTS = {
    "F": ("Original adult character: a woman in her twenties with shoulder-length chestnut hair tied with a ribbon, green eyes, wearing a "
          "navy coat over a white blouse and a long pleated skirt, standing in a sunlit greenhouse full of plants, one hand raised to "
          "brush the hair back, full body, looking at the viewer."),
    "M": ("Original adult character: a man in his late twenties with short spiky silver hair, amber eyes, wearing a black turtleneck, "
          "a grey long coat and dark trousers, standing on a quiet street in soft morning light, full body, looking at the viewer."),
}
STYLES = {
    "Q4": ("Style: vibrant high-quality anime key visual illustration, crisp clean line art, glossy vivid colours, smooth cel shading with "
           "soft gradients, luminous detailed eyes with bright highlights, shiny silky hair with light streaks, glowing light with soft "
           "bloom and floating light particles, polished professional finish."),
    "Q3": ("Style: polished semi-realistic anime illustration, clean thin line art, smooth flawless skin with soft airbrushed shading, "
           "natural blush, large expressive anime eyes with clear reflections, sleek glossy hair, soft lighting with a gentle rim light, "
           "clean fabric shading, refined high-resolution finish."),
    "Q34": ("Style: polished anime illustration between semi-realistic and vivid key visual, clean line art, smooth soft skin shading, "
            "detailed luminous eyes, glossy hair with fine strands, soft warm light with gentle bloom, pastel background colours, "
            "refined professional finish."),
}
TAIL = " No text, no lettering, no logo, no watermark, a single person."
SEEDS = (42, 43)
SIZE = (1104, 1472)          # 3:4 세로


def main():
    model, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    pipe = DiffusionPipeline.from_pretrained(str(model), torch_dtype=torch.bfloat16).to("cuda")
    records = []
    for subject, text in SUBJECTS.items():
        for key, style in STYLES.items():
            for seed in SEEDS:
                prompt = f"{text} {style}{TAIL}"
                t0 = time.time()
                image = pipe(prompt=prompt, negative_prompt=" ", width=SIZE[0], height=SIZE[1], num_inference_steps=50,
                             true_cfg_scale=4.0, generator=torch.Generator(device="cuda").manual_seed(seed)).images[0]
                name = f"{subject}-{key}-s{seed}.png"
                image.save(out / name)
                records.append(dict(file=name, subject=subject, style=key, seed=seed, prompt=prompt, seconds=round(time.time() - t0, 1)))
                print(name, records[-1]["seconds"], "s", flush=True)
    revision = (model.parent / "model-revision.txt").read_text().strip() if (model.parent / "model-revision.txt").exists() else ""
    (out / "drafts.json").write_text(json.dumps(dict(model="Qwen/Qwen-Image-2512", revision=revision, license="Apache-2.0", size=SIZE,
                                                     steps=50, true_cfg_scale=4.0, gpu=torch.cuda.get_device_name(0), records=records),
                                                ensure_ascii=False, indent=2), encoding="utf-8")
    print("DRAFTS_DONE", flush=True)


if __name__ == "__main__":
    main()
