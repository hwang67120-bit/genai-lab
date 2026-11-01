"""S32 자동 고치기 순환 시험(identity_autofix, 최대 2바퀴): 앞서 앱 흐름으로 만든 작업 폴더마다 원본·후보 태그 비교 → 잃은 태그 →
AI 태그 분류(tag_kinds)로 장면 태그 빼기 → '빠진 특징' 문장 붙여 같은 seed로 다시 → 아직 잃었으면 한 번 더. 사람이 고른 항목 없음.
잃은 태그가 없으면 다시 그리지 않는다(멀쩡한 캐릭터를 건드리지 않는지 = 회귀 확인). CCIP 차이(0.178 미만 같은 캐릭터)도 같이 기록.
바퀴 규칙(3판): 요청은 계속 묶어서(내려놓으면 털까지 다시 잃음), seed마다 첫 진단 태그를 가장 덜 잃은 바퀴의 후보(best_per_seed).
태그 종류: 환경 변수 S32_KINDS에 표(tag_kinds 판정 json)가 있으면 그것, 없으면 이 PC의 4B 분류기.
사용: python autofix_loop_s32.py <출력 폴더> <이름>=<작업 폴더(flux)> ...
"""
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
import gc, json, sys, time
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab"); sys.path.insert(0, r"D:\Code\genai-lab\genai_lab")
from PIL import Image, ImageDraw

from genai_lab.identity_autofix import (MAX_ROUNDS, PRESENT, best_per_seed, fixed_prompt, load_kinds, lost_tags, next_round, remaining,
                                       to_fix)


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    cases = {k: Path(v) for k, v in (a.split("=", 1) for a in sys.argv[2:])}
    import torch
    from genai_lab.clothing_analysis import ClothingDesignAnalysisSettings, WdTagSession
    from genai_lab.style_redraw_flux import GUIDANCE, MAX_SEQUENCE_LENGTH, OUTPUT_SIZE, SEEDS, STEPS, STYLE_LORA_WEIGHT
    from genai_lab.tag_kinds import TagKindClassifier
    from style_redraw_worker import MEMORY_CAP_GIB
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    first = json.loads((next(iter(cases.values())) / "request.json").read_text(encoding="utf-8"))
    model = Path(first["flux_model"])
    settings = ClothingDesignAnalysisSettings(cache_dir=Path(first["model_cache"]), local_files_only=True)
    state = {}
    with WdTagSession(settings) as t:
        for name, R in cases.items():
            prepared = json.loads((R / "prepare.json").read_text(encoding="utf-8"))
            original = dict(t.analyze(Image.open(R / "character.png").convert("RGB")).raw_general_scores)
            results = [dict(t.analyze(Image.open(R / f"result-raw-seed{s}.png").convert("RGB")).raw_general_scores) for s in SEEDS]
            state[name] = dict(R=R, prompt=prepared["prompt"], with_garment=bool(prepared.get("with_garment")), original=original,
                               rounds=[dict(prompt=prepared["prompt"], files=[str(R / f"result-raw-seed{s}.png") for s in SEEDS], scores=results)])
    started = time.time()
    table = load_kinds(os.environ.get("S32_KINDS", ""))
    if table:
        for s in state.values():
            s["kinds"] = {k: table.get(k, dict(kind="body")) for k, v in s["original"].items() if v >= PRESENT}
    else:
        with TagKindClassifier(model / "text_encoder", model / "tokenizer") as c:      # 원본 태그 전체를 한 번에(바퀴마다 다시 올리지 않게)
            for s in state.values():
                s["kinds"] = c.classify([k for k, v in s["original"].items() if v >= PRESENT])
    classify_seconds = round(time.time() - started, 1)
    import huggingface_hub.constants as hub
    hub.HF_HUB_CACHE = str(first["model_cache"])
    from imgutils.metrics import ccip_difference, ccip_extract_feature
    pipe = None
    for rnd in range(1, MAX_ROUNDS + 1):
        todo = {}
        for name, s in state.items():
            last = s["rounds"][-1]
            lost = lost_tags(s["original"], last["scores"])
            prev = s["rounds"][-2]["asked"] if len(s["rounds"]) > 1 else []
            asked = next_round(prev, lost, s["kinds"], s["with_garment"])
            fix = [t for t in asked if t not in prev]
            last.update(lost=lost, fix=fix)
            if fix:
                todo[name] = asked
        if not todo:
            break
        if pipe is None:
            from diffusers import Flux2KleinPipeline
            from diffusers.quantizers import PipelineQuantizationConfig
            quant = PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit", quant_kwargs={
                "load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16, "bnb_4bit_use_double_quant": True},
                components_to_quantize=["transformer", "text_encoder"])
            pipe = Flux2KleinPipeline.from_pretrained(str(model), torch_dtype=torch.bfloat16, quantization_config=quant, low_cpu_mem_usage=True,
                                                      local_files_only=True)
            lora = Path(first["flux_style_lora"])
            pipe.load_lora_weights(str(lora.parent), weight_name=lora.name, adapter_name="style")
            pipe.set_adapters(["style"], adapter_weights=[STYLE_LORA_WEIGHT]); pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling()
        for name, asked in todo.items():
            s = state[name]
            prompt = fixed_prompt(s["prompt"], asked)
            refs = [Image.open(s["R"] / "character.png").convert("RGB")] + (
                [Image.open(s["R"] / "garment.png").convert("RGB")] if s["with_garment"] else [])
            files = []
            for seed in SEEDS:
                t0 = time.time()
                image = pipe(image=refs, prompt=prompt, width=OUTPUT_SIZE[0], height=OUTPUT_SIZE[1], num_inference_steps=STEPS,
                             guidance_scale=GUIDANCE, max_sequence_length=MAX_SEQUENCE_LENGTH,
                             generator=torch.Generator(device="cpu").manual_seed(seed)).images[0]
                path = out / f"{name}-r{rnd}-seed{seed}.png"
                image.save(path); files.append(str(path))
                print(name, rnd, seed, round(time.time() - t0, 1), "s", flush=True)
            s["rounds"][-1]["asked"] = asked
            s["rounds"].append(dict(prompt=prompt, files=files))
        torch.cuda.empty_cache()
        with WdTagSession(settings) as t:
            for name in todo:
                last = state[name]["rounds"][-1]
                last["scores"] = [dict(t.analyze(Image.open(f).convert("RGB")).raw_general_scores) for f in last["files"]]
    pipe = None; gc.collect(); torch.cuda.empty_cache()
    report = {}
    for name, s in state.items():
        last = s["rounds"][-1]
        if "lost" not in last:
            last["lost"] = lost_tags(s["original"], last["scores"]); last["fix"] = to_fix(last["lost"], s["kinds"], s["with_garment"])
        base = ccip_extract_feature(Image.open(s["R"] / "character.png").convert("RGB"))
        rows = []
        watch = s["rounds"][0].get("asked") or []
        picks = best_per_seed(s["original"], [r["scores"] for r in s["rounds"]], watch)
        for i, rnd in enumerate(s["rounds"]):
            rows.append(dict(round=i, fix_asked=rnd.get("asked"), lost=rnd.get("lost"), fix_next=rnd.get("fix"),
                             still_lost_per_seed=[remaining(s["original"], sc, watch) for sc in rnd["scores"]],
                             ccip=[round(float(ccip_difference(base, ccip_extract_feature(Image.open(f).convert("RGB")))), 3) for f in rnd["files"]],
                             files=rnd["files"]))
        report[name] = dict(work=str(s["R"]), with_garment=s["with_garment"], rounds=rows, picks=picks,
                            kinds={k: v["kind"] for k, v in s["kinds"].items()})
        W, H = 300, 398
        sheet = Image.new("RGB", (W * (1 + len(SEEDS)), (H + 22) * len(s["rounds"])), "white"); dr = ImageDraw.Draw(sheet)
        for i, rnd in enumerate(s["rounds"]):
            y = i * (H + 22)
            sheet.paste(Image.open(s["R"] / "character.png").convert("RGB").resize((W, H)), (0, y + 22))
            dr.text((4, y + 5), f"{name} original", fill="black")
            for j, f in enumerate(rnd["files"]):
                sheet.paste(Image.open(f).convert("RGB").resize((W, H)), ((j + 1) * W, y + 22))
                chosen = picks[j] == i
                dr.text(((j + 1) * W + 4, y + 5), f"round {i} seed {SEEDS[j]} ccip {rows[i]['ccip'][j]}" + (" CHOSEN" if chosen else ""),
                        fill="red" if chosen else "black")
        sheet.save(out / f"{name}-rounds.png")
    (out / "autofix-loop.json").write_text(json.dumps(dict(classify_seconds=classify_seconds, cases=report), ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    for name, r in report.items():
        print(name, "picks", r["picks"],
              [dict(round=x["round"], asked=x["fix_asked"], still=x["still_lost_per_seed"], ccip=x["ccip"]) for x in r["rounds"]])


if __name__ == "__main__":
    main()
