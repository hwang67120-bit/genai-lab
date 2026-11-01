"""S32 태그 종류를 AI가 스스로 나눈다(사용자 11/1 "수동 조정이 아닌 AI가 스스로 고칠 수 있는 계산법").
데이터에서 배우는 방법은 지금 파이프라인이 자주 놓치는 특징(어두운 피부)을 '바뀌어도 되는 태그'로 배웠다 → 대신 FLUX.2-klein-4B 안의
문장 인코더(Qwen3-4B, Apache-2.0, 새로 받지 않음)에게 WD v3 일반 태그 전체를 한 번씩 물어 표로 만든다:
  body = 캐릭터 몸의 고유 특징(새 그림에서도 같아야 함) · outfit = 옷·장신구(옷을 바꾸지 않으면 유지) · scene = 장면·자세·표정·구도·배경·소품(바뀌어도 됨).
답은 첫 토큰 A/B/C의 확률(생성하지 않음, 순서 세 번·문맥 보정). 태그 어휘가 고정이라 표 하나로 앱 실행 중에는 언어 모델이 필요 없다. GPU 4비트, 상한 6.5GiB.
사용: python classify_tags_llm_s32.py <출력 json> [태그 수 제한]
"""
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
import csv, json, sys, time
from pathlib import Path
sys.path.insert(0, r"D:\Code\genai-lab"); sys.path.insert(0, r"D:\Code\genai-lab\genai_lab")

MODEL = Path(r"D:\genai-cache\models\FLUX.2-klein-4B-e7b7dc27")
TAGS_CSV = next(Path(r"D:\genai-cache\huggingface\models--SmilingWolf--wd-vit-tagger-v3\snapshots").glob("*/selected_tags.csv"))
# 종류 설명. 선택지 글자(A·B·C)는 순서를 바꿔 세 번 묻는다 — 작은 언어 모델은 첫 선택지를 고르는 버릇이 있어(첫 시험: 웃음·앉기까지 몸 특징),
# 순서마다 빈 태그('N/A')의 답 분포로 나눠 버릇을 빼고(문맥 보정) 세 번을 평균한다.
KINDS = {
    "body": ("a permanent trait of the character's own body that must look the same in any new picture of this character "
             "(species or body type, fur, scales, skin colour, face, nose or muzzle, hair colour, hair length, eye colour, ears, tail, wings, "
             "horns, body markings, scars, tattoos, build)"),
    "outfit": "clothing, footwear, jewellery or an accessory the character is wearing",
    "scene": ("something that changes from picture to picture (pose, action, gesture, expression, gaze, camera angle, framing, background, "
              "place, weather, lighting, objects held or nearby, other people, number of people, art style, image quality)"),
}
ORDERS = (("body", "outfit", "scene"), ("outfit", "scene", "body"), ("scene", "body", "outfit"))
QUESTION = ('An image tagger gave the tag "{tag}" to a picture of an illustrated character. Which kind of tag is it?\n'
            "A = {A}.\nB = {B}.\nC = {C}.\nAnswer with one letter: A, B or C.")
YES_NO = {
    "body": ('Is "{tag}" a physical feature of the own body of the character, such as hair colour, hair length, eye colour, skin colour, fur, '
             'scales, nose or muzzle, animal ears, tail, wings, horns, body build or body markings? Answer Yes or No.'),
    "outfit": ('Is "{tag}" something the character wears, such as clothing, shoes, a hat, jewellery or another accessory? '
               'Answer Yes or No.'),
}
BATCH = 32


def main():
    out = Path(sys.argv[1]); limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3ForCausalLM
    from style_redraw_worker import MEMORY_CAP_GIB
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP_GIB * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    with TAGS_CSV.open(encoding="utf-8") as f:
        tags = [row["name"] for row in csv.DictReader(f) if row["category"] == "0"]
    tags = tags[:limit] if limit else tags
    tok = AutoTokenizer.from_pretrained(str(MODEL / "tokenizer"), local_files_only=True)
    tok.padding_side = "left"
    model = Qwen3ForCausalLM.from_pretrained(str(MODEL / "text_encoder"), local_files_only=True, device_map="cuda:0",
                                             quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                                                    bnb_4bit_compute_dtype=torch.bfloat16),
                                             torch_dtype=torch.bfloat16).eval()
    letters = [tok.encode(k, add_special_tokens=False)[0] for k in "ABC"]

    def ask(group, order):
        texts = [tok.apply_chat_template([dict(role="user", content=QUESTION.format(tag=g.replace("_", " "), **{k: KINDS[c] for k, c in zip("ABC", order)}))],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False) for g in group]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda:0")
        with torch.no_grad():
            logits = model(**enc, logits_to_keep=1).logits[:, -1, :]   # 마지막 자리만(전체 어휘 × 길이는 메모리 초과)
        return torch.softmax(logits[:, letters].float(), dim=-1).cpu()

    yes_no = [tok.encode(k, add_special_tokens=False)[0] for k in ("Yes", "No")]

    def ask_yes(group, question):
        texts = [tok.apply_chat_template([dict(role="user", content=question.format(tag=g.replace("_", " ")))], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False) for g in group]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda:0")
        with torch.no_grad():
            logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
        return torch.softmax(logits[:, yes_no].float(), dim=-1)[:, 0].cpu()

    null = {k: float(ask_yes(["N/A"], q)[0]) for k, q in YES_NO.items()}
    started = time.time()
    table = {}
    for n in range(0, len(tags), BATCH):
        group = tags[n:n + BATCH]
        answers = {}
        for k, q in YES_NO.items():
            yes = ask_yes(group, q)
            # 문맥 보정: 빈 태그의 '예' 비율로 나눈 승산비 → 0~1
            odds = (yes / (1 - yes).clamp(min=1e-6)) / (null[k] / max(1 - null[k], 1e-6))
            answers[k] = (odds / (1 + odds)).tolist()
        for i, g in enumerate(group):
            body, outfit = answers["body"][i], answers["outfit"][i]
            kind = "body" if body >= 0.5 and body >= outfit else "outfit" if outfit >= 0.5 else "scene"
            table[g] = dict(kind=kind, body=round(body, 3), outfit=round(outfit, 3))
        if n % (BATCH * 20) == 0:
            print(n, len(tags), round(time.time() - started, 1), "s", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(schema=1, made=time.strftime("%Y-%m-%d %H:%M:%S"), model="FLUX.2-klein-4B e7b7dc27 text_encoder (Qwen3-4B, Apache-2.0), 4-bit",
                                   tagger=str(TAGS_CSV.parent.name), questions=YES_NO, calibration="예/아니오 두 질문, 빈 태그(N/A)로 문맥 보정", seconds=round(time.time() - started, 1),
                                   peak_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2), table=table), ensure_ascii=False, indent=0),
                   encoding="utf-8")
    print("DONE", len(table), "tags", {k: sum(v["kind"] == k for v in table.values()) for k in KINDS}, flush=True)


if __name__ == "__main__":
    main()
