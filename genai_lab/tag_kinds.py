"""태그 종류를 AI가 스스로 나눈다(S32, 사용자 11/1 "수동 조정이 아닌 AI가 스스로 고칠 수 있는 계산법").

FLUX.2-klein-4B 안의 문장 인코더(Qwen3-4B, Apache-2.0)에게 태그마다 '몸의 고유 특징(body) · 입은 것(outfit) · 장면에 따라 바뀌는 것(scene)'
중 무엇인지 묻는다. 답은 첫 토큰 A/B/C의 확률(생성하지 않음). 작은 모델은 첫 선택지를 고르는 버릇이 있어 선택지 순서를 세 번 바꾸고,
순서마다 빈 태그('N/A')의 답 분포로 나눠(문맥 보정) 평균한다. 예/아니오 두 질문 방식은 거의 모두 '예'로 기울어 버렸다(S32 시험).
"""
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


class TagKindClassifier:
    """문장 인코더를 4비트로 올려 태그 종류를 판정한다. with 문으로 쓰고 나오면 GPU 메모리를 돌려준다."""

    def __init__(self, text_encoder_dir, tokenizer_dir, batch=32, four_bit=True):
        """four_bit=False: 큰 GPU(클라우드)에서 다른 Qwen3 언어 모델로 표 전체를 만들 때(bf16)."""
        self.text_encoder_dir, self.tokenizer_dir, self.batch, self.four_bit = str(text_encoder_dir), str(tokenizer_dir), batch, four_bit

    def __enter__(self):
        import torch
        from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3ForCausalLM
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(self.tokenizer_dir, local_files_only=True)
        self.tok.padding_side = "left"
        quant = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
                 if self.four_bit else None)
        self.model = Qwen3ForCausalLM.from_pretrained(self.text_encoder_dir, local_files_only=True, device_map="cuda:0", dtype=torch.bfloat16,
                                                      quantization_config=quant).eval()
        self.letters = [self.tok.encode(k, add_special_tokens=False)[0] for k in "ABC"]
        self.null = {order: self._ask(["N/A"], order)[0] for order in ORDERS}
        return self

    def __exit__(self, *exc):
        import gc
        self.model = None
        gc.collect()
        self.torch.cuda.empty_cache()

    def _ask(self, tags, order):
        texts = [self.tok.apply_chat_template(
            [dict(role="user", content=QUESTION.format(tag=t.replace("_", " "), **{k: KINDS[c] for k, c in zip("ABC", order)}))],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for t in tags]
        enc = self.tok(texts, return_tensors="pt", padding=True).to("cuda:0")
        with self.torch.no_grad():
            logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :]      # 마지막 자리만(전체 어휘 × 길이는 메모리 초과)
        return self.torch.softmax(logits[:, self.letters].float(), dim=-1).cpu()

    def classify(self, tags):
        """태그 목록 → {태그: dict(kind, body, outfit, scene)}."""
        table = {}
        tags = list(dict.fromkeys(tags))
        for n in range(0, len(tags), self.batch):
            group = tags[n:n + self.batch]
            total = self.torch.zeros(len(group), 3)
            for order in ORDERS:
                p = self._ask(group, order) / self.null[order]
                p = p / p.sum(dim=1, keepdim=True)
                total += p[:, [order.index(k) for k in KINDS]]
            total /= len(ORDERS)
            for t, values in zip(group, total.tolist()):
                row = {k: round(v, 3) for k, v in zip(KINDS, values)}
                table[t] = dict(kind=max(row, key=row.get), **row)
        return table
