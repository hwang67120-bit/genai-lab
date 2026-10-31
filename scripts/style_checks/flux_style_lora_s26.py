"""S26 그림체 LoRA 3단계(D-234 다음): 사용자가 S25에서 고른 Qwen-Image Q4 그림으로 FLUX.2-klein 그림체 LoRA 학습 자료를 만든다(이 PC).
BFL·HF 안내: 기본판(FLUX.2-klein-base-4B)으로 학습하고 빠른 판(4단계)에 붙인다. 설명문에는 그려진 내용만 쓰고 그림체는 쓰지 않는다
(그림체는 그림에서 배우게) — 그래서 S25 문장에서 그림체 문장·끝 문장을 빼고 맨 앞에 트리거 낱말을 둔다.
학습은 diffusers v0.38.0의 examples/dreambooth/train_dreambooth_lora_flux2_klein.py(이 PC·Pod와 같은 diffusers)로 클라우드에서.
사용: python flux_style_lora_s26.py <S25 폴더> <자료 폴더> Q003 Q007 ...   (고른 번호)
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from qwen_image_dataset_s25 import STYLE, TAIL, plan   # noqa: E402

TRIGGER = "gnlbq4 style illustration."


def caption(prompt):
    return f"{TRIGGER} {prompt.replace(' ' + STYLE, '').replace(TAIL, '').strip()}"


def main():
    source, out, picks = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    prompts = {item["id"]: item["prompt"] for item in plan()}
    excluded = set(json.loads((source / "dataset.json").read_text(encoding="utf-8"))["excluded"])
    bad = [p for p in picks if p not in prompts or p in excluded or not (source / f"{p}.png").exists()]
    if bad:
        raise SystemExit(f"쓸 수 없는 번호: {bad}")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in picks:
        shutil.copyfile(source / f"{p}.png", out / f"{p}.png")
        rows.append(dict(file_name=f"{p}.png", text=caption(prompts[p]),
                         sha256=hashlib.sha256((out / f"{p}.png").read_bytes()).hexdigest()))
    with (out / "metadata.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(dict(file_name=r["file_name"], text=r["text"]), ensure_ascii=False) + "\n")
    (out / "provenance.json").write_text(json.dumps(dict(
        source="S25 Qwen-Image-2512 (Apache-2.0) 오리지널 성인 캐릭터, 사용자 선택", trigger=TRIGGER, picks=picks, records=rows),
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("DATASET", len(rows), out, flush=True)


if __name__ == "__main__":
    main()
