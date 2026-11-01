"""S43 그림체 LoRA 2판 자료: 1판에서 사용자가 고른 S25 그림(Q…)과 2판 S40에서 고른 그림(R…)을 합쳐 그림 + metadata.jsonl.
설명문 규칙은 1판(S26)과 같다: 맨 앞 트리거 낱말, 그림체 문장·끝 문장은 빼고 그려진 내용만.
사용: python flux_style_lora_data_s43.py <자료 폴더> <S25 폴더> <S40 폴더> Q001 Q002 … R003 R007 …
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import qwen_image_dataset_s25 as s25  # noqa: E402

S25_PLAN = {item["id"]: item["prompt"] for item in s25.plan()}          # S40을 가져오기 전에(그쪽이 s25.plan을 바꾼다)
import qwen_image_dataset_s40 as s40  # noqa: E402
S40_PLAN = {item["id"]: item["prompt"] for item in s40.plan()}
from flux_style_lora_s26 import TRIGGER  # noqa: E402


def caption(prompt):
    return f"{TRIGGER} {prompt.replace(' ' + s25.STYLE, '').replace(s25.TAIL, '').strip()}"


def main():
    out, s25_dir, s40_dir, picks = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:]
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in picks:
        plan, folder = (S25_PLAN, s25_dir) if p.startswith("Q") else (S40_PLAN, s40_dir)
        excluded = set(json.loads((folder / "dataset.json").read_text(encoding="utf-8"))["excluded"])
        if p not in plan or p in excluded or not (folder / f"{p}.png").exists():
            raise SystemExit(f"쓸 수 없는 번호: {p}")
        shutil.copyfile(folder / f"{p}.png", out / f"{p}.png")
        rows.append(dict(file_name=f"{p}.png", text=caption(plan[p]), sha256=hashlib.sha256((out / f"{p}.png").read_bytes()).hexdigest()))
    with (out / "metadata.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(dict(file_name=r["file_name"], text=r["text"]), ensure_ascii=False) + "\n")
    (out / "provenance.json").write_text(json.dumps(dict(
        source="S25·S40 Qwen-Image-2512 (Apache-2.0) 오리지널 성인 캐릭터, 사용자 선택", trigger=TRIGGER, picks=picks, records=rows),
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("DATASET", len(rows), out, flush=True)


if __name__ == "__main__":
    main()
