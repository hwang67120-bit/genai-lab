"""S33 태그 종류 표(클라우드 A100): Qwen3-14B(Apache-2.0, 40c06982, bf16)로 WD v3 일반 태그 전체를 tag_kinds와 같은 질문·보정으로 나눈다.
이 PC의 4B 문장 인코더는 F1 태그 34개 중 33개를 맞혔지만 처음 300개에서 하늘·모자도 몸으로 봤다 → 큰 모델로 한 번 만들어 표로 쓴다
(앱 실행 중에는 언어 모델 없음).
사용: python tag_kinds_pod_s33.py <모델 폴더> <selected_tags.csv> <출력 json>
"""
import csv, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from tag_kinds import KINDS, ORDERS, QUESTION, TagKindClassifier

model, tags_csv, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
with tags_csv.open(encoding="utf-8") as f:
    tags = [row["name"] for row in csv.DictReader(f) if row["category"] == "0"]
started = time.time()
with TagKindClassifier(model, model, batch=64, four_bit=False) as c:
    table = c.classify(tags)
    import torch
    peak = round(torch.cuda.max_memory_reserved() / 2**30, 2)
revision = (model / "revision.txt").read_text().strip() if (model / "revision.txt").exists() else ""
out.write_text(json.dumps(dict(schema=1, model="Qwen/Qwen3-14B", revision=revision, license="Apache-2.0", tagger="SmilingWolf/wd-vit-tagger-v3",
                               question=QUESTION, kinds_described=KINDS, orders=ORDERS, calibration="문맥 보정(N/A), 순서 세 번 평균",
                               seconds=round(time.time() - started, 1), peak_gib=peak, table=table), ensure_ascii=False), encoding="utf-8")
print("DONE", len(table), {k: sum(v["kind"] == k for v in table.values()) for k in KINDS}, round(time.time() - started, 1), "s", flush=True)
