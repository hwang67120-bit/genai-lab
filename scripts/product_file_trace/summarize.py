"""관찰 원자료(opens.jsonl 여러 개) → 단계별·분류별 파일 목록(JSON). 실행 환경 파일은 묶어서 센다.

사용: python summarize.py <출력 json> <opens.jsonl> [<opens.jsonl> ...]
"""
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


def classify(p):
    low = p.lower()
    if re.search(r"\\(venv|catvton-venv|qwen-test-venv)\\", low) or "site-packages" in low or "\\python3" in low:
        return "environment"
    if "\\downloads\\" in low or "\\onedrive\\" in low:
        return "user_input"
    if "character-memory" in low or "\\appdata\\roaming\\genailab" in low:
        return "user_data"
    if low.endswith((".onnx", ".safetensors", ".pt", ".pth", ".ckpt", ".gguf", ".bin", ".msgpack")) or "model-cache\\hub" in low \
            or "genai-cache\\huggingface" in low or "genai-cache\\models" in low or "genai-cache\\tools" in low:
        return "model"
    if "\\outputs\\product-file-trace" in low or "\\appdata\\local\\temp" in low:
        return "work_output"
    if "\\outputs\\" in low:
        return "existing_results"
    if "genai-lab" in low:
        return "project"
    return "other"


def main(argv):
    rows = []
    for log in argv[1:]:
        scenario = Path(log).parent.name
        stage = "setup"          # 첫 단계 표시 전(준비·사본 복사 등). 앞 파일의 단계를 이어 붙이지 않는다
        for line in open(log, encoding="utf-8"):
            d = json.loads(line)
            if "stage" in d:
                stage = d["stage"]
                continue
            if "path" not in d:          # popen(하위 프로세스 기록)은 실행 환경 요약(env_report.py)에서 본다
                continue
            rows.append(dict(d, stage=f"{scenario}:{stage}"))
    files = defaultdict(lambda: dict(stages=set(), kinds=set()))
    for r in rows:
        p = os.path.normpath(r["path"].replace("/", "\\"))
        files[p]["stages"].add(r["stage"])
        files[p]["kinds"].add(r["kind"])
    result = defaultdict(dict)
    env_roots = defaultdict(int)
    for p, v in sorted(files.items()):
        c = classify(p)
        if c == "environment":
            m = re.search(r"^(.*?\\(venv|catvton-venv|qwen-test-venv))\\", p, re.I)
            env_roots[m.group(1) if m else "기타"] += 1
            continue
        if c == "work_output":
            continue
        result[c][p] = dict(stages=sorted(v["stages"]), kinds=sorted(v["kinds"]))
    result["environment_file_counts"] = dict(env_roots)
    Path(argv[0]).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    for c in ("model", "project", "user_data", "existing_results", "user_input", "other"):
        for p, v in result.get(c, {}).items():
            print(c, "|", ",".join(v["kinds"]), "|", " ".join(v["stages"]), "|", p)
    print("environment", json.dumps(result["environment_file_counts"], ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
