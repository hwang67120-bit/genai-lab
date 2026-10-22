"""손 개수 검사 오프라인 채점(GPU 없음, 2026-10-23). 사용자 판정이 있는 결과 그림마다 손을 세어,
'손이 3개 이상이면 구조 이상'이라는 규칙이 실제 이상(요청 34 B1: 손 3개)을 잡는지, 사람 통과 그림에 잘못 걸리지 않는지 센다.

손 검출: imgutils.detect.detect_hands(deepghs/anime_hand_detection hand_detect_v1.0_s, rev dba2c5be, 2026-10-23 사용자 승인 내려받기).
사용: python scripts/replay_hand_count.py <출력 json> <ledger.json> [<ledger.json> ...]   (HF_HUB_CACHE=D:/genai-cache/huggingface, HF_HUB_OFFLINE=1)
"""
import json
import sys
from collections import Counter
from pathlib import Path

THRESHOLDS = (0.35, 0.5)       # 검출 신뢰도 문턱(0.35 = imgutils 기본)
LIMIT = 2                      # 손이 이보다 많으면 구조 이상 후보


def main(argv):
    from PIL import Image
    from imgutils.detect import detect_hands
    out, ledgers = Path(argv[0]), argv[1:]
    rows = []
    for ledger in ledgers:
        for entry in json.loads(Path(ledger).read_text(encoding="utf-8")):
            target = entry.get("target") or {}
            image = Path(target.get("image", ""))
            if not image.is_file() or entry.get("quality_pass") is None:
                continue
            with Image.open(image) as picture:
                found = detect_hands(picture.convert("RGB"), conf_threshold=min(THRESHOLDS))
            scores = sorted((round(float(s), 3) for _, _, s in found), reverse=True)
            rows.append(dict(ledger=str(ledger), case=entry.get("code") or entry["case"], attempt=target.get("attempt"),
                             variant=target.get("variant"), quality_pass=entry["quality_pass"], failure_type=entry.get("failure_type"),
                             note=entry.get("note", "")[:80], hands={str(t): sum(s >= t for s in scores) for t in THRESHOLDS}, scores=scores))
    summary = {}
    for t in THRESHOLDS:
        key = str(t)
        flagged = [r for r in rows if r["hands"][key] > LIMIT]
        summary[key] = dict(rows=len(rows), distribution=dict(sorted(Counter(r["hands"][key] for r in rows).items())),
                            flagged=len(flagged), flagged_on_user_pass=sum(r["quality_pass"] is True for r in flagged),
                            flagged_on_user_fail=sum(r["quality_pass"] is False for r in flagged),
                            flagged_cases=[(r["case"], r["attempt"], r["variant"], r["quality_pass"], r["hands"][key]) for r in flagged])
    out.write_text(json.dumps(dict(summary=summary, rows=rows), ensure_ascii=False, indent=1), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
