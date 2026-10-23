"""격리 검증 기록 판정(D-174): 배포 후보 프로세스가 연 파일·연결 시도·실제 Python을 금지 위치 기준으로 나눈다.

허용: 배포 후보 폴더, 모델 폴더, 가짜 사용자 폴더, Windows 폴더, 시험 입력(저장소 outputs의 기존 결과·사용자 Downloads 참고 그림, 읽기만).
금지: 시스템 Python, 개발 환경(venv), 공유 캐시(D:\\genai-cache\\huggingface 등), 실제 사용자 폴더(.cache·AppData), 저장소 코드·설정.
사용: python analyze_isolated_trace.py <trace 결과 폴더> <배포 후보> <모델 폴더> <가짜 사용자 폴더들(;로)> <출력 json>
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def norm(p):
    return os.path.normcase(os.path.normpath(str(p).replace("/", "\\")))


def main(argv):
    folder, candidate, model_home, profiles, out = Path(argv[0]), argv[1], argv[2], argv[3].split(";"), Path(argv[4])
    user = norm(Path.home())
    allowed = [norm(candidate), norm(model_home)] + [norm(p) for p in profiles] + [norm(os.environ.get("SYSTEMROOT", r"C:\Windows"))]
    test_inputs = [norm(r"Z:\genai-lab\outputs"), norm(r"\\192.168.0.109\win_g\genai-lab\outputs"), norm(Path.home() / "Downloads")]
    forbidden = {
        "system_python": [norm(r"C:\Program Files\Python310")],
        "dev_env": [norm(r"D:\genai-cache\venv"), norm(r"D:\genai-cache\catvton-venv"), norm(r"D:\genai-cache\qwen-test-venv"),
                    norm(r"D:\Model\genai-cache\venv"), norm(r"D:\Model\genai-cache\catvton-venv")],
        "shared_cache": [norm(r"D:\genai-cache"), norm(r"D:\Model\genai-cache"), norm(r"\\192.168.0.109\win_g\genai-lab\outputs\pose-identity-20260927")],
        "repo_code": [norm(r"Z:\genai-lab"), norm(r"\\192.168.0.109\win_g\genai-lab")],
        "real_user": [user],
    }
    files, nets, popens = defaultdict(set), [], []
    log = folder / "opens.jsonl"
    for line in log.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("kind") in ("net", "net-sample"):
            nets.append(r)
        elif r.get("kind") == "popen":
            popens.append(r)
        elif "path" in r:
            files[norm(r["path"])].add(r["kind"])
    verdict = defaultdict(list)
    for p in sorted(files):
        if p in ("nul",) or p.startswith(norm(r"\\.\\")):
            continue
        if any(p.startswith(t) for t in test_inputs):
            verdict["test_input"].append(p)
            continue
        if any(p.startswith(a) for a in allowed):
            verdict["allowed"].append(p)
            continue
        hit = next((k for k, roots in forbidden.items() if any(p.startswith(r) for r in roots)), "other")
        verdict[hit].append(p)
    processes = []
    for f in sorted(folder.glob("modules-*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        outside = sorted({norm(v) for v in d["modules"].values() if not norm(v).startswith(norm(candidate)) and os.path.isabs(v)})
        processes.append(dict(pid=d["pid"], argv=d["argv"][:3], executable=d["executable"], base_prefix=d["base_prefix"],
                              in_candidate=norm(d["base_prefix"]).startswith(norm(candidate)), modules_outside_candidate=outside[:10],
                              sys_path_outside=[p for p in d["path"] if p and not norm(p).startswith(norm(candidate))]))
    external = [n for n in nets if n["kind"] == "net-sample" and n.get("raddr") not in ("()", "None", "")]
    result = dict(counts={k: len(v) for k, v in verdict.items()}, forbidden={k: v[:20] for k, v in verdict.items() if k not in ("allowed", "test_input")},
                  test_input=verdict["test_input"][:10], processes=processes,
                  net_events=len([n for n in nets if n["kind"] == "net"]), net_event_examples=[n for n in nets if n["kind"] == "net"][:10],
                  net_samples_with_remote=external[:20], popens=[dict(args=p["args"][:3] if isinstance(p["args"], list) else p["args"], cwd=p["cwd"],
                                                                     pythonpath=p["env_pythonpath"]) for p in popens])
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result["counts"], ensure_ascii=False), "| 프로세스", len(processes), "모두 후보 안:", all(p["in_candidate"] for p in processes),
          "| net 이벤트", result["net_events"], "| 원격 연결 표본", len(external))


if __name__ == "__main__":
    main(sys.argv[1:])
