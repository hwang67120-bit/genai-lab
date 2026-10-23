"""요청 37 시간 분해(기존 기록만, 추정 없음). 사용: python timing37.py <P5 flow 폴더> <출력 json>

출처 구분:
  measured  작업 프로세스·작업실이 직접 잰 값(stage_seconds, steps, seconds, elapsed_seconds, tqdm 표시)
  derived   잰 값 두 개의 차(예: generation − Σsteps). 측정 지점 사이 간격만큼 오차가 있음(설명 포함)
  bounded   파일 기록 시각 사이의 간격 — 그 안에 여러 일이 섞여 있어 나눌 수 없음(내용 명시)
기록이 없는 구간은 값 없이 'not_recorded'로 둔다.
"""
import json
import os
import re
import sys
from pathlib import Path

SHARDS = re.compile(r"Loading checkpoint shards: 100%\|[#]+\| 4/4 \[(\d+):(\d+)<")


def mtime(path):
    return os.path.getmtime(path) if Path(path).exists() else None


def worker(directory):
    d = Path(directory)
    run = json.loads((d / "run.json").read_text(encoding="utf-8"))
    steps = [s["seconds"] for s in run.get("steps") or []]
    stage = run.get("stage_seconds") or {}
    out = dict(directory=str(d), seed=(run.get("call_arguments") or {}).get("seed"))
    out["worker_seconds"] = dict(value=run.get("seconds"), source="measured",
                                 covers="작업 프로세스 안 시계: SHA 검사·라이브러리 가져오기 뒤부터 raw 저장까지")
    out["text_encode"] = dict(value=stage.get("text_encode"), source="measured", covers="문자 인코더 불러오기 + 지시문·그림 인코딩 + 내리기")
    log = (d / "worker.log").read_text(encoding="utf-8", errors="replace") if (d / "worker.log").exists() else ""
    shards = SHARDS.findall(log)
    out["text_encoder_shard_load"] = (dict(value=int(shards[0][0]) * 60 + int(shards[0][1]), source="measured",
                                           covers="문자 인코더 체크포인트 4조각 읽기(tqdm 표시, 초 단위)") if shards else "not_recorded")
    if shards and stage.get("text_encode"):
        out["text_encode_rest"] = dict(value=stage["text_encode"] - out["text_encoder_shard_load"]["value"], source="derived",
                                       covers="text_encode − 조각 읽기 = 양자화 배치·인코딩·내리기(tqdm이 초 단위라 ±1초)")
    out["pipeline_load"] = dict(value=stage.get("pipeline_load"), source="measured", covers="GGUF transformer·VAE·파이프라인 불러오기와 오프로드 설정")
    out["generation"] = dict(value=stage.get("generation"), source="measured", covers="파이프라인 호출 전체(40단계 + VAE 디코드)")
    if steps:
        out["steps"] = dict(count=len(steps), total=sum(steps), step0=steps[0], mean_1_39=sum(steps[1:]) / max(1, len(steps) - 1),
                            min_1_39=min(steps[1:]) if len(steps) > 1 else None, max_1_39=max(steps[1:]) if len(steps) > 1 else None,
                            source="measured",
                            covers="단계마다 콜백 사이 시간. 직전 단계의 미리보기 그림·진행 기록 쓰기를 포함. step0은 입력 그림 전처리(VAE 인코딩 포함)와 첫 단계")
        if stage.get("generation"):
            out["decode_and_after"] = dict(value=stage["generation"] - sum(steps), source="derived",
                                           covers="generation − Σsteps = 마지막 단계 뒤 VAE 디코드·그림 변환(측정 시작점 차이 수 ms)")
    t_request, t_raw, t_launch = mtime(d / "request.json"), mtime(d / "raw.png"), mtime(d / "launcher.json")
    if t_request and t_raw and run.get("seconds"):
        out["before_worker_clock"] = dict(value=(t_raw - run["seconds"]) - t_request, source="bounded",
                                          covers="요청 파일 기록 → 작업 프로세스 시계 시작: 프로세스 시작·버전 확인·모델 SHA 검사(27개 + GGUF)·torch/diffusers 가져오기·CUDA 준비. 안에서 나눈 기록 없음")
    if t_raw and t_launch:
        out["after_raw"] = dict(value=t_launch - t_raw, source="bounded", covers="raw 저장 → 실행기 기록: 프로세스 종료·제품 크기 변환")
    if t_request and t_launch:
        out["worker_span"] = dict(value=t_launch - t_request, source="bounded", covers="요청 기록 → 실행기 기록(작업 프로세스 한 번 전체)")
    out["preview_overhead"] = "not_recorded"
    out["sha_check_alone"] = "not_recorded"
    return out


def stage_dir(directory):
    d = Path(directory)
    trace = json.loads((d / "trace.json").read_text(encoding="utf-8"))
    workers = [worker(q) for q in sorted(d.glob("qwen*")) if (q / "run.json").exists()]
    elapsed = trace.get("elapsed_seconds")
    spans = sum((w.get("worker_span") or {}).get("value") or 0 for w in workers)
    if elapsed is not None:
        stage_time = dict(value=elapsed, source="measured", covers="작업실이 잰 이 단계 전체")
    else:               # 의상 편집 단계 기록에는 단계 시계가 없다 → 단계 폴더 안 첫 파일 ~ 마지막 파일 시각
        times = [os.path.getmtime(p) for p in d.rglob("*") if p.is_file()]
        elapsed = max(times) - min(times)
        stage_time = dict(value=elapsed, source="bounded", covers="단계 시계 없음 — 단계 폴더 안 첫 파일 기록 ~ 마지막 파일 기록(첫 파일 전 준비 시간은 빠짐)")
    out = dict(directory=str(d), elapsed=stage_time,
               selected_attempt=trace.get("selected_attempt"), attempts=len(trace.get("attempts") or []), workers=workers)
    if elapsed is not None:
        out["outside_workers"] = dict(value=elapsed - spans, source="derived",
                                      covers="단계 전체 − 작업 프로세스 구간 합 = 같은 단계의 CPU 일(입력 준비·분할·특징·결과 분석·판정). 안에서 나눈 시간 기록 없음")
    reasons = []
    for a in trace.get("attempts") or []:
        j = a.get("judgement") or a.get("retry_reason") or {}
        reasons.append(str(j)[:160])
    out["attempt_notes"] = reasons
    return out


def main(argv):
    flow, output = Path(argv[0]), Path(argv[1])
    summary = json.loads((flow / "summary.json").read_text(encoding="utf-8"))
    rows = {}
    for case, rec in summary["cases"].items():
        run = Path(rec["run"])
        stages = {name: stage_dir(run / name) for name in ("outfit-edit", "free-generation") if (run / name / "trace.json").exists()}
        row = dict(total=dict(value=rec.get("seconds"), source="measured", covers="사용자 화면 '이미지 생성' → 결과(실행기 측정)"), stages=stages)
        known = sum(s["elapsed"]["value"] or 0 for s in stages.values())
        row["outside_stages"] = dict(value=(rec.get("seconds") or 0) - known, source="derived",
                                     covers="행 전체 − 단계 합 = 입력 분석(관절·특징)·장면 준비·저장 등. 안에서 나눈 기록 없음")
        rows[case] = row
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    for case, row in rows.items():
        print(f"== {case} 전체 {row['total']['value']}s, 단계 밖 {row['outside_stages']['value']:.0f}s")
        for name, s in row["stages"].items():
            print(f"  [{name}] 단계 {s['elapsed']['value']:.0f}s, 작업 프로세스 밖 {s['outside_workers']['value']:.0f}s, 시도 {s['attempts']} 선택 {s['selected_attempt']}")
            for w in s["workers"]:
                st = w.get("steps", {})
                print("    seed {seed}: 전 {pre:.0f} | 인코딩 {te:.1f} (조각읽기 {sh}) | 불러오기 {pl:.1f} | 생성 {gen:.1f} = 단계합 {tot:.1f} (step0 {s0:.1f}, 평균 {m:.1f}, {mn:.1f}~{mx:.1f}) + 디코드 {dec:.1f} | 후 {post:.0f} | 구간 {span:.0f}".format(
                    seed=w["seed"], pre=w["before_worker_clock"]["value"], te=w["text_encode"]["value"],
                    sh=(w["text_encoder_shard_load"] or {}).get("value") if isinstance(w["text_encoder_shard_load"], dict) else "-",
                    pl=w["pipeline_load"]["value"], gen=w["generation"]["value"], tot=st["total"], s0=st["step0"], m=st["mean_1_39"],
                    mn=st["min_1_39"], mx=st["max_1_39"], dec=w["decode_and_after"]["value"], post=w["after_raw"]["value"], span=w["worker_span"]["value"]))


if __name__ == "__main__":
    main(sys.argv[1:])
