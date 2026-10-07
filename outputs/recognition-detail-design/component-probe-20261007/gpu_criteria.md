# S06 SAM2 GPU 1회 비교 — 실행 전 고정
사용자 승인: 2026-10-07, 'S06을 SAM2 GPU로 1회 분석 — 마스크와 끝점 후보 비교'.
- 기존 캐시 facebook/sam2.1-hiera-tiny, 기존 이미지 구성 로더, float32, eval, inference_mode.
- CPU 시험과 같은 원본 SHA, 사용자 상자, 프로세서, multimask_output=False.
- CUDA 모델 forward 정확히 1회. 재시도·후보 추가·생성·Qwen·설치·다운로드 없음.
- 비교는 기존 CPU의 keep_parts 이후 마스크와 GPU의 같은 후처리 마스크. 원시 마스크의 CPU 기록은 없어 비교 불가.
- 마스크 픽셀 동일 여부·다른 픽셀 수·IoU, 조각 수·모든 끝점 좌표 동일 여부·말단 정답 상자 후보 포함을 기록.
- 값이 다르면 그대로 보고하고 허용 오차를 사후에 만들지 않는다.
- 연결 후보는 기하 추정. GPU 일치로 의미 해석·가림 복원·생성 개선을 주장하지 않는다.
- 적재/forward/전체 시간과 torch allocated/reserved 최고값을 기록. 외부 GPU 전체 피크와 공유메모리 피크는 미측정.
- 실패/OOM 시 오류와 실제 forward 호출 수를 run.json에 남기고 종료.
