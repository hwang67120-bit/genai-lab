"""2단계 결과 마무리 시험 스크립트: 눈 통계와 실행 후보 수집을 GPU 없이 확인한다."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from genai_lab.head_paste_rules import TAGS
from genai_lab.qwen_record_io import write_json

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_proportion_finishing", ROOT / "scripts/run_proportion_finishing.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def test_eye_stats_orders_two_largest_eyes_and_reads_iris_color():
    parts = {name: np.zeros((1024, 1024), bool) for name in TAGS}
    parts["eyes"][400:460, 300:360] = True
    parts["eyes"][400:460, 600:660] = True
    parts["eyes"][800:805, 500:505] = True  # 작은 조각은 무시
    crop = np.full((1024, 1024, 3), 255, np.uint8)
    crop[400:460, 300:360] = (40, 60, 200)   # 파란 눈
    crop[400:460, 600:660] = (200, 40, 40)   # 빨간 눈
    eyes = script.eye_stats(parts, crop)
    assert [e["area"] for e in eyes] == [3600, 3600]
    assert eyes[0]["center"][0] < eyes[1]["center"][0]
    assert eyes[0]["iris_lab"][2] < 0 < eyes[1]["iris_lab"][1]  # 왼쪽 파랑(b<0), 오른쪽 빨강(a>0)


def test_candidates_follow_preflight_seeds(tmp_path):
    generation = tmp_path / "run/generation"
    for seed in (5, 6):
        folder = generation / f"CONTOUR_{seed}"; folder.mkdir(parents=True)
        write_json(folder / "run.json", {"seed": seed})
    write_json(generation / "preflight.json", {"seeds": [5, 6]})
    found = list(script.candidates(tmp_path / "run", None))
    assert [seed for seed, *_ in found] == [5, 6]
    assert found[1][2] == generation / "seed-6" / "product.png"
