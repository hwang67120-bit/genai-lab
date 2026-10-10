"""머리 안쪽 선 시험 보조: 확인 머리카락으로 얼굴 정하기, 기존 실행 복원 인자, 얼굴 위치 측정 대상 수집."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from genai_lab.head_lines import confirmed_hair_face, CONFIRMED_HAIR_RULES
from genai_lab.proportion_inputs import sha
from genai_lab.qwen_record_io import write_json

ROOT = Path(__file__).resolve().parents[1]


def confirmation(folder, confirmed=True, tamper=False):
    folder.mkdir()
    H = np.zeros((1024, 1024), np.uint8); H[:512] = 255  # 자른 그림 위쪽 절반이 머리카락
    Image.fromarray(H).save(folder / "H.png")
    write_json(folder / "result.json", dict(H_file=str(folder / "H.png"), H_sha256=sha(folder / "H.png"), box=[100, 50, 300, 250]))
    write_json(folder / "confirmation.json", dict(confirmed=confirmed, reviewer="user",
               preview_sha256="0" * 64 if tamper else sha(folder / "result.json")))
    return {"directory": str(folder)}


def test_confirmed_hair_maps_back_to_box_and_face_is_rest_of_head(tmp_path):
    head = np.zeros((1232, 736), np.uint8); head[50:250, 100:300] = 255
    face, details = confirmed_hair_face(head, confirmation(tmp_path / "c"))
    assert not face[50:150, 100:300].any() and face[150:250, 100:300].all()
    assert details["method"] == CONFIRMED_HAIR_RULES["method"] and details["box"] == [100, 50, 300, 250]


@pytest.mark.parametrize("confirmed,tamper", [(False, False), (True, True)])
def test_unconfirmed_or_changed_hair_is_refused(tmp_path, confirmed, tamper):
    head = np.full((1232, 736), 255, np.uint8)
    with pytest.raises(ValueError, match="확인한 머리카락"):
        confirmed_hair_face(head, confirmation(tmp_path / "c", confirmed, tamper))


def test_measure_collects_conditions_and_matching_c0(tmp_path):
    spec = importlib.util.spec_from_file_location("measure_face_offset", ROOT / "scripts/measure_face_offset.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    for folder in ("trial/hair/CONTOUR_7", "trial/all/CONTOUR_7", "gen/CONTOUR_7"):
        (tmp_path / folder).mkdir(parents=True); (tmp_path / folder / "raw.png").write_bytes(b"x")
    found = module.raws(tmp_path / "trial", tmp_path / "gen")
    assert set(found) == {("hair", 7), ("all", 7), ("C0", 7)}
