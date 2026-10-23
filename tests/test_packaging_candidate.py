"""배포 후보 도구(D-175): 공식 내장형 zip 풀기, 파일 목록 고정·대조, P4 분할 비교, P5 장면 seed 고정."""
import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = load("build_candidate", "packaging/build_candidate.py")
freeze = load("freeze_candidate", "packaging/freeze_candidate.py")
inner = load("candidate_gpu_inner", "packaging/candidate_gpu_inner.py")


def test_embed_zip_is_extracted_and_its_pth_is_replaced_without_import_site(tmp_path):
    archive = tmp_path / "embed.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("python.exe", b"exe")
        z.writestr("python310.zip", b"stdlib")
        z.writestr("python310._pth", "python310.zip\n.\n\n# Uncomment to run site.main() automatically\n#import site\n")
    runtime = tmp_path / "runtime-main"
    build.extract_embed(archive, runtime)
    lines = (runtime / "python310._pth").read_text(encoding="utf-8").splitlines()
    assert lines == ["python310.zip", ".", "Lib\\site-packages", "..\\app"]
    assert (runtime / "python.exe").read_bytes() == b"exe" and (runtime / "Lib" / "site-packages").is_dir()


def test_build_refuses_an_existing_output_folder(tmp_path):
    with pytest.raises(SystemExit, match="덮어쓰지 않음"):
        build.main([str(tmp_path), str(ROOT), str(tmp_path), str(tmp_path), "--embed-zip", str(tmp_path / "none.zip")])


def test_freeze_and_verify_detect_changed_added_and_missing_files_but_ignore_pycache(tmp_path, capsys):
    candidate = tmp_path / "후보 폴더"
    (candidate / "app" / "__pycache__").mkdir(parents=True)
    (candidate / "app" / "a.py").write_text("a", encoding="utf-8")
    (candidate / "b.txt").write_text("b", encoding="utf-8")
    manifest = tmp_path / "m.json.gz"
    assert freeze.main(["freeze", str(candidate), str(manifest)]) == 0
    (candidate / "app" / "__pycache__" / "a.cpython-310.pyc").write_bytes(b"cache")
    assert freeze.main(["verify", str(candidate), str(manifest)]) == 0
    (candidate / "b.txt").write_text("changed", encoding="utf-8")
    assert freeze.main(["verify", str(candidate), str(manifest)]) == 1
    (candidate / "b.txt").write_text("b", encoding="utf-8")
    (candidate / "runtime-main").mkdir()
    (candidate / "runtime-main" / "sitecustomize.py").write_text("trace", encoding="utf-8")   # 기록 도구가 남은 상태
    assert freeze.main(["verify", str(candidate), str(manifest)]) == 1
    (candidate / "runtime-main" / "sitecustomize.py").unlink()
    (candidate / "app" / "a.py").unlink()
    assert freeze.main(["verify", str(candidate), str(manifest)]) == 1
    assert "빠짐 app/a.py" in capsys.readouterr().out.replace("   ", " ")


def test_segmentation_comparison_counts_differing_pixels(tmp_path):
    base = np.zeros((4, 4), dtype=np.uint8)
    np.savez(tmp_path / "dev.npz", protect=base, face=base)
    np.savez(tmp_path / "same.npz", protect=base, face=base)
    other = base.copy()
    other[0, :2] = 1
    np.savez(tmp_path / "other.npz", protect=other, face=base)
    assert inner.compare_parts(tmp_path / "same.npz", tmp_path / "dev.npz")["all_equal"]
    result = inner.compare_parts(tmp_path / "other.npz", tmp_path / "dev.npz")
    assert not result["all_equal"] and result["arrays"]["protect"]["differing"] == 2 and result["arrays"]["face"]["equal"]


def test_p5_cases_reproduce_the_request_34_scenes_from_their_seeds():
    from genai_lab.scene_composer import compose_scene
    cases = json.loads((ROOT / "outputs/packaging-gpu-20261023/cases-p5.json").read_text(encoding="utf-8"))
    assert [c["case"] for c in cases] == ["A3", "B2"] and cases[0]["garment"] is None and cases[1]["garment"]
    for case in cases:
        assert compose_scene(case["scene_seed"])["text"] == case["scene_text"]
