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


runner = load("run_candidate_gpu", "packaging/run_candidate_gpu.py")
SITE = dict(file="outfit_edit_worker.py", line=37, function="segment", exception="ValueError")
REFUSAL = "얼굴·머리를 찾지 못했습니다. 이 그림은 원본 편집형 의상 교체를 쓸 수 없습니다."


def refused(**changes):
    row = dict(state="refused", dev_error=REFUSAL, new_error=REFUSAL, dev_stage="parts", new_stage="parts", dev_site=SITE, new_site=SITE)
    row.update(changes)
    return row


@pytest.mark.parametrize("row, outcome", [
    (refused(), "pass"),                                                                   # 예정된 거절: 문구·단계·검사 같음
    (dict(state="ok", compare=dict(all_equal=True)), "pass"),
    (dict(state="ok", compare=dict(all_equal=False)), "review_needed"),                    # 배열 차이는 사람이 확인
    (dict(state="refusal_not_reproduced", dev_error=REFUSAL), "review_needed"),
    (refused(new_site=None), "review_needed"),                                             # 문구만 같고 자리를 확인 못 함
    (refused(new_site=dict(SITE, line=99)), "unexpected"),                                 # 같은 문구, 다른 검사
    (refused(new_error="CUDA error: out of memory"), "unexpected"),
    (refused(new_stage="load"), "unexpected"),
    (dict(state="failed", error="ImportError"), "unexpected"),
    (dict(state="skipped: canvas SHA 불일치"), "unexpected"),
])
def test_p4_outcomes_are_pass_review_needed_or_unexpected(row, outcome):
    assert inner.classify(row)[0] == outcome


def test_refusal_site_is_read_from_the_worker_traceback_whatever_the_message_encoding(tmp_path):
    log = tmp_path / "worker.log"
    worker = r"\\192.168.0.109\win_g\genai-lab\genai_lab\outfit_edit_worker.py"
    log.write_bytes(("Traceback (most recent call last):\n"
                     f'  File "{worker}", line 52, in main\n'
                     "    segment(json.loads(Path(filename).read_text(encoding=\"utf-8\")), directory)\n"
                     f'  File "{worker}", line 37, in segment\n').encode()
                    + "    raise ValueError(\"얼굴\")\nValueError: 얼굴·머리를 찾지 못했습니다.\n".encode("cp949"))
    assert inner.refusal_site(log) == SITE
    assert inner.refusal_site(tmp_path / "none.log") is None


def test_array_difference_evidence_separates_boundary_jitter_from_far_changes(tmp_path):
    from PIL import Image
    dev = np.zeros((40, 30), dtype=bool)
    dev[10:20, 10:20] = True
    new = dev.copy()
    new[10:20, 20] = True                     # 경계 1픽셀 흔들림
    new[30:35, 0:5] = True                    # 멀리 떨어진 곳이 새로 들어감(배경 포함 후보)
    new[10:12, 10:20] = False                 # 머리 위쪽 2줄이 빠짐(경계 안쪽이라 '먼 누락'은 아님)
    zero = np.zeros_like(dev)
    np.savez(tmp_path / "dev.npz", hair=dev, face=zero)
    np.savez(tmp_path / "new.npz", hair=new, face=zero)
    Image.new("RGB", (30, 40), "white").save(tmp_path / "canvas.png")
    evidence = inner.substantive_evidence(tmp_path / "new.npz", tmp_path / "dev.npz", tmp_path / "canvas.png", tmp_path)
    protect = evidence["regions"]["protect"]
    assert protect["added_far_px"] == 25 and protect["added_px"] == 35 and protect["missing_px"] == 20
    assert evidence["parts"]["hair"]["dev_px"] == 100 and "face" not in evidence["parts"]
    assert Path(evidence["pictures"][0]).is_file()


def test_p5_starts_only_after_p4_passes_or_its_review_items_are_resolved(tmp_path):
    manifest = tmp_path / "m.json.gz"
    manifest.write_bytes(b"x")
    folder = tmp_path / "p4"
    (folder / "segment").mkdir(parents=True)
    names = [name for name, _ in inner.P4_DEV]

    def write(run, outcomes):
        base = dict(stage="p4", dry=False, manifest=str(manifest), unchanged_after=True,
                    process=dict(exit_codes=dict(p4=0), normal_exit=True))
        (folder / "run-report.json").write_text(json.dumps(dict(base, **run)), encoding="utf-8")
        rows = {n: dict(outcome=o) for n, o in zip(names, outcomes)}
        (folder / "segment" / "inner-p4.json").write_text(json.dumps(dict(dry=False, **inner.p4_verdict(rows, dry=False))), encoding="utf-8")

    passing = ["pass"] * 5
    write({}, passing)
    assert runner.p4_gate(folder, manifest) == ""
    write({"process": dict(exit_codes=dict(p4=9), normal_exit=False)}, passing)          # 결과는 통과여도 프로세스 9 → 시작 안 함
    assert "정상 종료 아님" in runner.p4_gate(folder, manifest)
    write({}, passing[:-1] + ["unexpected"])
    assert "예상 밖" in runner.p4_gate(folder, manifest)
    write({"dry": True}, passing)
    assert "실제 P4" in runner.p4_gate(folder, manifest)
    write({"unchanged_after": False}, passing)
    assert "후보가 달라짐" in runner.p4_gate(folder, manifest)
    write({}, ["pass", "pass", "review_needed", "pass", "pass"])
    assert "검토 필요 미해결" in runner.p4_gate(folder, manifest)
    review = tmp_path / "review.json"
    review.write_text(json.dumps(dict(reviewer="Codex", rows={names[2]: dict(resolution="no_substantive_difference", evidence="")})),
                      encoding="utf-8")
    assert "검토 필요 미해결" in runner.p4_gate(folder, manifest, review)                    # 근거 없는 해결은 받지 않음
    review.write_text(json.dumps(dict(reviewer="Codex", rows={names[2]: dict(resolution="no_substantive_difference",
                                                                           evidence="diff-protect.png: 경계 1픽셀, 먼 차이 0")})),
                      encoding="utf-8")
    assert runner.p4_gate(folder, manifest, review) == ""
    with pytest.raises(SystemExit, match="--after-p4"):
        runner.main(["p5", str(tmp_path), str(manifest), str(tmp_path), str(tmp_path / "out"), str(ROOT), "cases.json"])


def test_p4_verdict_classifies_every_row_including_skipped_ones():
    names = [name for name, _ in inner.P4_DEV]
    rows = {names[0]: dict(state="skipped: canvas SHA 불일치"), names[1]: refused()}
    rows.update({n: dict(state="ok", compare=dict(all_equal=True)) for n in names[2:]})
    verdict = inner.p4_verdict(rows, dry=False)
    assert verdict["unexpected"] == [names[0]] and not verdict["p5_allowed"] and not verdict["p5_allowed_after_review"]
    assert verdict["outcomes"][names[1]] == "pass" and "예정된 거절" in rows[names[1]]["outcome_reason"]
    assert not inner.p4_verdict({n: dict(state="dry") for n in names}, dry=True)["p5_allowed"]


def test_p5_summary_reads_the_flow_state_of_each_row():
    data = dict(cases=dict(A3=dict(inputs_sha_match=True, flow=dict(state="ok")), B2=dict(state="skipped: 입력 SHA 다름")))
    assert runner.summarize_inner("p5", data) == dict(cases=dict(A3="ok", B2="skipped: 입력 SHA 다름"))
    assert runner.summarize_inner("p4", dict(outcomes={"a": "pass"}, p5_allowed=True)) == dict(outcomes={"a": "pass"}, p5_allowed=True)
