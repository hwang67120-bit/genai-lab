"""옷 참조 공통 전처리(D-117 검토): 배경 흰색화와 머리 제거가 옷 픽셀을 건드리지 않는지(GPU 없이)."""
import json
from pathlib import Path

import numpy as np

from genai_lab import garment_reference as gr


def scene(bg=(150, 142, 141), size=(400, 300)):
    """회색 배경 위 검은 옷(사각형)과 그 위의 흰 마네킹 머리(원)."""
    h, w = size
    image = np.full((h, w, 3), bg, np.uint8)
    image[120:380, 90:210] = (20, 20, 25)                       # 옷
    yy, xx = np.mgrid[:h, :w]
    image[((yy - 70) ** 2 + (xx - 150) ** 2) < 35 ** 2] = (235, 235, 235)   # 머리
    return image


def joints_for(neck_y=125, left=(210, 130), right=(90, 130)):
    return [dict(joint_name="neck", x=150, y=neck_y, detected=True),
            dict(joint_name="left_shoulder", x=left[0], y=left[1], detected=True),
            dict(joint_name="right_shoulder", x=right[0], y=right[1], detected=True)]


def test_background_becomes_white_and_garment_pixels_stay():
    image = scene()
    out, report = gr.whiten_background(image)
    assert report["status"] == "applied"
    assert (out[5, 5] == 255).all() and (out[395, 295] == 255).all()
    garment = image[130:370, 100:200]
    assert (out[130:370, 100:200] == garment).all()              # 옷 픽셀 무변경


def test_white_border_is_skipped_and_noise_image_stays_untouched():
    out, report = gr.whiten_background(scene(bg=(255, 255, 255)))
    assert report["status"] == "skipped" and report["reason"] == "white border"
    full = np.random.default_rng(0).integers(0, 255, (200, 150, 3), np.uint8)
    out, report = gr.whiten_background(full)
    assert report["status"] == "skipped" and out is full          # 분리 불가(전경 비율 밖 또는 grabcut 실패) → 그대로


def test_gradient_background_that_fills_the_frame_is_still_whitened():
    # 회색 비네트: 테두리 색 ±45 밖의 픽셀이 그림 전체에 퍼져 있어 내용 상자가 전체가 된다(작업실 23e2a7bc 사례, D-118)
    h, w = 400, 300
    yy, xx = np.mgrid[:h, :w]
    shade = (150 + 60 * ((yy / h) ** 2 + (xx / w) ** 2)).clip(0, 255).astype(np.uint8)
    image = np.stack([shade, shade, shade], 2)
    image[120:380, 90:210] = (20, 20, 25)
    out, report = gr.whiten_background(image)
    assert report["status"] == "applied (inset rect)"
    assert (out[130:370, 100:200] == image[130:370, 100:200]).all() and (out[5:20, 5:20] == 255).all()


def test_head_is_removed_above_the_neck_line_only_when_tags_say_head():
    image = scene()
    tags = dict(head_out_of_frame=0.0, no_humans=0.06)
    out, report = gr.remove_head(image, tags, joints_for())
    assert report["status"] == "applied" and report["line"] == 125 - 30      # 목 125 − 어깨 폭 120 × 0.25
    assert (out[:95] == 255).all() and (out[95:] == image[95:]).all()
    for tags in (dict(head_out_of_frame=0.42, no_humans=0.17), dict(head_out_of_frame=0.08, no_humans=0.83)):
        out, report = gr.remove_head(image, tags, joints_for())
        assert report["status"] == "skipped" and report["reason"] == "no head by tags" and out is image


def test_head_removal_needs_neck_and_a_line_inside_the_upper_part():
    image = scene()
    tags = dict(head_out_of_frame=0.0, no_humans=0.0)
    out, report = gr.remove_head(image, tags, [j for j in joints_for() if j["joint_name"] != "neck"])
    assert report["status"] == "skipped" and report["reason"] == "no neck line"
    out, report = gr.remove_head(image, tags, joints_for(neck_y=300))          # 선이 그림의 40% 아래 → 안 자름
    assert report["status"] == "skipped"
    out, report = gr.remove_head(image, tags, joints_for(neck_y=20))           # 선이 0 이하
    assert report["status"] == "skipped"


def test_clean_reports_unchanged_when_nothing_applies():
    image = scene(bg=(255, 255, 255))
    out, report = gr.clean_garment_reference(image, dict(head_out_of_frame=0.42, no_humans=0.17))
    assert report["changed"] is False and out is image
    out, report = gr.clean_garment_reference(scene(), dict(head_out_of_frame=0.0, no_humans=0.0), joints_for())
    assert report["changed"] is True and report["background"]["status"] == "applied" and report["head"]["status"] == "applied"
    # 머리 자리의 흰 여백은 잘라낸다. 옷(120~380행, 90~210열)은 잘린 상자 안에 그대로 있다
    x0, y0, x1, y1 = report["trim"]["box"]
    assert report["trim"]["status"] == "applied" and y0 < 120 - gr.TRIM_MARGIN < 120 and 380 + gr.TRIM_MARGIN <= y1 <= 380 + gr.TRIM_MARGIN + 2 * gr.GUARD
    assert (out[120 - y0:380 - y0, 90 - x0:210 - x0] == scene()[120:380, 90:210]).all()


def test_trim_keeps_garment_pixels_and_skips_when_no_margin():
    image = scene(bg=(255, 255, 255))
    ys, xs = np.where((image < gr.TRIM_WHITE).any(2))
    out, report = gr.trim_white(image)
    assert report["status"] == "applied"
    assert report["box"] == [xs.min() - 12, ys.min() - 12, xs.max() + 13, ys.max() + 13]
    assert out.shape == (ys.max() - ys.min() + 25, xs.max() - xs.min() + 25, 3)
    full = np.zeros((50, 40, 3), np.uint8)
    out, report = gr.trim_white(full)
    assert report["status"] == "skipped" and out is full


def test_result_notice_names_what_was_removed_and_stays_silent_otherwise():
    from genai_lab.outfit_edit_results import garment_reference_notice
    assert garment_reference_notice(None) == "" and garment_reference_notice(dict(changed=False)) == ""
    both = dict(changed=True, background=dict(status="applied"), head=dict(status="applied"))
    assert garment_reference_notice(both) == " 옷 참조의 배경을 흰색으로·머리 부분을 자동으로 뺐습니다(옷 자체는 그대로)."
    only_bg = dict(changed=True, background=dict(status="applied"), head=dict(status="skipped"))
    assert "머리" not in garment_reference_notice(only_bg)


def test_prepare_falls_back_to_the_original_file_when_analysis_fails(tmp_path, monkeypatch):
    from PIL import Image
    from genai_lab import clothing_analysis
    garment = tmp_path / "g.jpg"
    Image.fromarray(scene()).save(garment)
    class Broken:
        def __init__(self, settings): raise RuntimeError("tagger model missing")
    monkeypatch.setattr(clothing_analysis, "WdTagSession", Broken)
    runtime = type("R", (), dict(model_cache=tmp_path))()
    result = gr.prepare_garment_reference(garment, tmp_path / "ref", runtime)
    assert result["path"] == str(garment) and result["report"]["status"] == "failed" and "tagger" in result["report"]["error"]
    assert (tmp_path / "ref" / "garment_reference.json").is_file()


def test_studio_summary_counts_repeated_failure_types(tmp_path):
    import subprocess, sys as _sys
    from genai_lab.outfit_edit import write_json
    from genai_lab.outfit_judgement import record_judgement
    for name, judgement in (("a", dict(judge="user", same_character="yes", outfit="partial", original_garment_remaining="some", severe_defect="none")),
                            ("b", dict(judge="user", same_character="yes", outfit="same", original_garment_remaining="some", severe_defect="present", defect_note="손 사라짐")),
                            ("c", None)):
        run = tmp_path / name / "outfit-edit"
        run.mkdir(parents=True)
        write_json(run / "trace.json", dict(judgement=dict(composited=True)))
        if judgement:
            record_judgement(run, judgement)
    script = Path(__file__).resolve().parents[1] / "scripts" / "studio_judgement_summary.py"
    out = subprocess.run([_sys.executable, "-B", str(script), "--root", str(tmp_path), "--json"], capture_output=True, text=True,
                         check=True, encoding="utf-8")
    result = json.loads(out.stdout)
    assert (result["total"], result["judged"], result["quality_pass"]) == (3, 2, 0)
    assert result["failures"] == {"original_garment_remaining": 2, "severe_defect": 1} and result["defect_notes"] == ["손 사라짐"]
