"""옷 참조 공통 전처리(D-117 검토): 배경 흰색화와 머리 제거가 옷 픽셀을 건드리지 않는지(GPU 없이)."""
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


def test_white_border_and_full_frame_content_are_skipped():
    out, report = gr.whiten_background(scene(bg=(255, 255, 255)))
    assert report["status"] == "skipped" and report["reason"] == "white border"
    full = np.random.default_rng(0).integers(0, 255, (200, 150, 3), np.uint8)
    out, report = gr.whiten_background(full)
    assert report["status"] == "skipped" and report["reason"] == "content touches every border"
    assert out is full


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
