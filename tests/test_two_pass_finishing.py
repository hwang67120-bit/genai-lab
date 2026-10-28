"""원본 비율 참고 결과의 마무리 입력: 보고 있는 그림(머리 붙이기 전·후)의 전체 원본을 쓰고, 검증도 같은 원본으로 한다."""
from dataclasses import replace
import json
from pathlib import Path
import pytest
from PIL import Image
from genai_lab.studio_generation import StudioResults
from genai_lab.proportion_inputs import sha
from genai_lab.qwen_record_io import write_json
from test_studio_proportion import two_pass_batch


def head_product(folder, result, *, with_redraw=True):
    folder.mkdir()
    product = folder / "product.png"; Image.new("RGB", (16, 24), (4, 5, 6)).save(product)
    redraw = folder / "raw_redraw.png"; Image.new("RGB", (16, 24), (9, 8, 7)).save(redraw)
    record = folder / "run.json"
    data = dict(status="review_pending", unet_calls=[{"batch": 2, "adapter": True}] * 28,
                callback=[{"index": i, "mode": "overwrite" if i < 24 else "blend"} for i in range(28)],
                adapter_calls=1, max_reserved_gib=4.67)
    if with_redraw:
        data["result_sha256"] = sha(redraw)
    write_json(record, data)
    info = dict(directory=str(folder), status="awaiting_user_review", raw_sha256=result.verify_original(),
                before_sha256=result.verify_current(), product_file=str(product), product_sha256=sha(product),
                checks=dict(outside_equal=True, face_equal=True), redraw_record=str(record), manifest={str(record): sha(record)})
    write_json(folder / "result.json", info)
    return info


def pasted_results(tmp_path, with_redraw=True):
    batch = two_pass_batch(tmp_path / "generation")
    result = StudioResults(batch)
    items = []
    for i in range(len(batch.candidates)):
        result.select(i)
        info = head_product(tmp_path / f"head-{i}", result, with_redraw=with_redraw)
        items.append(dict(seed=batch.candidates[i].seed, status="completed", directory=info["directory"]))
    return StudioResults(replace(batch, auto_head=dict(status="awaiting_user_review", items=items)))


def test_before_view_and_plain_two_pass_finish_the_generated_raw(tmp_path):
    result = StudioResults(two_pass_batch(tmp_path / "generation"))
    assert not result.finishing_unavailable_reason
    assert result.finishing_candidate(tmp_path / "unused") is result.candidate.raw
    assert result.finishing_raw_sha() == result.verify_original()
    assert not (tmp_path / "unused").exists()


def test_pasted_view_finishes_the_pasted_full_image_without_touching_original_record(tmp_path):
    result = pasted_results(tmp_path)
    original_record = result.candidate.raw.record_path.read_bytes()
    assert result.head_view == "pasted" and not result.finishing_unavailable_reason
    candidate = result.finishing_candidate(tmp_path / "source")
    redraw = Path(result.active_head["directory"]) / "raw_redraw.png"
    assert candidate.path == redraw and candidate.record["raw_sha256"] == sha(redraw)
    assert candidate.record["finishing_source"]["original_raw_sha256"] == result.verify_original()
    assert json.loads(candidate.record_path.read_text(encoding="utf-8"))["raw_sha256"] == sha(redraw)
    assert result.finishing_raw_sha() == sha(redraw)
    assert result.candidate.raw.record_path.read_bytes() == original_record
    result.toggle_head_view()
    assert result.head_view == "before" and result.finishing_raw_sha() == result.verify_original()


def test_pasted_view_without_redraw_source_explains_and_blocks(tmp_path):
    result = pasted_results(tmp_path, with_redraw=False)
    assert "붙이기 전 보기" in result.finishing_unavailable_reason
    result.toggle_head_view()
    assert not result.finishing_unavailable_reason


def test_changed_redraw_image_is_rejected(tmp_path):
    result = pasted_results(tmp_path)
    (Path(result.active_head["directory"]) / "raw_redraw.png").write_bytes(b"changed")
    assert result.finishing_unavailable_reason
    with pytest.raises(ValueError):
        result.finishing_candidate(tmp_path / "source")


def test_adoption_and_verification_check_finishing_against_the_viewed_source(tmp_path, monkeypatch):
    from genai_lab import studio_optional_finishing as optional
    result = pasted_results(tmp_path)
    seen = []
    def fake_info(directory, raw_sha, before_sha):
        seen.append((raw_sha, before_sha))
        return {"directory": str(directory), "product_file": str(result.active_head["product_file"]), "product_sha256": "f" * 64,
                "finishing": {}, "background": {}}
    monkeypatch.setattr(optional, "finishing_info", fake_info)
    before = result.verify_basis()
    result.adopt_finishing(tmp_path / "finish")
    redraw_sha = sha(Path(result.active_head["directory"]) / "raw_redraw.png")
    assert seen[-1] == (redraw_sha, before)
    result.verify_basis()
    assert seen[-1] == (redraw_sha, before)
