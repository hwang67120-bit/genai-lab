"""Qwen 비교 스크립트의 순수 함수: 평행 이동 측정·보정과 마스크 안 합성을 GPU 없이 확인한다."""
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("preserve_qwen_edit", ROOT / "scripts/preserve_qwen_edit.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def textured(seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (1024 // 16, 768 // 16, 3)).astype(np.uint8)
    return np.kron(base, np.ones((16, 16, 1), np.uint8))


def test_shift_is_measured_and_undone():
    original = textured()
    edited = np.roll(original, (6, -4), axis=(0, 1))   # 아래로 6, 왼쪽으로 4
    area = np.ones(original.shape[:2], bool)
    dx, dy, _ = script.estimate_shift(original, edited, area)
    assert abs(dx + 4) < 1 and abs(dy - 6) < 1
    back = script.align(edited, dx, dy)
    inner = (slice(40, -40), slice(40, -40))
    assert np.abs(back[inner].astype(int) - original[inner]).mean() < 3


def test_composite_changes_only_inside_mask():
    person = textured(1)
    edited = textured(2)
    change = np.zeros(person.shape[:2], bool); change[300:700, 200:500] = True
    result = script.composite(person, edited, change)
    assert np.array_equal(result[~change], person[~change])
    assert not np.array_equal(result[change], person[change])


def test_head_composite_keeps_head_core_and_drops_original_background():
    person = textured(3)
    edited = textured(4)
    protect = np.zeros(person.shape[:2], bool); protect[100:400, 250:550] = True
    result, core = script.head_composite(person, edited, protect)
    assert core.any() and np.array_equal(result[core], person[core])
    assert np.array_equal(result[~protect], edited[~protect])          # 머리 밖은 원본 배경·소품이 남지 않는다
    edge = protect & ~(np.roll(protect, 3, 0) & np.roll(protect, -3, 0))
    assert not np.array_equal(result[edge], person[edge])               # 가장자리는 Qwen 쪽과 섞인다


def test_border_white_ratio():
    image = np.full((1024, 768, 3), 255, np.uint8)
    assert script.border_white_ratio(image) == 1.0
    image[:, :20] = (200, 40, 90)
    assert 0 < script.border_white_ratio(image) < 1


class FakeCuda:
    def __init__(self, reserved):
        self.reserved, self.fraction = reserved, None

    def max_memory_reserved(self):
        return self.reserved * 2 ** 30

    def memory_reserved(self):
        return self.reserved * 2 ** 30

    def memory_allocated(self):
        return 2 ** 30

    def reset_peak_memory_stats(self):
        pass

    def get_device_properties(self, device):
        return type("Props", (), {"total_memory": 8 * 2 ** 30})()

    def set_per_process_memory_fraction(self, fraction, device):
        self.fraction = fraction


def test_memory_guard_caps_allocation_records_stages_and_stops(tmp_path):
    guard = script.MemoryGuard(FakeCuda(6.0), 6.5, tmp_path / "memory.json")
    assert guard.apply_cap() == guard.cuda.fraction == 6.5 / 8   # 한도 이상은 할당 자체가 실패하게 상한을 건다
    callback = guard.step_callback("W3")
    assert callback(None, 0, None, {"latents": 1}) == {"latents": 1}
    guard.cuda.reserved = 7.0
    try:
        callback(None, 1, None, {})
        raise AssertionError("한도 초과인데 멈추지 않았습니다")
    except MemoryError as error:
        assert "W3/step1" in str(error)
    import json
    log = json.loads((tmp_path / "memory.json").read_text(encoding="utf-8"))
    assert [s["stage"] for s in log["stages"]] == ["W3/step0(조건 그림 VAE 인코딩 포함)", "W3/step1"]
    assert log["limit_gib"] == 6.5 and "W3/step1" in log["stopped"]


def test_memory_guard_records_where_allocation_failed(tmp_path):
    guard = script.MemoryGuard(FakeCuda(5.0), 6.5, tmp_path / "memory.json")
    guard.check("W3/step39")
    guard.fail("CUDA out of memory")
    import json
    log = json.loads((tmp_path / "memory.json").read_text(encoding="utf-8"))
    assert "W3/step39" in log["stopped"] and "out of memory" in log["stopped"]   # 디코딩 중 실패면 마지막 단계 다음으로 남는다


def fake_parts():
    shape = (1024, 768)
    parts = {k: np.zeros(shape, bool) for k in script.HEAD_KEEP + script.HEAD_ACCESSORIES}
    parts["hair"][100:300, 300:500] = True
    parts["face"][200:300, 350:450] = True
    parts["headwear"][90:120, 300:500] = True      # 머리띠·헤드셋
    parts["earwear"][220:240, 290:300] = True
    return parts


def test_head_accessories_toggle_only_changes_head_accessories():
    parts = fake_parts()
    protect = np.logical_or.reduce(list(parts.values()))
    keep = script.head_protect(parts, protect, "keep")
    remove = script.head_protect(parts, protect, "remove")
    worn = parts["headwear"] | parts["earwear"]
    assert np.array_equal(keep, protect)
    assert not (remove & worn).any()                 # 끄면 머리 장식은 원본에서 가져오지 않는다
    assert np.array_equal(keep ^ remove, worn)       # 차이는 머리 장식뿐(머리카락과 겹친 장식 픽셀도 장식으로 본다)


def test_head_protect_rejects_mismatched_segmentation():
    parts = fake_parts()
    protect = np.logical_or.reduce(list(parts.values()))
    protect[600:700, 100:200] = True
    try:
        script.head_protect(parts, protect, "keep")
        raise AssertionError("분할 불일치를 통과했습니다")
    except ValueError:
        pass


def test_prompts_differ_only_in_head_accessory_sentence():
    keep, remove = script.PROMPT_HEAD["keep"], script.PROMPT_HEAD["remove"]
    assert "head accessories" in keep and "remove every head accessory" in remove.lower()
    assert keep.startswith(script.OUTFIT_HEAD) and remove.startswith(script.OUTFIT_HEAD)


def test_failure_during_first_load_is_named_and_measured(tmp_path):
    guard = script.MemoryGuard(FakeCuda(5.7), 6.5, tmp_path / "memory.json")
    guard.begin("text_encoder_load")
    guard.fail("CUDA out of memory. Tried to allocate 1.02 GiB")
    import json
    log = json.loads((tmp_path / "memory.json").read_text(encoding="utf-8"))
    assert log["stopped"].startswith("'text_encoder_load' 중")
    assert log["stages"] == [dict(stage="text_encoder_load(실패)", allocated_gib=1.0, reserved_gib=5.7,
                                  stage_peak_reserved_gib=5.7)]


def test_failure_measurement_errors_are_recorded_as_unmeasured(tmp_path):
    class Broken(FakeCuda):
        def memory_allocated(self):
            raise RuntimeError("CUDA 오류")
    guard = script.MemoryGuard(Broken(5.0), 6.5, tmp_path / "memory.json")
    guard.begin("transformer_load")
    guard.fail("out of memory")
    assert "unmeasured" in guard.stages[-1] and guard.stages[-1]["stage"] == "transformer_load(실패)"


def test_text_encoder_keeps_lm_head_off_gpu():
    assert script.text_encoder_device_map() == {"model": 0, "lm_head": "cpu"}
