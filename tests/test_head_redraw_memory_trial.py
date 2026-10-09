"""머리 재생성 시험의 CPU 보호 계약과 메모리 기록 저장을 검증한다."""
import json
from types import SimpleNamespace
from scripts import head_redraw_memory_trial as trial


def test_latent_and_pixel_protection_on_cpu():
    trial.check_cpu()


def test_memory_snapshot_is_saved_without_cuda(tmp_path):
    cuda = SimpleNamespace(memory_allocated=lambda: 100, memory_reserved=lambda: 200,
                           max_memory_allocated=lambda: 150, max_memory_reserved=lambda: 250)
    parameter = SimpleNamespace(device=SimpleNamespace(type="cuda"),
                                numel=lambda: 10, element_size=lambda: 2)
    model = SimpleNamespace(parameters=lambda: iter([parameter]))
    pipe = SimpleNamespace(**{name: model for name in
        ("text_encoder", "text_encoder_2", "image_encoder", "adapter", "unet", "vae")})
    row = trial.memory_snapshot(SimpleNamespace(cuda=cuda), pipe, tmp_path, "계산 후")
    saved = json.loads((tmp_path / "memory.jsonl").read_text(encoding="utf-8"))
    assert saved == row
    assert saved["allocated_bytes"] == 100
    assert saved["reserved_bytes"] == 200
    assert saved["peak_allocated_bytes"] == 150
    assert saved["peak_reserved_bytes"] == 250
    assert saved["gpu_parameter_bytes"]["unet"] == 20
