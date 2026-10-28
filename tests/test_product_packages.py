"""제품 패키지 목록(D-170): 모두 고정 버전, 관찰한 핵심 패키지 포함, safetensors는 확인된 버전 그대로."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_product_packages", ROOT / "scripts/check_product_packages.py")
check_product_packages = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_product_packages)


@pytest.mark.parametrize("env, required", [
    ("main", {"torch": "2.9.1+cu128", "diffusers": "0.38.0", "transformers": "4.57.6", "bitsandbytes": "0.50.2", "gguf": "0.19.0",
              "safetensors": "0.9.0rc1", "PySide6": "6.12.0", "onnxruntime": "1.23.2", "dghs-imgutils": None}),
    ("pose", {"easy_dwpose": "1.0.2", "onnxruntime": "1.23.2", "numpy": "1.26.4"}),
])
def test_lists_are_fully_pinned_and_keep_the_observed_versions(env, required):
    pins = check_product_packages.read_requirements(ROOT / f"packaging/requirements-{env}.txt")
    for name, version in required.items():
        assert name in pins and (version is None or pins[name] == version)


def test_qwen_version_contract_matches_the_main_list():
    from genai_lab.qwen_pose_settings import EXPECTED_VERSIONS
    pins = check_product_packages.read_requirements(ROOT / "packaging/requirements-main.txt")
    assert all(pins[name] == version for name, version in EXPECTED_VERSIONS.items())


def test_unpinned_lines_are_rejected(tmp_path):
    bad = tmp_path / "r.txt"
    bad.write_text("# x\nnumpy>=2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="고정 버전"):
        check_product_packages.read_requirements(bad)
