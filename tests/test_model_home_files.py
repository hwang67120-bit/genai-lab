"""모델 폴더 필수 파일 목록 초안(D-171): 배치(model_home.LAYOUT)와 맞고, 파일마다 크기·SHA가 있으며, Qwen은 기존 명세를 가리킨다."""
import json
import re
from pathlib import Path

from genai_lab import model_home

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "configs/model_home_files.json").read_text(encoding="utf-8"))


def test_every_component_sits_in_the_model_home_layout_with_sizes_and_hashes():
    roots = {model_home.LAYOUT[k] for k in ("model_cache", "pose_models", "head_cache", "head_semantic_repo")}
    for component in MANIFEST["components"]:
        assert component["files"] and component["missing_message"]
        assert any(component["folder"] == r or component["folder"].startswith(r + "/") for r in roots), component["folder"]
        for f in component["files"]:
            assert f["path"].startswith(component["folder"] + "/")
            assert f["size"] > 0 or component["key"] == "see_through_code"     # 코드 저장소에는 빈 파일(__init__.py 등)이 있다
            assert re.fullmatch(r"[0-9a-f]{64}", f["sha256"])


def test_observed_models_and_the_head_segmentation_checkpoint_are_listed():
    paths = {f["path"] for c in MANIFEST["components"] for f in c["files"]}
    assert any(p.endswith("wd-vit-tagger-v3/snapshots/7f6b584d0bd3f55c4531f14ba3d4761b2bccdc0f/model.onnx") for p in paths)
    assert {"dwpose/checkpoints/dw-ll_ucoco_384.onnx", "dwpose/checkpoints/yolox_l.onnx"} <= paths
    assert "hf/" + model_home.L2D_SAM_SNAPSHOT in paths


def test_qwen_points_at_the_existing_manifest_and_layout():
    qwen = MANIFEST["qwen"]
    assert qwen["folder"] == model_home.LAYOUT["qwen_model_root"] and qwen["gguf"] == model_home.LAYOUT["qwen_gguf"]
    assert (ROOT / qwen["manifest"]).is_file()
