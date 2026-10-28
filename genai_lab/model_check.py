"""필수 모델 검사(D-172): 준비 화면과 실제 실행이 같이 쓰는 하나의 검사.

목록: configs/model_home_files.json(분석 모델·머리 분할·see-through 코드: 경로·크기·SHA) + configs/qwen_model_files.json(Qwen 27개) + GGUF.
- 경로는 실행 설정(StudioRuntime·Qwen 설정)이 실제로 쓰는 위치에서 찾는다 — 모델 폴더(GENAI_MODEL_HOME)든 기존 경로든 같은 검사.
- 기본(빠른 검사): 파일이 있는지·크기가 같은지. deep=True(준비 화면): 분석 모델 SHA까지(Qwen SHA는 기존 validate_model_files).
- 작업 프로세스가 실행마다 하는 Qwen SHA 검사는 그대로 둔다(이 검사로 대신하지 않는다).
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "model_home_files.json"

LABELS = {
    "wd_tagger": "캐릭터 태그 분석 모델(WD 태거)",
    "anime_seg": "캐릭터 외곽 분할 모델(anime-seg)",
    "dwpose": "자세 분석 모델(DWPose)",
    "head_detection": "머리 검출 모델",
    "head_segmentation": "머리 분할 모델(l2d_sam)",
    "see_through_code": "머리 분할 도구 코드(see-through)",
    "qwen": "이미지 생성 모델(Qwen-Image-Edit-2511)",
}
# 사용자 화면 흐름별 필수 구성(CPU 관찰·정적 확인, INSTALLER_PACKAGING_PREP 3-1)
FLOWS = {
    "outfit": ("wd_tagger", "anime_seg", "dwpose", "head_detection", "head_segmentation", "see_through_code", "qwen"),
    "staging": ("wd_tagger", "dwpose", "head_segmentation", "see_through_code", "qwen"),
}
ACTION = ("할 일: 모델 준비를 다시 하거나(내려받기 또는 기존 모델 폴더 지정), 설정한 모델 폴더(GENAI_MODEL_HOME)가 "
          "맞는지 확인하세요. 파일을 고치거나 옮긴 뒤에는 다시 '이미지 생성'을 누르세요.")


class MissingModelFiles(RuntimeError):
    """필수 모델이 없거나 다르다. 성공으로 넘어가지 않는다."""

    def __init__(self, result):
        self.result = result
        super().__init__(message(result))


def component_location(key, folder, runtime):
    """목록의 모델 폴더 기준 상대 위치 → 이 실행 설정이 실제로 쓰는 위치."""
    if key == "wd_tagger" or key == "anime_seg" or key == "head_segmentation":
        return Path(runtime.model_cache) / Path(folder).name
    if key == "dwpose":
        return Path(runtime.pose_models)
    if key == "head_detection":
        return Path(runtime.head_cache) / Path(folder).name
    if key == "see_through_code":
        return Path(runtime.head_semantic_repo)
    raise ValueError(f"알 수 없는 구성: {key}")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_models(runtime, settings, flow, *, deep=False, manifest_path=MANIFEST, progress=None):
    """flow('outfit'|'staging')에 필요한 모델 → dict(ok, problems, locations). problems: 구성·경로·문제(없음/크기 다름/SHA 다름).
    progress: 구성마다 진행 문장을 받는 함수(준비 화면용, 검사 내용은 같다)."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    by_key = {c["key"]: c for c in manifest["components"]}
    problems, locations = [], {}
    for key in FLOWS[flow]:
        if progress is not None:
            progress(f"{LABELS[key]} 확인 중" + (" — 약 28GB SHA 대조, 몇 분 걸립니다" if key == "qwen" and deep else ""))
        if key == "qwen":
            problems += _check_qwen(settings, deep, locations)
            continue
        component = by_key[key]
        base = component_location(key, component["folder"], runtime)
        locations[key] = str(base if key != "head_segmentation" else runtime.head_semantic_checkpoint)
        for f in component["files"]:
            relative = Path(f["path"]).relative_to(component["folder"])
            path = Path(runtime.head_semantic_checkpoint) if key == "head_segmentation" else base / relative
            if not path.is_file():
                problems.append(dict(component=key, label=LABELS[key], path=str(path), problem="없음"))
            elif path.stat().st_size != f["size"]:
                problems.append(dict(component=key, label=LABELS[key], path=str(path), problem="크기 다름"))
            elif deep and _sha(path) != f["sha256"]:
                problems.append(dict(component=key, label=LABELS[key], path=str(path), problem="SHA 다름"))
    return dict(ok=not problems, flow=flow, deep=deep, problems=problems, locations=locations)


def _check_qwen(settings, deep, locations):
    root, gguf = Path(settings.model_root), Path(settings.gguf_file)
    locations["qwen"] = str(root)
    problems = []
    manifest = Path(settings.artifact_manifest)
    files = json.loads(manifest.read_text(encoding="utf-8"))["files"] if manifest.is_file() else {}
    if not files:
        problems.append(dict(component="qwen", label=LABELS["qwen"], path=str(manifest), problem="없음"))
    for relative in files:
        if not (root / relative).is_file():
            problems.append(dict(component="qwen", label=LABELS["qwen"], path=str(root / relative), problem="없음"))
    if not gguf.is_file():
        problems.append(dict(component="qwen", label=LABELS["qwen"], path=str(gguf), problem="없음"))
    if deep and not problems:
        from genai_lab.qwen_pose_settings import validate_model_files
        try:
            validate_model_files(settings)
        except ValueError as error:
            problems.append(dict(component="qwen", label=LABELS["qwen"], path=str(root), problem=f"SHA 다름: {error}"))
    return problems


def message(result, limit=3):
    """사용자에게 보일 안내: 필요한 모델 · 확인한 경로 · 할 일."""
    lines = ["필요한 모델 파일이 없거나 달라서 시작하지 않았습니다."]
    for key in dict.fromkeys(p["component"] for p in result["problems"]):
        items = [p for p in result["problems"] if p["component"] == key]
        lines.append(f"- {LABELS[key]}: {len(items)}개 파일 {items[0]['problem']}")
        lines.append(f"  확인한 위치: {result['locations'].get(key, '')}")
        for p in items[:limit]:
            lines.append(f"  · {p['path']} ({p['problem']})")
        if len(items) > limit:
            lines.append(f"  · …외 {len(items) - limit}개")
    lines.append(ACTION)
    return "\n".join(lines)


def require_product_models(runtime, settings, *, garment):
    """사용자 화면 '이미지 생성' 시작 전 검사. 문제가 있으면 MissingModelFiles로 멈춘다(빠른 검사)."""
    result = check_models(runtime, settings, "outfit" if garment else "staging")
    if not result["ok"]:
        raise MissingModelFiles(result)
    return result
