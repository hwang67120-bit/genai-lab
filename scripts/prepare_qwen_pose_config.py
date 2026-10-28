"""이미 준비된 로컬 Qwen 환경에서 CPU로 잠금을 만든다. 설치·다운로드·모델 가져오기·추론은 없다. 파일을 덮어쓰지 않는다. 실행 컴퓨터에서 한 번
수행하고 사용 전 경로·해시를 검토한다.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from genai_lab.qwen_pose_settings import QwenPoseSettings, EXPECTED_VERSIONS, validate_model_files
from genai_lab.qwen_preservation import file_sha
from genai_lab.qwen_pose_edit import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, dest="python_executable")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--analysis-cache", default="")
    parser.add_argument("--output", required=True, help="New directory for runtime.json + model-files.json")
    args = parser.parse_args(argv)
    # 설치 패키지 정보만 확인하며 여기서 torch를 가져올 필요는 없다.
    code = "import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in " + repr(list(EXPECTED_VERSIONS)) + "}))"
    result = subprocess.run([args.python_executable, "-c", code], capture_output=True, text=True, check=True)
    if json.loads(result.stdout) != EXPECTED_VERSIONS:
        raise ValueError("기존 Qwen 환경 버전이 시험 기준과 다릅니다. 설치/업데이트하지 않습니다.")
    root = Path(args.model_root).resolve()
    if not root.is_dir(): raise FileNotFoundError(root)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    entries = {p.relative_to(root).as_posix(): file_sha(p) for p in sorted(root.rglob("*"))
               if p.is_file() and ".cache" not in p.relative_to(root).parts}
    manifest = output / "model-files.json"
    write_json(manifest, {"files": entries, "provenance": "Local provisioned files; verify against approved trial cache"})
    settings = QwenPoseSettings(str(Path(args.python_executable).resolve()), str(root),
        str(Path(args.gguf).resolve()), str(manifest), file_sha(manifest),
        analysis_cache_dir=args.analysis_cache)
    validate_model_files(settings)
    write_json(output / "runtime.json", settings.record())
    print(output / "runtime.json")


if __name__ == "__main__":
    main()
