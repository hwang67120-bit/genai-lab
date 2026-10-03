"""Prepare approved B replay for Claude; --execute explicitly enables the GPU worker.

Default is CPU preparation only. No downloads, retrials or reference C support.
"""
import argparse
from pathlib import Path, PureWindowsPath
import json

from genai_lab.qwen_preservation import PreservationItem, PreservationSpec, file_sha
from genai_lab.qwen_pose_prompt import PoseEditInstructions, assemble_pose_edit_prompt
from genai_lab.qwen_pose_settings import QwenPoseSettings
from genai_lab.qwen_pose_edit import PoseEditWorkflow, make_request, run_pose_edit, write_json

HANDS = "Both arms hang down at the sides of the body, and each hand rests lightly beside the hip, next to the side seam of the shorts."


def repository_path(value, root):
    # Trial files contain host drive letters. Resolve only their outputs-relative part.
    parts = PureWindowsPath(value).parts
    if "outputs" not in parts: raise ValueError("시험 입력 경로가 outputs 아래가 아닙니다.")
    relative = parts[parts.index("outputs"):]
    path = root.joinpath(*relative).resolve()
    if not path.is_relative_to(root.resolve()): raise ValueError("시험 입력 경로가 저장소 밖입니다.")
    return path


def prepare_replay(root, character, settings):
    trial = root / "outputs/qwen-pose-identity-20261003"
    source = json.loads((trial / "preservation.json").read_text(encoding="utf-8"))["characters"][character]
    prompts = json.loads((trial / "prompts.json").read_text(encoding="utf-8"))
    basis = repository_path(source["start"], root)
    if file_sha(basis) != source["start_sha256"]: raise ValueError("시험 기준 이미지 SHA 불일치")
    items = tuple(PreservationItem(x["id"], "style" if "그림체" in x["kind"] else x["kind"],
        x["fact"], x["english"], x["source"], source["start_sha256"], x["check_at"],
        "confirmed", x["approval"], approved_gender=True) for x in source["items"])
    spec = PreservationSpec(str(basis), source["start_sha256"], "시험 사용자 확인 기록 재현", items)
    instructions = PoseEditInstructions("front-facing", HANDS, "시험 사용자 확인 기록 재현")
    if assemble_pose_edit_prompt(spec, instructions)["positive"] != prompts["prompts"][character]["B"]:
        raise ValueError("시험 B 전문 불일치")
    manifest = json.loads((trial / "manifest.json").read_text(encoding="utf-8"))
    skeleton = repository_path(manifest["skeleton"]["path"], root)
    request = make_request(spec, skeleton, manifest["skeleton"]["sha256"], settings=settings,
                           instructions=instructions, seed=209212001)
    return spec, request


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--character", choices=("raccoon", "ordinary-female"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    spec, request = prepare_replay(root, args.character, QwenPoseSettings.load(args.runtime))
    if not args.execute:
        out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
        write_json(out / "prepared-request.json", request)
        print("CPU 준비 완료; 생성하지 않음:", out)
        return
    # This CLI owns no generation/analysis models. Approved trial analysis is replayed.
    class ApprovedAnalysis:
        def analyze(self, *args): return []
        def close(self): pass
    flow = PoseEditWorkflow()
    flow.release_generation(); flow.analyze(spec.image_path, ApprovedAnalysis); flow.confirm(spec)
    run_pose_edit(request, args.output, flow)


if __name__ == "__main__":
    main()
