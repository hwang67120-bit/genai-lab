"""Sequential GPU ownership and an explicit separate-process Qwen edit request."""
from dataclasses import asdict
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image
from genai_lab.qwen_preservation import PreservationSpec, file_sha, json_sha
from genai_lab.qwen_pose_prompt import PoseEditInstructions, assemble_pose_edit_prompt
from genai_lab.qwen_pose_settings import QwenPoseSettings, output_dimensions


from genai_lab.qwen_record_io import write_json, latest_progress, finalize_stopped_run


def assert_parent_gpu_released():
    """Do not initialize CUDA for this check. Allocations indicate live prior tensors."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_initialized():
        return {"cuda_initialized": False}
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    allocated = int(torch.cuda.memory_allocated())
    reserved = int(torch.cuda.memory_reserved())
    if allocated or reserved:
        raise RuntimeError(f"앞 단계 GPU가 해제되지 않았습니다: allocated={allocated}, reserved={reserved}")
    return {"cuda_initialized": True, "allocated": allocated, "reserved": reserved}


class PoseEditWorkflow:
    """Generation close -> analysis close -> user confirmation -> Qwen.

    Own all prior GPU resource close callbacks. Running upstream tasks must be
    rejected by the host before constructing this controller. Release failures
    are terminal; a boolean 'released' supplied by a caller is not accepted.
    """
    def __init__(self, generation_releases=(), *, gpu_probe=assert_parent_gpu_released):
        self.releases = list(generation_releases)
        self.gpu_probe = gpu_probe
        self.phase = "generation"
        self.events = []
        self.spec_sha256 = None
        self.analysis_image_sha256 = None

    def release_generation(self):
        if self.phase != "generation":
            raise RuntimeError("잘못된 단계 전환")
        try:
            for release in self.releases:
                release()
            self.releases.clear()
            self.events.append({"stage": "generation_released", "probe": self.gpu_probe()})
            self.phase = "ready_for_analysis"
        except BaseException:
            self.phase = "blocked"
            raise

    def analyze(self, image_path, analyzer_factory):
        if self.phase != "ready_for_analysis":
            raise RuntimeError("기준 이미지 생성 자원 해제가 먼저 필요합니다.")
        self.phase = "analysis"
        analyzer = None
        reports, failure = [], None
        try:
            data = Path(image_path).read_bytes()
            self.analysis_image_sha256 = hashlib.sha256(data).hexdigest()
            analyzer = analyzer_factory()
            with Image.open(io.BytesIO(data)) as source:
                with source.convert("RGB") as rgb:
                    reports = analyzer.analyze(rgb, self.analysis_image_sha256)
        except Exception as error:
            failure = {"type": type(error).__name__, "message": str(error)}
        finally:
            try:
                if analyzer is not None:
                    analyzer.close()
                analyzer = None
                self.events.append({"stage": "analysis_released", "probe": self.gpu_probe(), "failure": failure})
                self.phase = "awaiting_confirmation"
            except BaseException:
                self.phase = "blocked"
                raise
        return reports, failure

    def confirm(self, spec):
        if self.phase != "awaiting_confirmation":
            raise RuntimeError("분석 해제 후에만 사용자 확인이 가능합니다.")
        spec.verify_image()
        if spec.image_sha256 != self.analysis_image_sha256:
            raise ValueError("분석 기준 이미지가 변경됐습니다. 새 이미지 확인과 분석이 필요합니다.")
        assemble_pose_edit_prompt(spec)
        self.spec_sha256 = spec.sha256
        self.phase = "confirmed"

    def before_launch(self, spec):
        if self.phase != "confirmed" or self.spec_sha256 != spec.sha256:
            raise RuntimeError("현재 명세에 대한 사용자 확인이 필요합니다.")
        spec.verify_image()
        self.events.append({"stage": "before_qwen", "probe": self.gpu_probe()})
        self.phase = "qwen_running"


def make_request(spec, skeleton_path, skeleton_sha256, *, instructions=PoseEditInstructions(),
                 settings, seed=None):
    spec.verify_image()
    if file_sha(skeleton_path) != skeleton_sha256:
        raise ValueError("반전 전 골격 SHA 불일치")
    with Image.open(spec.image_path) as image:
        basis_size = image.size
        size = output_dimensions(*basis_size)
    with Image.open(skeleton_path) as image:
        if image.mode != "RGB":
            raise ValueError("정규화된 반전 전 RGB 골격이 필요합니다.")
    seed = settings.default_seed if seed is None else seed
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed 범위 오류")
    return {"schema_version": 1, "spec": spec.record(), "prompt": assemble_pose_edit_prompt(spec, instructions),
            "instructions": asdict(instructions), "skeleton_path": str(skeleton_path),
            "skeleton_sha256": skeleton_sha256, "skeleton_color_order": "openpose_rgb_before_t2i_reversal",
            "settings": settings.record(), "seed": seed, "output_size": list(size), "basis_size": list(basis_size), "images": 2}


def product_preview(raw_path, destination, *, basis_size):
    """Map raw pixels to the approved basis canvas; never alter raw.

    Pillow resize uses pixel centers: x_out = (x_in + .5)*sx - .5
    (and likewise for y). This restores the canvas mapping, not changes
    made by the model. No letterboxing or inferred feature alignment.
    https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize
    """
    if (not isinstance(basis_size, (tuple, list)) or len(basis_size) != 2
            or any(type(v) is not int or v <= 0 for v in basis_size)):
        raise ValueError("기준 이미지 크기는 양의 정수 2개가 필요합니다.")
    raw_path, destination = Path(raw_path), Path(destination)
    if (raw_path.resolve() == destination.resolve()
            or (destination.exists() and raw_path.samefile(destination))):
        raise ValueError("Qwen 원시 결과를 미리보기로 덮어쓸 수 없습니다.")
    size = tuple(basis_size)
    with Image.open(raw_path) as raw:
        raw_size = raw.size
        with raw.convert("RGB") as rgb, rgb.resize(size, Image.Resampling.LANCZOS) as resized:
            resized.save(destination)
    return {"version": 2, "kind": "resize_to_approved_basis", "raw_size": list(raw_size),
            "size": list(size), "offset": [0, 0], "canvas": list(size),
            "scale": [size[0] / raw_size[0], size[1] / raw_size[1]],
            "pixel_coordinates": "centers: (input + 0.5) * scale - 0.5",
            "filter": "LANCZOS"}


def run_pose_edit(request, directory, workflow, *, cancelled=lambda: False, popen=subprocess.Popen,
                  on_progress=lambda _: None):
    spec = PreservationSpec.from_record(request["spec"])
    return _run_image_edit(request, directory, workflow, spec, cancelled=cancelled,
                           popen=popen, on_progress=on_progress)


def _run_image_edit(request, directory, workflow, spec, *, cancelled=lambda: False,
                    popen=subprocess.Popen, on_progress=lambda _: None):
    """Shared process lifecycle; caller validates its own pose or tail contract."""
    settings = QwenPoseSettings(**request["settings"])
    if cancelled():
        raise RuntimeError("실행 전 취소")
    if not Path(settings.python_executable).is_file():
        raise FileNotFoundError("설정된 Qwen Python 실행 파일이 없습니다. 설치하지 않습니다.")
    # Bind output coordinates to the approved bytes, before starting the worker.
    basis_data = Path(spec.image_path).read_bytes()
    if hashlib.sha256(basis_data).hexdigest() != spec.image_sha256:
        raise ValueError("승인 기준 이미지 SHA 불일치")
    with Image.open(io.BytesIO(basis_data)) as basis:
        basis_size = basis.size
    if "basis_size" in request and request["basis_size"] != list(basis_size):
        raise ValueError("요청의 기준 이미지 크기가 승인 이미지와 다릅니다.")
    workflow.before_launch(spec)
    directory = Path(directory).resolve()
    process = None
    created = False
    record = {"status": "starting", "request_sha256": json_sha(request),
              "gpu_sequence": workflow.events, "final_return_eligible": False}
    try:
        directory.mkdir(parents=True, exist_ok=False)
        created = True
        write_json(directory / "request.json", request)
        write_json(directory / "launcher.json", record)
        command = [settings.python_executable, "-m", "genai_lab.qwen_pose_worker", str(directory / "request.json")]
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
        env.pop("PYTHONPATH", None)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        with (directory / "worker.log").open("w", encoding="utf-8") as log:
            process = popen(command, cwd=str(Path(__file__).resolve().parents[1]), env=env,
                            stdout=log, stderr=subprocess.STDOUT, shell=False, creationflags=creationflags)
            record.update(status="running", command=command, pid=process.pid)
            write_json(directory / "launcher.json", record)
            last_progress = None
            while process.poll() is None:
                if cancelled():
                    (directory / "cancel.request").touch()
                    process.terminate()
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise RuntimeError("사용자가 Qwen 실행을 취소했습니다.")
                progress = latest_progress(directory)
                if progress is not None and progress[0] != last_progress:
                    try:
                        on_progress(progress[1])
                        last_progress = progress[0]
                    except (OSError, ValueError): pass
                time.sleep(.2)
        if process.returncode != 0:
            raise RuntimeError(f"Qwen 프로세스 실패(exit={process.returncode}); worker.log/run.json 확인")
        result = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if result.get("status") != "completed" or result.get("request_sha256") != record["request_sha256"]:
            raise RuntimeError("불완전하거나 다른 요청의 Qwen 결과")
        if file_sha(directory / "raw.png") != result.get("raw_sha256"):
            raise RuntimeError("Qwen 원시 결과 SHA 불일치")
        derived = product_preview(directory / "raw.png", directory / "product.png", basis_size=basis_size)
        record.update(status="awaiting_review", raw_sha256=result["raw_sha256"], product_transform=derived,
                      product_sha256=file_sha(directory / "product.png"),
                      basis_sha256=spec.image_sha256)
        return directory / "product.png"
    except BaseException as error:
        record.update(status="cancelled" if cancelled() else "failed", error=str(error))
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        workflow.phase = "finished"
        if created:
            try:
                finalize_stopped_run(directory, record["request_sha256"],
                    cancelled=record["status"] == "cancelled", error=record.get("error", "Worker ended without a final record"),
                    returncode=process.returncode if process is not None else None)
            except BaseException as final_error:
                record.update(status="failed", final_record_error=str(final_error))
                raise
            finally:
                write_json(directory / "launcher.json", record)
