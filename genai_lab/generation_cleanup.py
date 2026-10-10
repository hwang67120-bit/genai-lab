"""취소한 생성의 GPU 참조와 소유한 출력만 폐기한다. 원본 입력은 건드리지 않는다."""
import gc
from pathlib import Path
import shutil
import sys
import traceback
import uuid


class GenerationCleanupError(RuntimeError):
    """정리가 불완전하면 다음 생성을 허용하지 않는다."""


def cleanup_steps(steps):
    """한 정리가 실패해도 나머지는 수행한다. 오류에는 모델 참조를 남기지 않는다."""
    errors = []
    for name, action in steps:
        try:
            action()
        except Exception as error:
            errors.append(f"{name}: {type(error).__name__}: {error}")
    return errors


def release_pipeline(pipe):
    """훅을 떼고 남은 가중치를 CPU로 내린다. 훅 제거만으로는 GPU가 비워지지 않는다."""
    steps = []
    for name, arguments in (("remove_all_hooks", ()), ("to", ("cpu",))):
        method = getattr(pipe, name, None)
        if callable(method):
            steps.append((name, lambda method=method, arguments=arguments: method(*arguments)))
    return cleanup_steps(steps)


def release_cuda_cache(torch):
    """모델 참조를 끊은 다음 캐시를 반환한다. CUDA를 새로 초기화하지 않는다."""
    gc.collect()
    if torch is None or not torch.cuda.is_initialized():
        return {"cuda_initialized": False, "allocated_bytes": 0, "reserved_bytes": 0}
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return {"cuda_initialized": True,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved()}


def detach_error_frames(error):
    """문자 오류 기록은 유지하고, 완료된 계산 프레임과 연쇄 오류의 모델 참조를 끊는다."""
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        pending.extend((current.__cause__, current.__context__))
        pending.extend(getattr(current, "exceptions", ()))
        if current.__traceback__ is not None:
            traceback.clear_frames(current.__traceback__)
            current.__traceback__ = None


class CancelledGeneration:
    """출력 폴더의 생성 직후 소유권을 기록하고, 취소한 해당 폴더만 지운다."""
    def __init__(self, run_directory):
        self.run_directory = Path(run_directory).resolve(strict=True)
        self.directory = self.run_directory / "generation"
        self.token = uuid.uuid4().hex
        self.claimed = False
        if self.directory.exists() or self.directory.is_symlink():
            raise FileExistsError(f"기존 생성 결과는 덮어쓰거나 폐기하지 않습니다: {self.directory}")

    def claim(self, directory):
        """실행기가 exist_ok=False로 새 폴더를 만든 뒤에만 호출한다."""
        directory = Path(directory)
        if directory.is_symlink() or directory.resolve(strict=True) != self.directory:
            raise GenerationCleanupError("생성 출력 경로가 실행 폴더 밖입니다.")
        with (directory / ".gui-generation-owner").open("x", encoding="utf-8") as stream:
            stream.write(self.token)
        self.claimed = True

    def discard_outputs(self):
        """삭제 전 모든 경로와 소유권을 확인한다. 링크나 다른 실행의 자료는 지우지 않는다."""
        if not self.claimed:
            return
        directory = self.directory
        if not directory.exists():
            raise GenerationCleanupError("소유한 생성 폴더를 확인할 수 없습니다.")
        paths = [directory]
        while paths:
            path = paths.pop()
            reparse = getattr(path.lstat(), "st_file_attributes", 0) & 0x400
            if path.is_symlink() or reparse or not path.resolve(strict=True).is_relative_to(directory):
                raise GenerationCleanupError(f"생성 폴더에 외부 연결 경로가 있습니다: {path}")
            if path.is_dir():
                paths.extend(path.iterdir())
        if directory.resolve(strict=True).parent != self.run_directory:
            raise GenerationCleanupError("생성 폴더의 상위 경로가 바뀌었습니다.")
        if (directory / ".gui-generation-owner").read_text(encoding="utf-8") != self.token:
            raise GenerationCleanupError("생성 폴더의 소유권이 바뀌었습니다.")
        shutil.rmtree(directory)
        self.claimed = False

    def cleanup(self):
        """계산이 종료된 뒤 호출한다. 실패해도 파일 정리와 메모리 정리를 각각 시도한다."""
        memory = {}
        def clear_memory():
            memory.update(release_cuda_cache(sys.modules.get("torch")))
            if memory["allocated_bytes"] or memory["reserved_bytes"]:
                raise GenerationCleanupError(f"취소 뒤에도 PyTorch GPU 메모리가 남아 있습니다: allocated={memory['allocated_bytes']}, reserved={memory['reserved_bytes']} bytes")
        errors = cleanup_steps((("GPU 정리", clear_memory), ("생성 결과 폐기", self.discard_outputs)))
        if errors:
            raise GenerationCleanupError("취소 정리를 마치지 못했습니다. 앱을 다시 시작해 주세요.\n" + "\n".join(errors))
        return {"status": "cancelled_cleaned", "outputs_removed": not self.claimed, **memory}
