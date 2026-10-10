"""취소 시 참조·캐시·출력 폐기를 실제 정리 메서드와 가짜 GPU로 검사한다."""
import gc
import sys
import threading
import weakref
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication
from genai_lab import generation_cleanup as cleanup
from genai_lab.onepass_generation import DiffusersOnePassBackend, OnePassCancelled
from genai_lab.proportion_backend import GuardedBase, ProportionBackend
from genai_lab.finishing_backend import FinishingBackend
from genai_lab.finishing_offload import FinishingOffload
from genai_lab.studio_controller import StudioTask


class Payload:
    pass


def fake_torch(events, *, allocated=0, reserved=0):
    return SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True,
        synchronize=lambda: events.append("synchronize"),
        empty_cache=lambda: events.append("empty_cache"),
        memory_allocated=lambda: allocated, memory_reserved=lambda: reserved))


def test_pipeline_is_released_before_cache_clear():
    events, references = [], []
    class Pipe:
        def remove_all_hooks(self): events.append("hooks")
        def to(self, device): events.append(device)
    backend = object.__new__(DiffusersOnePassBackend)
    backend.pipe = Pipe()
    references.append(weakref.ref(backend.pipe))
    backend.torch = fake_torch(events)
    def empty():
        assert references[0]() is None
        events.append("empty_cache")
    backend.torch.cuda.empty_cache = empty
    backend.close()
    assert backend.pipe is None
    assert events == ["hooks", "cpu", "synchronize", "empty_cache"]


def test_failed_hook_cleanup_still_moves_model_and_clears_cache():
    events = []
    class Pipe:
        def remove_all_hooks(self): raise RuntimeError("broken hook")
        def to(self, device): events.append(device)
    backend = object.__new__(DiffusersOnePassBackend)
    backend.pipe, backend.torch = Pipe(), fake_torch(events)
    with pytest.raises(cleanup.GenerationCleanupError, match="broken hook"):
        backend.close()
    assert backend.pipe is None
    assert events == ["cpu", "synchronize", "empty_cache"]


def test_guarded_base_drops_duplicate_pipeline_reference():
    class Backend:
        def __init__(self): self.pipe, self.callback_mode = Payload(), "test"
        def close(self): self.pipe = None
    backend = Backend()
    reference = weakref.ref(backend.pipe)
    guarded = GuardedBase(backend, None)
    guarded.close()
    assert guarded.pipe is None and reference() is None


def test_proportion_hook_failure_does_not_skip_pipeline_close():
    events = []
    def broken(): raise RuntimeError("adapter")
    backend = object.__new__(ProportionBackend)
    backend.adapter_handles = [SimpleNamespace(remove=broken), SimpleNamespace(remove=lambda: events.append("hook2"))]
    backend.pipe, backend.torch = None, fake_torch(events)
    with pytest.raises(cleanup.GenerationCleanupError, match="adapter"):
        backend.close()
    assert backend.adapter_handles == []
    assert events == ["hook2", "synchronize", "empty_cache"]


def test_finishing_failure_does_not_skip_other_resources():
    events = []
    def broken(): raise RuntimeError("offload")
    backend = object.__new__(FinishingBackend)
    backend.offload = SimpleNamespace(close=broken)
    backend.face_cache = SimpleNamespace(close=lambda: events.append("face_cache"))
    backend.embeds, backend.pipe, backend.torch = Payload(), None, fake_torch(events)
    reference = weakref.ref(backend.embeds)
    with pytest.raises(cleanup.GenerationCleanupError, match="offload"):
        backend.close()
    assert backend.offload is backend.face_cache is backend.embeds is backend.pipe is None
    assert reference() is None
    assert events == ["face_cache", "synchronize", "empty_cache"]


def test_finishing_offload_releases_all_hooks_after_one_failure():
    events = []
    def broken(): raise RuntimeError("first model")
    owner = FinishingOffload(Payload(), {})
    owner.model_hooks = [SimpleNamespace(offload=broken, remove=lambda: events.append("remove1")),
                         SimpleNamespace(offload=lambda: events.append("offload2"), remove=lambda: events.append("remove2"))]
    owner.groups = [(SimpleNamespace(offload_=lambda: events.append("group")), {}, "unet")]
    owner.observer_hooks = [SimpleNamespace(remove=lambda: events.append("observer"))]
    with pytest.raises(cleanup.GenerationCleanupError, match="first model"):
        owner.close()
    assert events == ["offload2", "group", "observer", "remove1", "remove2"]
    assert owner.pipe is None and not owner.groups and not owner.model_hooks and not owner.observer_hooks


def owned_job(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "inputs").mkdir()
    (run / "inputs" / "character.png").write_bytes(b"original")
    (tmp_path / "saved.png").write_bytes(b"previous result")
    job = cleanup.CancelledGeneration(run)
    job.directory.mkdir()
    job.claim(job.directory)
    (job.directory / "seed-1").mkdir()
    (job.directory / "seed-1" / "raw.png").write_bytes(b"cancelled image")
    return job


def test_cancel_discards_only_owned_generation(tmp_path, monkeypatch):
    job = owned_job(tmp_path)
    monkeypatch.setitem(sys.modules, "torch", fake_torch([]))
    report = job.cleanup()
    assert report["status"] == "cancelled_cleaned" and report["outputs_removed"]
    assert not job.directory.exists()
    assert (job.run_directory / "inputs" / "character.png").read_bytes() == b"original"
    assert (tmp_path / "saved.png").read_bytes() == b"previous result"


def test_cancel_before_directory_creation_keeps_unclaimed_files(tmp_path, monkeypatch):
    job = cleanup.CancelledGeneration(tmp_path)
    job.directory.mkdir()
    (job.directory / "other.png").write_bytes(b"not owned")
    monkeypatch.setitem(sys.modules, "torch", fake_torch([]))
    job.cleanup()
    assert (job.directory / "other.png").read_bytes() == b"not owned"


def test_existing_generation_is_never_claimed(tmp_path):
    (tmp_path / "generation").mkdir()
    with pytest.raises(FileExistsError): cleanup.CancelledGeneration(tmp_path)


def test_changed_owner_blocks_deletion(tmp_path, monkeypatch):
    job = owned_job(tmp_path)
    (job.directory / ".gui-generation-owner").write_text("other owner")
    monkeypatch.setitem(sys.modules, "torch", fake_torch([]))
    with pytest.raises(cleanup.GenerationCleanupError, match="소유권"):
        job.cleanup()
    assert (job.directory / "seed-1" / "raw.png").exists()


def test_link_guard_prevents_partial_deletion(tmp_path, monkeypatch):
    job = owned_job(tmp_path)
    kind = type(job.directory)
    original = kind.is_symlink
    monkeypatch.setattr(kind, "is_symlink", lambda path: path.name == "raw.png" or original(path))
    with pytest.raises(cleanup.GenerationCleanupError, match="외부 연결"):
        job.discard_outputs()
    assert (job.directory / "seed-1" / "raw.png").exists()


@pytest.mark.parametrize("allocated,reserved", [(1, 0), (0, 1)])
def test_remaining_gpu_memory_blocks_completion_but_discards_outputs(tmp_path, monkeypatch, allocated, reserved):
    job = owned_job(tmp_path)
    monkeypatch.setitem(sys.modules, "torch", fake_torch([], allocated=allocated, reserved=reserved))
    with pytest.raises(cleanup.GenerationCleanupError, match="메모리가 남아"):
        job.cleanup()
    assert not job.directory.exists()


def test_cuda_failure_does_not_skip_output_discard(tmp_path, monkeypatch):
    job = owned_job(tmp_path)
    torch = fake_torch([])
    def broken(): raise RuntimeError("cuda error")
    torch.cuda.synchronize = broken
    monkeypatch.setitem(sys.modules, "torch", torch)
    with pytest.raises(cleanup.GenerationCleanupError, match="cuda error"):
        job.cleanup()
    assert not job.directory.exists()


def test_cpu_only_cancel_does_not_initialize_cuda(tmp_path, monkeypatch):
    job = owned_job(tmp_path)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False)))
    assert job.cleanup()["cuda_initialized"] is False


def test_chained_error_frames_do_not_retain_models():
    references = []
    def inner():
        model = Payload()
        references.append(weakref.ref(model))
        raise RuntimeError("inner")
    def outer():
        try: inner()
        except Exception as error: raise OnePassCancelled("cancel") from error
    try: outer()
    except Exception as error:
        assert references[0]() is not None
        cleanup.detach_error_frames(error)
        gc.collect()
        assert references[0]() is None
        assert error.__traceback__ is None and error.__cause__.__traceback__ is None
        assert str(error.__cause__) == "inner"


@pytest.mark.parametrize("moment", ["before", "during", "after"])
def test_task_cancel_releases_references_before_cleanup(tmp_path, monkeypatch, moment):
    app = QApplication.instance() or QApplication([])
    job = cleanup.CancelledGeneration(tmp_path)
    references = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch([]))
    def action(cancelled, progress):
        if moment == "before" and cancelled(): raise OnePassCancelled("before")
        job.directory.mkdir()
        job.claim(job.directory)
        (job.directory / "raw.png").write_bytes(b"partial")
        model = Payload()
        references.append(weakref.ref(model))
        task.stop.set()
        if moment == "during": raise OnePassCancelled("during")
        return model
    def on_cancel():
        gc.collect()
        assert not references or references[0]() is None
        assert task.action is None and task.result is None
        return job.cleanup()
    task = StudioTask(action, None, on_cancel=on_cancel)
    if moment == "before": task.stop.set()
    task.run()
    assert isinstance(task.error, OnePassCancelled)
    assert task.cleanup_report["status"] == "cancelled_cleaned"
    assert task.result is None and task.action is None and task.on_cancel is None
    assert not job.directory.exists()


def test_cleanup_runs_before_worker_finished_signal(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    begun, release = threading.Event(), threading.Event()
    def action(cancelled, progress):
        while not cancelled(): begun.wait(.01)
        raise OnePassCancelled("cancel")
    def on_cancel():
        begun.set()
        assert release.wait(5)
        return {"status": "cancelled_cleaned"}
    task = StudioTask(action, None, on_cancel=on_cancel)
    task.start()
    try:
        task.stop.set()
        assert begun.wait(5)
        assert task.isRunning()
        assert task.cleanup_report is None
    finally:
        release.set()
        assert task.wait(5000)
    assert task.cleanup_report["status"] == "cancelled_cleaned"


def test_task_cleanup_failure_is_not_reported_as_cancel_success():
    app = QApplication.instance() or QApplication([])
    def action(cancelled, progress): raise OnePassCancelled("cancel")
    def broken(): raise cleanup.GenerationCleanupError("cleanup failed")
    task = StudioTask(action, None, on_cancel=broken)
    task.run()
    assert isinstance(task.error, cleanup.GenerationCleanupError)
    assert task.cleanup_report is None and task.result is None
    assert "cleanup failed" in task.detail
    assert task.error.__traceback__ is None


def test_normal_completion_does_not_discard_results():
    app = QApplication.instance() or QApplication([])
    result = Payload()
    task = StudioTask(lambda cancel, progress: result, None,
                      on_cancel=lambda: pytest.fail("normal completion cannot discard"))
    task.run()
    assert task.result is result and task.error is None and task.cleanup_report is None


def test_model_close_failure_is_preserved_even_if_final_cache_clear_succeeds():
    app = QApplication.instance() or QApplication([])
    def action(cancelled, progress):
        task.stop.set()
        raise cleanup.GenerationCleanupError("model close failed")
    task = StudioTask(action, None, on_cancel=lambda: {"status": "cancelled_cleaned"})
    task.run()
    assert isinstance(task.error, cleanup.GenerationCleanupError)
    assert task.cleanup_report["status"] == "cancel_cleanup_failed"
    assert task.on_cancel is None


def test_cancel_request_after_completed_work_is_not_accepted():
    app = QApplication.instance() or QApplication([])
    value = Payload()
    task = StudioTask(lambda cancelled, progress: value, None,
                      on_cancel=lambda: pytest.fail("completed result cannot discard"))
    task.run()
    assert task.request_cancel() is False
    assert not task.stop.is_set() and task.result is value


def test_accepted_cancel_request_is_applied_before_completion(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    job = cleanup.CancelledGeneration(tmp_path)
    monkeypatch.setitem(sys.modules, "torch", fake_torch([]))
    def action(cancelled, progress):
        job.directory.mkdir()
        job.claim(job.directory)
        (job.directory / "raw.png").write_bytes(b"cancelled")
        assert task.request_cancel() is True
        return Payload()
    task = StudioTask(action, None, on_cancel=job.cleanup)
    task.run()
    assert isinstance(task.error, OnePassCancelled) and task.result is None
    assert not job.directory.exists()
    assert task.request_cancel() is False


def workspace_torch(events, *, model_bytes=0):
    """라이브러리 작업 공간과 모델 할당을 구분하는 가짜 GPU다."""
    memory = {"workspace": 9568256, "model": model_bytes, "reserved": 23068672}
    torch = fake_torch(events)
    torch.cuda.memory_allocated = lambda: memory["workspace"] + memory["model"]
    torch.cuda.memory_reserved = lambda: memory["reserved"]
    def clear_workspace():
        events.append("workspace")
        memory["workspace"] = 0
    def empty_cache():
        events.append("empty_cache")
        memory["reserved"] = memory["workspace"] + memory["model"]
    torch.cuda.empty_cache = empty_cache
    torch._C = SimpleNamespace(_cuda_clearCublasWorkspaces=clear_workspace)
    return torch


def test_library_workspace_is_released_before_allocator_cache():
    events = []
    report = cleanup.release_cuda_cache(workspace_torch(events))
    assert events == ["synchronize", "workspace", "empty_cache"]
    assert report["before_workspace_cleanup"]["allocated_bytes"] == 9568256
    assert report["allocated_bytes"] == report["reserved_bytes"] == 0
    assert report["cublas_workspace_cleanup"] == "completed"


def test_cancel_releases_library_workspace_and_discards_outputs(tmp_path, monkeypatch):
    job = owned_job(tmp_path)
    monkeypatch.setitem(sys.modules, "torch", workspace_torch([]))
    report = job.cleanup()
    assert report["status"] == "cancelled_cleaned"
    assert report["allocated_bytes"] == report["reserved_bytes"] == 0
    assert not job.directory.exists()


def test_workspace_cleanup_does_not_hide_live_model_memory(tmp_path, monkeypatch):
    job = owned_job(tmp_path)
    monkeypatch.setitem(sys.modules, "torch", workspace_torch([], model_bytes=512))
    with pytest.raises(cleanup.GenerationCleanupError, match="allocated=512"):
        job.cleanup()
    assert not job.directory.exists()


def test_missing_workspace_api_keeps_strict_memory_check(tmp_path, monkeypatch):
    job = owned_job(tmp_path)
    monkeypatch.setitem(sys.modules, "torch", fake_torch([], allocated=9568256, reserved=23068672))
    with pytest.raises(cleanup.GenerationCleanupError, match="메모리가 남아"):
        job.cleanup()


def test_workspace_release_failure_still_returns_allocator_cache():
    events = []
    torch = workspace_torch(events)
    def broken():
        events.append("workspace")
        raise RuntimeError("workspace release failed")
    torch._C._cuda_clearCublasWorkspaces = broken
    with pytest.raises(RuntimeError, match="workspace release failed"):
        cleanup.release_cuda_cache(torch)
    assert events == ["synchronize", "workspace", "empty_cache"]


def test_cpu_only_cleanup_does_not_call_workspace_api():
    def forbidden():
        raise AssertionError("CPU 정리에서 CUDA를 초기화하면 안 된다.")
    torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False),
                            _C=SimpleNamespace(_cuda_clearCublasWorkspaces=forbidden))
    assert cleanup.release_cuda_cache(torch)["cuda_initialized"] is False
