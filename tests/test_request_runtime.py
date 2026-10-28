import pytest

from genai_lab.request_runtime import reset_request_runtime


class CachedPipeline:
    def __init__(self):
        self.released = False

    def maybe_free_model_hooks(self):
        self.released = True


class Pipeline:
    def __init__(self):
        self._interrupt = True
        self.scale = None
        self.hooks_released = False
        self._genai_lab_part_inpaint_pipeline = CachedPipeline()

    def maybe_free_model_hooks(self):
        self.hooks_released = True

    def set_ip_adapter_scale(self, value):
        self.scale = value


def test_request_boundary_reset_clears_mutable_state():
    pipeline = Pipeline()
    cached = pipeline._genai_lab_part_inpaint_pipeline

    report = reset_request_runtime(pipeline, boundary="request_end")

    assert report["status"] == "completed"
    assert pipeline.hooks_released is True
    assert pipeline._interrupt is False
    assert pipeline.scale == 0.0
    assert cached.released is True
    assert not hasattr(pipeline, "_genai_lab_part_inpaint_pipeline")


def test_request_boundary_reset_accepts_missing_pipeline():
    report = reset_request_runtime(None, boundary="request_start")
    assert report["status"] == "not_required"


def test_request_boundary_reset_rejects_unknown_boundary():
    with pytest.raises(ValueError, match="지원하지 않는"):
        reset_request_runtime(None, boundary="candidate")
