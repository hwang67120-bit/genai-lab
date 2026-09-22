import gc
import json
import subprocess
import sys
from threading import Event, Thread
from types import SimpleNamespace
import weakref

import pytest

from scripts.inpaint_runtime_probe import RuntimeProbe, sample_resources
from genai_lab.inpaint_runtime_monitor import RuntimeLogTail


def records(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def test_advice_preserves_output_nested_order_and_detaches(tmp_path):
    class Model:
        def forward(self, value, *, delta=1):
            return value + delta
    model = Model()
    path = tmp_path / 'runtime_trace.jsonl'
    probe = RuntimeProbe(path)
    probe.wrap(model, 'forward', 'unet.forward')
    with probe.measure('pipeline.generate'):
        assert model.forward(2, delta=3) == 5
    probe.close()
    assert 'forward' not in vars(model)
    assert model.forward(4) == 5
    events = [(r['stage'], r['event']) for r in records(path) if r['event'] in ('start', 'completed')]
    assert events == [('pipeline.generate', 'start'), ('unet.forward', 'start'),
                      ('unet.forward', 'completed'), ('pipeline.generate', 'completed')]


def test_exception_identity_and_no_tensor_retention(tmp_path):
    error = RuntimeError('original failure')
    class Model:
        def forward(self, value):
            raise error
    path = tmp_path / 'trace.jsonl'
    probe = RuntimeProbe(path)
    model = Model()
    probe.wrap(model, 'forward', 'unet.forward')
    with pytest.raises(RuntimeError) as caught:
        model.forward(object())
    assert caught.value is error
    probe.close()
    assert any(r['event'] == 'failed' and r['error_type'] == 'RuntimeError' for r in records(path))
    # Probe keeps no model references after detach.
    ref = weakref.ref(model)
    del caught
    error.__traceback__ = None
    del model
    gc.collect()
    assert ref() is None


def test_live_heartbeat_during_blocked_operation(tmp_path):
    path = tmp_path / 'trace.jsonl'
    entered, release = Event(), Event()
    probe = RuntimeProbe(path, heartbeat_seconds=0.05)
    def work():
        with probe.measure('vae.encode'):
            entered.set()
            release.wait(2)
    worker = Thread(target=work)
    worker.start()
    assert entered.wait(1)
    # Wait on the evidence, not on GPU work.
    found = Event()
    for _ in range(50):
        if path.exists():
            try:
                if any(r.get('active') for r in records(path) if r['event'] == 'heartbeat'):
                    found.set()
                    break
            except ValueError:
                pass
        found.wait(0.02)
    release.set()
    worker.join(2)
    probe.close()
    assert found.is_set()
    assert any(r.get('active', [{}])[0].get('stage') == 'vae.encode'
               for r in records(path) if r['event'] == 'heartbeat' and r.get('active'))


def test_bad_log_path_cannot_break_generation(tmp_path):
    probe = RuntimeProbe(tmp_path)  # directory is not a writable log file
    with probe.measure('test'):
        result = 7
    probe.close()
    assert result == 7


def test_cuda_probe_does_not_initialize_or_synchronize():
    cuda = SimpleNamespace(is_initialized=lambda: False)
    info = sample_resources(SimpleNamespace(cuda=cuda))
    assert info['cuda_metrics'] == 'not_initialized'


def test_tail_keeps_partial_line_and_does_not_replay(tmp_path, capsys):
    path = tmp_path / 'trace.jsonl'
    tail = RuntimeLogTail(path)
    assert tail.poll() == []
    path.write_bytes(b'{"event":"start","stage":"vae.encode","resources":{"pid":123}}')
    assert tail.poll() == []
    with path.open('ab') as stream:
        stream.write(b'\ninvalid\n')
    assert len(tail.poll()) == 1
    assert tail.poll() == []
    assert 'vae.encode' in capsys.readouterr().err


def test_offload_hook_is_preserved_and_measured(tmp_path):
    order = []
    class Hook:
        def pre_forward(self, module, *args, **kwargs):
            order.append('transfer')
            return args, kwargs
        def post_forward(self, module, output):
            return output
    class Model:
        def __init__(self):
            self._hf_hook = Hook()
        def forward(self, x):
            self._hf_hook.pre_forward(self, x)
            order.append('compute')
            return self._hf_hook.post_forward(self, x + 1)
    model = Model()
    probe = RuntimeProbe(tmp_path / 'trace.jsonl')
    probe.attach_pipeline(SimpleNamespace(unet=model))
    assert model.forward(1) == 2
    probe.close()
    assert order == ['transfer', 'compute']
    assert any(r['stage'] == 'unet.offload_pre_forward' and r['event'] == 'completed'
               for r in records(tmp_path / 'trace.jsonl'))


def test_classmethod_wrapper_restores_descriptor(tmp_path):
    class Loader:
        @classmethod
        def from_pretrained(cls, name):
            return cls, name
    before = vars(Loader)['from_pretrained']
    probe = RuntimeProbe(tmp_path / 'trace.jsonl')
    probe.wrap(Loader, 'from_pretrained', 'model.load')
    assert Loader.from_pretrained('fake') == (Loader, 'fake')
    probe.close()
    assert vars(Loader)['from_pretrained'] is before


def test_child_stage_is_visible_before_child_finishes(tmp_path, capsys):
    path = tmp_path / 'runtime_trace.jsonl'
    script = (
        'import sys\n'
        'from scripts.inpaint_runtime_probe import RuntimeProbe\n'
        'p = RuntimeProbe(sys.argv[1], heartbeat_seconds=0.1)\n'
        'with p.measure("vae.encode"):\n'
        '    sys.stdin.readline()\n'
        'p.close()\n'
    )
    child = subprocess.Popen([sys.executable, '-c', script, str(path)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True)
    tail = RuntimeLogTail(path)
    seen = []
    try:
        for _ in range(150):
            seen.extend(tail.poll())
            if any(r['stage'] == 'vae.encode' and r['event'] == 'start' for r in seen):
                break
            Event().wait(0.02)
        assert any(r['stage'] == 'vae.encode' and r['event'] == 'start' for r in seen)
        assert child.poll() is None
        child.communicate('\n', timeout=5)
        seen.extend(tail.poll())
        assert any(r['stage'] == 'vae.encode' and r['event'] == 'completed' for r in seen)
        assert 'vae.encode' in capsys.readouterr().err
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=5)
