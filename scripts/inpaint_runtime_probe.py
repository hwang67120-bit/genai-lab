"""Lightweight, removable runtime advice. No tensor capture or CUDA synchronization."""

from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
from queue import Queue, Empty, Full
import sys
from threading import Event, Lock, Thread, get_ident
from time import perf_counter, time

try:
    import psutil
except ImportError:
    psutil = None


def sample_resources(torch_module=None):
    values = {'pid': os.getpid()}
    if psutil is None:
        values['cpu_metrics'] = 'psutil_unavailable'
    else:
        try:
            proc = psutil.Process()
            mem = proc.memory_info()
            values.update(rss_mib=round(mem.rss / 1048576, 1),
                          private_mib=round(getattr(mem, 'private', mem.vms) / 1048576, 1),
                          private_metric='private' if hasattr(mem, 'private') else 'vms',
                          system_available_mib=round(psutil.virtual_memory().available / 1048576, 1))
            io = proc.io_counters()
            values.update(io_read_bytes=io.read_bytes, io_write_bytes=io.write_bytes)
        except Exception as error:
            values['cpu_metrics_error'] = type(error).__name__
    if torch_module is not None:
        try:
            cuda = torch_module.cuda
            if cuda.is_initialized():
                values.update(cuda_allocated_mib=round(cuda.memory_allocated() / 1048576, 1),
                              cuda_reserved_mib=round(cuda.memory_reserved() / 1048576, 1),
                              cuda_peak_allocated_mib=round(cuda.max_memory_allocated() / 1048576, 1))
            else:
                values['cuda_metrics'] = 'not_initialized'
        except Exception as error:
            values['cuda_metrics_error'] = type(error).__name__
    return values


class RuntimeProbe:
    def __init__(self, path, torch_module=None, heartbeat_seconds=10.0):
        self.path = Path(path)
        self.torch = torch_module
        self.interval = max(0.05, heartbeat_seconds)
        self.started = perf_counter()
        self.queue = Queue(maxsize=2048)
        self.stop = Event()
        self.lock = Lock()
        self.active = {}
        self.counts = {}
        self.originals = []
        self.dropped = 0
        self.thread = Thread(target=self._writer, name='inpaint-runtime-probe', daemon=True)
        self.thread.start()
        self.emit('probe_started', 'runtime')

    def emit(self, event, stage, **extra):
        # Fail-open: diagnostics must not turn a valid generation into a failure.
        try:
            payload = dict(event=event, stage=stage, timestamp=time(),
                           elapsed=round(perf_counter() - self.started, 4),
                           resources=sample_resources(self.torch), dropped_events=self.dropped,
                           **extra)
            self.queue.put_nowait(payload)
        except Full:
            self.dropped += 1
        except Exception:
            pass

    @contextmanager
    def measure(self, stage):
        started = perf_counter()
        thread_id = get_ident()
        with self.lock:
            self.counts[stage] = self.counts.get(stage, 0) + 1
            call = self.counts[stage]
            entry = {'stage': stage, 'call': call, 'started': started}
            self.active.setdefault(thread_id, []).append(entry)
        self.emit('start', stage, call=call)
        try:
            yield
        except BaseException as error:
            self.emit('failed', stage, call=call, duration=round(perf_counter() - started, 4),
                      error_type=type(error).__name__, error=str(error)[:1000])
            raise
        else:
            self.emit('completed', stage, call=call, duration=round(perf_counter() - started, 4))
        finally:
            with self.lock:
                stack = self.active.get(thread_id, [])
                if entry in stack:
                    stack.remove(entry)
                if not stack:
                    self.active.pop(thread_id, None)

    def wrap(self, obj, method, stage):
        if obj is None:
            return
        try:
            original = getattr(obj, method, None)
            if not callable(original):
                self.emit('unavailable', stage)
                return
            own = method in vars(obj)
            previous = vars(obj).get(method)
            @wraps(original)
            def advised(*args, **kwargs):
                with self.measure(stage):
                    return original(*args, **kwargs)
            setattr(obj, method, advised)
            self.originals.append((obj, method, own, previous))
        except Exception as error:
            self.emit('attach_failed', stage, error_type=type(error).__name__)

    def attach_pipeline(self, pipeline):
        for name in ('encode_prompt', 'prepare_latents', 'prepare_mask_latents',
                     'prepare_control_image', '_encode_vae_image'):
            self.wrap(pipeline, name, 'pipeline.' + name)
        for name in ('text_encoder', 'text_encoder_2', 'vae', 'controlnet', 'unet'):
            module = getattr(pipeline, name, None)
            methods = ('encode', 'decode') if name == 'vae' else ('forward',)
            for method in methods:
                self.wrap(module, method, name + '.' + method)
            hook = getattr(module, '_hf_hook', None)
            for method in ('pre_forward', 'post_forward'):
                self.wrap(hook, method, name + '.offload_' + method)

    def detach(self):
        for obj, method, own, previous in reversed(self.originals):
            try:
                if own:
                    setattr(obj, method, previous)
                else:
                    delattr(obj, method)
            except Exception:
                pass
        self.originals.clear()

    def _writer(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a', encoding='utf-8') as stream:
                next_heartbeat = perf_counter() + self.interval
                while not self.stop.is_set() or not self.queue.empty():
                    try:
                        payload = self.queue.get(timeout=max(0.01, min(0.25, next_heartbeat - perf_counter())))
                    except Empty:
                        payload = None
                    if payload is not None:
                        stream.write(json.dumps(payload, ensure_ascii=False) + '\n')
                        stream.flush()
                    if not self.stop.is_set() and perf_counter() >= next_heartbeat:
                        with self.lock:
                            active = [dict(stage=e['stage'], call=e['call'],
                                           active_seconds=round(perf_counter() - e['started'], 1))
                                      for stack in self.active.values() for e in stack]
                        self.emit('heartbeat', 'runtime', active=active)
                        next_heartbeat = perf_counter() + self.interval
        except Exception as error:
            try:
                print(f'[runtime probe unavailable] {type(error).__name__}', file=sys.stderr, flush=True)
            except Exception:
                pass

    def close(self):
        self.detach()
        self.emit('probe_closed', 'runtime')
        self.stop.set()
        self.thread.join(timeout=2.0)
