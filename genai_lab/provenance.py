"""읽기 전용 실행 출처 기록이다. 텐서 계산·난수 호출·모델 로드는 없다. 참고:
https://docs.python.org/3/library/stdtypes.html#mapping-types-dict 및
https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html. 설정을 읽었다는 기록은 이후 결과에
영향을 줬다는 증거가 아니다.
"""
from __future__ import annotations

import atexit
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import weakref
from collections.abc import ItemsView, ValuesView

_ROOT = Path(__file__).resolve().parents[1]
_MISSING = object()
_UNTRACKED = {}


def plain(value):
    """메타데이터만 저장한다. 텐서나 이미지 내용을 문자열로 만들거나 복사하지 않는다."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in dict.items(value)}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    return {"object_type": type(value).__module__ + "." + type(value).__qualname__}


def flatten(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for k, v in dict.items(value):
            name = f"{prefix}.{k}" if prefix else str(k)
            result[name] = plain(v)
            result.update(flatten(v, name))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            if isinstance(v, dict):
                result.update(flatten(v, f"{prefix}.{i}"))
    return result


def _serialization_access():
    # 설정 복사와 기록 직렬화가 모든 말단 값을 읽어 버리지 않게 한다.
    frame = sys._getframe(1)
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        if module in {"json.encoder", "copy"}:
            return True
        if (module == "genai_lab.generation_replay"
                and frame.f_code.co_name == "_json_safe"):
            return True
        frame = frame.f_back
    return False


class _Items(ItemsView):
    def __iter__(self):
        for key in dict.__iter__(self._mapping):
            yield key, self._mapping[key]


class _Values(ValuesView):
    def __iter__(self):
        for key in dict.__iter__(self._mapping):
            yield self._mapping[key]


class TrackedConfig(dict):
    """값·순서·예외·기본값을 유지하는 사전 호환 트리다."""

    def __init__(self, value=(), *, recorder=None, path=""):
        self.recorder = recorder
        self.path = path
        super().__init__()
        iterator = dict.items(value) if isinstance(value, dict) else value
        for key, item in iterator:
            self[key] = item

    def _path(self, key):
        return f"{self.path}.{key}" if self.path else str(key)

    def __setitem__(self, key, value):
        def wrap(v, p):
            if isinstance(v, dict):
                if isinstance(v, TrackedConfig) and v.recorder is self.recorder and v.path == p:
                    return v
                return TrackedConfig(v, recorder=self.recorder, path=p)
            if isinstance(v, list):
                # 목록 객체는 유지한다. 사전 요소에만 경로가 필요하다.
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        v[i] = wrap(item, f"{p}.{i}")
            return v
        dict.__setitem__(self, key, wrap(value, self._path(key)))

    def __getitem__(self, key):
        value = dict.__getitem__(self, key)
        if self.recorder is not None and not _serialization_access():
            self.recorder.access(self._path(key), value)
        return value

    def get(self, key, default=None):
        if dict.__contains__(self, key):
            return self[key]
        if self.recorder is not None and not _serialization_access():
            self.recorder.defaults[self._path(key)] = plain(default)
        return default

    def __iter__(self):
        # CPython의 dict(config)와 **config가 항목 조회를 우회하는 것도 막는다.
        return dict.__iter__(self)

    def items(self):
        return _Items(self)

    def values(self):
        return _Values(self)

    def setdefault(self, key, default=None):
        if not dict.__contains__(self, key):
            self[key] = default
        return self[key]

    def update(self, other=(), **kwargs):
        iterator = other.items() if hasattr(other, "items") else other
        for key, value in iterator:
            self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def pop(self, key, *default):
        if dict.__contains__(self, key):
            self[key]
        return dict.pop(self, key, *default)

    def copy(self):
        return TrackedConfig(self, recorder=self.recorder, path=self.path)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        result = TrackedConfig(recorder=self.recorder, path=self.path)
        memo[id(self)] = result
        for key, value in dict.items(self):
            result[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
        return result


class RunProvenance:
    def __init__(self, config, path=None, root=None):
        self.root = Path(root or _ROOT)
        self.config_path = str(path or self.root / "configs/animagine.yaml")
        declared = plain(config)
        source = Path(self.config_path)
        self.source_sha256 = None
        if source.is_file():
            raw = source.read_bytes()
            self.source_sha256 = hashlib.sha256(raw).hexdigest()
            if source.suffix.lower() in {".yaml", ".yml"}:
                import yaml
                declared = yaml.safe_load(raw)
            elif source.suffix.lower() == ".json":
                declared = json.loads(raw)
        self.declared = flatten(declared)
        self.read = {}
        self.history = {}
        self.defaults = {}
        self.loaded = {}
        self.snapshots = []
        self.gates = []
        self.attempts = []
        self.branches = []
        self.events = []
        self.errors = []
        self.completed = False
        self.path = None
        self.run_id = None
        self.code = {}
        self.handles = []
        self.hooked = set()
        self.bound = False
        self._exit_registered = False

    def access(self, key, value):
        snapshot = plain(value)
        self.read[key] = snapshot
        versions = self.history.setdefault(key, [])
        if not versions or versions[-1] != snapshot:
            versions.append(snapshot)

    def start(self):
        if self.path is not None:
            return
        self.run_id = f"load-{time.time_ns()}-{os.getpid()}"
        self.path = self.root / "outputs/provenance-runs" / self.run_id / "run_provenance.json"
        sources = {}
        for folder in ("genai_lab", "scripts"):
            for p in sorted((self.root / folder).glob("*.py")):
                sources[str(p.relative_to(self.root))] = hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (self.root / "run.py", self.root / "gui_main.py"):
            if p.is_file():
                sources[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        self.code = {"source_sha256": sources}
        try:
            self.code["commit"] = subprocess.check_output(
                ["git", "-C", str(self.root), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            self.code["commit"] = None
        if not self._exit_registered:
            atexit.register(self.flush)
            self._exit_registered = True
        self.flush()

    def bind(self, context):
        self.start()
        old_path = self.path
        self.run_id = context.run_id
        self.path = Path(context.run_directory) / "run_provenance.json"
        self.bound = True
        self.flush()
        # 초기 로드 실패 기록을 남기고 실행 폴더가 정해지면 기록 위치를 옮긴다.
        if old_path != self.path:
            old_path.write_text(json.dumps({
                "schema_version": 1, "relocated_to": str(self.path),
                "run_id": self.run_id}, indent=2) + "\n", encoding="utf-8")

    def record(self):
        return {
            "schema_version": 1, "run_id": self.run_id,
            "completed": self.completed, "code": self.code,
            "config": {
                "path": self.config_path, "sha256": self.source_sha256,
                "declared_count": len(self.declared),
                "declared": self.declared, "read": self.read,
                "read_history": self.history, "default_reads": self.defaults,
                "unread": sorted(set(self.declared) - set(self.read)),
                "tracking_available": getattr(self, "tracking_available", True),
                "semantics": "Observed dict value accesses, not causal usage; parent access does not read descendants. copy.copy/deepcopy and JSON serialization excluded; dict()/unpacking count as value access.",
            },
            "loaded": self.loaded, "loaded_snapshots": self.snapshots,
            "gates": self.gates, "attempts": self.attempts,
            "implementation_choices": self.branches,
            "events": self.events, "observation_errors": self.errors,
        }

    def flush(self):
        if self.path is None:
            return
        # 진단 입출력 오류가 생성 예외를 덮어쓰면 안 된다.
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(self.record(), ensure_ascii=False,
                                           indent=2) + "\n", encoding="utf-8")
            temporary.replace(self.path)
        except (OSError, TypeError, ValueError) as error:
            self.errors.append({"operation": "flush", "error": type(error).__name__})

    def event(self, stage, status, **details):
        self.events.append({"stage": stage, "status": status, **plain(details)})
        if status in {"failed", "cancelled"}:
            self.completed = False
            # 중단 여부는 판정 문구가 아닌 실제로 관찰한 실행 실패를 뜻한다.
            for gate in self.gates:
                gate["aborted"] = True
        if stage == "finalize_selected_candidate" and status == "completed":
            self.completed = True
            self.remove_hooks()
        self.flush()

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.hooked.clear()

    def gate(self, name, report, *, scope):
        if report is None:
            return
        verdict = report.get("status", report.get("action", "not_reported"))
        self.gates.append({
            "name": name, "scope": scope, "verdict": verdict,
            "reported_action": report.get("action"), "action": "none",
            "retry_count": 0, "aborted": False, "observed_actions": [],
        })

    def action(self, name, scope, action):
        for gate in reversed(self.gates):
            if gate["name"] == name and gate["scope"] == scope:
                gate["action"] = action
                gate["observed_actions"].append(action)
                return

    def attempt(self, candidate, integrity_attempt, retry_phase):
        self.attempts.append({"candidate": candidate,
                              "integrity_attempt": integrity_attempt,
                              "retry_phase": retry_phase})
        # 설정된 횟수나 RETRY 판정이 아닌 실제 호출 횟수를 센다.
        if integrity_attempt > 1:
            for gate in self.gates:
                if gate["scope"] == f"base:{candidate}" and gate["name"] == "integrity":
                    gate["retry_count"] += 1
                    gate["action"] = "retry"
                    gate["observed_actions"].append("retry")
        elif retry_phase is not None:
            # 묶음 재시도를 특정 검사 하나의 책임으로 단정할 수 없다.
            # 이전 후보를 실제로 격리한 검사만 대상이 된다.
            for gate in self.gates:
                if (gate["scope"].startswith("base:")
                        and "quarantined" in gate["observed_actions"]):
                    gate["retry_count"] += 1
                    gate["action"] = "retry"
                    gate["retry_count_scope"] = "subsequent_batch_retry_calls_not_unique_causality"
        self.flush()

    def pipeline(self, pipe, stage):
        self.start()
        unet = getattr(pipe, "unet", None)
        encoder = getattr(pipe, "image_encoder", None)
        config = getattr(pipe, "config", {})
        def cfg(obj, name):
            return getattr(getattr(obj, "config", None), name, None)
        components = {}
        for name in ("unet", "vae", "text_encoder", "text_encoder_2",
                     "image_encoder", "controlnet"):
            mod = getattr(pipe, name, None)
            if mod is None:
                continue
            params = getattr(mod, "parameters", lambda: ())()
            locations = sorted({(str(p.dtype), str(p.device)) for p in params})
            components[name] = [{"dtype": d, "device": v} for d, v in locations]
        scales = {name: plain(proc.scale) for name, proc in
                  getattr(unet, "attn_processors", {}).items() if hasattr(proc, "scale")}
        control = getattr(pipe, "controlnet", None)
        controls = (getattr(control, "nets", [control]) if control is not None else [])
        loras = []
        for name in ("unet", "text_encoder", "text_encoder_2"):
            for adapter, settings in getattr(getattr(pipe, name, None), "peft_config", {}).items():
                loras.append({"component": name, "adapter": adapter,
                              "base_model_name_or_path": getattr(settings, "base_model_name_or_path", None)})
        receipt = getattr(pipe, "_provenance_adapter_receipt", None)
        if receipt is None and unet is not None:
            receipt = getattr(unet, "_provenance_adapter_receipt", None)
        snap = {
            "stage": stage, "pipeline_class": type(pipe).__name__,
            "unet_in_channels": cfg(unet, "in_channels"),
            "base_model": config.get("_name_or_path") if hasattr(config, "get") else None,
            "ip_adapter_weight": receipt.get("weight_name") if receipt else None,
            "adapter_load_receipt": plain(receipt),
            "ip_adapter_scale": scales,
            "image_encoder_hidden_size": cfg(encoder, "hidden_size"),
            "scheduler_class": type(getattr(pipe, "scheduler", None)).__name__,
            "components": components, "lora": loras,
            "controlnet": [{"class": type(m).__name__, "model": cfg(m, "_name_or_path")} for m in controls],
            "ip_adapter_tokens": None, "image_projection_outputs": [],
            "vae_tiling": getattr(getattr(pipe, "vae", None), "use_tiling", None),
            "padding_mask_crop": None,
        }
        self.snapshots.append(snap)
        self.loaded = snap
        pipe._run_provenance = self
        projections = getattr(getattr(unet, "encoder_hid_proj", None), "image_projection_layers", ())
        for i, projection in enumerate(projections):
            identity = id(projection)
            if identity in self.hooked or not hasattr(projection, "register_forward_hook"):
                continue
            self.hooked.add(identity)
            owner = weakref.ref(self)
            def observed(module, args, output, index=i):
                rec = owner()
                if rec is not None:
                    shape = list(output.shape) if hasattr(output, "shape") else None
                    observation = {"layer": index, "class": type(module).__name__, "shape": shape}
                    current = rec.loaded
                    if observation not in current["image_projection_outputs"]:
                        current["image_projection_outputs"].append(observation)
                        current["ip_adapter_tokens"] = shape[-2] if shape and len(shape) >= 3 else None
                # None을 반환한다. 결과 교체·분리·복제·보관을 하지 않는다.
                return None
            self.handles.append(projection.register_forward_hook(observed))
        self.flush()


def recorder(config):
    if isinstance(config, TrackedConfig):
        return config.recorder
    entry = _UNTRACKED.get(id(config))
    if entry is not None and entry[0] is config:
        return entry[1]
    if isinstance(config, dict):
        for value in dict.values(config):
            if isinstance(value, TrackedConfig):
                return value.recorder
    return None


def track_config(config, path=None, root=None):
    if isinstance(config, TrackedConfig):
        if not config.recorder.completed:
            return config
        path = path or config.recorder.config_path
        root = root or config.recorder.root
    return TrackedConfig(config, recorder=RunProvenance(config, path, root))


def observe_pipeline(config, pipeline, stage, arguments=None):
    rec = recorder(config)
    if rec is not None:
        rec.pipeline(pipeline, stage)
        if arguments is not None:
            rec.loaded["call_arguments"] = {
                k: plain(v) for k, v in arguments.items()
                if k not in {"generator", "image", "mask_image", "ip_adapter_image",
                             "callback_on_step_end", "prompt_embeds", "pooled_prompt_embeds",
                             "negative_prompt_embeds", "negative_pooled_prompt_embeds"}}
            rec.loaded["input_image_size"] = list(arguments["image"].size) if hasattr(arguments.get("image"), "size") else None
            rec.loaded["padding_mask_crop"] = arguments.get("padding_mask_crop")
            rec.flush()


def observe_gate(config, name, report, scope):
    rec = recorder(config)
    if rec is not None:
        rec.gate(name, report, scope=scope)


def observe_action(config, name, scope, action):
    rec = recorder(config)
    if rec is not None:
        rec.action(name, scope, action)


def observe_attempt(config, candidate, integrity_attempt, retry_phase):
    rec = recorder(config)
    if rec is not None:
        rec.attempt(candidate, integrity_attempt, retry_phase)


def adapter_loaded(pipeline, **receipt):
    # Diffusers 모듈 상태에는 가중치 파일명이 남지 않는다. 이 확인 기록은
    # 실제 로더가 성공한 뒤에만 붙이며 파일명을 추정하지 않는다.
    pipeline._provenance_adapter_receipt = receipt
    unet = getattr(pipeline, "unet", None)
    if unet is not None:
        unet._provenance_adapter_receipt = receipt


def observe_choice(pipeline, name, choice):
    rec = getattr(pipeline, "_run_provenance", None)
    if rec is not None:
        rec.branches.append({"name": name, "choice": choice})


def ensure_recorder(config, root=None):
    """호출부가 소유한 별칭을 유지한다. 설정 로더는 추적 사전을 제공한다. 일반 사전의 읽기 추적 범위는 명시적으로 확인 불가다. 읽지 않은 목록을 사용하지 않은
    설정으로 단정하지 않는다.
    """
    rec = recorder(config)
    if rec is None:
        rec = RunProvenance(config, root=root)
        rec.tracking_available = False
        _UNTRACKED[id(config)] = (config, rec)
    return rec
