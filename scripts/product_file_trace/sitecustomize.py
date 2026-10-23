"""파일 열기 관찰(제품 의존 파일 목록 조사용, 2026-10-23). GENAI_OPEN_TRACE가 있을 때만 켠다.

- 파이썬 open 감사 이벤트 + 네이티브 코드가 직접 여는 경로(onnxruntime·safetensors·huggingface_hub 반환 경로·torch.load)를 감싸 기록한다.
- 그래도 C 확장이 자체로 여는 파일은 빠질 수 있다 → 관찰 목록은 '최소 목록'이다.
"""
import os
import sys

_LOG = os.environ.get("GENAI_OPEN_TRACE")
if _LOG:
    import importlib.abc
    import importlib.machinery
    import json
    import threading

    _lock = threading.Lock()
    _skip_ext = (".py", ".pyc", ".pyd", ".dll", ".pth")

    def _write(kind, path):
        try:
            p = os.fspath(path)
        except TypeError:
            return
        if not isinstance(p, str) or p.endswith(_skip_ext) or "__pycache__" in p:
            return
        with _lock:
            with open(_LOG, "a", encoding="utf-8") as f:   # 감사 훅 재귀 방지: 아래에서 _LOG는 거른다
                f.write(json.dumps(dict(pid=os.getpid(), argv0=sys.argv[0] if sys.argv else "", kind=kind, path=p),
                                   ensure_ascii=False) + "\n")

    def _hook(event, args):
        if event == "open" and args and args[0] != _LOG and isinstance(args[0], (str, bytes, os.PathLike)):
            mode = args[1] if len(args) > 1 and isinstance(args[1], str) else "r"
            if "r" in mode and "+" not in mode and "w" not in mode and "a" not in mode:
                _write("open", args[0] if not isinstance(args[0], bytes) else args[0].decode(errors="replace"))

    sys.addaudithook(_hook)

    def _wrap(module, name, kind, arg_index=0, result=False):
        original = getattr(module, name, None)
        if original is None or getattr(original, "_traced", False):
            return
        def wrapper(*a, **k):
            if not result:
                target = a[arg_index] if len(a) > arg_index else k.get("path_or_bytes") or k.get("filename") or k.get("f")
                if isinstance(target, (str, os.PathLike)):
                    _write(kind, target)
            out = original(*a, **k)
            if result and isinstance(out, (str, os.PathLike)):
                _write(kind, out)
            return out
        wrapper._traced = True
        setattr(module, name, wrapper)

    _TARGETS = {
        "onnxruntime": lambda m: _wrap(m, "InferenceSession", "onnx"),
        "safetensors": lambda m: _wrap(m, "safe_open", "safetensors"),
        "safetensors.torch": lambda m: _wrap(m, "load_file", "safetensors"),
        "huggingface_hub": lambda m: (_wrap(m, "hf_hub_download", "hf_download", result=True),
                                      _wrap(m, "snapshot_download", "hf_snapshot", result=True)),
        "torch": lambda m: _wrap(m, "load", "torch_load"),
    }

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname not in _TARGETS:
                return None
            for finder in sys.meta_path:
                if finder is self:
                    continue
                spec = finder.find_spec(fullname, path, target) if hasattr(finder, "find_spec") else None
                if spec is not None and spec.loader is not None:
                    loader = spec.loader
                    original_exec = loader.exec_module
                    def exec_module(module, _orig=original_exec, _name=fullname):
                        _orig(module)
                        try:
                            _TARGETS[_name](module)
                        except Exception:
                            pass
                    loader.exec_module = exec_module
                    return spec
            return None

    sys.meta_path.insert(0, _Finder())

    # 선택: 열린 핸들 표본(GENAI_OPEN_TRACE_HANDLES=1). C 확장이 직접 연 파일 중 '열려 있는 동안' 잡히는 것만 보인다
    # (열었다 바로 닫은 파일·핸들 없이 매핑된 파일은 빠진다). 위 기록에 없는 파일이 여기서 나오면 관찰 도구가 놓친 범위다.
    if os.environ.get("GENAI_OPEN_TRACE_HANDLES") == "1":
        def _sample():
            import time
            try:
                import psutil
            except ImportError:
                return
            proc, seen = psutil.Process(os.getpid()), set()
            while True:
                try:
                    for item in proc.open_files():
                        if item.path not in seen:
                            seen.add(item.path)
                            _write("handle", item.path)
                except Exception:
                    pass
                time.sleep(0.2)
        threading.Thread(target=_sample, name="open-trace-handles", daemon=True).start()
