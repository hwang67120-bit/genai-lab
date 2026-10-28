"""로컬 인식 모델을 명시적으로 준비한다. 앱 실행 중에는 다운로드하지 않는다."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
FILES = ("README.md", "config.json", "generation_config.json", "preprocessor_config.json",
         "video_preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
         "chat_template.json", "vocab.json", "merges.txt", "model.safetensors")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""): digest.update(block)
    return digest.hexdigest()


def download_file(root, entry):
    name, size = entry["rfilename"], entry["size"]
    target = root / name
    expected = entry.get("lfs", {}).get("sha256")
    if target.exists() and target.stat().st_size == size:
        digest = sha(target)
        if expected is None or expected == digest:
            return {"size":size, "sha256":digest, "remote_lfs_sha256":expected}
        raise ValueError("Existing model hash mismatch: " + name)
    partial = target.with_suffix(target.suffix + ".partial")
    url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"
    # Windows 기본 인증서 저장소를 사용한다. TLS 검증을 끄지 않는다.
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as stream:
        total, milestone = 0, 0
        while True:
            block = response.read(8*1024*1024)
            if not block: break
            stream.write(block)
            total += len(block)
            if total // (256*1024*1024) > milestone:
                milestone = total // (256*1024*1024)
                print(f"{name}: {total}/{size}", flush=True)
    if partial.stat().st_size != size: raise ValueError("Download size mismatch: " + name)
    digest = sha(partial)
    if expected and digest != expected: raise ValueError("LFS hash mismatch: " + name)
    partial.replace(target)
    print("verified " + name, flush=True)
    return {"size":size, "sha256":digest, "remote_lfs_sha256":expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--settings", required=True)
    args = parser.parse_args()
    root, settings = Path(args.root).resolve(), Path(args.settings).resolve()
    if settings.exists(): raise FileExistsError("Do not overwrite runtime settings: " + str(settings))
    root.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as response: metadata=json.load(response)
    if metadata["sha"] != REVISION: raise ValueError("Revision mismatch")
    by_name={e["rfilename"]:e for e in metadata["siblings"]}
    files={name:download_file(root, by_name[name]) for name in FILES}
    manifest=root / "recognition-manifest.json"
    manifest.write_text(json.dumps({"model_id":MODEL,"revision":REVISION,"license":"Apache-2.0",
                                    "files":files},indent=2)+"\n",encoding="utf-8")
    config={"python_executable":sys.executable,"model_root":str(root),"model_revision":REVISION,
            "manifest_sha256":sha(manifest),"cache_dir":str(root.parent / "tail-recognition-cache"),
            "device":"cuda"}
    settings.parent.mkdir(parents=True,exist_ok=True)
    settings.write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"settings":str(settings),"total_bytes":sum(f["size"] for f in files.values()),
                      "manifest_sha256":sha(manifest)},indent=2),flush=True)

if __name__ == "__main__": main()
