"""허깅페이스 인증정보 차단(D-172): 앱·하위 프로세스가 사용자 토큰 파일·토큰 변수를 읽지 않는다. 사용자 파일은 건드리지 않는다."""
import os
import subprocess
import sys
from pathlib import Path

from genai_lab.child_env import child_environment, hf_token_path, isolate_hf_credentials


def test_isolation_moves_the_token_path_to_app_data_and_drops_token_variables():
    env = {"APPDATA": "C:\\Users\\u\\AppData\\Roaming", "HF_TOKEN": "hf_secret", "HUGGING_FACE_HUB_TOKEN": "hf_old"}
    path = isolate_hf_credentials(env)
    assert path == os.path.join("C:\\Users\\u\\AppData\\Roaming", "GenAILab", "hf-token-disabled", "token")
    assert env["HF_TOKEN_PATH"] == path and "HF_TOKEN" not in env and "HUGGING_FACE_HUB_TOKEN" not in env


def test_child_environment_never_carries_tokens_and_points_at_the_app_path():
    env = child_environment({"APPDATA": "A", "HF_TOKEN": "hf_secret", "HF_TOKEN_PATH": "C:\\Users\\u\\.cache\\huggingface\\token"})
    assert "HF_TOKEN" not in env and env["HF_TOKEN_PATH"] == hf_token_path({"APPDATA": "A"})


def test_a_child_using_huggingface_hub_does_not_open_the_user_token_file(tmp_path):
    user_home = tmp_path / "user-hf"
    user_home.mkdir()
    (user_home / "token").write_text("hf_user_token", encoding="utf-8")        # 시험용 '사용자' 토큰(진짜 사용자 파일 아님)
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import sys\n"
        "opened = []\n"
        "sys.addaudithook(lambda e, a: opened.append(str(a[0])) if e == 'open' and a and isinstance(a[0], str) else None)\n"
        "from huggingface_hub.utils import get_token\n"
        "print(repr(get_token()))\n"
        "print(any(p.endswith('user-hf' + __import__('os').sep + 'token') or p.endswith('user-hf/token') for p in opened))\n",
        encoding="utf-8")
    parent = dict(os.environ, HF_HOME=str(user_home), APPDATA=str(tmp_path / "appdata"))
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH"):     # 이 시험 프로세스가 이미 차단돼 있어도 대조가 되게
        parent.pop(name, None)
    before = subprocess.run([sys.executable, str(probe)], env=dict(parent, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True, timeout=120)
    after = subprocess.run([sys.executable, str(probe)], env=child_environment(parent), capture_output=True, text=True, timeout=120)
    assert before.stdout.split() == ["'hf_user_token'", "True"]               # 막지 않으면 사용자 토큰을 읽는다(대조)
    assert after.stdout.split() == ["None", "False"], after.stderr             # 막으면 열지도 않는다
    assert (user_home / "token").read_text(encoding="utf-8") == "hf_user_token"     # 파일은 그대로
    assert not Path(hf_token_path({"APPDATA": str(tmp_path / "appdata")})).exists()


def test_the_studio_isolates_credentials_before_heavy_imports():
    source = (Path(__file__).resolve().parents[1] / "gui_main.py").read_text(encoding="utf-8")
    assert source.index("isolate_hf_credentials()") < source.index("import torch")
