# D: 고장(2026-10-16) 임시 복구 1단계: Python 환경 3개. 사용자 승인 "진행"(목록 확인 후).
# 설치 위치 C:\genai-temp-d\genai-cache, 가상 드라이브 subst D: C:\genai-temp-d 로 기존 D:/genai-cache 경로 유지.
# 인증서: Avast가 HTTPS를 가로채므로 Windows 신뢰 저장소를 PEM으로 내보내 검증에 사용(검증을 끄지 않음).
$ErrorActionPreference = "Stop"
$pem = "C:\genai-temp-d\genai-cache\windows-roots.pem"
$env:PIP_CERT = $pem; $env:SSL_CERT_FILE = $pem; $env:REQUESTS_CA_BUNDLE = $pem
$py = "C:\Program Files\Python310\python.exe"
$root = "D:\genai-cache"
$req = "\\192.168.0.109\win_g\genai-lab\requirements.txt"
function Step($name, [scriptblock]$body) { "=== $name $(Get-Date -Format HH:mm:ss)"; & $body; if ($LASTEXITCODE -ne 0) { throw "$name 실패 (exit $LASTEXITCODE)" } }

# 1) 운영 venv: torch cu128 + 프로젝트 의존성 + Qwen 실행에 필요한 고정 버전(작업 프로세스가 버전을 대조한다)
Step "main torch" { & "$root\venv\Scripts\python.exe" -m pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 --index-url https://download.pytorch.org/whl/cu128 }
Step "main requirements" { & "$root\venv\Scripts\python.exe" -m pip install -r $req "transformers==4.57.6" "bitsandbytes==0.50.2" "gguf==0.19.0" }

# 2) Qwen 시험 venv: 예전과 같이 운영 venv 패키지를 참조(.pth) + gguf 고정
if (-not (Test-Path "$root\qwen-test-venv\Scripts\python.exe")) { Step "qwen venv" { & $py -m venv "$root\qwen-test-venv" } }
Set-Content -Encoding ascii "$root\qwen-test-venv\Lib\site-packages\operation-venv.pth" "D:\genai-cache\venv\Lib\site-packages"
Step "qwen gguf" { & "$root\qwen-test-venv\Scripts\python.exe" -m pip install "gguf==0.19.0" }

# 3) 관절 분석 venv(easy-dwpose, CPU)
if (-not (Test-Path "$root\catvton-venv\Scripts\python.exe")) { Step "pose venv" { & $py -m venv "$root\catvton-venv" } }
Step "pose packages" { & "$root\catvton-venv\Scripts\python.exe" -m pip install "easy-dwpose==1.0.2" onnxruntime numpy pillow opencv-python }
"=== ENV SETUP DONE $(Get-Date -Format HH:mm:ss)"
