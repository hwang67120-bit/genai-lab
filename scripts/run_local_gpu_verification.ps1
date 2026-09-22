param(
    [Parameter(Mandatory=$true)]
    [ValidateNotNullOrEmpty()]
    [string]$Prompt,
    [switch]$ProbeOnly
)

$ErrorActionPreference = "Stop"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:DIFFUSERS_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"

$Python = "D:\genai-cache\venv\Scripts\python.exe"
$ProjectRoot = "\\192.168.0.109\win_g\genai-lab"
$Character = "C:\Users\user\Downloads\참조 이미지\HFTK9dCbgAAkxKb.png"
$Garment = "C:\Users\user\Downloads\참조 의상\19f3058071f8221e4d563036a40a03b5.jpg"
$Cache = "C:\Users\user\.cache\huggingface"
$Seed = 24681357
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$RunRoot = Join-Path $ProjectRoot "outputs\local-gpu-verification-$Stamp"
New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null

$EnvironmentRecord = [ordered]@{
    version = "local_gpu_verification_v1"
    execution_scope = "runtime_smoke"
    product_pipeline_equivalent = $false
    final_return_eligible = $false
    started_at = (Get-Date).ToString("o")
    selected_model = "FLUX.2-klein-4B"
    execution_mode = "offline_local_diffusers_pytorch_cuda"
    python = $Python
    project_root = $ProjectRoot
    cache_dir = $Cache
    character_reference = $Character
    garment_reference = $Garment
    prompt = $Prompt
    seed = $Seed
    offline_environment = [ordered]@{
        HF_HUB_OFFLINE = $env:HF_HUB_OFFLINE
        TRANSFORMERS_OFFLINE = $env:TRANSFORMERS_OFFLINE
        DIFFUSERS_OFFLINE = $env:DIFFUSERS_OFFLINE
        HF_HUB_DISABLE_TELEMETRY = $env:HF_HUB_DISABLE_TELEMETRY
    }
    gpu_before = (& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1 | Out-String).Trim()
}
$EnvironmentPath = Join-Path $RunRoot "environment.json"
[IO.File]::WriteAllText($EnvironmentPath, ($EnvironmentRecord | ConvertTo-Json -Depth 8), (New-Object Text.UTF8Encoding($false)))

if ($ProbeOnly) {
    Write-Host "검증 실행기 프로브 완료: $RunRoot" -ForegroundColor Green
    exit 0
}

$GpuLog = Join-Path $RunRoot "gpu-monitor.csv"
$GpuErrorLog = Join-Path $RunRoot "gpu-monitor-error.log"
$MonitorArguments = @("--query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total", "--format=csv", "--loop-ms=1000")
$Monitor = Start-Process -FilePath "nvidia-smi" -ArgumentList $MonitorArguments -WindowStyle Hidden -RedirectStandardOutput $GpuLog -RedirectStandardError $GpuErrorLog -PassThru

function Invoke-Smoke {
    param(
        [string]$Name,
        [string]$ScriptPath,
        [string[]]$ModelArguments
    )

    $OutputDirectory = Join-Path $RunRoot $Name
    New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
    $Arguments = @("-u", $ScriptPath, "--character", $Character, "--garment", $Garment, "--output-dir", $OutputDirectory, "--cache-dir", $Cache, "--seed", "$Seed", "--prompt", $Prompt, "--width", "320", "--height", "512", "--local-files-only") + $ModelArguments

    $Invocation = [ordered]@{
        model = $Name
        executable = $Python
        arguments = $Arguments
        started_at = (Get-Date).ToString("o")
    }
    $InvocationPath = Join-Path $OutputDirectory "invocation.json"
    [IO.File]::WriteAllText($InvocationPath, ($Invocation | ConvertTo-Json -Depth 8), (New-Object Text.UTF8Encoding($false)))

    Write-Host ""
    Write-Host "[$Name] 오프라인 로컬 GPU 실행 시작" -ForegroundColor Cyan
    & $Python @Arguments
    $ExitCode = $LASTEXITCODE
    $Invocation.finished_at = (Get-Date).ToString("o")
    $Invocation.exit_code = $ExitCode
    [IO.File]::WriteAllText($InvocationPath, ($Invocation | ConvertTo-Json -Depth 8), (New-Object Text.UTF8Encoding($false)))
    if ($ExitCode -ne 0) {
        throw "$Name 실행 실패: exit code $ExitCode"
    }
}

try {
    Invoke-Smoke -Name "flux2-klein" -ScriptPath (Join-Path $ProjectRoot "scripts\flux2_klein_smoke.py") -ModelArguments @("--steps", "4", "--guidance-scale", "1.0")
}
finally {
    if ($null -ne $Monitor -and -not $Monitor.HasExited) {
        Stop-Process -Id $Monitor.Id -Force
        $Monitor.WaitForExit()
    }
}

Write-Host ""
Write-Host "검증 완료: $RunRoot" -ForegroundColor Green
Write-Host "environment.json, gpu-monitor.csv, invocation.json, 결과 PNG/JSON을 확인하세요."

