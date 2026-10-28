@echo off
chcp 65001 >nul
set "VERIFY_PROMPT="
set /p "VERIFY_PROMPT=이번 검증에 사용할 캐릭터와 의상 지시문을 입력하세요: "
if not defined VERIFY_PROMPT (
  echo 프롬프트가 비어 있어 실행을 중단합니다.
  pause
  exit /b 2
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "\\192.168.0.109\win_g\genai-lab\scripts\run_local_gpu_verification.ps1" -Prompt "%VERIFY_PROMPT%"
set "VERIFY_EXIT=%ERRORLEVEL%"
echo.
if "%VERIFY_EXIT%"=="0" (
  echo 로컬 GPU 검증이 완료되었습니다.
) else (
  echo 로컬 GPU 검증이 실패했습니다. 종료 코드: %VERIFY_EXIT%
)
pause
exit /b %VERIFY_EXIT%
