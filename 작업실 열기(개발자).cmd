@echo off
chcp 65001 >nul
rem 개발·평가 화면(확인 창·체크 판정·진단)으로 작업실 열기. 사용자 화면은 '작업실 열기.cmd'.
set GENAI_STUDIO_DEVELOPER=1
"D:\genai-cache\venv\Scripts\python.exe" "%~dp0gui_main.py"
if errorlevel 1 (
  echo.
  echo 작업실이 오류로 끝났습니다. 위 내용을 Claude에게 보내 주세요.
  pause
)
