@echo off
chcp 65001 >nul
rem 캐릭터 작업실 실행(더블클릭). 네트워크 경로에서도 이 파일 옆의 gui_main.py를 연다.
"D:\genai-cache\venv\Scripts\python.exe" "%~dp0gui_main.py"
if errorlevel 1 (
  echo.
  echo 작업실이 오류로 끝났습니다. 위 내용을 Claude에게 보내 주세요.
  pause
)
