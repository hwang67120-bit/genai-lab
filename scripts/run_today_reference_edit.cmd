@echo off
setlocal
set "PYTHON=D:\genai-cache\venv\Scripts\python.exe"
set "SCRIPT=\\192.168.0.109\win_g\genai-lab\scripts\run_today_reference_edit.py"
set "OUT=\\192.168.0.109\win_g\genai-lab\outputs\today-reference-edit-20260918"

"%PYTHON%" "%SCRIPT%" ^
  --output-dir "%OUT%" ^
  --base-model-id "cagliostrolab/animagine-xl-3.1" ^
  --seeds 1109384701 ^
  --reuse-analysis ^
  --inference-width 512 ^
  --steps 18 ^
  --strength 0.82 ^
  --guidance 7.0 ^
  --adapter-scale 0.55 ^
  --prompt "masterpiece, best quality, 1boy, male focus, masculine silhouette, solo, full body, standing, fitted dark teal school blazer, white collared shirt, separate very short dark teal pleated mini skirt, visible skirt waistband, straight skirt hem, skirt covering pelvis, bare upper thighs, simple tailored outfit, plain white background" ^
  --negative-prompt "1girl, woman, female, breasts, pants, trousers, shorts, jeans, leggings, bodysuit, leotard, swimsuit, underwear, highleg, loincloth, cape, cloak, long skirt, extra person, changed hair, changed animal ears, changed tail, scenery"

if errorlevel 1 exit /b %errorlevel%
echo.
echo Review: %OUT%\review-sheet.png
endlocal
