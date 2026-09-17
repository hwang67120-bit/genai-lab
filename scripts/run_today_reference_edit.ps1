param([switch]$PrepareOnly, [switch]$ReuseAnalysis)

$python = 'D:\genai-cache\venv\Scripts\python.exe'
$project = '\\192.168.0.109\win_g\genai-lab'
$arguments = @(
    "$project\scripts\run_today_reference_edit.py",
    '--output-dir', "$project\outputs\today-reference-edit-20260918",
    '--base-model-id', 'cagliostrolab/animagine-xl-3.1',
    '--seeds', '1109384701',
    '--inference-width', '512',
    '--steps', '18',
    '--strength', '0.82',
    '--guidance', '7.0',
    '--adapter-scale', '0.55',
    '--prompt', 'masterpiece, best quality, 1boy, male focus, masculine silhouette, solo, full body, standing, fitted dark teal school blazer, white collared shirt, separate very short dark teal pleated mini skirt, visible skirt waistband, straight skirt hem, skirt covering pelvis, bare upper thighs, simple tailored outfit, plain white background',
    '--negative-prompt', '1girl, woman, female, breasts, pants, trousers, shorts, jeans, leggings, bodysuit, leotard, swimsuit, underwear, highleg, loincloth, cape, cloak, long skirt, extra person, changed hair, changed animal ears, changed tail, scenery'
)
if ($PrepareOnly) { $arguments += '--prepare-only' }
if ($ReuseAnalysis) { $arguments += '--reuse-analysis' }
& $python @arguments
exit $LASTEXITCODE