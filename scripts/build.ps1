# Builds dist\DigUp\ (DigUp.exe + digup-cli.exe) and, with Inno Setup installed, an installer in dist\.
#   powershell -ExecutionPolicy Bypass -File scripts\build.ps1 [-SkipTests]
param([switch]$SkipTests)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path .venv)) { uv venv --python 3.12 .venv }
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
if (-not $SkipTests) {
    .venv\Scripts\python.exe -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "tests failed" }
}
.venv\Scripts\pyinstaller.exe --noconfirm --clean --distpath dist --workpath build packaging\digup.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$iscc = Get-Command iscc.exe -ErrorAction SilentlyContinue
if (-not $iscc) { $iscc = Get-Item "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" -ErrorAction SilentlyContinue }
if ($iscc) {
    & $iscc packaging\installer.iss
    Write-Host "Installer: dist\DigUp-Setup.exe"
} else {
    Write-Host "Built dist\DigUp\DigUp.exe (install Inno Setup 6 for an installer)"
}
