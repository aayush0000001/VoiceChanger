# Builds V-COMMS into dist\V-COMMS (PyInstaller) and, if Inno Setup is
# installed, into dist\installer\V-COMMS-Setup-<version>.exe.
#
# Usage (from anywhere):
#   powershell -ExecutionPolicy Bypass -File packaging\build.ps1 [-Version 1.0.0] [-Python <path to x64 python.exe>] [-SkipInstaller]
#
# Uses its own venv (.venv-build) so it never touches your dev .venv.

param(
    [string]$Version = "1.0.0",
    [string]$Python = "",
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv-build"
$VenvPy = Join-Path $Venv "Scripts\python.exe"

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe failed with exit code $LASTEXITCODE" }
}

# 1. Build venv. The app ships as x64 (the dev setup and its DirectML
#    wheels are x64), so prefer the x64 Python even on Windows on Arm.
if (-not (Test-Path $VenvPy)) {
    if (-not $Python) {
        $candidate = Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe"
        if (Test-Path $candidate) { $Python = $candidate } else { $Python = "python" }
    }
    Write-Host "Creating build venv with $Python"
    Invoke-Checked $Python @("-m", "venv", $Venv)
}

Invoke-Checked $VenvPy @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked $VenvPy @("-m", "pip", "install", "-r", (Join-Path $Root "requirements.txt"))
Invoke-Checked $VenvPy @("-m", "pip", "install", "-r", (Join-Path $PSScriptRoot "requirements-build.txt"))

# 2. App bundle -> dist\V-COMMS
Push-Location $Root
try {
    Invoke-Checked $VenvPy @("-m", "PyInstaller", "--noconfirm", "--clean",
        "--distpath", "dist", "--workpath", "build",
        (Join-Path $PSScriptRoot "vcomms.spec"))
} finally {
    Pop-Location
}
Write-Host "App bundle: $(Join-Path $Root 'dist\V-COMMS\V-COMMS.exe')"

if ($SkipInstaller) { return }

# 3. Installer -> dist\installer
$iscc = (Get-Command iscc.exe -ErrorAction SilentlyContinue).Source
if (-not $iscc) {
    $iscc = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
        (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe")
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $iscc) {
    Write-Warning "Inno Setup 6 not found, skipping the installer. Install it with: winget install JRSoftware.InnoSetup"
    return
}
Invoke-Checked $iscc @("/DAppVersion=$Version", (Join-Path $PSScriptRoot "vcomms.iss"))
Write-Host "Installer: $(Join-Path $Root "dist\installer\V-COMMS-Setup-$Version.exe")"
