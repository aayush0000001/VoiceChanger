# Builds V-COMMS into dist\V-COMMS (PyInstaller) and dist\installer\V-COMMS-Setup-<version>.exe (Inno Setup).
#
# Usage (from anywhere):
#   powershell -ExecutionPolicy Bypass -File packaging\build.ps1 [-Python <path to x64 python.exe>] [-SkipInstaller]
#
# The version comes from appinfo.py (APP_VERSION), so the exe, the installer and Apps always agree.
# Uses its own venv (.venv-build) so it never touches your dev .venv.
# VB-CABLE: downloads VB-Audio's official driver pack once into build\vbcable, checks its SHA-256 and that every
# setup program is signed by VB-Audio's author, and bundles it (VB-Audio's licensing page allows this for
# donationware use, with credit). Update $VBCableSha256 when VB-Audio publishes a new pack.

param(
    [string]$Python = "",
    [switch]$SkipInstaller,
    [string]$Version = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv-build"
$VenvPy = Join-Path $Venv "Scripts\python.exe"
$VBCableUrl = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack45.zip"
$VBCableSha256 = "B950E39F01AF1D04EA623C8F6D8EB9B6EA5C477C637295FABF20631C85116BFB"
$VBCableSigner = "BUREL VINCENT"

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe failed with exit code $LASTEXITCODE" }
}

$appVersion = ([regex]::Match((Get-Content (Join-Path $Root "appinfo.py") -Raw), 'APP_VERSION\s*=\s*"([^"]+)"')).Groups[1].Value
if (-not $Version) { $Version = $appVersion }
if ($Version -ne $appVersion) { throw "-Version $Version does not match APP_VERSION $appVersion in appinfo.py; change appinfo.py instead." }
Write-Host "Building V-COMMS $Version"

# 1. Build venv. The app ships as x64 (the dev setup and its DirectML wheels are x64), so prefer the x64 Python even
#    on Windows on Arm.
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

# 3. VB-CABLE driver pack -> build\vbcable (verified)
$cableDir = Join-Path $Root "build\vbcable"
$cableZip = Join-Path $Root "build\VBCABLE_Driver_Pack.zip"
if (-not (Test-Path (Join-Path $cableDir "VBCABLE_Setup_x64.exe"))) {
    New-Item -ItemType Directory -Force (Split-Path $cableZip) | Out-Null
    Write-Host "Downloading VB-CABLE from $VBCableUrl"
    Invoke-WebRequest -Uri $VBCableUrl -OutFile $cableZip -UseBasicParsing
    $hash = (Get-FileHash $cableZip -Algorithm SHA256).Hash
    if ($hash -ne $VBCableSha256) { Remove-Item $cableZip; throw "VB-CABLE pack hash $hash is not the pinned $VBCableSha256" }
    Expand-Archive $cableZip -DestinationPath $cableDir -Force
}
foreach ($exe in Get-ChildItem $cableDir -Filter "*.exe") {
    $sig = Get-AuthenticodeSignature $exe.FullName
    if ($sig.Status -ne "Valid" -or $sig.SignerCertificate.Subject -notmatch $VBCableSigner) {
        throw "$($exe.Name) is not validly signed by VB-Audio ($($sig.Status), $($sig.SignerCertificate.Subject))"
    }
}

# 4. Installer -> dist\installer
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
Invoke-Checked $iscc @("/DAppVersion=$Version", "/DVBCableDir=$cableDir", (Join-Path $PSScriptRoot "vcomms.iss"))
Write-Host "Installer: $(Join-Path $Root "dist\installer\V-COMMS-Setup-$Version.exe")"
