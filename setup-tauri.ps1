# WordScript — Tauri dev environment setup (Windows)
# Run once after cloning: .\setup-tauri.ps1
# Requires: Node.js 18+ already installed (https://nodejs.org)

$ErrorActionPreference = "Stop"
Write-Host "WordScript Tauri Setup (Windows)" -ForegroundColor Cyan

# ── 1. Rust ───────────────────────────────────────────────────────────────────
if (-not (Get-Command rustc -ErrorAction SilentlyContinue)) {
    Write-Host "[1/5] Installing Rust via winget..." -ForegroundColor Yellow
    winget install --id Rustlang.Rustup --silent --accept-source-agreements --accept-package-agreements
    # Reload PATH so cargo is found in this session
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + `
                [System.Environment]::GetEnvironmentVariable("PATH", "User")
    if (-not (Get-Command rustc -ErrorAction SilentlyContinue)) {
        Write-Host "Rust installed — please RESTART this terminal then re-run this script." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[1/5] Rust already installed: $(rustc --version)" -ForegroundColor Green
}

# ── 2. MSVC Build Tools (required by Rust on Windows) ─────────────────────────
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vsWhere)) {
    Write-Host "[2/5] Installing Visual Studio Build Tools (C++ workload)..." -ForegroundColor Yellow
    Write-Host "      This may take 5-10 minutes..." -ForegroundColor Gray
    winget install --id Microsoft.VisualStudio.2022.BuildTools --silent `
        --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --quiet --norestart" `
        --accept-source-agreements --accept-package-agreements
    Write-Host "[2/5] Build Tools installed." -ForegroundColor Green
} else {
    Write-Host "[2/5] Visual Studio Build Tools already present." -ForegroundColor Green
}

# ── 3. WebView2 (usually pre-installed on Win 10/11 2004+) ────────────────────
Write-Host "[3/5] WebView2 check..." -ForegroundColor Yellow
$wv2 = Get-ItemProperty -Path "HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" -ErrorAction SilentlyContinue
if ($wv2) {
    Write-Host "[3/5] WebView2 already installed ($($wv2.pv))." -ForegroundColor Green
} else {
    Write-Host "[3/5] Installing WebView2 Runtime..." -ForegroundColor Yellow
    $wv2Url = "https://go.microsoft.com/fwlink/p/?LinkId=2124703"
    $wv2Installer = "$env:TEMP\MicrosoftEdgeWebview2Setup.exe"
    Invoke-WebRequest -Uri $wv2Url -OutFile $wv2Installer -UseBasicParsing
    Start-Process $wv2Installer -ArgumentList "/silent /install" -Wait
    Write-Host "[3/5] WebView2 installed." -ForegroundColor Green
}

# ── 4. npm install ────────────────────────────────────────────────────────────
Write-Host "[4/5] Installing npm dependencies..." -ForegroundColor Yellow
npm install
Write-Host "[4/5] npm dependencies installed." -ForegroundColor Green

# ── 5. Generate Tauri icons from assets/logo.png ─────────────────────────────
Write-Host "[5/5] Generating app icons..." -ForegroundColor Yellow
if (Test-Path "assets/logo.png") {
    npx tauri icon assets/logo.png
    Write-Host "[5/5] Icons generated in src-tauri/icons/." -ForegroundColor Green
} else {
    Write-Host "[5/5] WARNING: assets/logo.png not found — icons not generated." -ForegroundColor Yellow
    Write-Host "      Create a 1024x1024 PNG at assets/logo.png then run: npx tauri icon assets/logo.png" -ForegroundColor Gray
    # Create placeholder tray icon so the build doesn't fail
    New-Item -ItemType Directory -Force -Path "src-tauri/icons" | Out-Null
}

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Start dev server:    npm run tauri dev" -ForegroundColor Cyan
Write-Host "  Build release:       npm run tauri build" -ForegroundColor Cyan
Write-Host ""
Write-Host "Note: In dev mode, Tauri spawns Python directly (python -m wordscript sidecar)." -ForegroundColor Gray
Write-Host "      Make sure your .venv is activated or Python deps are installed globally." -ForegroundColor Gray
