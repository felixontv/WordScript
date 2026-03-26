# WordScript — Build Python sidecar binary for Tauri (Windows)
# Run from repo root after `pip install pyinstaller` and all Python deps.
# Output: src-tauri/binaries/wordscript-sidecar-x86_64-pc-windows-msvc.exe
#
# Tauri expects the sidecar binary at:
#   src-tauri/binaries/<name>-<target-triple><.exe>
# where <name> matches externalBin in tauri.conf.json ("wordscript-sidecar").

$ErrorActionPreference = "Stop"
Write-Host "Building WordScript Python sidecar..." -ForegroundColor Cyan

# ── 1. PyInstaller build ──────────────────────────────────────────────────────
pyinstaller WordScript.spec --distpath dist-python
if ($LASTEXITCODE -ne 0) {
    Write-Host "PyInstaller build failed." -ForegroundColor Red
    exit 1
}

# ── 2. Determine Rust target triple ───────────────────────────────────────────
$triple = "x86_64-pc-windows-msvc"
if (Get-Command rustc -ErrorAction SilentlyContinue) {
    $tripleRaw = rustc -vV 2>$null | Select-String "host:" | ForEach-Object { $_ -replace "host:\s*", "" }
    if ($tripleRaw) { $triple = $tripleRaw.Trim() }
}

# ── 3. Copy to src-tauri/binaries/ ───────────────────────────────────────────
$src  = "dist-python\WordScript-windows.exe"
$dest = "src-tauri\binaries\wordscript-sidecar-$triple.exe"
New-Item -ItemType Directory -Force -Path "src-tauri\binaries" | Out-Null
Copy-Item -Force $src $dest
Write-Host "Sidecar ready: $dest" -ForegroundColor Green

Write-Host ""
Write-Host "Now run:  npm run tauri build" -ForegroundColor Cyan
