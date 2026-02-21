# WordScript — First-time setup
# Run this once after cloning: .\setup.ps1

Write-Host "Setting up WordScript..." -ForegroundColor Cyan

# Activate git hooks
git config core.hooksPath hooks
Write-Host "[OK] Git hooks activated (BUILD_ID will auto-update on every commit)" -ForegroundColor Green

# Install Python dependencies
if (Test-Path ".venv") {
  Write-Host "[OK] .venv already exists, skipping creation" -ForegroundColor Yellow
}
else {
  python -m venv .venv
  Write-Host "[OK] Created .venv" -ForegroundColor Green
}

& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt --quiet
Write-Host "[OK] Python dependencies installed" -ForegroundColor Green

# Config
if (Test-Path "config.json") {
  Write-Host "[OK] config.json already exists" -ForegroundColor Yellow
}
else {
  Copy-Item config.example.json config.json
  Write-Host "[OK] config.json created from example — add your Groq API key!" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Setup complete. Edit config.json with your API key, then run: python speech_to_text.py" -ForegroundColor Cyan
