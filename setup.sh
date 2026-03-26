#!/usr/bin/env bash
# WordScript — First-time setup (Linux / macOS)
# Run this once after cloning: bash setup.sh

set -e

echo "Setting up WordScript..."

# Activate git hooks
git config core.hooksPath hooks
chmod +x hooks/pre-commit
echo "[OK] Git hooks activated (BUILD_ID will auto-update on every commit)"

# System dependencies
if [[ "$(uname -s)" == "Linux" ]]; then
    if command -v apt-get &>/dev/null; then
        echo "Installing system dependencies via apt-get..."
        sudo apt-get update -qq
        sudo apt-get install -y python3-tk libportaudio2 portaudio19-dev python3-venv
        echo "[OK] System dependencies installed"
    else
        echo "[WARN] Non-Debian Linux: ensure python3-tk and portaudio19-dev are installed manually"
    fi
elif [[ "$(uname -s)" == "Darwin" ]]; then
    if command -v brew &>/dev/null; then
        echo "Installing system dependencies via Homebrew..."
        brew install portaudio
        # python-tk is bundled with python.org builds; only needed for Homebrew Python
        if python3 -c "import tkinter" 2>/dev/null; then
            echo "[OK] tkinter already available"
        else
            brew install python-tk
        fi
        echo "[OK] macOS system dependencies installed"
    else
        echo "[WARN] Homebrew not found. Install it from https://brew.sh then re-run this script."
        echo "       Homebrew is required to install PortAudio (sounddevice dependency)."
        exit 1
    fi
fi

# Virtual environment
if [[ -d ".venv" ]]; then
    echo "[OK] .venv already exists, skipping creation"
else
    python3 -m venv .venv
    echo "[OK] Created .venv"
fi

source .venv/bin/activate
pip install -r requirements.txt --quiet
echo "[OK] Python dependencies installed"

# Config
if [[ -f "config.json" ]]; then
    echo "[OK] config.json already exists"
else
    cp config.example.json config.json
    echo "[OK] config.json created from example — add your Groq API key!"
fi

echo ""
echo "Setup complete. Edit config.json with your API key, then run: python speech_to_text.py"
