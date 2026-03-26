#!/usr/bin/env bash
# WordScript — Tauri dev environment setup (Linux / macOS)
# Run once after cloning: bash setup-tauri.sh
set -euo pipefail

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; GRAY='\033[0;37m'; NC='\033[0m'

echo -e "${CYAN}WordScript Tauri Setup (Linux / macOS)${NC}"

OS="$(uname -s)"

# ── 1. Rust ──────────────────────────────────────────────────────────────────
if command -v rustc &>/dev/null; then
  echo -e "${GREEN}[1/5] Rust already installed: $(rustc --version)${NC}"
else
  echo -e "${YELLOW}[1/5] Installing Rust via rustup...${NC}"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --quiet
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
  echo -e "${GREEN}[1/5] Rust installed: $(rustc --version)${NC}"
fi

# ── 2. System dependencies ────────────────────────────────────────────────────
echo -e "${YELLOW}[2/5] Installing system dependencies...${NC}"
if [[ "$OS" == "Darwin" ]]; then
  if ! command -v brew &>/dev/null; then
    echo -e "${RED}Homebrew not found. Install it from https://brew.sh then re-run.${NC}"
    exit 1
  fi
  brew install portaudio
  # Tauri on macOS needs Xcode CLT (usually already present)
  if ! xcode-select -p &>/dev/null; then
    echo -e "${YELLOW}      Installing Xcode Command Line Tools...${NC}"
    xcode-select --install
  fi
  echo -e "${GREEN}[2/5] macOS dependencies ready.${NC}"
else
  # Linux — detect package manager
  if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends \
      libwebkit2gtk-4.1-dev libssl-dev libgtk-3-dev libayatana-appindicator3-dev \
      librsvg2-dev portaudio19-dev python3-dev build-essential curl wget file
  elif command -v dnf &>/dev/null; then
    sudo dnf install -y webkit2gtk4.1-devel openssl-devel gtk3-devel \
      libappindicator-gtk3-devel librsvg2-devel portaudio-devel python3-devel \
      gcc curl wget file
  elif command -v pacman &>/dev/null; then
    sudo pacman -Syu --noconfirm webkit2gtk-4.1 openssl gtk3 \
      libappindicator-gtk3 librsvg portaudio python base-devel curl wget
  else
    echo -e "${YELLOW}      Unknown package manager — install Tauri Linux deps manually:${NC}"
    echo -e "${GRAY}      https://tauri.app/start/prerequisites/#linux${NC}"
  fi
  echo -e "${GREEN}[2/5] Linux dependencies ready.${NC}"
fi

# ── 3. Node.js check ──────────────────────────────────────────────────────────
if command -v node &>/dev/null; then
  NODE_VER=$(node -e "process.stdout.write(process.versions.node)")
  NODE_MAJOR="${NODE_VER%%.*}"
  if [[ "$NODE_MAJOR" -lt 18 ]]; then
    echo -e "${RED}[3/5] Node.js $NODE_VER found but 18+ is required. Please upgrade.${NC}"
    exit 1
  fi
  echo -e "${GREEN}[3/5] Node.js $NODE_VER already installed.${NC}"
else
  echo -e "${RED}[3/5] Node.js not found. Install v18+ from https://nodejs.org then re-run.${NC}"
  exit 1
fi

# ── 4. npm install ────────────────────────────────────────────────────────────
echo -e "${YELLOW}[4/5] Installing npm dependencies...${NC}"
npm install
echo -e "${GREEN}[4/5] npm dependencies installed.${NC}"

# ── 5. Generate Tauri icons from assets/logo.png ──────────────────────────────
echo -e "${YELLOW}[5/5] Generating app icons...${NC}"
if [[ -f "assets/logo.png" ]]; then
  npx tauri icon assets/logo.png
  echo -e "${GREEN}[5/5] Icons generated in src-tauri/icons/.${NC}"
else
  echo -e "${YELLOW}[5/5] WARNING: assets/logo.png not found — icons not generated.${NC}"
  echo -e "${GRAY}      Create a 1024×1024 PNG at assets/logo.png then run: npx tauri icon assets/logo.png${NC}"
  mkdir -p src-tauri/icons
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo -e "${CYAN}  Start dev server:    npm run tauri dev${NC}"
echo -e "${CYAN}  Build release:       npm run tauri build${NC}"
echo ""
echo -e "${GRAY}Note: In dev mode, Tauri spawns Python directly (python -m wordscript sidecar).${NC}"
echo -e "${GRAY}      Make sure your .venv is activated or Python deps are installed globally.${NC}"
