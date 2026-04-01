#!/usr/bin/env bash
# WordScript — Build Python sidecar binary for Tauri (Linux / macOS)
# Run from repo root after `pip install pyinstaller` and all Python deps.
# Output: src-tauri/binaries/wordscript-sidecar-<target-triple>
set -euo pipefail

CYAN='\033[0;36m'; GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
echo -e "${CYAN}Building WordScript Python sidecar...${NC}"

# ── 1. PyInstaller build ──────────────────────────────────────────────────────
pyinstaller WordScript.spec --distpath dist-python

# ── 2. Determine Rust target triple ───────────────────────────────────────────
if command -v rustc &>/dev/null; then
  TRIPLE=$(rustc -vV 2>/dev/null | awk '/^host:/{print $2}')
else
  OS="$(uname -s)"
  ARCH="$(uname -m)"
  case "$OS-$ARCH" in
    Linux-x86_64)   TRIPLE="x86_64-unknown-linux-gnu" ;;
    Linux-aarch64)  TRIPLE="aarch64-unknown-linux-gnu" ;;
    Darwin-x86_64)  TRIPLE="x86_64-apple-darwin" ;;
    Darwin-arm64)   TRIPLE="aarch64-apple-darwin" ;;
    *)              TRIPLE="unknown-unknown-unknown" ;;
  esac
fi

# ── 3. Copy to src-tauri/binaries/ ───────────────────────────────────────────
case "$(uname -s)" in
  Linux)  SRC="dist-python/WordScript-linux" ;;
  Darwin) SRC="dist-python/WordScript-macos" ;;
  *)      echo -e "${RED}Unknown OS — cannot determine sidecar binary name.${NC}"; exit 1 ;;
esac
DEST="src-tauri/binaries/wordscript-sidecar-$TRIPLE"
mkdir -p src-tauri/binaries
cp -f "$SRC" "$DEST"
chmod +x "$DEST"
echo -e "${GREEN}Sidecar ready: $DEST${NC}"

echo ""
echo -e "${CYAN}Now run:  npm run tauri build${NC}"
