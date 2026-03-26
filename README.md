# WordScript

![version](https://img.shields.io/badge/version-v0.2.0--alpha-orange)
![status](https://img.shields.io/badge/status-alpha-yellow)
![license](https://img.shields.io/badge/license-MIT-blue)

<p align="center">
  <img src="assets/ws-logo.png" alt="WordScript Logo" width="160">
</p>

**One shortcut. Speak. Done.**

WordScript is a lightweight desktop app that turns speech into text — instantly pasted where your cursor is. Press a hotkey, talk, release. No browser tabs, no app switching, no copy-paste.

Built with Tauri v2 + React (frontend) and a Python sidecar (Groq Whisper + LLM post-processing).

---

## What works (v0.2.0-alpha)

- Global hotkey to start/stop recording (tap or hold mode)
- Always-on-top pill overlay with live audio waveform visualizer
- Transcription via Groq Whisper API (~1s turnaround)
- AI post-correction: filter filler words (ähm, äh, um…) and/or professionalize text via LLM
- Auto-paste into any focused application
- Mute toggle (click mic icon in the overlay)
- Full settings UI — API keys, models, language, hotkey, audio device, all configurable in-app
- System tray icon with Settings and Quit
- Audio feedback on start / stop / abort / error
- Per-user config: `%APPDATA%\WordScript\` (Windows), `~/.config/WordScript/` (Linux), `~/Library/Application Support/WordScript/` (macOS)

## What doesn't work yet

- AI assistant mode (planned)
- Auto-updater (notification only — manual download for now)
- macOS / Linux installers (Windows NSIS installer available; other platforms: run from source)

---

## Quick Start

### Download & Install (Windows)

Grab the latest installer from the [Releases page](https://github.com/felixontv/WordScript/releases):

1. Download `WordScript_0.2.0-alpha_x64-setup.exe`
2. Run the installer
3. WordScript starts in the system tray
4. Click the tray icon → **Settings**, enter your [Groq API key](https://console.groq.com/keys), hit Save

### Run from Source (all platforms)

**Requirements:** Python 3.10+, Node.js 18+, Rust

```bash
git clone https://github.com/felixontv/WordScript.git
cd WordScript

# Install Python dependencies
pip install -r requirements.txt

# Install Node dependencies
npm install

# Development mode (Tauri + Vite + Python sidecar)
npm run tauri dev
```

### Build installer from source

```bash
# 1. Build Python sidecar binary
.\build-sidecar.ps1        # Windows
# or
./build-sidecar.sh         # Linux / macOS

# 2. Build Tauri app
npm run tauri build
```

Output: `src-tauri/target/release/bundle/nsis/WordScript_0.2.0-alpha_x64-setup.exe`

---

## Usage

1. **Launch WordScript** — a pill-shaped overlay appears near the bottom of your screen
2. **Press the hotkey** (`Ctrl + Left Win` on Windows, `Ctrl + Cmd` on macOS) to start recording
3. **Press again** (tap mode) or **release** (hold mode) to stop
4. The transcription is auto-pasted into whatever window is focused — typically within 1 second
5. **Click the mic icon** in the overlay to mute/unmute without stopping recording
6. **Click the chevron** (`▾`) to open Settings

To abort a recording: `Ctrl + Alt` (Windows/Linux) or `Ctrl + Cmd` while recording

---

## Settings

All settings live in the in-app Settings panel (tray icon → Settings, or click `▾` on the overlay).

| Setting | Default | Description |
|---|---|---|
| Groq API Key | _(empty)_ | Required — get one at https://console.groq.com/keys |
| Whisper Model | `whisper-large-v3-turbo` | Speech recognition model |
| Language | _(auto)_ | Language code (`en`, `de`, `fr`, …) or empty for auto-detect |
| Enable AI post-correction | On | Run LLM on every transcription |
| Filter filler words | On | Remove ähm, äh, um, uh, hmm… |
| Professionalize text | Off | Restructure rambling sentences into clean prose |
| Correction Model | `llama-3.1-8b-instant` | LLM used for post-correction |
| Hotkey | `ctrl_l+win` | Global keyboard shortcut |
| Activation Mode | `tap` | `tap` = toggle on/off, `hold` = push-to-talk |
| Audio Device | _(system default)_ | Select a specific microphone |
| Auto-paste | On | Paste transcription automatically |
| Play Sounds | On | Audio feedback on start / stop / error |

---

## Hotkey Reference

Combine keys with `+`. Example: `ctrl_l+win`, `ctrl_l+alt_l+space`

| Key | Description |
|---|---|
| `ctrl_l` / `ctrl_r` | Left / Right Ctrl |
| `alt_l` / `alt_r` | Left / Right Alt |
| `shift_l` / `shift_r` | Left / Right Shift |
| `win` / `cmd` | Windows / Command key |
| `f1` – `f12` | Function keys |

---

## Architecture

```
WordScript
├── src/                     React + TypeScript frontend (Tauri webview)
│   ├── windows/
│   │   ├── OverlayWindow    Pill overlay — waveform, mute, drag
│   │   └── SettingsWindow   Full settings UI
│   └── hooks/useSidecar     IPC bridge to Python backend
├── src-tauri/               Rust / Tauri shell
│   └── src/lib.rs           Sidecar spawn, overlay visibility, tray
├── wordscript/              Python backend (sidecar)
│   ├── sidecar.py           IPC loop — receives commands, emits events
│   ├── transcription.py     Groq Whisper + LLM post-correction
│   ├── recorder.py          Microphone capture → WAV
│   ├── hotkeys.py           Global hotkey listener (pynput)
│   ├── paster.py            Clipboard + paste simulation
│   └── config.py            Config dataclass, per-OS path resolution
└── build-sidecar.ps1 / .sh  Build Python binary for bundling
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| No audio device found | Check system sound settings; select the correct mic in Settings |
| Hotkey doesn't respond | Try running as Administrator — some apps block Win key combos |
| Transcription errors | Verify your Groq API key and internet connection at console.groq.com |
| Overlay covered by other windows | Fixed in v0.2.0-alpha — overlay re-applies always-on-top on every activation |
| Post-correction not applied | Enable "AI post-correction" in Settings → API & Models; check that your Groq key is valid |
| App won't start | Check `%APPDATA%\WordScript\wordscript.log` for errors |

---

## License

[MIT](LICENSE)
