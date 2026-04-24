# WordScript — Development Guide

## Voraussetzungen

- Node.js 18+
- Rust + Cargo (Tauri 2)
- Python 3.10+ mit venv
- Linux: `libwebkit2gtk-4.1-0`, `libayatana-appindicator3-1`, `libxdo3`, `wl-clipboard` (Wayland)

## Setup

```bash
# 1. Node-Dependencies
npm install

# 2. Python venv + Dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Dev-Server starten
npm run tauri dev
```

## Wie der Dev-Mode funktioniert

`npm run tauri dev` startet:
1. **Vite** Dev-Server auf `localhost:1420`
2. **Cargo** kompiliert `src-tauri/` und startet die Desktop-App
3. **Tauri** spawnt den Python-Sidecar:
   - Dev: `.venv/bin/python -m wordscript sidecar` (Pfad via `CARGO_MANIFEST_DIR`)
   - Prod: Gebundelte PyInstaller-Binary

### Wichtig: Rust-Rebuild

Tauri's Hot-Reload erkennt Dateiänderungen in `src-tauri/src/`. Falls Änderungen nicht übernommen werden:
```bash
cd src-tauri && touch src/lib.rs && cargo build --no-default-features
```

## Projektstruktur

```
WordScript/
├── src/                    # React Frontend (TypeScript)
│   ├── hooks/              # useSidecar State-Machine
│   ├── windows/            # Overlay + Settings Windows
│   └── types/              # IPC-Typen
├── src-tauri/              # Rust/Tauri Backend
│   └── src/lib.rs          # Sidecar-Spawn, Event-Routing
├── wordscript/             # Python Sidecar
│   ├── sidecar.py          # Headless Backend
│   ├── transcription.py    # Groq Whisper + LLM
│   ├── recorder.py         # Audio-Aufnahme
│   ├── hotkey.py           # Globaler Hotkey
│   ├── paster.py           # Clipboard + Paste
│   └── config.py           # Konfiguration
├── requirements.txt        # Python-Dependencies
└── docs/                   # Dokumentation
```

## Production Build

```bash
# Sidecar-Binary bauen
./build-sidecar.sh          # Linux/macOS
./build-sidecar.ps1         # Windows

# App bauen
npm run tauri build
```

## Debugging

### Python-Sidecar separat testen
```bash
source .venv/bin/activate
python -m wordscript sidecar
# JSON-Commands über stdin senden, z.B.:
# {"cmd": "reload_config"}
```

### Häufige Probleme

| Problem | Ursache | Lösung |
|---|---|---|
| "No module named wordscript" | System-Python statt venv | `source .venv/bin/activate && pip install -r requirements.txt` |
| 60s Timeout bei Transkription | IPv6-Connect-Timeout | Bereits behoben: IPv4 erzwungen in `transcription.py` |
| "Connecting to backend..." | Sidecar startet nicht | Logs prüfen: `npm run tauri dev` Output |
| Kein Audio | PulseAudio/PipeWire fehlt | `pactl info` prüfen |
| Clipboard leer (Wayland) | `wl-copy` fehlt | `sudo pacman -S wl-clipboard` |
