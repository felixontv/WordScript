# WordScript — Architektur

## Überblick

Desktop Speech-to-Text App: Globaler Hotkey → Mikrofon-Aufnahme → Groq Whisper API → Text in Zwischenablage + Auto-Paste.

## Drei-Schichten-Architektur

```
┌─────────────────────────────────────────┐
│  React Frontend (Vite + TypeScript)     │
│  - Overlay Window (Waveform-Visualizer) │
│  - Settings Window (6 Tabs)             │
│  - useSidecar Hook (State Machine)      │
└──────────────┬──────────────────────────┘
               │ listen("py-event") / invoke("send_to_python")
┌──────────────┴──────────────────────────┐
│  Rust / Tauri 2                         │
│  - Spawnt Python-Sidecar                │
│  - Parst stdout JSON → emit("py-event") │
│  - Overlay Visibility Management        │
│  - System Tray, Singleton Instance      │
└──────────────┬──────────────────────────┘
               │ stdin/stdout JSON-IPC
┌──────────────┴──────────────────────────┐
│  Python Sidecar                         │
│  - HotkeyManager (pynput)              │
│  - AudioRecorder (sounddevice, 16kHz)   │
│  - TranscriptionService (Groq Whisper)  │
│  - TextPaster (pyperclip + pynput)      │
│  - LLM Post-Korrektur (optional)        │
└─────────────────────────────────────────┘
```

## Datenfluss

1. **Hotkey** (pynput) → `recorder.start()`
2. **Recording** → 16kHz/mono/int16, Silence-Detection, Max-Timer
3. **Transkription** → Groq Whisper API (IPv4 erzwungen, 55s Timeout, 0 Retries)
4. **Paste** → pyperclip + Ctrl+V Simulation (Wayland: nur Clipboard)
5. **Optional:** LLM-Korrektur aktualisiert Clipboard im Hintergrund
6. **IPC** → `stdout JSON` → Rust parst → `emit("py-event")` → React UI

## Schlüsseldateien

| Datei | Verantwortung |
|---|---|
| `src-tauri/src/lib.rs` | Tauri-Setup, Sidecar-Spawn, Event-Forwarding |
| `wordscript/sidecar.py` | Headless Backend, Command-Dispatch, IPC |
| `wordscript/transcription.py` | Groq Whisper + LLM-Korrektur |
| `wordscript/recorder.py` | Audio-Aufnahme, Silence-Detection |
| `wordscript/hotkey.py` | Globaler Hotkey (tap/hold), Debounce |
| `wordscript/paster.py` | Clipboard + Auto-Paste |
| `wordscript/config.py` | Config-Laden/Speichern, Platform-Defaults |
| `wordscript/ipc.py` | JSON-IPC über stdin/stdout |
| `src/hooks/useSidecar.ts` | React State-Machine für Backend-Events |

## Plattform-Besonderheiten

### Linux
- **Wayland-Workaround:** App erzwingt `GDK_BACKEND=x11` und entfernt `WAYLAND_DISPLAY` in `main.rs` — WebKitGTK + transparente Fenster crashen auf nativem Wayland (Gdk Error 71). Läuft statt dessen via XWayland.
- **Overlay-Steuerung:** `set_position()` (on-/off-screen) statt `show()`/`hide()`/`set_always_on_top()` — letztere crashen auf Wayland.
- **Settings-Fenster:** `minimize()`/`unminimize()` statt `hide()`/`show()`, startet mit `visible: true`.
- Wayland: Auto-Paste deaktiviert (nur Clipboard via `wl-copy`)
- Hotkey-Debounce 300ms (Compositor synthetic events)
- Clipboard-Backends: `xclip`, `xsel`, `wl-copy`

### Config-Pfad
- **Einheitlich** (Dev + Prod): `~/.config/WordScript/config.json` (Linux), `%APPDATA%\WordScript` (Win), `~/Library/Application Support/WordScript` (macOS)
- Migration: Bei erstem Frozen-Run wird alte Config neben der exe kopiert

### Dev-Mode
- Rust spawnt `.venv/bin/python -m wordscript sidecar` (Pfad via `CARGO_MANIFEST_DIR`)
- Fallback auf System-`python` wenn kein venv vorhanden

### Produktion
- PyInstaller-Binary als Tauri Sidecar gebundelt (`wordscript-sidecar-<triple>`)

## Bekannte Design-Entscheidungen

- **IPv4 erzwungen** für alle Groq API-Calls (verhindert IPv6-Timeout auf allen Plattformen)
- **Keine Retries** bei API-Fehlern — direktes Error-Feedback an User
- **60s Wall-Clock-Timeout** als Guard gegen hängende Verbindungen (zusätzlich zum SDK-Timeout)
- **Hallucination-Filtering** für Whisper (exakte Matches + Regex) und LLM (Länge, Blacklist, Overlap)
