# WordScript

![version](https://img.shields.io/badge/version-v0.0.1--alpha-orange)
![status](https://img.shields.io/badge/status-alpha-yellow)
![license](https://img.shields.io/badge/license-MIT-blue)
![build](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Ffelixontv%2FWordScript%2Fmaster%2Fbuild_info.json&query=%24.build&label=build&color=lightgrey)

**One shortcut. Two modes. Always ready.**

WordScript is a lightweight, always-on desktop utility that lets you interact with your computer using voice or text — fast, without switching apps or opening a browser tab.

- **Transcription mode** — press a hotkey, speak, text gets pasted instantly into whatever is focused
- **Assistant mode** _(planned)_ — same shortcut, different mode: ask a question and get a concise AI answer, optionally with screenshot context

**Target platform:** Cross-platform Electron app (Windows, macOS, Linux).  
**Current state:** Working Python prototype (Windows) — validates the core logic before the Electron rebuild.

---

## Roadmap

| # | Feature | Status |
|---|---|---|
| ✅ | Global hotkey transcription via Groq Whisper | Done |
| ✅ | Auto-paste into focused app | Done |
| ✅ | Tap / hold mode, multilingual, tray icon | Done |
| 🔲 | Audio visualizer overlay while recording | Planned |
| 🔲 | AI assistant mode (voice or text input) | Planned |
| 🔲 | Screenshot context for visual Q&A | Planned |
| 🔲 | Switchable AI backends (Groq, Claude, local) | Planned |
| 🔲 | Electron rebuild — cross-platform | Planned |

See [VISION.md](VISION.md) for the full design rationale and open questions.

---

## Quick Start (Python Prototype)

### 1. One-time setup after cloning

```powershell
.\setup.ps1
```

> Installs dependencies, creates `config.json`, and activates git hooks so `BUILD_ID` updates automatically on every commit.

### 2. Configure

Copy the example config and fill in your keys:

```bash
cp config.example.json config.json
```

Edit **config.json** and set your Groq API key (get one at https://console.groq.com).  
Adjust hotkey, activation mode, and audio settings as needed.

| Setting                 | Default                  | Description                                                        |
| ----------------------- | ------------------------ | ------------------------------------------------------------------ |
| `groq_api_key`          | _(your key)_             | Groq API key from https://console.groq.com                         |
| `model`                 | `whisper-large-v3-turbo` | Whisper model to use                                               |
| `language`              | `""` (empty)             | Language code (`en`, `de`, `fr`, etc.) or empty for auto-detection |
| `hotkey`                | `ctrl_l+win`             | Global hotkey combo                                                |
| `activation_mode`       | `tap`                    | `tap` = toggle on/off, `hold` = hold to record                     |
| `sample_rate`           | `16000`                  | Audio sample rate in Hz                                            |
| `max_recording_seconds` | `120`                    | Max recording duration                                             |
| `auto_paste`            | `true`                   | Auto Ctrl+V after transcription                                    |
| `show_tray_icon`        | `true`                   | Show system tray icon                                              |
| `play_sounds`           | `true`                   | Beep feedback on start/stop                                        |

### 3. Run

```bash
python speech_to_text.py
```

A tray icon appears (green = idle, red = recording). Press the hotkey to start/stop. Transcription is pasted automatically within ~1 second.

**Multilingual:** Leave `language` empty for auto-detection across 90+ languages. Set `"language": "en"` or `"de"` to force a specific one.

---

## Usage

1. **Start the script** — it sits silently in the background.
2. **Press Ctrl+Left Win** (or your configured hotkey) to start recording. Short beep confirms.
3. **Press the hotkey again** (tap mode) or **release it** (hold mode) to stop. Two beeps confirm.
4. Audio is sent to Groq Whisper → transcription auto-pasted into the active window.

---

## Planned Shortcut Design

| Shortcut | Action |
| --- | --- |
| `Ctrl + Win` | Transcribe (current) |
| `Ctrl + Alt + Win` | Ask AI — voice or text |
| `Ctrl + Shift + Win` | Ask AI + screenshot context |

---

## Hotkey Options

| Key name              | Actual key         |
| --------------------- | ------------------ |
| `ctrl_l` / `ctrl_r`   | Left / Right Ctrl  |
| `alt_l` / `alt_r`     | Left / Right Alt   |
| `shift_l` / `shift_r` | Left / Right Shift |
| `win` / `cmd`         | Windows key        |
| `f1`–`f12`            | Function keys      |
| Any single char       | e.g. `t`, `r`      |

---

## Run at Startup (Optional)

1. Press **Win+R** → `shell:startup` → Enter
2. Create a shortcut to `pythonw speech_to_text.py`, set "Start in" to this folder

---

## Build as .exe (Optional)

```bash
pip install pyinstaller
pyinstaller WordScript.spec
```

Output: `dist/WordScript.exe`. Place `config.json` in the same directory.

---

## Troubleshooting

| Problem | Solution |
| --- | --- |
| "No audio input device found" | Check system sound settings → Input |
| No beep sounds | Check `play_sounds` config, ensure sound card is active |
| Hotkey doesn't work | Try running as Administrator. Some apps intercept Win key combos. |
| Transcription errors | Check Groq API key and internet connection |
| Tray icon missing | Ensure `pystray` and `Pillow` are installed |

---

## Architecture (Python Prototype)

```
SpeechToTextApp          ← main orchestrator
├── AudioRecorder        ← sounddevice microphone capture → WAV bytes
├── TranscriptionService ← Groq Whisper API client
├── TextPaster           ← clipboard + Ctrl+V simulation
├── HotkeyManager        ← pynput global keyboard listener
├── TrayIcon             ← pystray system tray (optional)
└── SoundFeedback        ← winsound beeps (optional)
```


