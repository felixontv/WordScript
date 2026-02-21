# WordScript — Vision & Roadmap

## What it is

WordScript is a lightweight, always-on desktop utility (Windows, macOS, Linux) that lets you interact with your computer using voice or text — fast, via a global hotkey. No context switching, no open browser tab.

**Core idea:** One shortcut. Two modes. Always ready.

---

## Current State

- Global hotkey triggers microphone recording
- Audio is transcribed via Groq Whisper API (ultra-fast)
- Transcribed text is auto-pasted into the focused application
- System tray icon, tap/hold modes, multilingual support

---

## Planned Features

### 1. Audio Visualizer
A minimal overlay (bottom of screen) showing a live waveform or level indicator while recording — so you always have visual confirmation that the mic is active.

**Options:**
- Waveform bar overlay (simple, no dependencies)
- Animated ring around tray icon (even lighter)

### 2. AI Voice / Text Assistant Mode
A second mode switchable via hotkey: instead of transcribing and pasting, the recorded/typed input is sent to an AI and a concise answer is returned.

**Use cases:**
- Quick question about something on screen (with optional screenshot context)
- Summarize, explain, rewrite selected text
- Short tasks: "What does this error mean?", "Translate this paragraph"

**Input:** Voice (via Whisper) or typed text in a small popup  
**Output:** Short AI response shown in an overlay or copied to clipboard

### 3. Screen Context (Visual Q&A)
Optionally attach a screenshot (full screen or selected region) as context when sending a question to the AI — enabling questions like *"What is this UI element doing?"* or *"Explain this chart."*

### 4. Switchable Backends

| Use case | Backend |
|---|---|
| Transcription | Groq Whisper (current) |
| Quick AI answers | Groq LLaMA / Claude Haiku via API |
| Deep tasks / coding | Claude Opus / Sonnet via API |
| Self-hosted / offline | Open WebUI + local model |

The goal is to stay fast and cheap for quick interactions, and only escalate to a heavier model when needed.

---

## Shortcut Design (Concept)

| Shortcut | Action |
|---|---|
| `Ctrl + Win` | Transcribe (current behavior) |
| `Ctrl + Alt + Win` | Ask AI (voice or text input) |
| `Ctrl + Shift + Win` | Ask AI with screenshot context |

---

## Integration into Other Projects

WordScript is designed to be self-contained and embeddable:

- **As a subprocess:** Any Python or Electron app can spawn it and read output via stdout/named pipe
- **As a module:** Core transcription and AI query logic can be imported directly
- **As a REST microservice:** A lightweight local HTTP server mode (`--serve`) could expose endpoints for other tools to call

**VS Code integration** (via Continue / Cline):  
The assistant mode can complement AI coding tools — use WordScript for quick voice queries, Continue/Cline for deep in-editor tasks. Clear boundary: WordScript = ambient assistant, Continue/Cline = code agent.

---

## Open Questions

- Overlay UI: `tkinter` (built-in) vs. `PyQt6` vs. a small web-based popup (Flask + browser)
- AI response: show inline overlay vs. paste into focused window vs. speak via TTS
- When does a "quick task" justify switching to Claude Desktop / a full agent?

---

## Platform Target

WordScript will be rebuilt as an **Electron app** to run natively on Windows, macOS, and Linux.

- Current Python prototype validates the core logic
- Electron provides a proper cross-platform shell, global hotkey support, system tray, and overlay UI
- Core transcription / AI logic stays in Python (called as a subprocess or sidecar) or gets ported to Node.js where feasible
- Distribution: packaged binaries per platform via `electron-builder`

---

## Non-Goals (for now)

- No persistent chat history
- No autonomous multi-step agent actions
- No heavy GUI / settings panel
