# WordScript — Speech-to-Text with Groq Whisper

![version](https://img.shields.io/badge/version-v0.0.1--alpha-orange)
![status](https://img.shields.io/badge/status-alpha-yellow)
![license](https://img.shields.io/badge/license-MIT-blue)

System-wide speech-to-text for Windows 11 using Groq's ultra-fast Whisper API.  
Press a global hotkey to record from your microphone, then the transcribed text is automatically pasted into whatever app is focused.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

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

The script runs in the background. A green tray icon appears; it turns red while recording.

**✨ Multilingual Support:** Language is set to empty (`""`) by default, enabling automatic language detection. Whisper will automatically detect and transcribe in any of the 90+ supported languages (English, German, Spanish, French, Chinese, Japanese, etc.). To force a specific language, set `"language": "en"` or `"de"` etc. in [config.json](config.json).

---

## Usage

1. **Start the script** — it sits silently in the background.
2. **Press Ctrl+Left Win** (or your configured hotkey) to start recording. You'll hear a short beep.
3. **Press the hotkey again** (tap mode) or **release it** (hold mode) to stop. Two short beeps confirm stop.
4. The audio is sent to Groq Whisper. The transcription is auto-pasted (Ctrl+V) into the active window within ~1 second.

---

## Hotkey Options

Combine any of these keys with `+`:

| Key name              | Actual key         |
| --------------------- | ------------------ |
| `ctrl_l` / `ctrl_r`   | Left / Right Ctrl  |
| `alt_l` / `alt_r`     | Left / Right Alt   |
| `shift_l` / `shift_r` | Left / Right Shift |
| `win` / `cmd`         | Windows key        |
| `space`               | Spacebar           |
| `f1`–`f12`            | Function keys      |
| Any single char       | e.g. `t`, `r`      |

Example: `"hotkey": "ctrl_l+shift_l+r"` → Ctrl+Shift+R

---

## Run at Startup (Optional)

1. Press **Win+R**, type `shell:startup`, press Enter.
2. Create a shortcut to: `pythonw speech_to_text.py` (use `pythonw` to hide the console window).
3. Set the shortcut's "Start in" directory to this folder.

Or use Task Scheduler for more control.

---

## Build as .exe (Optional)

```bash
pip install pyinstaller
pyinstaller WordScript.spec
```

The executable will be in the `dist/` folder as `WordScript.exe`. The `config.json` must be in the same directory as the `.exe`.

---

## Troubleshooting

| Problem                       | Solution                                                                                      |
| ----------------------------- | --------------------------------------------------------------------------------------------- |
| "No audio input device found" | Check Windows sound settings → Input. Make sure a microphone is connected and set as default. |
| No beep sounds                | Windows beep requires a sound card. Check `play_sounds` config.                               |
| Hotkey doesn't work           | Run as Administrator. Some apps intercept Win key combos. Try a different hotkey.             |
| Transcription errors          | Check your Groq API key and internet connection. Check console logs.                          |
| Tray icon missing             | Ensure `pystray` and `Pillow` are installed. Set `show_tray_icon: true`.                      |

---

## Architecture

```
SpeechToTextApp          ← main orchestrator
├── AudioRecorder        ← sounddevice microphone capture → WAV bytes
├── TranscriptionService ← Groq Whisper API client
├── TextPaster           ← clipboard + Ctrl+V simulation
├── HotkeyManager        ← pynput global keyboard listener
├── TrayIcon             ← pystray system tray (optional)
└── SoundFeedback        ← winsound beeps (optional)
```
