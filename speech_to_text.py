"""
Speech-to-Text System using Groq Whisper API
=============================================
A lightweight, production-ready Windows 11 system-wide speech-to-text tool.
Press a global hotkey to record audio from the microphone, transcribe it via
Groq's Whisper API, and auto-paste the result into the active application.

Author: Auto-generated
License: MIT
"""

import io
import json
import logging
import os
import socket
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import numpy as np
import pyautogui
import pyperclip
import sounddevice as sd
from groq import Groq
from pynput import keyboard

# Optional imports for tray icon and sound feedback
try:
    import pystray
    from PIL import Image, ImageDraw, ImageFont
    TRAY_AVAILABLE = True
except ImportError:
    TRAY_AVAILABLE = False

try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Get the directory where config.json is located
# For .exe: use the directory where the .exe is located
# For script: use script's directory
if getattr(sys, 'frozen', False):
    # Running as compiled .exe - look in the same directory as the .exe
    EXE_DIR = Path(sys.executable).parent
    CONFIG_FILE = EXE_DIR / "config.json"
    LOG_FILE = EXE_DIR / "wordscript.log"
else:
    # Running as Python script
    CONFIG_FILE = Path(__file__).parent / "config.json"
    LOG_FILE = Path(__file__).parent / "wordscript.log"

@dataclass
class Config:
    """Application configuration loaded from config.json."""
    groq_api_key: str = ""
    model: str = "whisper-large-v3-turbo"  # Schnell und gut genug
    language: str = ""  # Auto-Detection für alle Sprachen
    prompt: str = ""  # Kontext für bessere Erkennung
    
    # AI-Korrektur nach Transkription
    post_process: bool = True  # Text durch LLM korrigieren lassen
    correction_model: str = "llama-3.3-70b-versatile"  # Groq LLM für Korrektur

    hotkey: str = "ctrl_l+win"          # pynput key combo string
    activation_mode: str = "tap"        # "tap" or "hold"

    sample_rate: int = 24000            # 24kHz for better quality (Whisper supports 16-48kHz)
    channels: int = 1
    dtype: str = "int16"
    audio_device: str = ""  # Leer = Standard-Mikrofon, sonst Name des Geräts
    max_recording_seconds: int = 120
    silence_timeout_seconds: int = 30

    auto_paste: bool = True
    show_tray_icon: bool = True
    play_sounds: bool = True
    log_level: str = "INFO"
    temp_audio_dir: str = ""

    @classmethod
    def load(cls, path: Path = CONFIG_FILE) -> "Config":
        """Load configuration from a JSON file, falling back to defaults."""
        cfg = cls()
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key, value in data.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
        else:
            logging.warning("Config file not found at %s — using defaults.", path)
        return cfg


# ---------------------------------------------------------------------------
# Audio Recorder
# ---------------------------------------------------------------------------

class AudioRecorder:
    """Records audio from the default microphone using sounddevice."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("AudioRecorder")
        self._frames: List[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None
        self._recording = False
        self._lock = threading.Lock()

    # --- public API ---

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self) -> None:
        """Start recording audio from the microphone."""
        with self._lock:
            if self._recording:
                self.logger.warning("Already recording.")
                return
            self._frames.clear()
            self._recording = True

        self.logger.info("Recording started.")
        try:
            # Find device by name if specified
            device = None
            if self.config.audio_device:
                devices = sd.query_devices()
                for i, dev in enumerate(devices):
                    if self.config.audio_device.lower() in dev['name'].lower() and dev['max_input_channels'] > 0:
                        device = i
                        self.logger.info("Using audio device: %s", dev['name'])
                        break
                if device is None:
                    self.logger.warning("Device '%s' not found, using default", self.config.audio_device)
            
            self._stream = sd.InputStream(
                device=device,
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype,
                callback=self._audio_callback,
            )
            self._stream.start()
        except sd.PortAudioError as exc:
            self._recording = False
            self.logger.error("Failed to open audio device: %s", exc)
            raise RuntimeError(
                "No audio input device found. Check microphone settings."
            ) from exc

    def stop(self) -> bytes:
        """Stop recording and return the audio as WAV bytes."""
        with self._lock:
            if not self._recording:
                self.logger.warning("Not currently recording.")
                return b""
            self._recording = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self.logger.info("Recording stopped (%d chunks captured).", len(self._frames))
        return self._build_wav()

    # --- internals ---

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:  # noqa: ANN001
        """sounddevice callback — accumulates raw audio frames."""
        if status:
            self.logger.debug("Audio callback status: %s", status)
        if self._recording:
            self._frames.append(indata.copy())

    def _build_wav(self) -> bytes:
        """Encode accumulated frames as a WAV file in memory."""
        if not self._frames:
            return b""
        audio_data = np.concatenate(self._frames, axis=0)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.config.channels)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(self.config.sample_rate)
            wf.writeframes(audio_data.tobytes())
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Transcription Service (Groq Whisper)
# ---------------------------------------------------------------------------

class TranscriptionService:
    """Sends audio to the Groq Whisper API and returns transcribed text."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("TranscriptionService")
        if not config.groq_api_key:
            raise ValueError("Groq API key is missing. Set it in config.json.")
        self._client = Groq(api_key=config.groq_api_key)

    def transcribe(self, wav_bytes: bytes) -> str:
        """Send WAV audio bytes to Groq and return the transcription text."""
        if not wav_bytes:
            return ""

        audio_size_kb = len(wav_bytes) / 1024
        self.logger.info("Sending %.1f KB to Groq Whisper (model: %s)...", audio_size_kb, self.config.model)
        start = time.perf_counter()

        try:
            # Groq SDK expects a file-like tuple: (filename, file_bytes, mime)
            params = {
                "file": ("recording.wav", wav_bytes),
                "model": self.config.model,
                "response_format": "text",
                "temperature": 0.0,  # Deterministisch = genauer
            }
            # Only add language if specified (empty string enables auto-detection)
            if self.config.language:
                params["language"] = self.config.language
            
            # Add prompt for context if specified
            if self.config.prompt:
                params["prompt"] = self.config.prompt
            
            transcription = self._client.audio.transcriptions.create(**params)
            elapsed = time.perf_counter() - start
            text = transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
            
            # Filter obvious hallucinations (Whisper training data artifacts)
            hallucinations = [
                "thanks for watching", "thank you for watching",
                "thank you", "thanks",
                "vielen dank", "vielen dank fürs zuschauen",
                "vielen dank für ihre aufmerksamkeit",
                "danke schön", "danke fürs zuschauen", "danke",
                "bitte abonnieren", "nicht vergessen zu abonnieren",
                "untertitel von", "untertitel der amara.org-community",
                ".", "",
            ]
            if text.lower() in hallucinations or (len(text) <= 2 and text in [".", ".."]):
                self.logger.info("Filtered likely hallucination: '%s'", text)
                return ""
            
            self.logger.info("✓ Whisper (%.2fs): %s", elapsed, text[:100])
            
            # Post-process with LLM if enabled
            if self.config.post_process and text:
                text = self._correct_with_llm(text)
            
            return text

        except Exception as exc:
            # Check for rate limiting
            error_str = str(exc)
            if "429" in error_str or "rate_limit" in error_str.lower():
                self.logger.error("⚠ RATE LIMIT EXCEEDED: %s", exc)
                return "[Rate limit erreicht - bitte warten]"
            else:
                self.logger.error("❌ Groq API error: %s", exc)
                return f"[Transcription error: {exc}]"
    
    def _correct_with_llm(self, text: str) -> str:
        """Use Groq LLM to correct transcription errors and improve text quality."""
        try:
            start = time.perf_counter()
            
            system_prompt = """Du bist ein stummer Textkorrektur-Filter. Du gibst AUSSCHLIESSLICH den korrigierten Text zurück.

REGELN:
- Gib NUR den korrigierten Text aus, NICHTS anderes
- KEINE Kommentare, KEINE Erklärungen, KEINE Antworten
- NICHT auf den Inhalt reagieren oder antworten
- NICHT sagen "Hier ist der korrigierte Text" oder ähnliches
- NICHT sagen "Ich verstehe" oder "Der Text ist zu kurz"
- Wenn der Text bereits korrekt ist, gib ihn unverändert zurück
- Wenn du nichts korrigieren musst, gib den Originaltext zurück
- Behalte die Sprache bei (DE, EN oder gemischt) — übersetze NIEMALS
- Korrigiere nur offensichtliche Tippfehler und Grammatikfehler
- NIEMALS Wörter, Phrasen oder Satzteile entfernen oder weglassen
- NIEMALS den Text kürzen oder zusammenfassen — jedes Wort des Originals bleibt erhalten
- NIEMALS Satzteile vereinfachen oder umformulieren, die korrekt sind
- Behalte alle Wörter, auch wenn sie aus Sicht des Stils weglassbar wären

BEISPIELE:
Input: "Ich geh jetz zum Meeting"
Output: "Ich geh jetzt zum Meeting"

Input: "Ok"
Output: "Ok"

Input: "Was machst du?"
Output: "Was machst du?"

Input: "I need to fix this bevor wir deployen"
Output: "I need to fix this bevor wir deployen"

Input: "Das ist, würde ich sagen, eigentlich ziemlich interessant und auch relevant"
Output: "Das ist, würde ich sagen, eigentlich ziemlich interessant und auch relevant"

Du antwortest NIEMALS auf Fragen oder Inhalte. Du bist ein Filter, kein Assistent."""

            response = self._client.chat.completions.create(
                model=self.config.correction_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.0,
                max_tokens=len(text) + 100,  # Nur minimal mehr als der Input
            )
            
            corrected = response.choices[0].message.content.strip()
            elapsed = time.perf_counter() - start
            
            # Sicherheitscheck: LLM darf nicht als Assistent antworten
            assistant_phrases = [
                "ich verstehe", "verstanden", "hier ist", "der text",
                "ich bin bereit", "gerne", "natürlich", "als ki",
                "entschuldigung", "leider", "ich kann", "möchtest du",
                "bitte", "danke für", "zu kurz", "nicht genug"
            ]
            corrected_lower = corrected.lower()

            # Wenn Antwort viel länger ist oder Assistenten-Phrasen enthält → Original behalten
            if len(corrected) > len(text) * 2:
                self.logger.warning("LLM-Antwort zu lang, ignoriert: %s", corrected[:50])
                return text

            # Wenn Antwort deutlich kürzer ist → LLM hat Text gekürzt → Original behalten
            if len(text) > 20 and len(corrected) < len(text) * 0.85:
                self.logger.warning("LLM hat Text gekürzt (%.0f%%), ignoriert: %s → %s",
                                    len(corrected)/len(text)*100, text[:50], corrected[:50])
                return text
            
            for phrase in assistant_phrases:
                if phrase in corrected_lower and phrase not in text.lower():
                    self.logger.warning("LLM antwortet als Assistent, ignoriert: %s", corrected[:50])
                    return text
            
            if corrected and corrected != text:
                self.logger.info("✓ LLM-Korrektur (%.2fs): %s → %s", elapsed, text[:50], corrected[:50])
                return corrected
            else:
                self.logger.info("✓ LLM: Keine Korrektur nötig (%.2fs)", elapsed)
                return text
                
        except Exception as exc:
            self.logger.warning("LLM-Korrektur fehlgeschlagen: %s", exc)
            return text  # Bei Fehler Original zurückgeben


# ---------------------------------------------------------------------------
# Text Paster
# ---------------------------------------------------------------------------

class TextPaster:
    """Copies text to the clipboard and optionally simulates Ctrl+V."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("TextPaster")

    def paste(self, text: str) -> None:
        """Put text on the clipboard and optionally auto-paste via Ctrl+V."""
        if not text:
            return
        
        # Add space after sentence-ending punctuation for proper separation
        if text and text[-1] in '.!?':
            text = text + ' '
        
        pyperclip.copy(text)
        self.logger.info("Text copied to clipboard.")

        if self.config.auto_paste:
            # Small delay to ensure the target window is focused
            time.sleep(0.15)
            pyautogui.hotkey("ctrl", "v")
            self.logger.info("Auto-pasted into active window.")


# ---------------------------------------------------------------------------
# Sound Feedback
# ---------------------------------------------------------------------------

class SoundFeedback:
    """Plays smooth, pleasant beep sounds using WAV playback for reliability."""
    
    _lock = threading.Lock()  # Prevent overlapping sounds
    _sample_rate = 44100
    
    @staticmethod
    def _generate_smooth_wav(frequency: float, duration_ms: int, volume: float = 0.3) -> bytes:
        """Generate a smooth sine wave WAV with fade-in/out for clippy, smooth sound."""
        duration_sec = duration_ms / 1000.0
        samples = int(SoundFeedback._sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, samples, False)
        
        # Generate sine wave
        tone = np.sin(2 * np.pi * frequency * t)
        
        # Apply smooth envelope with 20% fade for extra smoothness
        fade_samples = int(samples * 0.20)
        if fade_samples > 0:
            fade_in = np.linspace(0, 1, fade_samples) ** 2  # Quadratic for smoother start
            fade_out = np.linspace(1, 0, fade_samples) ** 2  # Quadratic for smoother end
            tone[:fade_samples] *= fade_in
            tone[-fade_samples:] *= fade_out
        
        # Convert to int16 PCM
        audio_data = (tone * volume * 32767).astype(np.int16)
        
        # Build WAV file in memory
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SoundFeedback._sample_rate)
            wf.writeframes(audio_data.tobytes())
        
        return buf.getvalue()
    
    @staticmethod
    def _play_wav(wav_bytes: bytes) -> None:
        """Play WAV bytes using winsound.PlaySound (reliable and smooth)."""
        try:
            if WINSOUND_AVAILABLE and wav_bytes:
                # SND_MEMORY plays from memory, SND_NODEFAULT prevents system beep on error
                winsound.PlaySound(wav_bytes, winsound.SND_MEMORY | winsound.SND_NODEFAULT)
        except Exception:
            pass
    
    @staticmethod
    def play_startup() -> None:
        """Smooth ascending tones — app ready."""
        def _sound():
            with SoundFeedback._lock:
                # C5 → E5 ascending
                SoundFeedback._play_wav(SoundFeedback._generate_smooth_wav(523, 150, 0.25))
                time.sleep(0.02)
                SoundFeedback._play_wav(SoundFeedback._generate_smooth_wav(659, 150, 0.25))
        threading.Thread(target=_sound, daemon=False).start()
    
    @staticmethod
    def play_start() -> None:
        """Crisp clean beep — recording started."""
        with SoundFeedback._lock:
            SoundFeedback._play_wav(SoundFeedback._generate_smooth_wav(880, 120, 0.28))  # A5
    
    @staticmethod
    def play_stop() -> None:
        """Smooth confirming beep — recording stopped."""
        with SoundFeedback._lock:
            SoundFeedback._play_wav(SoundFeedback._generate_smooth_wav(659, 140, 0.26))  # E5
    
    @staticmethod
    def play_abort() -> None:
        """Smooth descending beeps — recording aborted."""
        def _sound():
            with SoundFeedback._lock:
                # C5 → G4 descending
                SoundFeedback._play_wav(SoundFeedback._generate_smooth_wav(523, 100, 0.25))
                time.sleep(0.02)
                SoundFeedback._play_wav(SoundFeedback._generate_smooth_wav(392, 100, 0.25))
        threading.Thread(target=_sound, daemon=False).start()
    
    @staticmethod
    def play_error() -> None:
        """Low descending tone — error occurred."""
        def _sound():
            with SoundFeedback._lock:
                # B4 → G4 descending
                SoundFeedback._play_wav(SoundFeedback._generate_smooth_wav(494, 130, 0.25))
                time.sleep(0.03)
                SoundFeedback._play_wav(SoundFeedback._generate_smooth_wav(392, 130, 0.25))
        threading.Thread(target=_sound, daemon=False).start()

# ---------------------------------------------------------------------------
# System Tray Icon
# ---------------------------------------------------------------------------

class TrayIcon:
    """Optional system tray icon with status indication and quit option."""

    def __init__(self, on_quit_callback):
        self._on_quit = on_quit_callback
        self._icon: Optional[pystray.Icon] = None
        self._recording = False
        self.logger = logging.getLogger("TrayIcon")

    def start(self) -> None:
        if not TRAY_AVAILABLE:
            self.logger.warning("pystray/Pillow not installed — tray icon disabled.")
            return
        self._icon = pystray.Icon(
            "WordScript",
            self._create_image(recording=False),
            "WordScript (Idle)",
            menu=pystray.Menu(
                pystray.MenuItem("WordScript", None, enabled=False),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Quit", self._quit),
            ),
        )
        threading.Thread(target=self._icon.run, daemon=True).start()
        self.logger.info("Tray icon started.")

    def set_recording(self, recording: bool) -> None:
        self._recording = recording
        if self._icon:
            self._icon.icon = self._create_image(recording)
            self._icon.title = "WordScript (Recording...)" if recording else "WordScript (Idle)"

    def stop(self) -> None:
        if self._icon:
            self._icon.stop()

    def _quit(self, icon, item) -> None:  # noqa: ANN001
        self._on_quit()

    @staticmethod
    def _create_image(recording: bool) -> "Image.Image":
        """Generate a simple colored circle icon: red = recording, green = idle."""
        size = 64
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        color = (220, 40, 40, 255) if recording else (40, 180, 60, 255)
        draw.ellipse([4, 4, size - 4, size - 4], fill=color)
        # Draw "STT" text in center
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()
        draw.text((size // 2, size // 2), "STT", fill=(255, 255, 255, 255), anchor="mm", font=font)
        return img


# ---------------------------------------------------------------------------
# Hotkey Manager
# ---------------------------------------------------------------------------

class HotkeyManager:
    """Global hotkey listener using pynput, supporting tap and hold modes."""

    # Map config key names to pynput key objects
    KEY_MAP = {
        "ctrl": keyboard.Key.ctrl_l,
        "ctrl_l": keyboard.Key.ctrl_l,
        "ctrl_r": keyboard.Key.ctrl_r,
        "alt": keyboard.Key.alt_l,
        "alt_l": keyboard.Key.alt_l,
        "alt_r": keyboard.Key.alt_r,
        "shift": keyboard.Key.shift_l,
        "shift_l": keyboard.Key.shift_l,
        "shift_r": keyboard.Key.shift_r,
        "win": keyboard.Key.cmd,
        "cmd": keyboard.Key.cmd,
        "space": keyboard.Key.space,
        "f1": keyboard.Key.f1,
        "f2": keyboard.Key.f2,
        "f3": keyboard.Key.f3,
        "f4": keyboard.Key.f4,
        "f5": keyboard.Key.f5,
        "f6": keyboard.Key.f6,
        "f7": keyboard.Key.f7,
        "f8": keyboard.Key.f8,
        "f9": keyboard.Key.f9,
        "f10": keyboard.Key.f10,
        "f11": keyboard.Key.f11,
        "f12": keyboard.Key.f12,
    }

    def __init__(self, config: Config, on_activate, on_deactivate, on_abort):
        self.config = config
        self.logger = logging.getLogger("HotkeyManager")
        self._on_activate = on_activate
        self._on_deactivate = on_deactivate
        self._on_abort = on_abort

        self._hotkey_keys = self._parse_hotkey(config.hotkey)
        # Abort hotkey: Ctrl+Alt
        self._abort_keys = {keyboard.Key.ctrl_l, keyboard.Key.alt_l}
        self._pressed_keys: set = set()
        self._hotkey_active = False  # Is the hotkey combo currently held?
        self._abort_active = False   # Is the abort combo currently held?
        self._toggled_on = False     # For tap mode: is recording toggled on?
        self._listener: Optional[keyboard.Listener] = None

        self.logger.info(
            "Hotkey: %s | Mode: %s | Abort: Ctrl+Alt",
            config.hotkey,
            config.activation_mode,
        )

    def start(self) -> None:
        """Start the global keyboard listener."""
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            suppress=False,
        )
        self._listener.daemon = True
        self._listener.start()
        self.logger.info("Hotkey listener started.")

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()

    def _parse_hotkey(self, hotkey_str: str) -> set:
        """Convert a hotkey config string like 'ctrl_l+win' into a set of pynput keys."""
        keys = set()
        for part in hotkey_str.lower().split("+"):
            part = part.strip()
            if part in self.KEY_MAP:
                keys.add(self.KEY_MAP[part])
            elif len(part) == 1:
                keys.add(keyboard.KeyCode.from_char(part))
            else:
                self.logger.warning("Unknown key in hotkey config: '%s'", part)
        return keys

    def _normalize_key(self, key) -> keyboard.Key:
        """Normalize a key event to its base form for reliable matching."""
        # Map right-side modifiers to their left equivalents for simpler matching
        RIGHT_TO_LEFT = {
            keyboard.Key.ctrl_r: keyboard.Key.ctrl_l,
            keyboard.Key.alt_r: keyboard.Key.alt_l,
            keyboard.Key.shift_r: keyboard.Key.shift_l,
            keyboard.Key.cmd_r: keyboard.Key.cmd,
        }
        if key in RIGHT_TO_LEFT:
            return RIGHT_TO_LEFT[key]
        return key

    def _on_press(self, key) -> None:
        normalized = self._normalize_key(key)
        self._pressed_keys.add(normalized)

        # Check abort hotkey first (Ctrl+Alt)
        if self._abort_keys.issubset(self._pressed_keys):
            if not self._abort_active:
                self._abort_active = True
                self._handle_abort_press()
        # Then check main hotkey
        elif self._hotkey_keys.issubset(self._pressed_keys):
            if not self._hotkey_active:
                self._hotkey_active = True
                self._handle_hotkey_press()

    def _on_release(self, key) -> None:
        normalized = self._normalize_key(key)
        self._pressed_keys.discard(normalized)
        # Also discard the original key in case it wasn't normalized
        self._pressed_keys.discard(key)

        if not self._abort_keys.issubset(self._pressed_keys):
            if self._abort_active:
                self._abort_active = False
        
        if not self._hotkey_keys.issubset(self._pressed_keys):
            if self._hotkey_active:
                self._hotkey_active = False
                self._handle_hotkey_release()

    def _handle_hotkey_press(self) -> None:
        if self.config.activation_mode == "hold":
            self._on_activate()
        elif self.config.activation_mode == "tap":
            if self._toggled_on:
                self._toggled_on = False
                self._on_deactivate()
            else:
                self._toggled_on = True
                self._on_activate()

    def _handle_hotkey_release(self) -> None:
        if self.config.activation_mode == "hold":
            self._on_deactivate()
    
    def _handle_abort_press(self) -> None:
        """Handle Ctrl+Alt abort hotkey."""
        self.logger.info("Abort hotkey pressed")
        self._on_abort()


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

class SpeechToTextApp:
    """
    Main orchestrator: ties together recording, transcription, hotkeys,
    clipboard pasting, tray icon, and sound feedback.
    """

    def __init__(self):
        self.config = Config.load()
        self.logger = logging.getLogger("App")

        # Components
        self.recorder = AudioRecorder(self.config)
        self.transcriber = TranscriptionService(self.config)
        self.paster = TextPaster(self.config)
        self.sounds = SoundFeedback()
        self.hotkeys = HotkeyManager(
            self.config,
            on_activate=self._start_recording,
            on_deactivate=self._stop_recording,
            on_abort=self._abort_recording,
        )
        self.tray: Optional[TrayIcon] = None
        self._running = True
        self._transcription_lock = threading.Lock()

    # --- lifecycle ---

    def run(self) -> None:
        """Start all components and block until shutdown."""
        self.logger.info("=" * 50)
        self.logger.info("Speech-to-Text starting up...")
        self.logger.info("  Hotkey     : %s", self.config.hotkey)
        self.logger.info("  Mode       : %s", self.config.activation_mode)
        self.logger.info("  Model      : %s", self.config.model)
        self.logger.info("  Auto-paste : %s", self.config.auto_paste)
        self.logger.info("=" * 50)

        self.hotkeys.start()

        if self.config.show_tray_icon and TRAY_AVAILABLE:
            self.tray = TrayIcon(on_quit_callback=self.shutdown)
            self.tray.start()

        # Play startup sound to indicate ready
        if self.config.play_sounds:
            time.sleep(0.5)  # Brief delay for tray icon to appear
            self.sounds.play_startup()

        # Keep the main thread alive
        try:
            while self._running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user.")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Gracefully stop everything."""
        self.logger.info("Shutting down...")
        self._running = False
        if self.recorder.is_recording:
            self.recorder.stop()
        self.hotkeys.stop()
        if self.tray:
            self.tray.stop()
        self.logger.info("Goodbye.")
        os._exit(0)  # Force exit to kill any lingering threads

    # --- recording callbacks ---

    def _start_recording(self) -> None:
        """Called by HotkeyManager when hotkey activates."""
        try:
            self.recorder.start()
            if self.config.play_sounds:
                self.sounds.play_start()
            if self.tray:
                self.tray.set_recording(True)
        except RuntimeError as exc:
            self.logger.error(str(exc))
            if self.config.play_sounds:
                self.sounds.play_error()

    def _stop_recording(self) -> None:
        """Called by HotkeyManager when hotkey deactivates."""
        if not self.recorder.is_recording:
            return
            
        wav_bytes = self.recorder.stop()
        if self.config.play_sounds:
            self.sounds.play_stop()
        if self.tray:
            self.tray.set_recording(False)

        if not wav_bytes:
            self.logger.warning("No audio captured.")
            return

        # Transcribe in a background thread so the hotkey listener isn't blocked
        threading.Thread(
            target=self._transcribe_and_paste,
            args=(wav_bytes,),
            daemon=True,
        ).start()
    
    def _abort_recording(self) -> None:
        """Called by HotkeyManager when Ctrl+Alt is pressed to abort."""
        if not self.recorder.is_recording:
            return
        
        self.logger.info("Recording aborted by user")
        self.recorder.stop()  # Just stop, don't transcribe
        if self.config.play_sounds:
            self.sounds.play_abort()
        if self.tray:
            self.tray.set_recording(False)
        # Reset tap mode if active
        self.hotkeys._toggled_on = False

    def _transcribe_and_paste(self, wav_bytes: bytes) -> None:
        """Send audio to Groq and paste the result."""
        with self._transcription_lock:
            text = self.transcriber.transcribe(wav_bytes)
            if text and not text.startswith("[Transcription error"):
                self.paster.paste(text)
            elif text.startswith("[Transcription error"):
                self.logger.error(text)
                if self.config.play_sounds:
                    self.sounds.play_error()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Global singleton socket to prevent multiple instances
_singleton_socket = None

def _setup_logging(log_level: str) -> None:
    """Configure logging to both file and console with detailed format."""
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # File handler - keep last 5MB, rotate
    try:
        file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        print(f"[OK] Logging to: {LOG_FILE}")
    except Exception as exc:
        print(f"[WARNING] Could not create log file: {exc}")
    
    # Console handler for errors only (if not already added)
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

def main() -> None:
    """Entry point for the speech-to-text application."""
    global _singleton_socket
    
    # Prevent multiple instances using a socket lock
    try:
        _singleton_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _singleton_socket.bind(('127.0.0.1', 48127))  # Random high port for lock
    except OSError:
        print("[INFO] WordScript is already running.")
        print("[INFO] Only one instance can run at a time.")
        sys.exit(0)
    
    # Quick sanity checks
    if not CONFIG_FILE.exists():
        print(f"[ERROR] Config file not found: {CONFIG_FILE}")
        print("Create config.json with at least your Groq API key.")
        sys.exit(1)

    # Load config early to get log level
    config = Config.load()
    _setup_logging(config.log_level)
    
    logger = logging.getLogger("Main")
    logger.info("="*60)
    logger.info("WordScript starting...")
    logger.info("Model: %s | Language: %s | Sample Rate: %d Hz", 
                config.model, config.language or "auto", config.sample_rate)

    # Check audio devices early
    try:
        devices = sd.query_devices()
        default_input = sd.query_devices(kind="input")
        logger.info("Audio input: %s", default_input['name'])
        print(f"[OK] Default input device: {default_input['name']}")
    except Exception as exc:
        logger.error("No audio input device: %s", exc)
        print(f"[ERROR] No audio input device found: {exc}")
        sys.exit(1)

    app = SpeechToTextApp()
    app.run()


if __name__ == "__main__":
    main()
