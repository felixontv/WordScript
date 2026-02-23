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
import queue
import socket
import sys
import tempfile
import threading
import time
import urllib.request
import wave
import webbrowser
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

# ---------------------------------------------------------------------------
# App identity
# ---------------------------------------------------------------------------

APP_VERSION  = "0.1.0-alpha"              # bump on each release
GITHUB_REPO  = "felixontv/WordScript"     # owner/repo on GitHub

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _resolve_user_data_dir() -> Path:
    """Return the per-user config directory, following OS conventions.

    Windows : %APPDATA%\\WordScript          (C:\\Users\\<user>\\AppData\\Roaming\\WordScript)
    macOS   : ~/Library/Application Support/WordScript
    Linux   : $XDG_CONFIG_HOME/WordScript   (defaults to ~/.config/WordScript)
    Dev run : next to the script            (convenient, no side-effects)
    """
    if not getattr(sys, 'frozen', False):
        # Running as plain Python script — keep data local for development
        return Path(__file__).parent

    plat = sys.platform
    if plat == "win32":
        base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    elif plat == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        # Linux / BSD — respect XDG_CONFIG_HOME
        xdg = os.environ.get("XDG_CONFIG_HOME", "")
        base = Path(xdg) if xdg else Path.home() / ".config"

    return base / "WordScript"


USER_DATA_DIR = _resolve_user_data_dir()

USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = USER_DATA_DIR / "config.json"
LOG_FILE    = USER_DATA_DIR / "wordscript.log"

# One-time migration: if no config exists yet in the user dir but one sits
# next to the EXE (old behaviour), copy it over so settings aren't lost.
if getattr(sys, 'frozen', False) and not CONFIG_FILE.exists():
    _old_config = Path(sys.executable).parent / "config.json"
    if _old_config.exists():
        import shutil
        shutil.copy2(_old_config, CONFIG_FILE)

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

    hotkey: str = ("ctrl_l+cmd" if sys.platform == "darwin"
                   else "ctrl_l+alt_l" if sys.platform != "win32"
                   else "ctrl_l+win")    # pynput key combo string
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

    def save(self, path: Path = CONFIG_FILE) -> None:
        """Persist current configuration back to JSON."""
        import dataclasses
        data = dataclasses.asdict(self)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logging.getLogger("Config").info("Config saved to %s", path)


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
        self._muted = False
        self._lock = threading.Lock()
        self.level_queue: queue.Queue = queue.Queue()

    @property
    def muted(self) -> bool:
        return self._muted

    def toggle_mute(self) -> bool:
        """Toggle mute state. Returns True if now muted."""
        self._muted = not self._muted
        self.logger.info("Microphone %s", "muted" if self._muted else "unmuted")
        return self._muted

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
            # Discard any stale level data from a previous recording
            while not self.level_queue.empty():
                try:
                    self.level_queue.get_nowait()
                except queue.Empty:
                    break
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
            if self._muted:
                # Record silence so timeline stays consistent
                silent = np.zeros_like(indata)
                self._frames.append(silent)
                self.level_queue.put(0.0)
            else:
                self._frames.append(indata.copy())
                peak = float(np.max(np.abs(indata.astype(np.float32))))
                if self.config.dtype == "int16":
                    peak /= 32767.0
                self.level_queue.put(min(1.0, peak))

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
        self._client = (
            Groq(api_key=config.groq_api_key, max_retries=0, timeout=15.0)
            if config.groq_api_key else None
        )
        if not config.groq_api_key:
            self.logger.warning("No Groq API key set — open Settings (chevron) to enter yours.")

    def reload_api_key(self) -> None:
        """Re-initialise the Groq client after an API-key change at runtime."""
        if self.config.groq_api_key:
            self._client = Groq(
                api_key=self.config.groq_api_key,
                max_retries=0,
                timeout=15.0,
            )
            self.logger.info("Groq client reloaded with new API key.")
        else:
            self._client = None

    def transcribe(self, wav_bytes: bytes) -> str:
        """Send WAV audio bytes to Groq and return the transcription text."""
        if not wav_bytes:
            return ""
        if not self._client:
            self.logger.error("No API key — open Settings (chevron button) to enter your Groq key.")
            return ""  # silent — user already sees the settings prompt

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
    
    # ------------------------------------------------------------------
    # Wörter-Überlappungs-Check: Prüft ob der korrigierte Text wirklich
    # vom Original abgeleitet ist (und nicht eine Assistenten-Antwort)
    # ------------------------------------------------------------------
    @staticmethod
    def _word_overlap_ok(original: str, corrected: str) -> bool:
        """Return True if corrected shares enough words with original."""
        orig_words = set(original.lower().split())
        if len(orig_words) < 5:
            return True  # Kurze Texte nicht prüfen (zu unzuverlässig)
        corr_words = set(corrected.lower().split())
        overlap = len(orig_words & corr_words) / len(orig_words)
        return overlap >= 0.55  # mind. 55% der Originalwörter müssen enthalten sein

    def _correct_with_llm(self, text: str) -> str:
        """Use Groq LLM to correct transcription errors and improve text quality."""
        try:
            start = time.perf_counter()

            system_prompt = ("Du bist ein stummer Textkorrektur-Filter. "
                "Gib AUSSCHLIESSLICH den korrigierten Text zurück — "
                "KEINE Kommentare, Erklärungen oder Antworten. "
                "Sprache beibehalten (DE/EN/gemischt), niemals übersetzen. "
                "Nur Tippfehler und Grammatik korrigieren; "
                "niemals Wörter entfernen, kürzen oder umformulieren. "
                "Kurzer Input (1-5 Wörter): exakt zurückgeben. "
                "Bei korrektem Text: Originaltext Zeichen für Zeichen zurück. "
                "Du bist ein Filter, kein Assistent.")

            response = self._client.chat.completions.create(
                model=self.config.correction_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.0,
                max_tokens=max(len(text), 20),  # Kein Spielraum für Blabber
            )

            corrected = response.choices[0].message.content.strip()
            elapsed = time.perf_counter() - start

            # --- Layer 1: Längenprüfung ---
            # Antwort darf nicht mehr als 40% länger sein als der Input
            if len(corrected) > len(text) * 1.4 + 30:
                self.logger.warning("LLM-Antwort zu lang (%d→%d), ignoriert: %s",
                                    len(text), len(corrected), corrected[:60])
                return text

            # Antwort darf nicht deutlich kürzer sein (LLM hat Text gekürzt)
            if len(text) > 20 and len(corrected) < len(text) * 0.85:
                self.logger.warning("LLM hat Text gekürzt (%.0f%%), ignoriert: %s → %s",
                                    len(corrected) / len(text) * 100, text[:50], corrected[:50])
                return text

            corrected_lower = corrected.lower()
            text_lower = text.lower()

            # --- Layer 2: Erweiterte Assistenten-Phrasen-Blacklist ---
            # Enthält Phrasen, die ein "antwortendes" LLM benutzt, aber kein Korrektur-Filter
            assistant_phrases = [
                # Deutsch – explizite Assistenten-Reaktionen
                "ich verstehe", "verstanden", "hier ist", "der text",
                "ich bin bereit", "als ki", "als sprachmodell",
                "entschuldigung", "leider", "ich kann", "möchtest du",
                "danke für", "zu kurz", "nicht genug",
                "du musst", "sie müssen", "bitte geben", "bitte eingeben",
                "bitte gib", "bitte schreib", "bitte tippe",
                "damit ich", "damit ich es", "um das zu", "um es zu",
                "ich benötige", "ich brauche", "ich habe nichts",
                "es gibt nichts", "kein text", "keinen text",
                "der eingabe", "die eingabe", "keine eingabe",
                "vielen dank", "herzlichen dank", "danke schön",
                "gerne helfe", "gerne korrigiere", "ich helfe",
                "hier der korrigier", "hier ist der korrigier",
                "natürlich", "selbstverständlich", "sicher",
                "ich werde", "ich würde", "ich habe",
                "bitte beachte", "bitte beachten",
                "kein problem", "no problem",
                # Englisch – explizite Assistenten-Reaktionen
                "i understand", "here is", "here's", "the text",
                "i'm ready", "as an ai", "as a language",
                "sorry", "unfortunately", "i can", "would you",
                "thank you for", "too short", "not enough",
                "you must", "you need to", "please enter", "please provide",
                "please type", "please write", "so that i can",
                "in order to", "i need", "i require", "there is nothing",
                "no text", "no input", "the input",
                "of course", "certainly", "sure",
                "i will", "i would", "i have",
                "please note", "no problem",
            ]

            for phrase in assistant_phrases:
                if phrase in corrected_lower and phrase not in text_lower:
                    self.logger.warning("LLM antwortet als Assistent [phrase='%s'], ignoriert: %s",
                                        phrase, corrected[:60])
                    return text

            # --- Layer 3: Verdächtige Satzanfänge ---
            # Wenn der korrigierte Text mit Mustern beginnt, die typisch für Assistenten sind
            suspicious_starts = [
                "ich ", "sie ", "du ", "bitte ", "danke", "vielen",
                "here ", "i ", "you ", "please ", "thank", "sure,",
                "of course", "certainly", "natürlich", "selbstverständlich",
            ]
            for start_phrase in suspicious_starts:
                if corrected_lower.startswith(start_phrase) and not text_lower.startswith(start_phrase):
                    self.logger.warning("LLM beginnt Antwort verdächtig [start='%s'], ignoriert: %s",
                                        start_phrase, corrected[:60])
                    return text

            # --- Layer 4: Wort-Overlap-Check ---
            # Der korrigierte Text muss zu mindestens 55% aus Originalwörtern bestehen
            if not self._word_overlap_ok(text, corrected):
                self.logger.warning("LLM-Wortüberlappung zu gering, ignoriert: %s → %s",
                                    text[:50], corrected[:50])
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
            # macOS uses Cmd+V; Windows and Linux use Ctrl+V
            if sys.platform == "darwin":
                pyautogui.hotkey("command", "v")
            else:
                pyautogui.hotkey("ctrl", "v")
            self.logger.info("Auto-pasted into active window.")


# ---------------------------------------------------------------------------
# Sound Feedback
# ---------------------------------------------------------------------------

class SoundFeedback:
    """Plays smooth notification sounds via sounddevice.play() — pre-generated at startup."""

    _SR = 44100  # sample rate for all sounds

    # Pre-generated audio arrays (filled on first use)
    _cache: dict = {}
    _cache_lock = threading.Lock()

    # ── Audio generation ─────────────────────────────────────────────────

    @classmethod
    def _tone(cls, freq: float, dur_ms: int, vol: float = 0.28,
              fade_pct: float = 0.35) -> np.ndarray:
        """Generate a smooth sine tone as a float32 array ready for sd.play()."""
        n = int(cls._SR * dur_ms / 1000)
        t = np.linspace(0, dur_ms / 1000, n, endpoint=False)
        wave_data = np.sin(2 * np.pi * freq * t).astype(np.float32) * vol

        fade = max(1, int(n * fade_pct))
        ramp_in  = (np.linspace(0.0, 1.0, fade) ** 2).astype(np.float32)
        ramp_out = (np.linspace(1.0, 0.0, fade) ** 2).astype(np.float32)
        wave_data[:fade]  *= ramp_in
        wave_data[-fade:] *= ramp_out
        return wave_data

    @classmethod
    def _concat(cls, *parts: np.ndarray, gap_ms: int = 40) -> np.ndarray:
        """Join multiple tones with a silence gap between them."""
        gap = np.zeros(int(cls._SR * gap_ms / 1000), dtype=np.float32)
        joined: list[np.ndarray] = []
        for i, p in enumerate(parts):
            joined.append(p)
            if i < len(parts) - 1:
                joined.append(gap)
        return np.concatenate(joined)

    @classmethod
    def _build_cache(cls) -> None:
        with cls._cache_lock:
            if cls._cache:
                return
            # startup  — C5 ↗ E5
            cls._cache["startup"] = cls._concat(
                cls._tone(523, 130, 0.22),
                cls._tone(659, 160, 0.22),
                gap_ms=35,
            )
            # start    — A5 single crisp click
            cls._cache["start"] = cls._tone(880, 110, 0.26, fade_pct=0.40)

            # stop     — E5 → G5 smooth confirm
            cls._cache["stop"] = cls._concat(
                cls._tone(659, 110, 0.24),
                cls._tone(784, 140, 0.24),
                gap_ms=30,
            )
            # abort    — C5 ↘ G4
            cls._cache["abort"] = cls._concat(
                cls._tone(523, 100, 0.22),
                cls._tone(392, 110, 0.22),
                gap_ms=35,
            )
            # error    — low double knock
            cls._cache["error"] = cls._concat(
                cls._tone(330, 120, 0.22),
                cls._tone(262, 140, 0.22),
                gap_ms=40,
            )

    # ── Playback ─────────────────────────────────────────────────────────

    @classmethod
    def _play(cls, key: str, blocking: bool = False) -> None:
        """Play a pre-cached sound. Non-blocking by default."""
        cls._build_cache()
        audio = cls._cache.get(key)
        if audio is None:
            return
        def _run():
            try:
                sd.play(audio, samplerate=cls._SR)
                if blocking:
                    sd.wait()
            except Exception:
                pass
        if blocking:
            _run()
        else:
            threading.Thread(target=_run, daemon=True).start()

    # ── Public API ────────────────────────────────────────────────────────

    @classmethod
    def play_startup(cls) -> None:
        cls._play("startup")

    @classmethod
    def play_start(cls) -> None:
        cls._play("start")

    @classmethod
    def play_stop(cls) -> None:
        cls._play("stop")

    @classmethod
    def play_abort(cls) -> None:
        cls._play("abort")

    @classmethod
    def play_error(cls) -> None:
        cls._play("error")

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

    def show_update_notice(self, latest_version: str, download_url: str) -> None:
        """Inject an 'Update available' entry at the top of the tray menu."""
        if not TRAY_AVAILABLE or not self._icon:
            return
        label = f"Update available: v{latest_version}  →  Download"
        def _open_browser(icon, item):  # noqa: ANN001
            webbrowser.open(download_url)
        self._icon.menu = pystray.Menu(
            pystray.MenuItem(label, _open_browser),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("WordScript", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._quit),
        )
        self._icon.title = f"WordScript — {label}"
        self.logger.info("Update notice shown in tray: %s", label)

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
# Update Checker
# ---------------------------------------------------------------------------

try:
    from packaging.version import Version as _Version
    def _parse_version(tag: str) -> _Version:
        """Parse a version tag using PEP 440 (handles pre-release suffixes like -alpha)."""
        clean = tag.lstrip("vV").strip()
        # Normalise hyphen-style pre-release: 1.0.0-alpha → 1.0.0a0
        for alias, pep in (("alpha", "a0"), ("beta", "b0"), ("rc", "rc0")):
            clean = clean.replace(f"-{alias}", alias)
        try:
            return _Version(clean)
        except Exception:
            return _Version("0")
except ImportError:
    # Fallback: plain tuple comparison (no pre-release awareness)
    def _parse_version(tag: str) -> tuple:  # type: ignore[misc]
        clean = tag.lstrip("vV").strip().split("-")[0]
        try:
            return tuple(int(x) for x in clean.split("."))
        except ValueError:
            return (0,)


def _notify_update(latest_tag: str, download_url: str) -> None:
    """Show a native OS toast notification about the update (best-effort, silent on failure)."""
    try:
        from plyer import notification
        notification.notify(
            title="WordScript — Update available",
            message=(
                f"{latest_tag} is available.\n"
                "Open the tray icon menu to download."
            ),
            app_name="WordScript",
            timeout=10,
        )
    except Exception:
        pass


def check_for_update(tray: "Optional[TrayIcon]" = None) -> None:
    """Check GitHub releases for a newer version in a daemon background thread.

    - Silent on network errors / timeouts — never crashes the app.
    - Uses packaging.version for correct semver + pre-release comparison.
    - Shows a native OS toast notification and updates the tray menu.
    """
    def _worker():
        try:
            url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": f"WordScript/{APP_VERSION}"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode())

            latest_tag   = data.get("tag_name", "")   # e.g. "v0.1.1-alpha"
            download_url = data.get("html_url", "")    # GitHub release page URL

            if not latest_tag:
                return

            current = _parse_version(APP_VERSION)
            latest  = _parse_version(latest_tag)

            log = logging.getLogger("UpdateChecker")
            if latest > current:
                log.info("New version available: %s (current: %s)", latest_tag, APP_VERSION)
                dl = download_url or f"https://github.com/{GITHUB_REPO}/releases"
                if tray:
                    tray.show_update_notice(latest_tag.lstrip("vV"), dl)
                _notify_update(latest_tag, dl)
            else:
                log.info("Already up to date (%s).", APP_VERSION)
        except Exception:  # network error, timeout, JSON parse — all silent
            pass

    threading.Thread(target=_worker, daemon=True, name="UpdateChecker").start()


# ---------------------------------------------------------------------------
# Audio Visualizer Overlay + Settings Modal
# ---------------------------------------------------------------------------

def _draw_rounded_rect(canvas, x1, y1, x2, y2, r, **kwargs):
    """Draw a filled rounded rectangle on *canvas*."""
    pts = [
        x1 + r, y1,
        x2 - r, y1,
        x2,     y1,
        x2,     y1 + r,
        x2,     y2 - r,
        x2,     y2,
        x2 - r, y2,
        x1 + r, y2,
        x1,     y2,
        x1,     y2 - r,
        x1,     y1 + r,
        x1,     y1,
    ]
    return canvas.create_polygon(pts, smooth=True, **kwargs)


def _draw_mic_icon(canvas, cx, cy, color="#ffffff", tag="mic"):
    """Draw a simple microphone icon centred at (cx, cy)."""
    bw, bh = 8, 12
    r = bw // 2
    canvas.create_arc(cx - r, cy - bh // 2 - r,
                      cx + r, cy - bh // 2 + r,
                      start=0, extent=180, fill=color, outline="", tags=tag)
    canvas.create_rectangle(cx - r, cy - bh // 2,
                             cx + r, cy + bh // 2,
                             fill=color, outline="", tags=tag)
    canvas.create_arc(cx - r, cy + bh // 2 - r,
                      cx + r, cy + bh // 2 + r,
                      start=180, extent=180, fill=color, outline="", tags=tag)
    sr = 10
    canvas.create_arc(cx - sr, cy - sr // 2,
                      cx + sr, cy + bh // 2 + sr,
                      start=0, extent=-180, style="arc",
                      outline=color, width=2, tags=tag)
    canvas.create_line(cx, cy + bh // 2 + sr,
                       cx, cy + bh // 2 + sr + 4,
                       fill=color, width=2, tags=tag)
    canvas.create_line(cx - 5, cy + bh // 2 + sr + 4,
                       cx + 5, cy + bh // 2 + sr + 4,
                       fill=color, width=2, tags=tag)


def _open_settings_modal(root_tk, config_ref: list, on_saved=None) -> None:
    """Open a tabbed dark settings window — no scroll canvas, no dropdown z-order bugs."""
    try:
        import tkinter as tk
        from tkinter import ttk
        import webbrowser
    except ImportError:
        return

    cfg: "Config" = config_ref[0]
    _no_key = not cfg.groq_api_key

    modal = tk.Toplevel(root_tk)
    modal.title("WordScript – Settings")
    modal.resizable(True, True)
    modal.attributes("-topmost", True)
    modal.attributes("-alpha", 0.96)
    # NOTE: no grab_set() — keeps the visualizer interactive while settings is open

    # ── Palette ───────────────────────────────────────────────────────────
    BG      = "#0c0c0c"
    SURFACE = "#161616"
    FG      = "#d4d4d4"
    FG_DIM  = "#555555"
    ACCENT  = "#ffffff"
    BTN_BG  = "#1c1c1c"
    BORDER  = "#282828"
    GREEN   = "#34d058"
    RED     = "#ff4444"
    TAB_ACT = "#1e1e1e"

    modal.configure(bg=BG)

    # ── Styles ────────────────────────────────────────────────────────────
    style = ttk.Style(modal)
    style.theme_use("clam")
    style.configure("Dark.TFrame",      background=BG)
    style.configure("Dark.TLabel",      background=BG, foreground=FG,
                                        font=("Segoe UI", 10))
    style.configure("Dim.TLabel",       background=BG, foreground=FG_DIM,
                                        font=("Segoe UI", 8))
    style.configure("Head.TLabel",      background=BG, foreground=ACCENT,
                                        font=("Segoe UI", 13, "bold"))
    style.configure("Section.TLabel",   background=BG, foreground="#666666",
                                        font=("Segoe UI", 8, "bold"))
    style.configure("Dark.TCheckbutton", background=BG, foreground=FG,
                    font=("Segoe UI", 10), padding=(4, 2))
    style.map("Dark.TCheckbutton",
              background=[("active", BG)], foreground=[("active", ACCENT)])
    style.configure("Dark.TCombobox",
                    fieldbackground=SURFACE, background=SURFACE,
                    foreground=FG, selectbackground=SURFACE,
                    selectforeground=FG, arrowcolor=FG,
                    borderwidth=0, padding=(8, 5))
    style.map("Dark.TCombobox",
              fieldbackground=[("readonly", SURFACE), ("focus", SURFACE)],
              selectbackground=[("readonly", SURFACE)],
              bordercolor=[("focus", BORDER)])
    style.configure("Dark.TEntry",
                    fieldbackground=SURFACE, foreground=FG,
                    insertcolor=FG, borderwidth=0, padding=(8, 5))
    style.map("Dark.TEntry", bordercolor=[("focus", BORDER)])
    # Notebook (tabs)
    style.configure("Dark.TNotebook",
                    background=BG, borderwidth=0, tabmargins=[0, 0, 0, 0])
    style.configure("Dark.TNotebook.Tab",
                    background=SURFACE, foreground=FG_DIM,
                    font=("Segoe UI", 9), padding=(14, 6),
                    borderwidth=0)
    style.map("Dark.TNotebook.Tab",
              background=[("selected", TAB_ACT)],
              foreground=[("selected", ACCENT)])

    # ── Header ────────────────────────────────────────────────────────────
    header_frame = ttk.Frame(modal, style="Dark.TFrame", padding=(24, 18, 24, 0))
    header_frame.pack(fill="x")
    ttk.Label(header_frame, text="Settings", style="Head.TLabel").pack(side="left")

    # First-launch banner
    if _no_key:
        banner = tk.Frame(modal, bg="#1a1200", padx=16, pady=10)
        banner.pack(fill="x", padx=24, pady=(8, 0))
        tk.Label(banner,
                 text="⚠️  No API key found — enter yours on the API tab to get started.",
                 bg="#1a1200", fg="#ffcc44", font=("Segoe UI", 9)).pack(anchor="w")
        link = tk.Label(banner, text="Get a free key at console.groq.com  ↗",
                        bg="#1a1200", fg="#6eb5ff",
                        font=("Segoe UI", 9, "underline"), cursor="hand2")
        link.pack(anchor="w", pady=(3, 0))
        link.bind("<Button-1>", lambda _e: webbrowser.open("https://console.groq.com/keys"))

    # ── Notebook ──────────────────────────────────────────────────────────
    nb = ttk.Notebook(modal, style="Dark.TNotebook")
    nb.pack(fill="both", expand=True, padx=0, pady=(10, 0))

    # ── Shared helpers that work on any parent frame ───────────────────────
    def _field(parent, r, label, widget_fn):
        ttk.Label(parent, text=label, style="Dark.TLabel").grid(
            row=r, column=0, sticky="w", padx=(0, 20), pady=7)
        w = widget_fn(parent)
        w.grid(row=r, column=1, sticky="ew", pady=7, ipady=1)
        parent.columnconfigure(1, weight=1)
        return w

    def _check(parent, r, label, var):
        ttk.Checkbutton(parent, text=label, variable=var,
                        style="Dark.TCheckbutton").grid(
            row=r, column=0, columnspan=2, sticky="w", pady=5)

    def _sep(parent, r):
        tk.Frame(parent, bg=BORDER, height=1).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=(10, 8))

    def _tab(title):
        f = ttk.Frame(nb, style="Dark.TFrame", padding=(24, 16))
        nb.add(f, text=f"  {title}  ")
        return f

    # ════════════════════════════════════════════════════════════════════
    # TAB 1 — API & Models
    # ════════════════════════════════════════════════════════════════════
    t1 = _tab("API & Models")

    api_key_var = tk.StringVar(value=cfg.groq_api_key or "")
    api_entry = _field(t1, 0, "Groq API Key",
                       lambda p: ttk.Entry(p, textvariable=api_key_var,
                                           show="•", width=36, style="Dark.TEntry"))
    show_key_var = tk.BooleanVar(value=False)
    def _toggle_key():
        api_entry.config(show="" if show_key_var.get() else "•")
    ttk.Checkbutton(t1, text="Show key", variable=show_key_var,
                    command=_toggle_key, style="Dark.TCheckbutton").grid(
        row=1, column=1, sticky="w", pady=(0, 4))

    _sep(t1, 2)

    model_var = tk.StringVar(value=cfg.model)
    _field(t1, 3, "Whisper Model",
           lambda p: ttk.Combobox(p, textvariable=model_var,
                                  values=["whisper-large-v3-turbo",
                                          "whisper-large-v3",
                                          "distil-whisper-large-v3-en"],
                                  state="readonly", width=30, style="Dark.TCombobox"))

    lang_var = tk.StringVar(value=cfg.language if cfg.language else "Auto")
    _field(t1, 4, "Language",
           lambda p: ttk.Combobox(p, textvariable=lang_var,
                                  values=["Auto", "en", "de", "fr", "es",
                                          "it", "pt", "nl", "pl", "ru",
                                          "ja", "ko", "zh"],
                                  state="readonly", width=30, style="Dark.TCombobox"))

    _sep(t1, 5)

    post_var = tk.BooleanVar(value=cfg.post_process)
    _check(t1, 6, "Enable AI post-correction", post_var)

    corr_model_var = tk.StringVar(value=cfg.correction_model)
    _field(t1, 7, "Correction Model",
           lambda p: ttk.Combobox(p, textvariable=corr_model_var,
                                  values=["llama-3.3-70b-versatile",
                                          "llama-3.1-8b-instant",
                                          "mixtral-8x7b-32768",
                                          "gemma2-9b-it"],
                                  state="readonly", width=30, style="Dark.TCombobox"))

    # ════════════════════════════════════════════════════════════════════
    # TAB 2 — Prompt  (big expandable text area)
    # ════════════════════════════════════════════════════════════════════
    t2 = _tab("Prompt")

    ttk.Label(t2, text="Context Prompt", style="Head.TLabel").grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
    ttk.Label(t2,
              text="Optional context fed to Whisper for better accuracy.\n"
                   "List jargon, names, abbreviations or domain-specific words.",
              style="Dim.TLabel").grid(
        row=1, column=0, columnspan=2, sticky="w", pady=(0, 10))

    prompt_frame = tk.Frame(t2, bg=SURFACE, bd=0)
    prompt_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(0, 6))
    t2.rowconfigure(2, weight=1)
    t2.columnconfigure(0, weight=1)
    t2.columnconfigure(1, weight=1)

    prompt_text = tk.Text(
        prompt_frame, bg=SURFACE, fg=FG, insertbackground=FG,
        font=("Segoe UI", 10), relief="flat", bd=0,
        wrap="word", width=46, height=10,
        padx=10, pady=8,
        selectbackground="#333333", selectforeground=FG,
    )
    prompt_sb = ttk.Scrollbar(prompt_frame, orient="vertical",
                               command=prompt_text.yview)
    prompt_text.configure(yscrollcommand=prompt_sb.set)
    prompt_text.pack(side="left", fill="both", expand=True)
    prompt_sb.pack(side="right", fill="y")
    if cfg.prompt:
        prompt_text.insert("1.0", cfg.prompt)

    # ════════════════════════════════════════════════════════════════════
    # TAB 3 — Audio
    # ════════════════════════════════════════════════════════════════════
    t3 = _tab("Audio")

    _devices = ["Default (system microphone)"]
    try:
        for _dev in sd.query_devices():
            if _dev["max_input_channels"] > 0:
                _devices.append(_dev["name"])
    except Exception:
        pass

    _cur_dev = "Default (system microphone)" if not cfg.audio_device else cfg.audio_device
    device_var = tk.StringVar(value=_cur_dev)
    _field(t3, 0, "Input Device",
           lambda p: ttk.Combobox(p, textvariable=device_var,
                                  values=_devices, state="readonly",
                                  width=30, style="Dark.TCombobox"))

    sr_var = tk.StringVar(value=str(cfg.sample_rate))
    _field(t3, 1, "Sample Rate (Hz)",
           lambda p: ttk.Combobox(p, textvariable=sr_var,
                                  values=["16000", "24000", "44100", "48000"],
                                  state="readonly", width=30, style="Dark.TCombobox"))

    max_rec_var = tk.StringVar(value=str(cfg.max_recording_seconds))
    _field(t3, 2, "Max Recording (sec)",
           lambda p: ttk.Entry(p, textvariable=max_rec_var, width=30,
                               style="Dark.TEntry"))

    # ════════════════════════════════════════════════════════════════════
    # TAB 4 — Hotkey & Behavior
    # ════════════════════════════════════════════════════════════════════
    t4 = _tab("Hotkey & Behavior")

    hotkey_var = tk.StringVar(value=cfg.hotkey)
    _field(t4, 0, "Hotkey Combo",
           lambda p: ttk.Entry(p, textvariable=hotkey_var, width=30,
                               style="Dark.TEntry"))
    ttk.Label(t4, text="e.g.  ctrl_l+win  or  ctrl_l+alt_l+space",
              style="Dim.TLabel").grid(row=1, column=0, columnspan=2, sticky="w",
                                       pady=(0, 6))

    mode_var = tk.StringVar(value=cfg.activation_mode)
    _field(t4, 2, "Activation Mode",
           lambda p: ttk.Combobox(p, textvariable=mode_var,
                                  values=["tap", "hold"],
                                  state="readonly", width=30, style="Dark.TCombobox"))
    ttk.Label(t4, text="tap = press once to start, press again to stop\n"
                       "hold = hold key to record, release to stop",
              style="Dim.TLabel").grid(row=3, column=0, columnspan=2, sticky="w",
                                       pady=(0, 8))

    _sep(t4, 4)

    auto_paste_var = tk.BooleanVar(value=cfg.auto_paste)
    _check(t4, 5, "Auto-paste into active window (Ctrl+V)", auto_paste_var)

    play_sounds_var = tk.BooleanVar(value=cfg.play_sounds)
    _check(t4, 6, "Play sound feedback", play_sounds_var)

    tray_var = tk.BooleanVar(value=cfg.show_tray_icon)
    _check(t4, 7, "Show tray icon", tray_var)

    # ════════════════════════════════════════════════════════════════════
    # TAB 5 — AI Assistant (not yet implemented)
    # ════════════════════════════════════════════════════════════════════
    t5 = _tab("AI Assistant")

    ttk.Label(t5, text="Coming soon", style="Head.TLabel").grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))
    ttk.Label(t5,
              text="AI Assistant mode is not yet available in this version.\n\n"
                   "Planned features:\n"
                   "  ·  Ask questions by voice or text\n"
                   "  ·  Screenshot context for visual Q&A\n"
                   "  ·  Switchable backends (Groq, OpenAI, Anthropic, local)\n"
                   "  ·  Configurable tone and response style\n\n"
                   "Settings for this tab will appear here once the feature ships.",
              style="Dim.TLabel").grid(row=1, column=0, columnspan=2, sticky="w")

    # ── Bottom bar: Status + Save/Cancel ─────────────────────────────────
    bottom = ttk.Frame(modal, style="Dark.TFrame", padding=(24, 10, 24, 16))
    bottom.pack(fill="x")

    tk.Frame(modal, bg=BORDER, height=1).pack(fill="x")  # separator line

    status_var = tk.StringVar(value="")
    status_lbl = ttk.Label(bottom, textvariable=status_var, style="Dim.TLabel")
    status_lbl.pack(side="left", padx=(0, 16))

    btn_frame = tk.Frame(bottom, bg=BG)
    btn_frame.pack(side="right")

    def _save():
        cfg.groq_api_key       = api_key_var.get().strip()
        cfg.model              = model_var.get()
        cfg.language           = "" if lang_var.get() == "Auto" else lang_var.get()
        cfg.prompt             = prompt_text.get("1.0", "end").strip()
        cfg.post_process       = post_var.get()
        cfg.correction_model   = corr_model_var.get()
        _dev = device_var.get()
        cfg.audio_device       = "" if _dev.startswith("Default") else _dev
        try:
            cfg.sample_rate    = int(sr_var.get())
        except ValueError:
            pass
        try:
            cfg.max_recording_seconds = max(5, int(max_rec_var.get()))
        except ValueError:
            pass
        cfg.hotkey             = hotkey_var.get().strip()
        cfg.activation_mode    = mode_var.get()
        cfg.auto_paste         = auto_paste_var.get()
        cfg.play_sounds        = play_sounds_var.get()
        cfg.show_tray_icon     = tray_var.get()
        try:
            cfg.save()
            if on_saved:
                on_saved()
            status_var.set("✓  Saved")
            status_lbl.configure(foreground=GREEN)
            modal.after(700, modal.destroy)
        except Exception as exc:
            status_var.set(f"✗  {exc}")
            status_lbl.configure(foreground=RED)

    tk.Button(btn_frame, text="  Cancel  ",
              bg=BTN_BG, fg=FG, activebackground=BORDER,
              font=("Segoe UI", 10), relief="flat", cursor="hand2", bd=0,
              command=modal.destroy, padx=18, pady=7).pack(side="left", padx=(0, 10))

    tk.Button(btn_frame, text="  Save  ",
              bg=ACCENT, fg="#000000", activebackground="#d0d0d0",
              font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2", bd=0,
              command=_save, padx=18, pady=7).pack(side="left")

    # ── Size and centre ───────────────────────────────────────────────────
    modal.update_idletasks()
    mw = modal.winfo_reqwidth()
    mh = modal.winfo_reqheight()
    sw = modal.winfo_screenwidth()
    sh = modal.winfo_screenheight()
    modal.geometry(f"{max(mw, 520)}x{max(mh, 480)}"
                   f"+{(sw - max(mw, 520)) // 2}+{(sh - max(mh, 480)) // 2}")


class VisualizerOverlay:
    """Compact pill-shaped audio visualizer overlay — Liquid Glass dark theme.

    Layout:  [ 🎙  |||||||||||||||||||||||||||   ▾ ]

    – Mic icon on the left (click = mute/unmute toggle, red when muted)
    – Animated waveform bars in the centre (flat when muted)
    – Chevron (▾) on the right → opens Settings modal
    – Draggable via bar area

    The Tk root lives in a dedicated daemon thread.
    show()/hide() toggle visibility without destroying the window.
    """

    _BAR_COUNT  = 24
    _UPDATE_MS  = 33        # ~30 fps
    _W          = 296       # total pill width
    _H          = 52        # total pill height
    _R          = 26        # corner radius (full pill)
    _BG_PILL    = "#0c0c0c" # near-black pill
    _BG_WIN     = "#000001" # transparent key colour (must differ from _BG_PILL)
    _BAR_COLOR  = "#ffffff"
    _BAR_MUTED  = "#333333" # dim bars when muted
    _MIC_COLOR  = "#ffffff"
    _MIC_MUTED  = "#ff3b3b" # red when muted
    _DIV_COLOR  = "#2a2a2a"
    _CHEV_COLOR = "#666666"
    _CHEV_HOV   = "#ffffff"
    _PILL_ALPHA = 0.88

    _MIC_W      = 52
    _CHEV_W     = 44

    def __init__(self):
        self.logger = logging.getLogger("VisualizerOverlay")
        self._level_queue: Optional[queue.Queue] = None
        self._want_visible = False
        self._is_visible   = False
        self._muted        = False             # set by SpeechToTextApp
        self._config_ref: list = []            # [Config]
        self._on_mic_click = None              # callback: () -> None
        self._on_settings_saved = None         # callback: () -> None
        self._want_settings = False            # set True to open modal from any thread
        self._root_ref:   list = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def show(self, level_queue: queue.Queue) -> None:
        self._level_queue = level_queue
        self._want_visible = True

    def hide(self) -> None:
        self._want_visible = False

    def open_settings(self) -> None:
        """Request the settings modal to open from any thread (thread-safe)."""
        self._want_settings = True

    # --- internals (all tkinter calls inside the tk thread) ---

    def _run(self) -> None:
        try:
            import tkinter as tk
        except ImportError:
            self.logger.warning("tkinter not available — visualizer disabled.")
            return

        root = tk.Tk()
        self._root_ref.append(root)

        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", self._PILL_ALPHA)
        root.wm_attributes("-transparentcolor", self._BG_WIN)
        root.configure(bg=self._BG_WIN)

        W, H = self._W, self._H
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        x  = (sw - W) // 2
        y  = sh - H - 90
        root.geometry(f"{W}x{H}+{x}+{y}")

        canvas = tk.Canvas(root, width=W, height=H,
                           bg=self._BG_WIN, highlightthickness=0)
        canvas.pack()

        levels = [0.0] * self._BAR_COUNT

        # ── Draw helpers ───────────────────────────────────────────────────
        def _draw_pill():
            _draw_rounded_rect(canvas, 0, 0, W, H, self._R,
                               fill=self._BG_PILL, outline="")

        def _draw_divider():
            dx = W - self._CHEV_W
            canvas.create_line(dx, 13, dx, H - 13,
                               fill=self._DIV_COLOR, width=1)

        def _draw_chevron():
            cx = W - self._CHEV_W // 2
            cy = H // 2
            s  = 5
            canvas.create_polygon(
                cx - s, cy - 3,
                cx + s, cy - 3,
                cx,     cy + 4,
                fill=self._CHEV_COLOR, outline="",
                tags="chevron",
            )

        def _draw_chrome():
            canvas.delete("all")
            _draw_pill()
            _draw_divider()
            mc = self._MIC_MUTED if self._muted else self._MIC_COLOR
            _draw_mic_icon(canvas, self._MIC_W // 2, H // 2,
                           color=mc, tag="mic")
            # Draw a small slash across mic when muted
            if self._muted:
                mx, my = self._MIC_W // 2, H // 2
                canvas.create_line(mx - 8, my - 10, mx + 8, my + 10,
                                   fill=self._MIC_MUTED, width=2, tags="mic")
            _draw_chevron()

        _draw_chrome()

        # ── Hit zones ─────────────────────────────────────────────────────
        chev_x1 = W - self._CHEV_W
        mic_x2  = self._MIC_W

        # ── Hover ─────────────────────────────────────────────────────────
        def _on_motion(e):
            if chev_x1 <= e.x <= W:
                canvas.itemconfig("chevron", fill=self._CHEV_HOV)
                canvas.config(cursor="hand2")
            elif 0 <= e.x <= mic_x2:
                canvas.config(cursor="hand2")
                canvas.itemconfig("chevron", fill=self._CHEV_COLOR)
            else:
                canvas.itemconfig("chevron", fill=self._CHEV_COLOR)
                canvas.config(cursor="")

        def _on_leave(e):
            canvas.itemconfig("chevron", fill=self._CHEV_COLOR)
            canvas.config(cursor="")

        canvas.bind("<Motion>", _on_motion)
        canvas.bind("<Leave>",  _on_leave)

        # ── Drag + click (unified) ────────────────────────────────────────
        _drag = {"sx": 0, "sy": 0, "wx": 0, "wy": 0, "moved": False}

        def _press(e):
            _drag["sx"]    = e.x
            _drag["sy"]    = e.y
            _drag["wx"]    = e.x_root
            _drag["wy"]    = e.y_root
            _drag["moved"] = False

        def _motion(e):
            if abs(e.x_root - _drag["wx"]) > 4 or abs(e.y_root - _drag["wy"]) > 4:
                _drag["moved"] = True
            if _drag["moved"] and mic_x2 < _drag["sx"] < chev_x1:
                dx = e.x_root - _drag["wx"]
                dy = e.y_root - _drag["wy"]
                _drag["wx"] = e.x_root
                _drag["wy"] = e.y_root
                root.geometry(f"+{root.winfo_x() + dx}+{root.winfo_y() + dy}")

        def _release(e):
            if _drag["moved"]:
                return
            sx = _drag["sx"]
            # Chevron click → settings
            if chev_x1 <= sx <= W and self._config_ref:
                _open_settings_modal(root, self._config_ref,
                                     on_saved=self._on_settings_saved)
            # Mic click → toggle mute
            elif 0 <= sx <= mic_x2:
                if self._on_mic_click:
                    self._on_mic_click()

        canvas.bind("<ButtonPress-1>",   _press)
        canvas.bind("<B1-Motion>",       _motion)
        canvas.bind("<ButtonRelease-1>", _release)

        root.withdraw()

        # ── Bars geometry ─────────────────────────────────────────────────
        bar_x1  = self._MIC_W + 6
        bar_x2  = W - self._CHEV_W - 6
        bar_w_t = bar_x2 - bar_x1
        bar_gap = 2
        bar_w   = (bar_w_t - bar_gap * (self._BAR_COUNT - 1)) / self._BAR_COUNT

        # ── Tick ──────────────────────────────────────────────────────────
        def _tick():
            if self._want_visible and not self._is_visible:
                levels.clear()
                levels.extend([0.0] * self._BAR_COUNT)
                _draw_chrome()
                root.deiconify()
                self._is_visible = True

            elif not self._want_visible and self._is_visible:
                root.withdraw()
                self._is_visible = False

            if self._is_visible and self._level_queue:
                while True:
                    try:
                        levels.append(self._level_queue.get_nowait())
                    except queue.Empty:
                        break
                del levels[: len(levels) - self._BAR_COUNT]

                canvas.delete("all")
                _draw_pill()

                is_muted = self._muted
                bar_col = self._BAR_MUTED if is_muted else self._BAR_COLOR

                # Bars
                max_h = H - 16
                for i, lvl in enumerate(levels):
                    if is_muted:
                        bh = 3  # flat line when muted
                    else:
                        bh = max(3, int(lvl * max_h))
                    bx = bar_x1 + i * (bar_w + bar_gap)
                    by = (H - bh) / 2
                    rr = min(bar_w / 2, bh / 2, 3)
                    _draw_rounded_rect(canvas,
                                       bx, by, bx + bar_w, by + bh, rr,
                                       fill=bar_col, outline="")

                _draw_divider()

                # Mic icon — red with slash when muted, white when active
                mc = self._MIC_MUTED if is_muted else self._MIC_COLOR
                _draw_mic_icon(canvas, self._MIC_W // 2, H // 2,
                               color=mc, tag="mic")
                if is_muted:
                    mx, my = self._MIC_W // 2, H // 2
                    canvas.create_line(mx - 8, my - 10, mx + 8, my + 10,
                                       fill=self._MIC_MUTED, width=2, tags="mic")

                _draw_chevron()

            # Open settings if requested from outside the Tk thread
            if self._want_settings:
                self._want_settings = False
                if self._config_ref:
                    _open_settings_modal(root, self._config_ref,
                                         on_saved=self._on_settings_saved)

            root.after(self._UPDATE_MS, _tick)

        root.after(self._UPDATE_MS, _tick)
        root.mainloop()


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

    def reload_hotkey(self) -> None:
        """Re-parse the hotkey combo after a settings change at runtime."""
        self._hotkey_keys = self._parse_hotkey(self.config.hotkey)
        self.logger.info("Hotkey reloaded: %s", self.config.hotkey)

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
        self.visualizer = VisualizerOverlay()
        self.visualizer._config_ref.append(self.config)   # wire settings modal
        self.visualizer._on_mic_click = self._toggle_mute    # mic click = mute/unmute
        self.visualizer._on_settings_saved = self._reload_after_settings
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

        # Background update check — silent, non-blocking
        check_for_update(tray=self.tray)

        # Play startup sound to indicate ready
        if self.config.play_sounds:
            time.sleep(0.5)  # Brief delay for tray icon to appear
            self.sounds.play_startup()

        # First-launch prompt — open Settings automatically when no API key is set
        if not self.config.groq_api_key:
            self.logger.info("No API key found — opening Settings for first-time setup.")
            time.sleep(1.2)  # give the Tk root a moment to be alive
            self.visualizer.open_settings()

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
        self.visualizer.hide()
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
            self.visualizer.show(self.recorder.level_queue)
            if self.config.play_sounds:
                self.sounds.play_start()
            if self.tray:
                self.tray.set_recording(True)
        except RuntimeError as exc:
            self.logger.error(str(exc))
            if self.config.play_sounds:
                self.sounds.play_error()

    def _toggle_mute(self) -> None:
        """Called when the mic icon in the visualizer is clicked."""
        if not self.recorder.is_recording:
            return
        is_muted = self.recorder.toggle_mute()
        self.visualizer._muted = is_muted
        self.logger.info("Mic %s via visualizer", "muted" if is_muted else "unmuted")

    def _reload_after_settings(self) -> None:
        """Called after the settings modal saves — reloads live components."""
        # Reinit Groq client if API key changed
        self.transcriber.reload_api_key()
        # Re-parse hotkey combo if it changed
        self.hotkeys.reload_hotkey()
        self.logger.info("Settings reloaded: model=%s lang=%s hotkey=%s",
                         self.config.model, self.config.language or "auto",
                         self.config.hotkey)

    def _stop_recording(self) -> None:
        """Called by HotkeyManager when hotkey deactivates."""
        if not self.recorder.is_recording:
            return

        # Always unmute when stopping so next recording starts clean
        self.recorder._muted = False
        self.visualizer._muted = False

        self.visualizer.hide()
        wav_bytes = self.recorder.stop()
        if self.config.play_sounds:
            self.sounds.play_stop()
        if self.tray:
            self.tray.set_recording(False)
        # Reset tap mode so hotkey stays in sync
        self.hotkeys._toggled_on = False

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
        self.visualizer.hide()
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
