"""Main application orchestrator and entry point."""

import logging
import os
import socket
import sys
import threading
import time
from typing import Optional

import pyperclip
import sounddevice as sd

from .config import Config, LOG_FILE
from .constants import TRAY_AVAILABLE, _IS_WAYLAND
from .recorder import AudioRecorder
from .transcription import TranscriptionService
from .paster import TextPaster
from .sounds import SoundFeedback
from .tray import TrayIcon
from .updater import check_for_update
from .hotkey import HotkeyManager
from .ui.overlay import VisualizerOverlay


class SpeechToTextApp:
    """
    Main orchestrator: ties together recording, transcription, hotkeys,
    clipboard pasting, tray icon, and sound feedback.
    """

    def __init__(self):
        self.config = Config.load()
        self.logger = logging.getLogger("App")

        self.recorder    = AudioRecorder(self.config)
        self.transcriber = TranscriptionService(self.config)
        self.paster      = TextPaster(self.config)
        self.sounds      = SoundFeedback()
        self.visualizer  = VisualizerOverlay()
        self.visualizer._config_ref.append(self.config)
        self.visualizer._on_mic_click       = self._toggle_mute
        self.visualizer._on_settings_saved  = self._reload_after_settings
        self.hotkeys = HotkeyManager(
            self.config,
            on_activate=self._start_recording,
            on_deactivate=self._stop_recording,
            on_abort=self._abort_recording,
        )
        self.tray: Optional[TrayIcon] = None
        self._running = True
        self._max_rec_timer: Optional[threading.Timer] = None

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
            self.tray = TrayIcon(
                on_quit_callback=self.shutdown,
                on_settings_callback=self.visualizer.open_settings,
            )
            self.tray.start()

        check_for_update(tray=self.tray)

        if self.config.play_sounds:
            time.sleep(0.5)
            self.sounds.play_startup()

        if not self.config.groq_api_key:
            self.logger.info("No API key found — opening Settings for first-time setup.")
            self.visualizer._tk_ready.wait(timeout=8.0)
            self.visualizer.open_settings()

        try:
            while self._running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user.")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self.logger.info("Shutting down...")
        self._running = False
        self.visualizer.hide()
        if self.recorder.is_recording:
            self.recorder.stop()
        self.hotkeys.stop()
        if self.tray:
            self.tray.stop()
        self.logger.info("Goodbye.")
        os._exit(0)

    # --- recording callbacks ---

    def _start_recording(self) -> None:
        if not self.config.groq_api_key and self.config.backend == "groq":
            self.logger.error("No Groq API key configured — opening Settings.")
            if self.config.play_sounds:
                self.sounds.play_error()
            self.visualizer.open_settings()
            return
        try:
            self.recorder.start()
            self.visualizer.show(self.recorder.level_queue)
            if self.config.play_sounds:
                self.sounds.play_start()
            if self.tray:
                self.tray.set_recording(True)

            # Auto-stop at max recording time
            self._max_rec_timer = threading.Timer(
                self.config.max_recording_seconds, self._stop_recording
            )
            self._max_rec_timer.daemon = True
            self._max_rec_timer.start()

            # Silence auto-stop
            if self.config.silence_timeout_seconds > 0:
                threading.Thread(
                    target=self._monitor_silence, daemon=True
                ).start()

        except RuntimeError as exc:
            self.logger.error(str(exc))
            if self.config.play_sounds:
                self.sounds.play_error()

    def _toggle_mute(self) -> None:
        if not self.recorder.is_recording:
            return
        is_muted = self.recorder.toggle_mute()
        self.visualizer._muted = is_muted
        self.logger.info("Mic %s via visualizer", "muted" if is_muted else "unmuted")

    def _reload_after_settings(self) -> None:
        self.transcriber.reload_api_key()
        self.hotkeys.reload_hotkey()
        self.logger.info(
            "Settings reloaded: model=%s lang=%s hotkey=%s",
            self.config.model,
            self.config.language or "auto",
            self.config.hotkey,
        )

    def _cancel_timers(self) -> None:
        if self._max_rec_timer:
            self._max_rec_timer.cancel()
            self._max_rec_timer = None

    def _monitor_silence(self) -> None:
        """Auto-stop after silence_timeout_seconds of quiet microphone input."""
        grace = 3.0  # wait at least 3s before watching for silence
        time.sleep(grace)
        timeout = self.config.silence_timeout_seconds
        while self.recorder.is_recording:
            if self.recorder.silence_seconds >= timeout:
                self.logger.info(
                    "Silence timeout (%.0fs) — auto-stopping.", timeout
                )
                self._stop_recording()
                return
            time.sleep(0.25)

    def _stop_recording(self) -> None:
        if not self.recorder.is_recording:
            return

        self._cancel_timers()
        self.recorder._muted    = False
        self.visualizer._muted  = False

        wav_bytes = self.recorder.stop()
        if self.config.play_sounds:
            self.sounds.play_stop()
        if self.tray:
            self.tray.set_recording(False)
        self.hotkeys._toggled_on = False

        if not wav_bytes:
            self.logger.warning("No audio captured.")
            self.visualizer.hide()
            return

        self.visualizer.show_processing()
        threading.Thread(
            target=self._transcribe_and_paste,
            args=(wav_bytes,),
            daemon=True,
        ).start()

    def _abort_recording(self) -> None:
        if not self.recorder.is_recording:
            return
        self._cancel_timers()
        self.logger.info("Recording aborted by user")
        self.visualizer.hide()
        self.recorder.stop()
        if self.config.play_sounds:
            self.sounds.play_abort()
        if self.tray:
            self.tray.set_recording(False)
        self.hotkeys._toggled_on = False

    def _transcribe_and_paste(self, wav_bytes: bytes) -> None:
        """Transcribe in two stages: Whisper pastes immediately, LLM updates clipboard."""
        text = ""
        try:
            text = self.transcriber.transcribe(wav_bytes)
            if not text:
                self.logger.info(
                    "Transcription returned empty (no speech or hallucination filtered)."
                )
                return
            if text.startswith("["):
                self.logger.warning("Transcription error (not pasting): %s", text)
                if self.config.play_sounds:
                    self.sounds.play_error()
                return
            try:
                self.paster.paste(text)
            except Exception as paste_exc:
                self.logger.error("Failed to paste text: %s", paste_exc, exc_info=True)
                if self.config.play_sounds:
                    self.sounds.play_error()
                return
        except Exception as exc:
            self.logger.error("Unexpected error in transcription: %s", exc, exc_info=True)
            if self.config.play_sounds:
                self.sounds.play_error()
            return
        finally:
            self.visualizer.hide()

        # Stage 2: LLM correction — silently update clipboard only
        if self.config.post_process and text:
            try:
                word_count   = len(text.split())
                corr_model   = (
                    "llama-3.3-70b-versatile" if word_count > 300
                    else self.config.correction_model
                )
                corr_timeout = 30.0 if word_count > 300 else 8.0
                if word_count > 300:
                    self.logger.info(
                        "Long transcript (%d words) → using 70B correction model", word_count
                    )
                corrected = self.transcriber.correct(text, model=corr_model, timeout=corr_timeout)
                if corrected and corrected != text:
                    clip_text = corrected + (" " if corrected[-1] in ".!?" else "")
                    pyperclip.copy(clip_text)
                    self.logger.info("Clipboard silently updated with LLM-corrected text.")
            except Exception as exc:
                self.logger.warning("LLM correction failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Entry point helpers
# ---------------------------------------------------------------------------

_singleton_socket = None


def _setup_logging(log_level: str) -> None:
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    try:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        print(f"[OK] Logging to: {LOG_FILE}")
    except Exception as exc:
        print(f"[WARNING] Could not create log file: {exc}")

    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)


def _warn_platform_prerequisites(logger: logging.Logger) -> None:
    import shutil

    if sys.platform == "linux":
        if _IS_WAYLAND:
            logger.warning(
                "Wayland session detected. Global hotkeys require XWayland. "
                "Auto-paste via Ctrl+V is DISABLED — text goes to clipboard only."
            )
        has_clipboard = any(shutil.which(c) for c in ("xclip", "xsel", "wl-copy"))
        if not has_clipboard:
            logger.warning(
                "No clipboard backend found — pyperclip will fail. Install one:\n"
                "  X11    : sudo apt install xclip\n"
                "  Wayland: sudo apt install wl-clipboard"
            )
            print("[WARNING] No clipboard backend (xclip / wl-clipboard) found.")
    elif sys.platform == "darwin":
        logger.info(
            "macOS: If the hotkey doesn't respond, grant Input Monitoring permission: "
            "System Settings → Privacy & Security → Input Monitoring → enable WordScript."
        )
        logger.info(
            "macOS: If the microphone doesn't work, grant Microphone permission: "
            "System Settings → Privacy & Security → Microphone → enable WordScript."
        )


def main() -> None:
    global _singleton_socket

    try:
        _singleton_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _singleton_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _singleton_socket.bind(("127.0.0.1", 48127))
    except OSError:
        print("[INFO] WordScript is already running.")
        sys.exit(0)

    config = Config.load()
    _setup_logging(config.log_level)

    logger = logging.getLogger("Main")
    logger.info("=" * 60)
    logger.info("WordScript starting...")
    logger.info(
        "Model: %s | Language: %s | Sample Rate: %d Hz",
        config.model, config.language or "auto", config.sample_rate,
    )

    _warn_platform_prerequisites(logger)

    try:
        default_input = sd.query_devices(kind="input")
        logger.info("Audio input: %s", default_input["name"])
        print(f"[OK] Default input device: {default_input['name']}")
    except Exception as exc:
        logger.error("No audio input device: %s", exc)
        if sys.platform == "linux":
            print("[ERROR] No audio input device. Try: pulseaudio --start")
        elif sys.platform == "darwin":
            print("[ERROR] No audio input. Allow microphone: System Settings → Privacy → Microphone")
        else:
            print(f"[ERROR] No audio input device found: {exc}")
        sys.exit(1)

    SpeechToTextApp().run()
