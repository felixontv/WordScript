"""Headless WordScript for Tauri sidecar mode.

The Tauri frontend owns all UI (overlay, tray, settings window).
This module owns: global hotkeys, audio capture, transcription, clipboard paste.

Usage:
    python -m wordscript sidecar
"""

import dataclasses
import logging
import os
import sys
import threading
import time
from typing import Optional

import pyperclip

from .config import Config, LOG_FILE
from .constants import APP_VERSION
from .hotkey import HotkeyManager
from .ipc import IPCChannel
from .paster import TextPaster
from .recorder import AudioRecorder
from .sounds import SoundFeedback
from .transcription import TranscriptionService
from .updater import check_for_update


class SidecarApp:
    """Headless WordScript — no Tkinter, no tray, pure JSON IPC."""

    def __init__(self):
        self.config      = Config.load()
        self.logger      = logging.getLogger("Sidecar")
        self.ipc         = IPCChannel()
        self.recorder    = AudioRecorder(self.config)
        self.transcriber = TranscriptionService(self.config)
        self.paster      = TextPaster(self.config)
        self.sounds      = SoundFeedback()
        self.hotkeys     = HotkeyManager(
            self.config,
            on_activate=self._start_recording,
            on_deactivate=self._stop_recording,
            on_abort=self._abort_recording,
        )
        self._running         = True
        self._max_rec_timer: Optional[threading.Timer] = None

    # ── Lifecycle ───────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the sidecar. Blocks until shutdown."""
        self.logger.info("WordScript sidecar v%s starting", APP_VERSION)

        self.hotkeys.start()
        self.ipc.listen(self._handle_command)

        check_for_update(on_update=self._on_update_available)

        if self.config.play_sounds:
            time.sleep(0.3)
            self.sounds.play_startup()

        # Signal readiness to Tauri — send full config snapshot so the
        # frontend can populate the settings UI without a separate read.
        self.ipc.emit(
            "ready",
            version=APP_VERSION,
            config=dataclasses.asdict(self.config),
        )
        self.logger.info("Sidecar ready. Hotkey: %s  Mode: %s",
                         self.config.hotkey, self.config.activation_mode)

        try:
            while self._running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._running = False
        if self.recorder.is_recording:
            self.recorder.stop()
        self.hotkeys.stop()
        self.ipc.emit("shutdown")
        os._exit(0)

    # ── IPC command dispatcher ──────────────────────────────────────────

    def _handle_command(self, cmd: dict) -> None:
        action = cmd.get("cmd", "")
        self.logger.debug("IPC ← %s", action)

        # save_config carries a payload — handle separately
        if action == "save_config":
            config_data = cmd.get("config", {})
            threading.Thread(
                target=self._save_config, args=(config_data,), daemon=True
            ).start()
            return

        handlers = {
            "start_recording": self._start_recording,
            "stop_recording":  self._stop_recording,
            "abort_recording": self._abort_recording,
            "toggle_mute":     self._toggle_mute,
            "reload_config":   self._reload_config,
            "shutdown":        self.shutdown,
            "open_settings":   lambda: None,  # Tauri owns the settings window
            "pause_hotkey":    self.hotkeys.pause,
            "resume_hotkey":   self.hotkeys.resume,
        }
        handler = handlers.get(action)
        if handler:
            threading.Thread(target=handler, daemon=True).start()
        else:
            self.logger.warning("Unknown IPC command: '%s'", action)

    # ── Recording ───────────────────────────────────────────────────────

    def _start_recording(self) -> None:
        if not self.config.groq_api_key and self.config.backend == "groq":
            self.ipc.emit("error", message="No Groq API key — open Settings to add yours.")
            if self.config.play_sounds:
                self.sounds.play_error()
            return
        try:
            self.recorder.start()
            self.ipc.emit("recording_started")
            if self.config.play_sounds:
                self.sounds.play_start()

            # Hard time limit
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

            # Stream audio levels to overlay bars
            threading.Thread(
                target=self._stream_audio_levels, daemon=True
            ).start()

        except RuntimeError as exc:
            self.ipc.emit("error", message=str(exc))
            if self.config.play_sounds:
                self.sounds.play_error()

    def _stop_recording(self) -> None:
        if not self.recorder.is_recording:
            return
        self._cancel_timers()
        self.recorder._muted = False
        self.hotkeys._toggled_on = False

        wav_bytes = self.recorder.stop()
        if self.config.play_sounds:
            self.sounds.play_stop()
        self.ipc.emit("recording_stopped")

        if not wav_bytes:
            self.ipc.emit("empty")
            return

        self.ipc.emit("processing")
        threading.Thread(
            target=self._transcribe_and_paste,
            args=(wav_bytes,),
            daemon=True,
        ).start()

    def _abort_recording(self) -> None:
        if not self.recorder.is_recording:
            return
        self._cancel_timers()
        self.recorder.stop()
        self.hotkeys._toggled_on = False
        if self.config.play_sounds:
            self.sounds.play_abort()
        self.ipc.emit("recording_stopped")
        self.ipc.emit("empty")  # tells Rust to hide the overlay

    def _toggle_mute(self) -> None:
        if not self.recorder.is_recording:
            return
        muted = self.recorder.toggle_mute()
        self.ipc.emit("muted", muted=muted)

    # ── Timers ──────────────────────────────────────────────────────────

    def _cancel_timers(self) -> None:
        if self._max_rec_timer:
            self._max_rec_timer.cancel()
            self._max_rec_timer = None

    def _stream_audio_levels(self) -> None:
        """Drain level_queue and emit audio_level ~12×/s while recording."""
        while self.recorder.is_recording:
            level = 0.0
            try:
                while not self.recorder.level_queue.empty():
                    level = self.recorder.level_queue.get_nowait()
            except Exception:
                pass
            self.ipc.emit("audio_level", level=round(level, 3))
            time.sleep(0.08)

    def _monitor_silence(self) -> None:
        """Auto-stop after silence_timeout_seconds of quiet mic input."""
        time.sleep(3.0)  # grace: don't trigger immediately at start
        timeout = self.config.silence_timeout_seconds
        while self.recorder.is_recording:
            if self.recorder.silence_seconds >= timeout:
                self.logger.info("Silence timeout (%.0fs) — auto-stopping.", timeout)
                self._stop_recording()
                return
            time.sleep(0.25)

    # ── Transcription ────────────────────────────────────────────────────

    def _transcribe_and_paste(self, wav_bytes: bytes) -> None:
        text = ""
        try:
            text = self.transcriber.transcribe(wav_bytes)
            if not text:
                self.ipc.emit("empty")
                return
            if text.startswith("["):
                self.ipc.emit("error", message=text)
                if self.config.play_sounds:
                    self.sounds.play_error()
                return
            paste_ok = True
            try:
                self.paster.paste(text)
            except Exception:
                paste_ok = False

            # Always emit the transcription so the user can see the text.
            self.ipc.emit("transcription", text=text, corrected=False)

            if not paste_ok:
                if self.config.play_sounds:
                    self.sounds.play_error()
                self.ipc.emit("error", message=(
                    "Clipboard nicht verfügbar — Text transkribiert, aber nicht eingefügt. "
                    "Auf Wayland: sudo pacman -S wl-clipboard installieren, "
                    "dann die App neu starten."
                ))
                return  # skip LLM correction since paste failed

        except Exception as exc:
            self.ipc.emit("error", message=str(exc))
            if self.config.play_sounds:
                self.sounds.play_error()
            return

        # Stage 2: LLM post-correction — silently update clipboard
        if self.config.post_process and text:
            try:
                word_count   = len(text.split())
                corr_model   = (
                    "llama-3.3-70b-versatile" if word_count > 300
                    else self.config.correction_model
                )
                corr_timeout = 30.0 if word_count > 300 else 8.0
                corrected = self.transcriber.correct(
                    text, model=corr_model, timeout=corr_timeout
                )
                if corrected and corrected != text:
                    clip_text = corrected + (" " if corrected[-1] in ".!?" else "")
                    pyperclip.copy(clip_text)
                    self.ipc.emit("transcription", text=corrected, corrected=True)
            except Exception as exc:
                self.logger.warning("LLM correction failed (non-fatal): %s", exc)

    # ── Config reload ───────────────────────────────────────────────────

    def _save_config(self, config_data: dict) -> None:
        """Write updated config from Tauri settings UI and hot-reload."""
        try:
            cfg = self.config
            for key, value in config_data.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
            cfg.save()
            self._reload_config()
        except Exception as exc:
            self.logger.error("Failed to save config: %s", exc)
            self.ipc.emit("error", message=f"Config save failed: {exc}")

    def _reload_config(self) -> None:
        """Hot-reload config.json at runtime (called after Tauri saves settings)."""
        self.config = Config.load()
        self.transcriber.config = self.config
        self.transcriber.reload_api_key()
        self.hotkeys.config = self.config
        self.hotkeys.reload_hotkey()
        self.hotkeys.reload_abort_hotkey()
        self.ipc.emit(
            "ready",
            version=APP_VERSION,
            config=dataclasses.asdict(self.config),
        )
        self.logger.info("Config reloaded.")

    def _on_update_available(self, version: str, url: str) -> None:
        self.ipc.emit("update_available", version=version, url=url)
