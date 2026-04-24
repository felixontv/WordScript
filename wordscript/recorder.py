"""Audio recording via sounddevice."""

import io
import logging
import queue
import sys
import threading
import time
import wave
from typing import List, Optional

import numpy as np
import sounddevice as sd

from .config import Config


class AudioRecorder:
    """Records audio from the default microphone using sounddevice."""

    _VOICE_THRESHOLD = 0.02  # normalized peak above which audio counts as "voice"

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("AudioRecorder")
        self._frames: List[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None
        self._recording = False
        self._muted = False
        self._lock = threading.Lock()
        self.level_queue: queue.Queue = queue.Queue(maxsize=64)
        self._last_voice_time: float = 0.0

    @property
    def silence_seconds(self) -> float:
        """Seconds elapsed since last voice activity above threshold."""
        if not self._recording or self._last_voice_time == 0.0:
            return 0.0
        return time.perf_counter() - self._last_voice_time

    @property
    def muted(self) -> bool:
        return self._muted

    def toggle_mute(self) -> bool:
        """Toggle mute state. Returns True if now muted."""
        self._muted = not self._muted
        self.logger.info("Microphone %s", "muted" if self._muted else "unmuted")
        return self._muted

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
            while not self.level_queue.empty():
                try:
                    self.level_queue.get_nowait()
                except queue.Empty:
                    break
            self._last_voice_time = time.perf_counter()  # grace: treat start as voice
            self._recording = True

        self.logger.info("Recording started.")
        try:
            device = None
            if self.config.audio_device:
                devices = sd.query_devices()
                for i, dev in enumerate(devices):
                    if (self.config.audio_device.lower() in dev["name"].lower()
                            and dev["max_input_channels"] > 0):
                        device = i
                        self.logger.info("Using audio device: %s", dev["name"])
                        break
                if device is None:
                    self.logger.warning(
                        "Device '%s' not found, using default", self.config.audio_device
                    )

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
            if sys.platform == "linux":
                hint = (
                    "Check that PulseAudio or PipeWire is running and your user is in "
                    "the 'audio' group. Try: pulseaudio --start  OR  systemctl --user start pipewire"
                )
            elif sys.platform == "darwin":
                hint = (
                    "macOS microphone access may be blocked. Open System Settings → "
                    "Privacy & Security → Microphone and enable WordScript."
                )
            else:
                hint = "Check that the microphone is connected and not in use by another app."
            self.logger.error("Failed to open audio device: %s — %s", exc, hint)
            raise RuntimeError(f"No audio input device found. {hint}") from exc

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

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            self.logger.debug("Audio callback status: %s", status)
        if self._recording:
            if self._muted:
                self._frames.append(np.zeros_like(indata))
                try:
                    self.level_queue.put_nowait(0.0)
                except queue.Full:
                    pass
            else:
                self._frames.append(indata.copy())
                peak = float(np.max(np.abs(indata.astype(np.float32))))
                if self.config.dtype == "int16":
                    peak /= 32767.0
                if peak > self._VOICE_THRESHOLD:
                    self._last_voice_time = time.perf_counter()
                try:
                    self.level_queue.put_nowait(min(1.0, peak))
                except queue.Full:
                    pass

    def _build_wav(self) -> bytes:
        if not self._frames:
            return b""
        audio_data = np.concatenate(self._frames, axis=0)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.config.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.config.sample_rate)
            wf.writeframes(audio_data.tobytes())
        return buf.getvalue()
