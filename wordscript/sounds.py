"""Sound feedback — pre-generated tones played via sounddevice."""

import threading

import numpy as np
import sounddevice as sd


class SoundFeedback:
    """Plays smooth notification sounds — pre-generated at startup."""

    _SR = 44100

    _cache: dict = {}
    _cache_lock = threading.Lock()

    @classmethod
    def _tone(cls, freq: float, dur_ms: int, vol: float = 0.28,
               fade_pct: float = 0.35) -> np.ndarray:
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
        gap = np.zeros(int(cls._SR * gap_ms / 1000), dtype=np.float32)
        joined: list = []
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
            cls._cache["startup"] = cls._concat(
                cls._tone(523, 130, 0.22), cls._tone(659, 160, 0.22), gap_ms=35
            )
            cls._cache["start"] = cls._tone(880, 110, 0.26, fade_pct=0.40)
            cls._cache["stop"] = cls._concat(
                cls._tone(659, 110, 0.24), cls._tone(784, 140, 0.24), gap_ms=30
            )
            cls._cache["abort"] = cls._concat(
                cls._tone(523, 100, 0.22), cls._tone(392, 110, 0.22), gap_ms=35
            )
            cls._cache["error"] = cls._concat(
                cls._tone(330, 120, 0.22), cls._tone(262, 140, 0.22), gap_ms=40
            )

    @classmethod
    def _play(cls, key: str, blocking: bool = False) -> None:
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
