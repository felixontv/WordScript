"""Configuration dataclass and user-data directory resolution."""

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _resolve_user_data_dir() -> Path:
    """Return the per-user config directory, following OS conventions.

    Windows : %APPDATA%\\WordScript
    macOS   : ~/Library/Application Support/WordScript
    Linux   : $XDG_CONFIG_HOME/WordScript  (defaults to ~/.config/WordScript)
    Dev run : next to the script  (convenient, no side-effects)
    """
    if not getattr(sys, "frozen", False):
        return Path(__file__).parent.parent

    plat = sys.platform
    if plat == "win32":
        base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    elif plat == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME", "")
        base = Path(xdg) if xdg else Path.home() / ".config"

    return base / "WordScript"


USER_DATA_DIR = _resolve_user_data_dir()
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = USER_DATA_DIR / "config.json"
LOG_FILE    = USER_DATA_DIR / "wordscript.log"

# One-time migration: copy config from next to exe on first frozen run
if getattr(sys, "frozen", False) and not CONFIG_FILE.exists():
    _old_config = Path(sys.executable).parent / "config.json"
    if _old_config.exists():
        import shutil
        shutil.copy2(_old_config, CONFIG_FILE)


@dataclass
class Config:
    """Application configuration loaded from config.json."""

    groq_api_key: str = ""
    model: str = "whisper-large-v3-turbo"
    language: str = ""
    prompt: str = ""

    post_process: bool = True
    correction_model: str = "llama-3.1-8b-instant"
    filter_fillers: bool = True
    professionalize: bool = False

    backend: str = "groq"
    local_model: str = "base"

    hotkey: str = (
        "ctrl_l+cmd"  if sys.platform == "darwin"
        else "ctrl_l+win"   if sys.platform == "win32"
        else "ctrl_l+f9"
    )
    activation_mode: str = "tap"

    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"
    audio_device: str = ""
    max_recording_seconds: int = 720
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
