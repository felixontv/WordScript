"""App-wide constants and optional-dependency availability flags."""

import os
import sys

APP_VERSION = "0.2.0-alpha"
GITHUB_REPO = "felixontv/WordScript"

# True when running inside a Wayland session on Linux.
# pynput cannot inject keyboard events to foreign windows on Wayland, so
# auto-paste via simulated Ctrl+V is disabled; text goes to clipboard only.
_IS_WAYLAND = (
    sys.platform == "linux"
    and bool(
        os.environ.get("WAYLAND_DISPLAY")
        or os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
    )
)

try:
    import pystray  # noqa: F401
    from PIL import Image, ImageDraw, ImageFont  # noqa: F401
    TRAY_AVAILABLE = True
except ImportError:
    TRAY_AVAILABLE = False

try:
    from faster_whisper import WhisperModel  # noqa: F401
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
