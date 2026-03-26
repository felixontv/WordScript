"""
speech_to_text.py — PyInstaller entry-point shim
=================================================
All logic lives in the `wordscript` package.
This file exists only so PyInstaller's WordScript.spec has a single
top-level script to analyse.  It delegates entirely to __main__.py so
that CLI arguments (including `sidecar`) are handled correctly.
"""

# Force PyInstaller to include the full package by touching key imports.
# These are never called here — the imports are for the dependency scanner only.
from wordscript.app import main as _app_main, SpeechToTextApp  # noqa: F401
from wordscript.sidecar import SidecarApp                      # noqa: F401
from wordscript.ipc import IPCChannel                          # noqa: F401
from wordscript.constants import (                             # noqa: F401
    APP_VERSION, GITHUB_REPO, _IS_WAYLAND, TRAY_AVAILABLE,
)
from wordscript.config import Config, USER_DATA_DIR, CONFIG_FILE, LOG_FILE  # noqa: F401
from wordscript.recorder import AudioRecorder                  # noqa: F401
from wordscript.transcription import TranscriptionService      # noqa: F401
from wordscript.paster import TextPaster                       # noqa: F401
from wordscript.sounds import SoundFeedback                    # noqa: F401
from wordscript.tray import TrayIcon                           # noqa: F401
from wordscript.updater import check_for_update                # noqa: F401
from wordscript.hotkey import HotkeyManager                    # noqa: F401
from wordscript.ui.overlay import VisualizerOverlay            # noqa: F401
from wordscript.ui.settings import open_settings_modal as _open_settings_modal  # noqa: F401

if __name__ == "__main__":
    # Delegate to __main__.py so `wordscript-sidecar.exe sidecar` works correctly.
    from wordscript.__main__ import main
    main()
