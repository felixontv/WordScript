"""Clipboard write and auto-paste via pynput."""

import logging
import sys
import time

import pyperclip
from pynput import keyboard

from .config import Config
from .constants import _IS_WAYLAND


class TextPaster:
    """Copies text to the clipboard and optionally simulates Ctrl+V."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("TextPaster")
        self._kb = keyboard.Controller()

    def paste(self, text: str) -> None:
        """Put text on the clipboard and optionally auto-paste via Ctrl+V."""
        if not text:
            return

        if text and text[-1] in ".!?":
            text = text + " "

        for attempt in range(3):
            try:
                pyperclip.copy(text)
                break
            except Exception as exc:
                if attempt < 2:
                    time.sleep(0.05)
                else:
                    self.logger.error(
                        "Failed to copy to clipboard after 3 attempts: %s", exc
                    )
                    raise  # propagate so sidecar can show user-visible error
        self.logger.info("Text copied to clipboard.")

        if self.config.auto_paste:
            if _IS_WAYLAND:
                self.logger.info(
                    "Wayland: text copied to clipboard — press Ctrl+V to paste."
                )
            else:
                time.sleep(0.25)
                paste_key = (
                    keyboard.Key.cmd if sys.platform == "darwin" else keyboard.Key.ctrl
                )
                with self._kb.pressed(paste_key):
                    self._kb.press("v")
                    self._kb.release("v")
                self.logger.info("Auto-pasted into active window.")
