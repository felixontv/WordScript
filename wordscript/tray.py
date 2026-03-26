"""System tray icon."""

import logging
import sys
import threading
import webbrowser
from typing import Optional

from .constants import TRAY_AVAILABLE

if TRAY_AVAILABLE:
    import pystray
    from PIL import Image, ImageDraw, ImageFont


class TrayIcon:
    """Optional system tray icon with status indication and quit option."""

    def __init__(self, on_quit_callback, on_settings_callback=None):
        self._on_quit = on_quit_callback
        self._on_settings = on_settings_callback
        self._icon: Optional["pystray.Icon"] = None
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
                pystray.MenuItem("Settings", self._open_settings),
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
            self._icon.title = (
                "WordScript (Recording...)" if recording else "WordScript (Idle)"
            )

    def stop(self) -> None:
        if self._icon:
            self._icon.stop()

    def show_update_notice(self, latest_version: str, download_url: str) -> None:
        if not TRAY_AVAILABLE or not self._icon:
            return
        label = f"Update available: v{latest_version}  →  Download"

        def _open_browser(icon, item):
            webbrowser.open(download_url)

        self._icon.menu = pystray.Menu(
            pystray.MenuItem(label, _open_browser),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("WordScript", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Settings", self._open_settings),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._quit),
        )
        self._icon.title = f"WordScript — {label}"
        self.logger.info("Update notice shown in tray: %s", label)

    def _open_settings(self, icon, item) -> None:
        if self._on_settings:
            self._on_settings()

    def _quit(self, icon, item) -> None:
        self._on_quit()

    @staticmethod
    def _create_image(recording: bool) -> "Image.Image":
        size = 64
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        color = (220, 40, 40, 255) if recording else (40, 180, 60, 255)
        draw.ellipse([4, 4, size - 4, size - 4], fill=color)
        font = TrayIcon._load_font(16)
        draw.text(
            (size // 2, size // 2), "STT",
            fill=(255, 255, 255, 255), anchor="mm", font=font,
        )
        return img

    @staticmethod
    def _load_font(size: int = 16) -> "ImageFont.FreeTypeFont":
        if sys.platform == "win32":
            candidates = ["arial.ttf", r"C:\Windows\Fonts\arial.ttf"]
        elif sys.platform == "darwin":
            candidates = [
                "/System/Library/Fonts/Helvetica.ttc",
                "/Library/Fonts/Arial.ttf",
            ]
        else:
            candidates = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            ]
        for path in candidates:
            try:
                return ImageFont.truetype(path, size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default()
