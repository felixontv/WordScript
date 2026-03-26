"""GitHub release update checker."""

import json
import logging
import threading
import urllib.request
from typing import TYPE_CHECKING, Optional

from .constants import APP_VERSION, GITHUB_REPO

if TYPE_CHECKING:
    from .tray import TrayIcon

try:
    from packaging.version import Version as _Version

    def _parse_version(tag: str) -> _Version:
        clean = tag.lstrip("vV").strip()
        for alias, pep in (("alpha", "a0"), ("beta", "b0"), ("rc", "rc0")):
            clean = clean.replace(f"-{alias}", alias)
        try:
            return _Version(clean)
        except Exception:
            return _Version("0")

except ImportError:
    def _parse_version(tag: str) -> tuple:  # type: ignore[misc]
        clean = tag.lstrip("vV").strip().split("-")[0]
        try:
            return tuple(int(x) for x in clean.split("."))
        except ValueError:
            return (0,)


def _notify_update(latest_tag: str, download_url: str) -> None:
    """Show a native OS toast notification (best-effort, silent on failure)."""
    try:
        from plyer import notification
        notification.notify(
            title="WordScript — Update available",
            message=f"{latest_tag} is available.\nOpen the tray icon menu to download.",
            app_name="WordScript",
            timeout=10,
        )
    except Exception:
        pass


def check_for_update(
    tray: "Optional[TrayIcon]" = None,
    on_update=None,
) -> None:
    """Check GitHub releases for a newer version in a daemon background thread.

    Args:
        tray:      Optional TrayIcon — shows update notice in system tray.
        on_update: Optional callable(version: str, url: str) — called when a
                   newer version is found. Used by the Tauri sidecar to emit
                   an IPC event instead of modifying the tray.
    """

    def _worker():
        try:
            url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
            req = urllib.request.Request(
                url, headers={"User-Agent": f"WordScript/{APP_VERSION}"}
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode())

            latest_tag   = data.get("tag_name", "")
            download_url = data.get("html_url", "")
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
                if on_update:
                    on_update(latest_tag.lstrip("vV"), dl)
                _notify_update(latest_tag, dl)
            else:
                log.info("Already up to date (%s).", APP_VERSION)
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True, name="UpdateChecker").start()
