"""WordScript — lightweight speech-to-text desktop utility."""

from .app import main
from .sidecar import SidecarApp
from .ipc import IPCChannel

__all__ = ["main", "SidecarApp", "IPCChannel"]
