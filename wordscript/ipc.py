"""JSON IPC channel for Tauri sidecar communication.

Protocol (newline-delimited JSON):
  Python  →  Tauri  (stdout): event objects
  Tauri   →  Python (stdin):  command objects

Events emitted by Python:
  {"event": "ready",            "version": "...", "config": {...}}
  {"event": "recording_started"}
  {"event": "recording_stopped"}
  {"event": "processing"}
  {"event": "transcription",    "text": "...", "corrected": false}
  {"event": "empty"}
  {"event": "muted",            "muted": true}
  {"event": "error",            "message": "..."}
  {"event": "update_available", "version": "...", "url": "..."}
  {"event": "shutdown"}

Commands sent by Tauri:
  {"cmd": "start_recording"}
  {"cmd": "stop_recording"}
  {"cmd": "abort_recording"}
  {"cmd": "toggle_mute"}
  {"cmd": "reload_config"}
  {"cmd": "open_settings"}
  {"cmd": "shutdown"}
"""

import json
import sys
import threading
from typing import Callable


class IPCChannel:
    """Bidirectional JSON IPC over stdout / stdin."""

    def emit(self, event: str, **data) -> None:
        """Send an event to the Tauri frontend. Thread-safe."""
        msg = {"event": event, **data}
        # flush=True is critical — Tauri reads stdout line-by-line
        print(json.dumps(msg, ensure_ascii=False), flush=True)

    def listen(self, on_command: Callable[[dict], None]) -> None:
        """Start a daemon thread that reads JSON commands from stdin."""
        def _reader():
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    cmd = json.loads(line)
                    on_command(cmd)
                except json.JSONDecodeError:
                    pass  # ignore malformed input

        threading.Thread(target=_reader, daemon=True, name="IPCReader").start()
