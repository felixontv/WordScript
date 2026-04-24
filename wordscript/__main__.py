"""WordScript CLI — entry point for `python -m wordscript`.

Commands:
  (none)        Start the full desktop app with UI (default)
  run           Same as above, explicit
  sidecar       Headless mode for Tauri — JSON IPC over stdout/stdin
  transcribe    Record once, print transcription as JSON, exit
  config        Print current config as JSON and exit
  version       Print version and exit
"""

import argparse
import sys


def _cmd_run():
    from .app import main
    main()


def _cmd_sidecar():
    """Headless mode: no Tkinter, all state communicated via JSON IPC."""
    import logging
    import os
    import socket

    from .app import _setup_logging
    from .config import Config
    from .sidecar import SidecarApp

    # Singleton guard — same port as full app so both can't run simultaneously
    try:
        _sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _sock.bind(("127.0.0.1", 48127))
    except OSError:
        # Emit error as JSON so Tauri can surface it
        import json
        print(json.dumps({"event": "error", "message": "WordScript is already running."}),
              flush=True)
        sys.exit(0)

    config = Config.load()
    _setup_logging(config.log_level)
    SidecarApp().run()


def _cmd_transcribe():
    """Record once (until Enter or hotkey), print result as JSON, exit."""
    import json
    import logging
    import sounddevice as sd

    from .config import Config
    from .recorder import AudioRecorder
    from .transcription import TranscriptionService

    logging.basicConfig(level=logging.WARNING)
    config = Config.load()

    if not config.groq_api_key:
        print(json.dumps({"error": "No Groq API key. Run: python -m wordscript config"}),
              flush=True)
        sys.exit(1)

    recorder    = AudioRecorder(config)
    transcriber = TranscriptionService(config)

    print("Recording… press Enter to stop.", file=sys.stderr)
    recorder.start()
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        wav = recorder.stop()

    if not wav:
        print(json.dumps({"text": "", "corrected": ""}), flush=True)
        sys.exit(0)

    text = transcriber.transcribe(wav)
    corrected = ""
    if config.post_process and text and not text.startswith("["):
        corrected = transcriber.correct(text)

    print(json.dumps({
        "text":      text,
        "corrected": corrected if corrected and corrected != text else "",
    }), flush=True)


def _cmd_config():
    import dataclasses
    import json
    from .config import Config
    print(json.dumps(dataclasses.asdict(Config.load()), indent=2), flush=True)


def _cmd_version():
    from .constants import APP_VERSION
    print(APP_VERSION, flush=True)


def main():
    parser = argparse.ArgumentParser(
        prog="wordscript",
        description="WordScript — lightweight speech-to-text desktop utility.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("run",        help="Start with full desktop UI (default)")
    sub.add_parser("sidecar",    help="Headless mode — JSON IPC for Tauri frontend")
    sub.add_parser("transcribe", help="Record once, print JSON transcription, exit")
    sub.add_parser("config",     help="Print current config.json as JSON and exit")
    sub.add_parser("version",    help="Print version and exit")

    args = parser.parse_args()

    dispatch = {
        None:         _cmd_run,
        "run":        _cmd_run,
        "sidecar":    _cmd_sidecar,
        "transcribe": _cmd_transcribe,
        "config":     _cmd_config,
        "version":    _cmd_version,
    }
    dispatch[args.command]()


main()
