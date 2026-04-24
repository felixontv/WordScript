"""Global hotkey listener using pynput."""

import logging
import sys
import threading
import time
from typing import Optional

from pynput import keyboard

from .config import Config


class HotkeyManager:
    """Global hotkey listener, supporting tap and hold activation modes."""

    # Minimum seconds between consecutive hotkey fires. Prevents double-trigger
    # caused by compositor synthetic evdev events (e.g. KDE Ctrl+Shift layout switch).
    _HOTKEY_DEBOUNCE_S: float = 0.3

    # Hold mode: minimum hold duration before a key-release stops recording.
    # Filters synthetic releases from Wayland compositors that arrive within
    # milliseconds of the physical press.
    _HOLD_MIN_S: float = 0.3

    KEY_MAP = {
        "ctrl":    keyboard.Key.ctrl_l,
        "ctrl_l":  keyboard.Key.ctrl_l,
        "ctrl_r":  keyboard.Key.ctrl_r,
        "alt":     keyboard.Key.alt_l,
        "alt_l":   keyboard.Key.alt_l,
        "alt_r":   keyboard.Key.alt_r,
        "shift":   keyboard.Key.shift_l,
        "shift_l": keyboard.Key.shift_l,
        "shift_r": keyboard.Key.shift_r,
        "win":     keyboard.Key.cmd,
        "cmd":     keyboard.Key.cmd,
        "space":   keyboard.Key.space,
        "f1":  keyboard.Key.f1,  "f2":  keyboard.Key.f2,
        "f3":  keyboard.Key.f3,  "f4":  keyboard.Key.f4,
        "f5":  keyboard.Key.f5,  "f6":  keyboard.Key.f6,
        "f7":  keyboard.Key.f7,  "f8":  keyboard.Key.f8,
        "f9":  keyboard.Key.f9,  "f10": keyboard.Key.f10,
        "f11": keyboard.Key.f11, "f12": keyboard.Key.f12,
    }

    def __init__(self, config: Config, on_activate, on_deactivate, on_abort):
        self.config = config
        self.logger = logging.getLogger("HotkeyManager")
        self._on_activate   = on_activate
        self._on_deactivate = on_deactivate
        self._on_abort      = on_abort

        self._hotkey_keys   = self._parse_hotkey(config.hotkey)
        self._abort_keys    = self._parse_hotkey(config.abort_hotkey)
        self._pressed_keys: set = set()
        self._raw_pressed_keys: set = set()
        self._keys_lock     = threading.Lock()
        self._hotkey_active       = False
        self._abort_active        = False
        self._toggled_on          = False
        self._paused: bool        = False
        self._hold_pending_release: bool = False
        self._listener: Optional[keyboard.Listener] = None

        # Debounce / hold-mode timing state
        self._last_hotkey_press_time: float = 0.0
        self._hold_start_time:        float = 0.0
        self._hold_session:           int   = 0

        self.logger.info(
            "Hotkey: %s | Mode: %s | Abort: %s",
            config.hotkey, config.activation_mode, config.abort_hotkey,
        )

    def start(self) -> None:
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            suppress=False,
        )
        self._listener.daemon = True
        self._listener.start()
        self.logger.info("Hotkey listener started.")

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
        with self._keys_lock:
            self._pressed_keys.clear()
            self._raw_pressed_keys.clear()
        self._hotkey_active        = False
        self._abort_active         = False
        self._hold_pending_release = False
        self._hold_session        += 1   # invalidate any pending deferred-stop timers

    def reload_hotkey(self) -> None:
        self._hotkey_keys = self._parse_hotkey(self.config.hotkey)
        self.logger.info("Hotkey reloaded: %s", self.config.hotkey)

    def reload_abort_hotkey(self) -> None:
        self._abort_keys = self._parse_hotkey(self.config.abort_hotkey)
        self.logger.info("Abort hotkey reloaded: %s", self.config.abort_hotkey)

    def pause(self) -> None:
        self._paused = True
        self.logger.debug("Hotkey listener paused.")

    def resume(self) -> None:
        self._paused = False
        self.logger.debug("Hotkey listener resumed.")

    def _parse_hotkey(self, hotkey_str: str) -> set:
        keys = set()
        # Support both "+" and ", " / "," as separator (UI recorder uses ", ")
        import re
        parts = re.split(r'[+,]', hotkey_str.lower())
        for part in parts:
            part = part.strip()
            if part in self.KEY_MAP:
                keys.add(self.KEY_MAP[part])
            elif len(part) == 1:
                keys.add(keyboard.KeyCode.from_char(part))
            else:
                self.logger.warning("Unknown key in hotkey config: '%s'", part)
        if not keys:
            if sys.platform in ("win32", "darwin"):
                fallback = {keyboard.Key.ctrl_l, keyboard.Key.cmd}
            else:
                fallback = {keyboard.Key.ctrl_l, keyboard.Key.alt_l}
            self.logger.error(
                "Hotkey '%s' resolved to no valid keys — falling back to platform default.",
                hotkey_str,
            )
            return fallback
        return keys

    def _normalize_key(self, key):
        RIGHT_TO_LEFT = {
            keyboard.Key.ctrl_r:  keyboard.Key.ctrl_l,
            keyboard.Key.alt_r:   keyboard.Key.alt_l,
            keyboard.Key.shift_r: keyboard.Key.shift_l,
            keyboard.Key.cmd_r:   keyboard.Key.cmd,
        }
        return RIGHT_TO_LEFT.get(key, key)

    def _on_press(self, key) -> None:
        try:
            normalized   = self._normalize_key(key)
            fire_abort   = False
            fire_hotkey  = False

            with self._keys_lock:
                self._raw_pressed_keys.add(key)
                self._pressed_keys.add(normalized)

                if not self._paused:
                    _is_altgr = (
                        keyboard.Key.ctrl_r in self._raw_pressed_keys
                        and keyboard.Key.alt_r in self._raw_pressed_keys
                    )
                    abort_hit  = self._abort_keys.issubset(self._pressed_keys) and not _is_altgr
                    hotkey_hit = self._hotkey_keys.issubset(self._pressed_keys)

                    if abort_hit and not self._abort_active:
                        self._abort_active = True
                        fire_abort = True
                    elif hotkey_hit and not self._hotkey_active:
                        # 300 ms debounce: ignore rapid re-trigger from compositor
                        # synthetic evdev events (e.g. KDE Ctrl+Shift layout switch).
                        now = time.perf_counter()
                        if now - self._last_hotkey_press_time >= self._HOTKEY_DEBOUNCE_S:
                            self._last_hotkey_press_time = now
                            self._hotkey_active = True
                            fire_hotkey = True

            if fire_abort:
                threading.Thread(target=self._handle_abort_press, daemon=True).start()
            elif fire_hotkey:
                threading.Thread(target=self._handle_hotkey_press, daemon=True).start()
        except Exception as e:
            self.logger.error("_on_press error: %s", e, exc_info=True)

    def _on_release(self, key) -> None:
        try:
            normalized          = self._normalize_key(key)
            fire_hotkey_release = False

            with self._keys_lock:
                self._raw_pressed_keys.discard(key)
                self._pressed_keys.discard(normalized)
                self._pressed_keys.discard(key)
                abort_held  = self._abort_keys.issubset(self._pressed_keys)
                hotkey_held = self._hotkey_keys.issubset(self._pressed_keys)
                # True when none of the hotkey keys remain in pressed set
                hotkey_fully_released = self._hotkey_keys.isdisjoint(self._pressed_keys)

                if not abort_held and self._abort_active:
                    self._abort_active = False

                if not hotkey_held and self._hotkey_active:
                    # The full combo is no longer held — reset active flag and
                    # flush remaining hotkey keys from tracked set so they don't
                    # block the next press.
                    self._hotkey_active = False
                    for k in self._hotkey_keys:
                        self._pressed_keys.discard(k)

                    if self.config.activation_mode == "hold":
                        # Hold mode: only deactivate (stop recording) when ALL
                        # hotkey keys are gone, not when just one is released.
                        # This prevents premature stop with multi-key combos
                        # where Ctrl may lift a few ms before F9 or vice-versa.
                        if hotkey_fully_released:
                            fire_hotkey_release = True
                        else:
                            self._hold_pending_release = True
                    else:
                        # Tap mode: fire release so _handle_hotkey_release can
                        # run (it does nothing for tap, but keeps logic symmetric).
                        fire_hotkey_release = True

                elif self._hold_pending_release and hotkey_fully_released:
                    # A previous release partially lifted the combo; now the last
                    # hotkey key is gone — safe to deactivate.
                    self._hold_pending_release = False
                    fire_hotkey_release = True

            if fire_hotkey_release:
                threading.Thread(target=self._handle_hotkey_release, daemon=True).start()
        except Exception as e:
            self.logger.error("_on_release error: %s", e, exc_info=True)

    def _handle_hotkey_press(self) -> None:
        if self.config.activation_mode == "hold":
            self._hold_session  += 1
            self._hold_start_time = time.perf_counter()
            self._on_activate()
        elif self.config.activation_mode == "tap":
            if self._toggled_on:
                self._toggled_on = False
                self._on_deactivate()
            else:
                self._toggled_on = True
                self._on_activate()

    def _handle_hotkey_release(self) -> None:
        if self.config.activation_mode == "hold":
            held_for = time.perf_counter() - self._hold_start_time
            if held_for < self._HOLD_MIN_S:
                # Key released too quickly — likely a synthetic compositor event.
                # Schedule the actual stop for when the minimum hold time expires.
                # The session counter lets us cancel this if a new recording starts.
                session = self._hold_session
                delay   = self._HOLD_MIN_S - held_for

                def _deferred_stop(s: int = session) -> None:
                    if self._hold_session == s:
                        self._on_deactivate()

                t = threading.Timer(delay, _deferred_stop)
                t.daemon = True
                t.start()
            else:
                self._on_deactivate()

    def _handle_abort_press(self) -> None:
        self.logger.info("Abort hotkey pressed")
        self._on_abort()
