"""Global hotkey listener using pynput."""

import logging
import sys
import threading
from typing import Optional

from pynput import keyboard

from .config import Config


class HotkeyManager:
    """Global hotkey listener, supporting tap and hold activation modes."""

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
        self._hotkey_active = False
        self._abort_active  = False
        self._toggled_on    = False
        self._paused: bool  = False
        self._listener: Optional[keyboard.Listener] = None

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
        self._hotkey_active = False
        self._abort_active  = False

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
        for part in hotkey_str.lower().split("+"):
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

                if not abort_held and self._abort_active:
                    self._abort_active = False

                if not hotkey_held and self._hotkey_active:
                    self._hotkey_active = False
                    for k in self._hotkey_keys:
                        self._pressed_keys.discard(k)
                    fire_hotkey_release = True

            if fire_hotkey_release:
                threading.Thread(target=self._handle_hotkey_release, daemon=True).start()
        except Exception as e:
            self.logger.error("_on_release error: %s", e, exc_info=True)

    def _handle_hotkey_press(self) -> None:
        if self.config.activation_mode == "hold":
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
            self._on_deactivate()

    def _handle_abort_press(self) -> None:
        self.logger.info("Abort hotkey pressed")
        self._on_abort()
