"""Pill-shaped audio visualizer overlay."""

import logging
import math
import queue
import sys
import threading
import time
from typing import Optional

from .theme import UI_FONT
from .widgets import draw_rounded_rect, draw_mic_icon
from .settings import open_settings_modal


class VisualizerOverlay:
    """Compact pill-shaped audio visualizer overlay — Liquid Glass dark theme.

    Layout:  [ 🎙  |||||||||||||||||||||||||||   ▾ ]

    – Mic icon on the left (click = mute/unmute toggle, red when muted)
    – Animated waveform bars in the centre (flat when muted)
    – Countdown timer on the right (opens Settings on click)
    – Draggable via bar area
    """

    _BAR_COUNT  = 16
    _UPDATE_MS  = 33
    _W          = 296
    _H          = 52
    _R          = 26
    _BG_PILL    = "#0c0c0c"
    _BG_WIN     = "#000001"
    _BAR_COLOR  = "#ffffff"
    _BAR_MUTED  = "#333333"
    _MIC_COLOR  = "#ffffff"
    _MIC_MUTED  = "#ff3b3b"
    _DIV_COLOR  = "#2a2a2a"
    _CHEV_COLOR = "#666666"
    _CHEV_HOV   = "#ffffff"
    _PILL_ALPHA = 0.88

    _MIC_W  = 52
    _CHEV_W = 44

    def __init__(self):
        self.logger = logging.getLogger("VisualizerOverlay")
        self._level_queue: Optional[queue.Queue] = None
        self._want_visible = False
        self._is_visible   = False
        self._processing   = False
        self._processing_start: float = 0.0
        self._recording_start: float  = 0.0
        self._muted        = False
        self._config_ref: list = []
        self._on_mic_click = None
        self._on_settings_saved = None
        self._want_settings = False
        self._root_ref: list = []
        self._tk_ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def show(self, level_queue: queue.Queue) -> None:
        self._level_queue = level_queue
        self._processing = False
        self._recording_start = time.perf_counter()
        self._want_visible = True

    def show_processing(self) -> None:
        self._processing_start = time.perf_counter()
        self._processing = True
        self._want_visible = True

    def hide(self) -> None:
        self._want_visible = False
        self._processing = False

    def open_settings(self) -> None:
        """Request the settings modal to open from any thread (thread-safe)."""
        self._want_settings = True

    def _run(self) -> None:
        try:
            import tkinter as tk
        except ImportError:
            self.logger.warning("tkinter not available — visualizer disabled.")
            return

        root = tk.Tk()
        self._root_ref.append(root)

        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", self._PILL_ALPHA)

        if sys.platform == "win32":
            root.wm_attributes("-transparentcolor", self._BG_WIN)
            root.configure(bg=self._BG_WIN)
            canvas_bg = self._BG_WIN
        elif sys.platform == "darwin":
            try:
                root.wm_attributes("-transparent", True)
                root.configure(bg="systemTransparent")
                canvas_bg = "systemTransparent"
            except Exception:
                root.configure(bg=self._BG_PILL)
                canvas_bg = self._BG_PILL
        else:
            root.configure(bg=self._BG_PILL)
            canvas_bg = self._BG_PILL

        W, H = self._W, self._H
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.geometry(f"{W}x{H}+{(sw - W) // 2}+{sh - H - 90}")

        canvas = tk.Canvas(root, width=W, height=H,
                           bg=canvas_bg, highlightthickness=0)
        canvas.pack()

        levels = [0.0] * self._BAR_COUNT

        # ── Draw helpers ──────────────────────────────────────────────────
        def _draw_pill():
            draw_rounded_rect(canvas, 0, 0, W, H, self._R,
                              fill=self._BG_PILL, outline="")

        def _draw_divider():
            dx = W - self._CHEV_W
            canvas.create_line(dx, 13, dx, H - 13, fill=self._DIV_COLOR, width=1)

        def _draw_chevron():
            cx = W - self._CHEV_W // 2
            cy = H // 2
            s  = 5
            canvas.create_polygon(
                cx - s, cy - 3,
                cx + s, cy - 3,
                cx,     cy + 4,
                fill=self._CHEV_COLOR, outline="", tags="chevron",
            )

        def _draw_chrome():
            canvas.delete("all")
            _draw_pill()
            _draw_divider()
            mc = self._MIC_MUTED if self._muted else self._MIC_COLOR
            draw_mic_icon(canvas, self._MIC_W // 2, H // 2, color=mc, tag="mic")
            if self._muted:
                mx, my = self._MIC_W // 2, H // 2
                canvas.create_line(mx - 8, my - 10, mx + 8, my + 10,
                                   fill=self._MIC_MUTED, width=2, tags="mic")
            _draw_chevron()

        _draw_chrome()

        # ── Hit zones ─────────────────────────────────────────────────────
        chev_x1 = W - self._CHEV_W
        mic_x2  = self._MIC_W

        # ── Hover ─────────────────────────────────────────────────────────
        def _on_motion(e):
            if chev_x1 <= e.x <= W:
                canvas.itemconfig("chevron", fill=self._CHEV_HOV)
                canvas.config(cursor="hand2")
            elif 0 <= e.x <= mic_x2:
                canvas.config(cursor="hand2")
                canvas.itemconfig("chevron", fill=self._CHEV_COLOR)
            else:
                canvas.itemconfig("chevron", fill=self._CHEV_COLOR)
                canvas.config(cursor="")

        def _on_leave(e):
            canvas.itemconfig("chevron", fill=self._CHEV_COLOR)
            canvas.config(cursor="")

        canvas.bind("<Motion>", _on_motion)
        canvas.bind("<Leave>",  _on_leave)

        # ── Drag + click ──────────────────────────────────────────────────
        _drag = {"sx": 0, "sy": 0, "wx": 0, "wy": 0, "moved": False}

        def _press(e):
            _drag["sx"] = e.x
            _drag["sy"] = e.y
            _drag["wx"] = e.x_root
            _drag["wy"] = e.y_root
            _drag["moved"] = False

        def _motion(e):
            if abs(e.x_root - _drag["wx"]) > 4 or abs(e.y_root - _drag["wy"]) > 4:
                _drag["moved"] = True
            if _drag["moved"] and mic_x2 < _drag["sx"] < chev_x1:
                dx = e.x_root - _drag["wx"]
                dy = e.y_root - _drag["wy"]
                _drag["wx"] = e.x_root
                _drag["wy"] = e.y_root
                root.geometry(f"+{root.winfo_x() + dx}+{root.winfo_y() + dy}")

        def _release(e):
            if _drag["moved"]:
                return
            sx = _drag["sx"]
            if chev_x1 <= sx <= W and self._config_ref:
                open_settings_modal(root, self._config_ref,
                                    on_saved=self._on_settings_saved)
            elif 0 <= sx <= mic_x2:
                if self._on_mic_click:
                    self._on_mic_click()

        canvas.bind("<ButtonPress-1>",   _press)
        canvas.bind("<B1-Motion>",       _motion)
        canvas.bind("<ButtonRelease-1>", _release)

        root.withdraw()

        # ── Bars geometry ─────────────────────────────────────────────────
        bar_x1  = self._MIC_W + 6
        bar_x2  = W - self._CHEV_W - 6
        bar_w_t = bar_x2 - bar_x1
        bar_gap = 3
        bar_w   = (bar_w_t - bar_gap * (self._BAR_COUNT - 1)) / self._BAR_COUNT

        # ── Animation state ───────────────────────────────────────────────
        _smooth     = [0.0] * self._BAR_COUNT
        _alpha      = [0.0]
        _proc_frame = [0]
        _rec_frame  = [0]

        def _draw_processing():
            canvas.delete("all")
            _draw_pill()
            max_h   = H - 16
            frame   = _proc_frame[0]
            elapsed = time.perf_counter() - self._processing_start
            shimmer = (frame * 0.018) % 1.3 - 0.15
            for i in range(self._BAR_COUNT):
                t     = i / max(self._BAR_COUNT - 1, 1)
                phase = frame * 0.08 + i * 0.38
                base  = math.sin(phase) * 0.22 + 0.35
                dist  = abs(t - shimmer)
                glow  = max(0.0, 1.0 - dist * 5.5)
                lvl   = min(1.0, base + glow * 0.45)
                gray  = int(0x30 + glow * (0xD0 - 0x30))
                color = f"#{gray:02x}{gray:02x}{gray:02x}"
                bh    = max(3, int(lvl * max_h))
                bx    = bar_x1 + i * (bar_w + bar_gap)
                by    = (H - bh) / 2
                rr    = min(bar_w / 2, bh / 2, 3)
                draw_rounded_rect(canvas, bx, by, bx + bar_w, by + bh, rr,
                                  fill=color, outline="")
            _draw_divider()
            draw_mic_icon(canvas, self._MIC_W // 2, H // 2,
                          color="#444444", tag="mic")
            if elapsed >= 5:
                secs  = int(elapsed)
                label = f"{secs}s"
                fg    = "#ffaa00" if elapsed >= 15 else "#666666"
                canvas.create_text(
                    W - self._CHEV_W // 2, H // 2,
                    text=label, fill=fg,
                    font=(UI_FONT, 8), anchor="center",
                )
            else:
                _draw_chevron()
            _proc_frame[0] += 1

        def _draw_rec_dot(frame: int):
            pulse = math.sin(frame * 0.14) * 0.5 + 0.5
            r     = 3.0 + pulse * 1.8
            cx    = self._MIC_W - 9
            cy    = 9
            canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                fill="#ff3b3b", outline="", tags="recdot")

        def _tick():
            # Fade in / out
            target = self._PILL_ALPHA if self._want_visible else 0.0
            cur    = _alpha[0]
            if cur != target:
                step      = 0.10
                _alpha[0] = (
                    min(target, cur + step) if cur < target
                    else max(target, cur - step)
                )
                root.attributes("-alpha", _alpha[0])

            # Show / hide
            if self._want_visible and not self._is_visible:
                _alpha[0] = 0.0
                root.attributes("-alpha", 0.0)
                for j in range(self._BAR_COUNT):
                    _smooth[j] = 0.0
                _draw_chrome()
                root.deiconify()
                self._is_visible = True
            elif not self._want_visible and self._is_visible and _alpha[0] <= 0.0:
                root.withdraw()
                self._is_visible = False

            # Render
            if self._is_visible:
                if self._processing:
                    _draw_processing()
                elif self._level_queue:
                    while True:
                        try:
                            levels.append(self._level_queue.get_nowait())
                        except queue.Empty:
                            break
                    del levels[: len(levels) - self._BAR_COUNT]

                    canvas.delete("all")
                    _draw_pill()

                    is_muted = self._muted
                    bar_col  = self._BAR_MUTED if is_muted else self._BAR_COLOR
                    max_h    = H - 16

                    for i, raw in enumerate(levels):
                        if is_muted:
                            breathe    = math.sin(_rec_frame[0] * 0.04) * 0.5 + 0.5
                            _smooth[i] = 3.0 + breathe * 1.5
                            bh         = int(_smooth[i])
                        else:
                            if raw > _smooth[i]:
                                _smooth[i] = _smooth[i] * 0.45 + raw * 0.55
                            else:
                                _smooth[i] = _smooth[i] * 0.82 + raw * 0.18
                            bh = max(3, int(_smooth[i] * max_h))

                        bx = bar_x1 + i * (bar_w + bar_gap)
                        by = (H - bh) / 2
                        rr = min(bar_w / 2, bh / 2, 3)
                        draw_rounded_rect(canvas, bx, by, bx + bar_w, by + bh, rr,
                                          fill=bar_col, outline="")

                    _draw_divider()

                    mc = self._MIC_MUTED if is_muted else self._MIC_COLOR
                    draw_mic_icon(canvas, self._MIC_W // 2, H // 2,
                                  color=mc, tag="mic")
                    if is_muted:
                        mx, my = self._MIC_W // 2, H // 2
                        canvas.create_line(mx - 8, my - 10, mx + 8, my + 10,
                                           fill=self._MIC_MUTED, width=2, tags="mic")
                    else:
                        _draw_rec_dot(_rec_frame[0])

                    # Countdown timer (MM:SS remaining)
                    _elapsed_rec = time.perf_counter() - self._recording_start
                    _max_rec     = self._config_ref[0].max_recording_seconds if self._config_ref else 720
                    _remaining   = max(0, _max_rec - _elapsed_rec)
                    _mm, _ss     = int(_remaining) // 60, int(_remaining) % 60
                    _clr = (
                        "#ff3b3b" if _remaining <= 30
                        else "#ffaa00" if _remaining <= 120
                        else "#555555"
                    )
                    canvas.create_text(
                        W - self._CHEV_W // 2, H // 2,
                        text=f"{_mm}:{_ss:02d}", fill=_clr,
                        font=(UI_FONT, 8), anchor="center",
                    )
                    _rec_frame[0] += 1

            # Open settings if requested from outside Tk thread
            if self._want_settings:
                self._want_settings = False
                if self._config_ref:
                    open_settings_modal(root, self._config_ref,
                                        on_saved=self._on_settings_saved)

            root.after(self._UPDATE_MS, _tick)

        root.after(self._UPDATE_MS, _tick)
        self._tk_ready.set()
        root.mainloop()
