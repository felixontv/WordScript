"""Shared dark-theme color palette for all UI components."""

import sys

# Cross-platform UI font — Segoe UI (Win), SF Pro / Helvetica (macOS), DejaVu (Linux)
if sys.platform == "win32":
    UI_FONT = "Segoe UI"
elif sys.platform == "darwin":
    UI_FONT = "SF Pro Text"
else:
    UI_FONT = "DejaVu Sans"

THEME = {
    "BG":      "#0c0c0c",
    "SURFACE": "#161616",
    "SIDEBAR": "#111111",
    "FG":      "#d4d4d4",
    "FG_DIM":  "#555555",
    "ACCENT":  "#ffffff",
    "BTN_BG":  "#1c1c1c",
    "BORDER":  "#282828",
    "GREEN":   "#34d058",
    "RED":     "#ff4444",
    "NAV_ACT": "#1e1e1e",
}
