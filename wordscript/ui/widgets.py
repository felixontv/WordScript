"""Reusable tkinter canvas drawing helpers."""


def draw_rounded_rect(canvas, x1, y1, x2, y2, r, **kwargs):
    """Draw a filled rounded rectangle on *canvas*."""
    pts = [
        x1 + r, y1,
        x2 - r, y1,
        x2,     y1,
        x2,     y1 + r,
        x2,     y2 - r,
        x2,     y2,
        x2 - r, y2,
        x1 + r, y2,
        x1,     y2,
        x1,     y2 - r,
        x1,     y1 + r,
        x1,     y1,
    ]
    return canvas.create_polygon(pts, smooth=True, **kwargs)


def draw_mic_icon(canvas, cx, cy, color="#ffffff", tag="mic"):
    """Draw a simple microphone icon centred at (cx, cy)."""
    bw, bh = 8, 12
    r = bw // 2
    canvas.create_arc(
        cx - r, cy - bh // 2 - r,
        cx + r, cy - bh // 2 + r,
        start=0, extent=180, fill=color, outline="", tags=tag,
    )
    canvas.create_rectangle(
        cx - r, cy - bh // 2,
        cx + r, cy + bh // 2,
        fill=color, outline="", tags=tag,
    )
    canvas.create_arc(
        cx - r, cy + bh // 2 - r,
        cx + r, cy + bh // 2 + r,
        start=180, extent=180, fill=color, outline="", tags=tag,
    )
    sr = 10
    canvas.create_arc(
        cx - sr, cy - sr // 2,
        cx + sr, cy + bh // 2 + sr,
        start=0, extent=-180, style="arc",
        outline=color, width=2, tags=tag,
    )
    canvas.create_line(
        cx, cy + bh // 2 + sr,
        cx, cy + bh // 2 + sr + 4,
        fill=color, width=2, tags=tag,
    )
