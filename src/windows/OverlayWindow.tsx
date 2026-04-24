import { useEffect, useRef, useState } from "react";
import { getCurrentWindow, PhysicalPosition } from "@tauri-apps/api/window";
import { currentMonitor } from "@tauri-apps/api/window";
import { listen } from "@tauri-apps/api/event";
import { useSidecar } from "../hooks/useSidecar";
import "../styles/overlay.css";

const BAR_COUNT = 16;

export default function OverlayWindow() {
  const { state, sendCommand, openSettings } = useSidecar();
  const { status, muted } = state;
  const positionedRef = useRef(false);

  // Mark html element for overlay-specific CSS
  useEffect(() => {
    document.documentElement.classList.add("overlay-window");
    document.documentElement.classList.add("overlay-idle");
  }, []);

  // CSS-based visibility: idle overlay is transparent + click-through
  // This avoids GTK show()/hide() which crash under XWayland.
  const isActive = status === "recording" || status === "processing";
  useEffect(() => {
    if (isActive) {
      document.documentElement.classList.remove("overlay-idle");
      // Position center-bottom on first activation (only once)
      if (!positionedRef.current) {
        positionedRef.current = true;
        currentMonitor().then((monitor) => {
          if (monitor) {
            const sw = monitor.size.width;
            const sh = monitor.size.height;
            const x = Math.round((sw - 312) / 2);
            const y = sh - 68 - 90;
            getCurrentWindow().setPosition(new PhysicalPosition(x, y)).catch(() => {});
          }
        });
      }
    } else {
      // Brief delay so user sees the final state before hiding
      const t = setTimeout(() => {
        document.documentElement.classList.add("overlay-idle");
      }, 300);
      return () => clearTimeout(t);
    }
  }, [isActive]);

  // Reactive waveform bars driven by audio_level events from Python
  const [barHeights, setBarHeights] = useState<number[]>(Array(BAR_COUNT).fill(4));

  useEffect(() => {
    const unlisten = listen<{ event: string; level?: number }>("py-event", ({ payload }) => {
      if (payload.event === "audio_level" && typeof payload.level === "number") {
        // Power curve: stretches low levels so quiet speech still moves the bars
        const lv = Math.pow(Math.min(1, payload.level * 3), 0.5);
        setBarHeights(Array.from({ length: BAR_COUNT }, () =>
          Math.max(4, Math.min(28, 4 + lv * 24 * (0.4 + Math.random() * 0.6)))
        ));
      }
    });
    return () => { unlisten.then(fn => fn()); };
  }, []);

  // Reset bars when not actively recording
  useEffect(() => {
    if (status !== "recording" || muted) {
      setBarHeights(Array(BAR_COUNT).fill(4));
    }
  }, [status, muted]);

  // Elapsed seconds counter (shown during processing only)
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    if (status === "processing") {
      setElapsed(0);
      timerRef.current = window.setInterval(() => setElapsed(s => s + 1), 1000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
      setElapsed(0);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [status]);

  // Drag via Tauri native drag (far simpler than manual mousemove)
  const startDrag = async (e: React.MouseEvent) => {
    // Only drag from the bars zone (not mic or chevron)
    if ((e.target as HTMLElement).closest(".pill__mic, .pill__chevron")) return;
    await getCurrentWindow().startDragging();
  };

  const pillClass = [
    "pill",
    status === "recording" && !muted ? "pill--recording" : "",
    status === "recording" && muted   ? "pill--muted"     : "",
    status === "processing"           ? "pill--processing" : "",
  ].filter(Boolean).join(" ");

  // Right zone: elapsed counter while processing, otherwise settings chevron
  const rightContent = status === "processing"
    ? <span className={elapsed >= 10 ? "pill__proc-label pill__proc-label--slow" : "pill__proc-label"}>
        {elapsed >= 3 ? `${elapsed}s` : "▾"}
      </span>
    : <span style={{ fontSize: 10, color: "var(--fg-dim)" }}>▾</span>;

  return (
    <div className={pillClass} onMouseDown={startDrag}>
      {/* Mic icon */}
      <div
        className="pill__mic"
        onClick={() => status === "recording" && sendCommand({ cmd: "toggle_mute" })}
        title={muted ? "Unmute" : "Mute"}
      >
        <MicIcon muted={muted} />
        {status === "recording" && !muted && <div className="rec-dot" />}
      </div>

      {/* Waveform bars */}
      <div className="pill__bars">
        {Array.from({ length: BAR_COUNT }, (_, i) => (
          <div
            key={i}
            className={`bar${muted ? " bar--muted" : ""}`}
            style={{ height: barHeights[i] }}
          />
        ))}
      </div>

      {/* Divider + right zone */}
      <div className="pill__divider" />
      <div
        className="pill__chevron"
        onClick={openSettings}
        title="Settings"
      >
        {rightContent}
      </div>
    </div>
  );
}

// ── Mic SVG icon ──────────────────────────────────────────────────────────────

function MicIcon({ muted }: { muted: boolean }) {
  const color = muted ? "#ff3b3b" : "#ffffff";
  return (
    <svg width="18" height="20" viewBox="0 0 18 22" fill="none"
         style={{ opacity: 0.85 }}>
      {/* Capsule body */}
      <rect x="5" y="1" width="8" height="12" rx="4" fill={color} />
      {/* Stand arc */}
      <path d="M2 10 C2 16 16 16 16 10" stroke={color} strokeWidth="2"
            fill="none" strokeLinecap="round" />
      {/* Stem */}
      <line x1="9" y1="16" x2="9" y2="20" stroke={color} strokeWidth="2"
            strokeLinecap="round" />
      {/* Mute slash */}
      {muted && (
        <line x1="3" y1="3" x2="15" y2="18" stroke="#ff3b3b" strokeWidth="2"
              strokeLinecap="round" />
      )}
    </svg>
  );
}
