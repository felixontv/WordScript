import { useState, useEffect, useCallback, useRef } from "react";

// Browser event.code → pynput key name
const CODE_TO_PYNPUT: Record<string, string> = {
  ControlLeft: "ctrl_l", ControlRight: "ctrl_l",
  AltLeft: "alt_l",      AltRight: "alt_l",
  ShiftLeft: "shift_l",  ShiftRight: "shift_l",
  MetaLeft: "win",       MetaRight: "win",
  Space: "space",
  F1: "f1",  F2: "f2",  F3: "f3",  F4: "f4",
  F5: "f5",  F6: "f6",  F7: "f7",  F8: "f8",
  F9: "f9",  F10: "f10", F11: "f11", F12: "f12",
};

function codeToKey(code: string): string | null {
  if (CODE_TO_PYNPUT[code]) return CODE_TO_PYNPUT[code];
  if (/^Key[A-Z]$/.test(code)) return code[3].toLowerCase();
  if (/^Digit[0-9]$/.test(code)) return code[5];
  return null;
}

const DISPLAY: Record<string, string> = {
  ctrl_l: "Ctrl", alt_l: "Alt", shift_l: "Shift", win: "Win", cmd: "Cmd", space: "Space",
  f1: "F1", f2: "F2", f3: "F3", f4: "F4",  f5: "F5",  f6: "F6",
  f7: "F7", f8: "F8", f9: "F9", f10: "F10", f11: "F11", f12: "F12",
};

const MODIFIER_ORDER = ["ctrl_l", "alt_l", "shift_l", "win", "cmd"];

function sortKeys(keys: string[]): string[] {
  return [...keys].sort((a, b) => {
    const ai = MODIFIER_ORDER.indexOf(a);
    const bi = MODIFIER_ORDER.indexOf(b);
    if (ai !== -1 && bi !== -1) return ai - bi;
    if (ai !== -1) return -1;
    if (bi !== -1) return 1;
    return a.localeCompare(b);
  });
}

function parseHotkey(hotkey: string): string[] {
  return hotkey.split("+").map((k) => k.trim()).filter(Boolean);
}

interface Props {
  value: string;
  onChange: (value: string) => void;
}

export function HotkeyRecorder({ value, onChange }: Props) {
  const [recording, setRecording] = useState(false);
  const [liveKeys, setLiveKeys] = useState<string[]>([]);

  // Use refs so event handlers always see current values without re-registering
  const heldRef     = useRef(new Set<string>());
  const capturedRef = useRef<string[]>([]);

  const finalize = useCallback(() => {
    const keys = sortKeys(capturedRef.current);
    if (keys.length > 0) onChange(keys.join("+"));
    setRecording(false);
    setLiveKeys([]);
    heldRef.current.clear();
    capturedRef.current = [];
  }, [onChange]);

  const cancel = useCallback(() => {
    setRecording(false);
    setLiveKeys([]);
    heldRef.current.clear();
    capturedRef.current = [];
  }, []);

  useEffect(() => {
    if (!recording) return;

    const onKeyDown = (e: KeyboardEvent) => {
      e.preventDefault();
      e.stopPropagation();

      if (e.code === "Escape") { cancel(); return; }

      const key = codeToKey(e.code);
      if (!key || heldRef.current.has(key)) return;

      heldRef.current.add(key);
      if (!capturedRef.current.includes(key)) {
        capturedRef.current = [...capturedRef.current, key];
      }
      setLiveKeys(sortKeys([...heldRef.current]));
    };

    const onKeyUp = (e: KeyboardEvent) => {
      e.preventDefault();
      const key = codeToKey(e.code);
      if (key) heldRef.current.delete(key);
      setLiveKeys(sortKeys([...heldRef.current]));
      if (heldRef.current.size === 0 && capturedRef.current.length > 0) {
        finalize();
      }
    };

    window.addEventListener("keydown", onKeyDown, true);
    window.addEventListener("keyup",   onKeyUp,   true);
    return () => {
      window.removeEventListener("keydown", onKeyDown, true);
      window.removeEventListener("keyup",   onKeyUp,   true);
    };
  }, [recording, finalize, cancel]);

  const startRecording = () => {
    heldRef.current.clear();
    capturedRef.current = [];
    setLiveKeys([]);
    setRecording(true);
  };

  const displayedKeys = recording ? liveKeys : sortKeys(parseHotkey(value));

  return (
    <div
      className={`hotkey-recorder${recording ? " hotkey-recorder--recording" : ""}`}
      onClick={!recording ? startRecording : undefined}
      onBlur={() => { if (recording) cancel(); }}
      tabIndex={0}
      role="button"
      aria-label={recording ? "Recording shortcut, press keys" : "Click to record shortcut"}
    >
      {recording && displayedKeys.length === 0 ? (
        <span className="hotkey-recorder__hint">Press your shortcut…</span>
      ) : displayedKeys.length > 0 ? (
        <>
          {displayedKeys.map((k) => (
            <span key={k} className="hotkey-recorder__key">
              {DISPLAY[k] ?? k.toUpperCase()}
            </span>
          ))}
          {recording && (
            <span className="hotkey-recorder__hint" style={{ marginLeft: 6 }}>
              — release to confirm
            </span>
          )}
        </>
      ) : (
        <span className="hotkey-recorder__hint">Click to record</span>
      )}
    </div>
  );
}
