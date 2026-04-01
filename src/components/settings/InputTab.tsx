import type { AppConfig, PythonCommand } from "../../types/ipc";
import { HotkeyRecorder } from "./HotkeyRecorder";

interface Props {
  config: AppConfig;
  onChange: (p: Partial<AppConfig>) => void;
  sendCommand: (cmd: PythonCommand) => void;
}

export function InputTab({ config, onChange, sendCommand }: Props) {
  const maxMins = Math.floor(config.max_recording_seconds / 60);
  const maxSecs = config.max_recording_seconds % 60;

  return (
    <>
      <div className="tab__title">Input</div>

      <div className="form-section">Hotkey</div>
      <div className="form-row">
        <label>Activate Hotkey</label>
        <HotkeyRecorder
          value={config.hotkey}
          onChange={(hotkey) => onChange({ hotkey })}
          onStartRecording={() => sendCommand({ cmd: "pause_hotkey" })}
          onStopRecording={() => sendCommand({ cmd: "resume_hotkey" })}
        />
      </div>
      <p className="form-dim">Click the field and press your desired key combination.</p>

      <div className="form-row">
        <label>Abort Hotkey</label>
        <HotkeyRecorder
          value={config.abort_hotkey}
          onChange={(abort_hotkey) => onChange({ abort_hotkey })}
          onStartRecording={() => sendCommand({ cmd: "pause_hotkey" })}
          onStopRecording={() => sendCommand({ cmd: "resume_hotkey" })}
        />
      </div>
      <p className="form-dim">Hold during recording to discard and stop.</p>

      <div className="form-row">
        <label>Activation Mode</label>
        <select value={config.activation_mode}
          onChange={(e) => onChange({ activation_mode: e.target.value as "tap" | "hold" })}>
          <option value="tap">tap</option>
          <option value="hold">hold</option>
        </select>
      </div>
      <p className="form-dim">
        tap = press once to start, press again to stop<br />
        hold = hold key to record, release to stop
      </p>

      <div className="form-section">Audio</div>
      <div className="form-row">
        <label>Input Device</label>
        <input type="text" value={config.audio_device}
          placeholder="Default (system microphone)"
          onChange={(e) => onChange({ audio_device: e.target.value })} />
      </div>
      <p className="form-dim">Leave empty to use the system default microphone.</p>

      <div className="form-row">
        <label>Max recording</label>
        <input type="number" value={config.max_recording_seconds} min={10} max={3600}
          onChange={(e) => onChange({ max_recording_seconds: Number(e.target.value) })}
          style={{ maxWidth: 90 }}
        />
        <span style={{ fontSize: 12, color: "var(--fg-dim)", marginLeft: 8 }}>seconds ({maxMins}m {maxSecs}s)</span>
      </div>

      <div className="form-row">
        <label>Silence timeout</label>
        <input type="number" value={config.silence_timeout_seconds} min={0} max={300}
          onChange={(e) => onChange({ silence_timeout_seconds: Number(e.target.value) })}
          style={{ maxWidth: 90 }}
        />
        <span style={{ fontSize: 12, color: "var(--fg-dim)", marginLeft: 8 }}>seconds (0 = disabled)</span>
      </div>

      <div className="form-section">Behavior</div>
      <label className="form-check">
        <input type="checkbox" checked={config.auto_paste}
          onChange={(e) => onChange({ auto_paste: e.target.checked })} />
        <span>Auto-paste into active window (Ctrl+V / Cmd+V)</span>
      </label>
      <label className="form-check form-check--disabled">
        <input type="checkbox" defaultChecked disabled />
        <span>Voice auto stop (silence detection)</span>
      </label>
    </>
  );
}
