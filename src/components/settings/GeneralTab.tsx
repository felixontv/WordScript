import type { AppConfig } from "../../types/ipc";

interface Props { config: AppConfig; onChange: (p: Partial<AppConfig>) => void; }

export function GeneralTab({ config, onChange }: Props) {
  return (
    <>
      <div className="tab__title">General</div>

      <div className="form-section">Feedback</div>
      <label className="form-check">
        <input type="checkbox" checked={config.play_sounds}
          onChange={(e) => onChange({ play_sounds: e.target.checked })} />
        <span>Play sound feedback</span>
      </label>
      <label className="form-check">
        <input type="checkbox" checked={config.show_tray_icon}
          onChange={(e) => onChange({ show_tray_icon: e.target.checked })} />
        <span>Show tray icon</span>
      </label>

      <div className="form-section">Application</div>
      <label className="form-check form-check--disabled">
        <input type="checkbox" disabled /> <span>Start on system boot</span>
      </label>
      <label className="form-check form-check--disabled">
        <input type="checkbox" defaultChecked disabled /> <span>Minimize to system tray</span>
      </label>
      <label className="form-check form-check--disabled">
        <input type="checkbox" defaultChecked disabled /> <span>Show in taskbar</span>
      </label>

      <div className="form-section">Updates</div>
      <label className="form-check form-check--disabled">
        <input type="checkbox" defaultChecked disabled /> <span>Automatically download updates</span>
      </label>
      <p className="form-dim">Application behavior settings coming in a future release.</p>
    </>
  );
}
