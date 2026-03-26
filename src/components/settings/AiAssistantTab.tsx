import type { AppConfig } from "../../types/ipc";

interface Props { config: AppConfig; onChange: (p: Partial<AppConfig>) => void; }

export function AiAssistantTab({ config: _config, onChange: _onChange }: Props) {
  return (
    <>
      <div className="tab__title">AI Assistant</div>

      <div className="form-section">AI Tone</div>
      <div className="form-row">
        <label>Response Style</label>
        <select disabled>
          <option>Professional &amp; Concise</option>
          <option>Creative &amp; Flowing</option>
          <option>Technical &amp; Precise</option>
        </select>
      </div>

      <div className="form-section">Circle to Search</div>
      <label className="form-check form-check--disabled">
        <input type="checkbox" disabled /> <span>Enable Circle to Search</span>
      </label>
      <label className="form-check form-check--disabled">
        <input type="checkbox" disabled /> <span>Allow AI to read other open apps</span>
      </label>

      <div className="form-sep" />
      <p className="form-dim">AI Assistant features are planned for a future release.</p>
    </>
  );
}
