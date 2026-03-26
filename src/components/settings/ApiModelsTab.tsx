import { useState } from "react";
import type { AppConfig } from "../../types/ipc";

interface Props { config: AppConfig; onChange: (p: Partial<AppConfig>) => void; }

const WHISPER_MODELS = [
  "whisper-large-v3-turbo",
  "whisper-large-v3",
  "distil-whisper-large-v3-en",
];
const CORRECTION_MODELS = [
  "llama-3.3-70b-versatile",
  "llama-3.1-8b-instant",
  "mixtral-8x7b-32768",
  "gemma2-9b-it",
];
const LANGUAGES = ["Auto", "en", "de", "fr", "es", "it", "pt", "nl", "pl", "ru", "ja", "ko", "zh"];

export function ApiModelsTab({ config, onChange }: Props) {
  const [showKey, setShowKey] = useState(false);

  return (
    <>
      <div className="tab__title">API &amp; Models</div>

      <div className="form-section">Authentication</div>
      <div className="form-row">
        <label>Groq API Key</label>
        <input
          type={showKey ? "text" : "password"}
          value={config.groq_api_key}
          onChange={(e) => onChange({ groq_api_key: e.target.value.trim() })}
          placeholder="gsk_…"
          spellCheck={false}
        />
      </div>
      <label className="form-check" style={{ marginBottom: 14 }}>
        <input type="checkbox" checked={showKey}
          onChange={(e) => setShowKey(e.target.checked)} />
        <span>Show key</span>
      </label>

      <div className="form-section">Transcription</div>
      <div className="form-row">
        <label>Whisper Model</label>
        <select value={config.model}
          onChange={(e) => onChange({ model: e.target.value })}>
          {WHISPER_MODELS.map((m) => <option key={m}>{m}</option>)}
        </select>
      </div>
      <div className="form-row">
        <label>Language</label>
        <select
          value={config.language || "Auto"}
          onChange={(e) => onChange({ language: e.target.value === "Auto" ? "" : e.target.value })}
        >
          {LANGUAGES.map((l) => <option key={l}>{l}</option>)}
        </select>
      </div>

      <div className="form-section">Post-Correction</div>
      <label className="form-check" style={{ marginBottom: 10 }}>
        <input type="checkbox" checked={config.post_process}
          onChange={(e) => onChange({ post_process: e.target.checked })} />
        <span>Enable AI post-correction</span>
      </label>
      <label className={`form-check${!config.post_process ? " form-check--disabled" : ""}`} style={{ marginBottom: 8, marginLeft: 16 }}>
        <input type="checkbox" checked={config.filter_fillers} disabled={!config.post_process}
          onChange={(e) => onChange({ filter_fillers: e.target.checked })} />
        <span>Filter filler words (ähm, äh, um, uh…)</span>
      </label>
      <label className={`form-check${!config.post_process ? " form-check--disabled" : ""}`} style={{ marginBottom: 10, marginLeft: 16 }}>
        <input type="checkbox" checked={config.professionalize} disabled={!config.post_process}
          onChange={(e) => onChange({ professionalize: e.target.checked })} />
        <span>Professionalize text</span>
      </label>
      <div className="form-row">
        <label>Correction Model</label>
        <select value={config.correction_model}
          disabled={!config.post_process}
          onChange={(e) => onChange({ correction_model: e.target.value })}>
          {CORRECTION_MODELS.map((m) => <option key={m}>{m}</option>)}
        </select>
      </div>
      <p className="form-dim">Corrects transcription using an LLM after Whisper. Uses the same Groq key.</p>
    </>
  );
}
