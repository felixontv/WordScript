import type { AppConfig } from "../../types/ipc";

interface Props { config: AppConfig; onChange: (p: Partial<AppConfig>) => void; }

export function PromptsTab({ config, onChange }: Props) {
  return (
    <>
      <div className="tab__title">Prompts</div>

      <div className="form-section">Transcription Context</div>
      <p className="form-dim">
        Optional context fed to Whisper for better accuracy.<br />
        List jargon, names, abbreviations or domain-specific words.
      </p>
      <textarea
        className="form-textarea"
        value={config.prompt}
        rows={6}
        onChange={(e) => onChange({ prompt: e.target.value })}
        placeholder="e.g. WordScript, Groq, PyInstaller, Tauri…"
      />

      <div className="form-sep" />
      <div className="form-section">Prompt Library</div>
      <p className="form-dim">Pre-built presets (Ghostwriter, Code Auditor, SEO Optimizer…) — coming soon.</p>

      <div className="form-section">Personal Dictionary</div>
      <p className="form-dim">Custom word list for improved transcription accuracy — coming soon.</p>
    </>
  );
}
