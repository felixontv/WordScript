import { useState, useEffect } from "react";
import { useSidecar } from "../hooks/useSidecar";
import type { AppConfig } from "../types/ipc";
import { GeneralTab }     from "../components/settings/GeneralTab";
import { ApiModelsTab }   from "../components/settings/ApiModelsTab";
import { InputTab }       from "../components/settings/InputTab";
import { PromptsTab }     from "../components/settings/PromptsTab";
import { AiAssistantTab } from "../components/settings/AiAssistantTab";
import { AboutTab }       from "../components/settings/AboutTab";
import "../styles/settings.css";

const TABS = ["General", "API & Models", "Input", "Prompts", "AI Assistant", "About"] as const;
type Tab = (typeof TABS)[number];

export default function SettingsWindow() {
  const { state, saveConfig } = useSidecar();
  const [form, setForm]       = useState<AppConfig | null>(null);
  const [active, setActive]   = useState<Tab>("General");
  const [status, setStatus]   = useState<{ msg: string; ok: boolean } | null>(null);

  // Populate form when Python sends config
  useEffect(() => {
    if (state.config && !form) {
      setForm({ ...state.config });
      // If no API key, land on API & Models tab first
      if (!state.config.groq_api_key) setActive("API & Models");
    }
  }, [state.config, form]);

  // Keep form in sync if config reloads externally
  useEffect(() => {
    if (state.config) setForm({ ...state.config });
  }, [state.config]);

  const patch = (partial: Partial<AppConfig>) =>
    setForm((prev) => (prev ? { ...prev, ...partial } : prev));

  const handleSave = async () => {
    if (!form) return;
    try {
      await saveConfig(form);
      setStatus({ msg: "✓  Saved", ok: true });
      setTimeout(() => setStatus(null), 1500);
    } catch (e) {
      setStatus({ msg: `✗  ${e}`, ok: false });
    }
  };

  const handleCancel = async () => {
    // Hide settings window via Tauri API
    const { getCurrentWindow } = await import("@tauri-apps/api/window");
    getCurrentWindow().hide();
  };

  if (!form) {
    return (
      <div style={{
        height: "100vh", display: "flex", alignItems: "center",
        justifyContent: "center", background: "var(--bg)", color: "var(--fg-dim)",
        fontSize: 13,
      }}>
        Connecting to backend…
      </div>
    );
  }

  return (
    <div className="settings">
      {/* Header */}
      <div className="settings__header">
        <span className="settings__title">Settings</span>
      </div>

      {/* No-API-key banner */}
      {!form.groq_api_key && (
        <div className="settings__banner">
          <p>⚠  No API key found — enter yours on the API &amp; Models tab.</p>
          <a onClick={() => window.open("https://console.groq.com/keys")}>
            Get a free key at console.groq.com  ↗
          </a>
        </div>
      )}

      {/* Main: sidebar + content */}
      <div className="settings__body">
        <nav className="settings__sidebar">
          {TABS.map((tab) => (
            <button
              key={tab}
              className={`settings__sidebar-item${active === tab ? " settings__sidebar-item--active" : ""}`}
              onClick={() => setActive(tab)}
            >
              {tab}
            </button>
          ))}
        </nav>
        <div className="settings__sidebar-divider" />

        <div className="settings__content">
          <div className={`tab${active === "General"       ? " tab--active" : ""}`}>
            <GeneralTab config={form} onChange={patch} />
          </div>
          <div className={`tab${active === "API & Models"  ? " tab--active" : ""}`}>
            <ApiModelsTab config={form} onChange={patch} />
          </div>
          <div className={`tab${active === "Input"         ? " tab--active" : ""}`}>
            <InputTab config={form} onChange={patch} />
          </div>
          <div className={`tab${active === "Prompts"       ? " tab--active" : ""}`}>
            <PromptsTab config={form} onChange={patch} />
          </div>
          <div className={`tab${active === "AI Assistant"  ? " tab--active" : ""}`}>
            <AiAssistantTab config={form} onChange={patch} />
          </div>
          <div className={`tab${active === "About"         ? " tab--active" : ""}`}>
            <AboutTab />
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="settings__footer">
        <span className={`settings__footer-status${
          status ? (status.ok ? " settings__footer-status--ok" : " settings__footer-status--err") : ""
        }`}>
          {status?.msg ?? ""}
        </span>
        <div className="settings__footer-btns">
          <button className="btn btn--cancel" onClick={handleCancel}>Cancel</button>
          <button className="btn btn--save"   onClick={handleSave}>Save</button>
        </div>
      </div>
    </div>
  );
}
