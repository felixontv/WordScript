// ── Python → Tauri events (received via listen("py-event")) ──────────────────

export interface AppConfig {
  groq_api_key:            string;
  model:                   string;
  language:                string;
  prompt:                  string;
  post_process:            boolean;
  correction_model:        string;
  filter_fillers:          boolean;
  professionalize:         boolean;
  backend:                 string;
  local_model:             string;
  hotkey:                  string;
  activation_mode:         "tap" | "hold";
  sample_rate:             number;
  channels:                number;
  dtype:                   string;
  audio_device:            string;
  max_recording_seconds:   number;
  silence_timeout_seconds: number;
  auto_paste:              boolean;
  show_tray_icon:          boolean;
  play_sounds:             boolean;
  log_level:               string;
  temp_audio_dir:          string;
}

export type PythonEvent =
  | { event: "ready";            version: string; config: AppConfig }
  | { event: "recording_started" }
  | { event: "recording_stopped" }
  | { event: "processing" }
  | { event: "transcription";    text: string; corrected: boolean }
  | { event: "empty" }
  | { event: "muted";            muted: boolean }
  | { event: "error";            message: string }
  | { event: "audio_level";      level: number }
  | { event: "update_available"; version: string; url: string }
  | { event: "shutdown" };

// ── React → Tauri → Python commands ──────────────────────────────────────────

export type PythonCommand =
  | { cmd: "start_recording" }
  | { cmd: "stop_recording" }
  | { cmd: "abort_recording" }
  | { cmd: "toggle_mute" }
  | { cmd: "reload_config" }
  | { cmd: "save_config"; config: AppConfig }
  | { cmd: "open_settings" }
  | { cmd: "shutdown" };

// ── Sidecar state (derived in useSidecar) ────────────────────────────────────

export type SidecarStatus = "idle" | "recording" | "processing";

export interface SidecarState {
  status:            SidecarStatus;
  config:            AppConfig | null;
  muted:             boolean;
  lastTranscription: string | null;
  error:             string | null;
  recordingStartMs:  number | null;   // Date.now() when recording started
}
