import { useEffect, useReducer, useCallback } from "react";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import type {
  AppConfig,
  PythonCommand,
  PythonEvent,
  SidecarState,
} from "../types/ipc";

// ── State machine ─────────────────────────────────────────────────────────────

type Action =
  | { type: "READY";      config: AppConfig }
  | { type: "RECORDING_STARTED" }
  | { type: "RECORDING_STOPPED" }
  | { type: "PROCESSING" }
  | { type: "TRANSCRIPTION"; text: string }
  | { type: "EMPTY" }
  | { type: "MUTED"; muted: boolean }
  | { type: "ERROR"; message: string };

const initial: SidecarState = {
  status:            "idle",
  config:            null,
  muted:             false,
  lastTranscription: null,
  error:             null,
  recordingStartMs:  null,
};

function reducer(state: SidecarState, action: Action): SidecarState {
  switch (action.type) {
    case "READY":
      return { ...state, config: action.config, error: null };
    case "RECORDING_STARTED":
      return { ...state, status: "recording", muted: false, error: null, recordingStartMs: Date.now() };
    case "RECORDING_STOPPED":
      return { ...state, recordingStartMs: null };
    case "PROCESSING":
      return { ...state, status: "processing" };
    case "TRANSCRIPTION":
      return { ...state, status: "idle", lastTranscription: action.text };
    case "EMPTY":
      return { ...state, status: "idle" };
    case "MUTED":
      return { ...state, muted: action.muted };
    case "ERROR":
      return { ...state, status: "idle", error: action.message };
    default:
      return state;
  }
}

// ── Hook ─────────────────────────────────────────────────────────────────────

export function useSidecar() {
  const [state, dispatch] = useReducer(reducer, initial);

  useEffect(() => {
    // Subscribe to all Python events on the single "py-event" channel
    const unlisten = listen<PythonEvent>("py-event", ({ payload }) => {
      if (payload.event === "audio_level") return; // handled directly in OverlayWindow
      switch (payload.event) {
        case "ready":
          dispatch({ type: "READY", config: payload.config });
          break;
        case "recording_started":
          dispatch({ type: "RECORDING_STARTED" });
          break;
        case "recording_stopped":
          dispatch({ type: "RECORDING_STOPPED" });
          break;
        case "processing":
          dispatch({ type: "PROCESSING" });
          break;
        case "transcription":
          dispatch({ type: "TRANSCRIPTION", text: payload.text });
          break;
        case "empty":
          dispatch({ type: "EMPTY" });
          break;
        case "muted":
          dispatch({ type: "MUTED", muted: payload.muted });
          break;
        case "error":
          dispatch({ type: "ERROR", message: payload.message });
          break;
      }
    });
    // Request current state — fixes the race where WebView loads after `ready` was emitted
    const configTimer = window.setTimeout(() => {
      invoke("send_to_python", { cmd: JSON.stringify({ cmd: "reload_config" }) }).catch(() => {});
    }, 300);

    return () => {
      unlisten.then((fn) => fn());
      window.clearTimeout(configTimer);
    };
  }, []);

  const sendCommand = useCallback(async (cmd: PythonCommand) => {
    try {
      await invoke<void>("send_to_python", { cmd: JSON.stringify(cmd) });
    } catch (e) {
      console.error("sendCommand failed:", e);
    }
  }, []);

  const saveConfig = useCallback(async (config: AppConfig) => {
    try {
      await invoke<void>("save_config", { config });
    } catch (e) {
      console.error("saveConfig failed:", e);
    }
  }, []);

  const openSettings = useCallback(async () => {
    try {
      await invoke<void>("open_settings_window");
    } catch (e) {
      console.error("openSettings failed:", e);
    }
  }, []);

  return { state, sendCommand, saveConfig, openSettings };
}
