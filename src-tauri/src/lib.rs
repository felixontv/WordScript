use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager};
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::TrayIconBuilder;
use tauri_plugin_shell::ShellExt;
use tauri_plugin_shell::process::{CommandChild, CommandEvent};

// ── State ─────────────────────────────────────────────────────────────────────

struct SidecarState {
    child: Mutex<Option<CommandChild>>,
}

// ── Tauri Commands (callable from React via invoke()) ─────────────────────────

/// Write a JSON command to the Python sidecar's stdin.
/// The `cmd` argument is already a JSON string (e.g. `{"cmd":"start_recording"}`).
#[tauri::command]
async fn send_to_python(
    cmd: String,
    state: tauri::State<'_, SidecarState>,
) -> Result<(), String> {
    let mut guard = state.child.lock().unwrap();
    if let Some(child) = guard.as_mut() {
        let line = format!("{}\n", cmd.trim());
        child.write(line.as_bytes()).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Save config via Python (so Python's path resolver handles all platforms).
/// Passes the config object as a JSON payload inside a save_config command.
#[tauri::command]
async fn save_config(
    config: serde_json::Value,
    state: tauri::State<'_, SidecarState>,
) -> Result<(), String> {
    let payload = serde_json::json!({ "cmd": "save_config", "config": config });
    let line = format!("{}\n", payload);
    let mut guard = state.child.lock().unwrap();
    if let Some(child) = guard.as_mut() {
        child.write(line.as_bytes()).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Show (and focus) the settings window.
#[tauri::command]
async fn open_settings_window(app: AppHandle) -> Result<(), String> {
    if let Some(w) = app.get_webview_window("settings") {
        let _ = w.unminimize();
        let _ = w.set_focus();
    }
    Ok(())
}

// ── Sidecar spawning ──────────────────────────────────────────────────────────

fn spawn_sidecar(app: &AppHandle) {
    let handle = app.clone();

    // In development (TAURI_ENV=dev or debug build): run Python from the repo
    // In production: use the bundled sidecar binary (wordscript-sidecar-<triple>)
    let spawn_result = if cfg!(debug_assertions) {
        // Resolve project root from the Cargo manifest dir embedded at compile time
        let project_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap_or(std::path::Path::new("."));
        let venv_python = project_root.join(".venv/bin/python");
        let python_cmd = if venv_python.exists() {
            venv_python.to_string_lossy().to_string()
        } else {
            "python".to_string()
        };
        app.shell()
            .command(&python_cmd)
            .args(["-m", "wordscript", "sidecar"])
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .current_dir(project_root)
            .spawn()
    } else {
        app.shell()
            .sidecar("wordscript-sidecar")
            .unwrap()
            .args(["sidecar"])
            .spawn()
    };

    match spawn_result {
        Ok((mut rx, child)) => {
            *app.state::<SidecarState>().child.lock().unwrap() = Some(child);

            // Forward Python stdout events to all Tauri windows
            tauri::async_runtime::spawn(async move {
                while let Some(event) = rx.recv().await {
                    match event {
                        CommandEvent::Stdout(bytes) => {
                            let line = String::from_utf8_lossy(&bytes);
                            let line = line.trim();
                            if line.is_empty() { continue; }

                            if let Ok(value) = serde_json::from_str::<serde_json::Value>(line) {
                                let event_type = value
                                    .get("event")
                                    .and_then(|e| e.as_str())
                                    .unwrap_or("");

                                // Overlay visibility is driven purely by CSS in React.
                                // No GTK show()/hide()/set_size()/set_position() calls —
                                // those crash under XWayland after repeated cycles.
                                match event_type {
                                    "shutdown" => {
                                        handle.exit(0);
                                    }
                                    _ => {}
                                }

                            let _ = handle.emit("py-event", &value);
                            }
                        }
                        CommandEvent::Stderr(bytes) => {
                            eprintln!("[Python] {}", String::from_utf8_lossy(&bytes).trim());
                        }
                        CommandEvent::Error(e) => {
                            eprintln!("[Python Error] {}", e);
                            let _ = handle.emit("py-event", serde_json::json!({
                                "event": "error",
                                "message": format!("Sidecar process error: {}", e)
                            }));
                        }
                        CommandEvent::Terminated(status) => {
                            eprintln!("[Python] exited (code {:?})", status.code);
                        }
                        _ => {}
                    }
                }
            });
        }
        Err(e) => {
            eprintln!("[WordScript] Failed to spawn Python sidecar: {}", e);
            // Emit error so the UI can surface it
            let _ = app.emit("py-event", serde_json::json!({
                "event": "error",
                "message": format!("Could not start Python backend: {}", e)
            }));
        }
    }
}

// ── App entry ─────────────────────────────────────────────────────────────────

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            // Second instance tried to launch — focus settings window instead
            if let Some(w) = app.get_webview_window("settings") {
                let _ = w.unminimize();
                let _ = w.set_focus();
            }
        }))
        .plugin(tauri_plugin_shell::init())
        .manage(SidecarState { child: Mutex::new(None) })
        .setup(|app| {
            // ── System tray ───────────────────────────────────────────────
            let title   = MenuItem::with_id(app, "title",    "WordScript", false, None::<&str>)?;
            let sep1    = PredefinedMenuItem::separator(app)?;
            let settings = MenuItem::with_id(app, "settings", "Settings",  true,  None::<&str>)?;
            let sep2    = PredefinedMenuItem::separator(app)?;
            let quit    = MenuItem::with_id(app, "quit",     "Quit",       true,  None::<&str>)?;
            let menu    = Menu::with_items(app, &[&title, &sep1, &settings, &sep2, &quit])?;

            let tray_icon = app.default_window_icon()
                .cloned()
                .expect("No default window icon configured — add an icon to tauri.conf.json bundle.icon");
            TrayIconBuilder::new()
                .icon(tray_icon)
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "quit" => {
                        // Graceful shutdown: tell Python first
                        if let Some(child) = app.state::<SidecarState>()
                            .child.lock().unwrap().as_mut()
                        {
                            let _ = child.write(b"{\"cmd\":\"shutdown\"}\n");
                        }
                        app.exit(0);
                    }
                    "settings" => {
                        if let Some(w) = app.get_webview_window("settings") {
                            let _ = w.unminimize();
                            let _ = w.set_focus();
                        }
                    }
                    _ => {}
                })
                .build(app)?;

            // ── Settings window: minimize on close instead of destroy ────
            if let Some(settings) = app.get_webview_window("settings") {
                let s = settings.clone();
                settings.on_window_event(move |event| {
                    if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                        api.prevent_close();
                        let _ = s.minimize();
                    }
                });
            }

            // ── Spawn Python backend ──────────────────────────────────────
            spawn_sidecar(app.handle());

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            send_to_python,
            save_config,
            open_settings_window,
        ])
        .run(tauri::generate_context!())
        .expect("error while running WordScript");
}
