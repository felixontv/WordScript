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
/// Briefly sets always-on-top so it clears the overlay, then restores.
#[tauri::command]
async fn open_settings_window(app: AppHandle) -> Result<(), String> {
    if let Some(w) = app.get_webview_window("settings") {
        let _ = w.unminimize();
        let _ = w.show();
        let _ = w.set_always_on_top(true);
        let _ = w.set_focus();
        let w2 = w.clone();
        tauri::async_runtime::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
            let _ = w2.set_always_on_top(false);
        });
    }
    Ok(())
}

// ── Sidecar spawning ──────────────────────────────────────────────────────────

fn spawn_sidecar(app: &AppHandle) {
    let handle = app.clone();

    // In development (TAURI_ENV=dev or debug build): run Python from the repo
    // In production: use the bundled sidecar binary (wordscript-sidecar-<triple>)
    let spawn_result = if cfg!(debug_assertions) {
        app.shell()
            .command("python")
            .args(["-m", "wordscript", "sidecar"])
            .current_dir(std::env::current_dir().unwrap_or_default())
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

                                // Rust manages overlay visibility — React only renders state
                                match event_type {
                                    "recording_started" | "processing" => {
                                        if let Some(overlay) = handle.get_webview_window("overlay") {
                                            let _ = overlay.show();
                                            let _ = overlay.set_always_on_top(true);
                                        }
                                    }
                                    "transcription" | "empty" | "error" => {
                                        if let Some(overlay) = handle.get_webview_window("overlay") {
                                            // Small delay so the last state is visible briefly
                                            let ov = overlay.clone();
                                            tauri::async_runtime::spawn(async move {
                                                tokio::time::sleep(
                                                    std::time::Duration::from_millis(300)
                                                ).await;
                                                let _ = ov.hide();
                                            });
                                        }
                                    }
                                    "shutdown" => {
                                        handle.exit(0);
                                    }
                                    _ => {}
                                }

                                // Broadcast to all windows — React subscribes with listen("py-event")
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
                let _ = w.show();
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
                            let _ = w.show();
                            let _ = w.set_always_on_top(true);
                            let _ = w.set_focus();
                            let w2 = w.clone();
                            tauri::async_runtime::spawn(async move {
                                tokio::time::sleep(std::time::Duration::from_millis(150)).await;
                                let _ = w2.set_always_on_top(false);
                            });
                        }
                    }
                    _ => {}
                })
                .build(app)?;

            // ── Settings window: hide on close instead of destroy ─────────
            if let Some(settings) = app.get_webview_window("settings") {
                let s = settings.clone();
                settings.on_window_event(move |event| {
                    if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                        api.prevent_close();
                        let _ = s.hide();
                    }
                });
            }

            // ── Position overlay at bottom-centre of primary monitor ──────
            if let Some(overlay) = app.get_webview_window("overlay") {
                if let Ok(Some(monitor)) = overlay.primary_monitor() {
                    let w = 296i32;
                    let h = 52i32;
                    let sw = monitor.size().width as i32;
                    let sh = monitor.size().height as i32;
                    let x = (sw - w) / 2;
                    let y = sh - h - 90;
                    let _ = overlay.set_position(tauri::PhysicalPosition::new(x, y));
                }
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
