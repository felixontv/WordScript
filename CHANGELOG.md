v0.2.0-alpha

### Fixed
- **Linux/Wayland:** Fatal Gdk Error 71 Crash beim App-Start. Ursache: WebKitGTK + transparente/dekorationslose Fenster auf nativem Wayland. Fix: `WAYLAND_DISPLAY` wird in `main.rs` entfernt, `GDK_BACKEND=x11` erzwungen → App läuft stabil via XWayland.
- **Linux/Wayland:** Overlay `show()`/`hide()`/`set_always_on_top()` crashten unter Wayland. Fix: Overlay-Sichtbarkeit wird jetzt über `set_position()` (on-/off-screen) gesteuert statt show/hide.
- **Linux/Wayland:** Settings-Fenster `hide()`/`show()` crashte auf Wayland. Fix: `minimize()`/`unminimize()` statt hide/show, `visible: true` in tauri.conf.json.
- **Alle Plattformen:** Config-Pfad im Dev-Modus war `./config.json` (Projekt-Root) statt `~/.config/WordScript/config.json`. API-Key wurde nicht gefunden → kein Groq-Client → Transkription schlug immer fehl. Fix: Einheitlicher Config-Pfad für Dev und Production.
- **Alle Plattformen:** Groq API-Calls hingen ~20–60s wegen IPv6-Connect-Timeout. Fix: IPv4-Transport (`local_address='0.0.0.0'`) erzwungen (aus v0.1.6, jetzt wirksam durch Config-Path-Fix).

### Removed
- Debug-Code aus SettingsWindow.tsx (Event-Counter), lib.rs (Emit-Logging)

v0.1.6-alpha

### Fixed
- **Alle Plattformen:** Groq API-Calls hingen ~20–60s wegen IPv6-Connect-Timeout (`httpx` versucht zuerst AAAA-Records, fällt erst nach Timeout auf IPv4 zurück). Fix: IPv4-Transport (`local_address='0.0.0.0'`) plattformübergreifend erzwungen.
- **Dev-Mode (Linux):** Python-Sidecar startete mit System-Python statt venv — fehlende Dependencies. Fix: `CARGO_MANIFEST_DIR` resolves das Projekt-Root zur Compile-Zeit, findet `.venv/bin/python` zuverlässig.

v0.1.5-alpha

### Fixed
- Linux: Groq API-Calls hingen ~20–60s wegen fehlgeschlagenem IPv6-Fallback (`httpx` versucht zuerst AAAA-Records). Fix: IPv4-Transport (`local_address='0.0.0.0'`) auf Linux erzwungen.
