// Suppresses the terminal window on Windows in release builds
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    // Force X11/XWayland on Linux to avoid Wayland Gdk Error 71 crashes
    // (transparent/decorationless overlay + WebKitGTK bugs on native Wayland)
    #[cfg(target_os = "linux")]
    {
        unsafe {
            std::env::set_var("GDK_BACKEND", "x11");
            // Disable GPU compositing — GBM buffer creation fails under XWayland
            std::env::set_var("WEBKIT_DISABLE_COMPOSITING_MODE", "1");
            // WebKitGTK also reads WAYLAND_DISPLAY directly — hide it so it falls back to X11
            if std::env::var("WAYLAND_DISPLAY").is_ok() {
                std::env::set_var("WORDSCRIPT_WAS_WAYLAND", "1");
                std::env::remove_var("WAYLAND_DISPLAY");
            }
        }
    }

    // Prevent stale .pyc cache from causing intermittent Python import bugs
    unsafe { std::env::set_var("PYTHONDONTWRITEBYTECODE", "1"); }

    wordscript_lib::run();
}
