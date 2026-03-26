// Suppresses the terminal window on Windows in release builds
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    wordscript_lib::run();
}
