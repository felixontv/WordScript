import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(async () => ({
  plugins: [react()],
  // Tauri expects a fixed origin in dev; don't expose to network
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    watch: {
      // Don't trigger rebuilds when Rust files change
      ignored: ["**/src-tauri/**"],
    },
  },
  // Required for Tauri to load assets with relative paths
  base: "./",
  build: {
    // Tauri targets ES2021 minimum on all supported platforms
    target: ["es2021", "chrome105", "safari15"],
    // Don't minify for better debuggability (Tauri bundles the whole thing anyway)
    minify: !process.env.TAURI_DEBUG ? "esbuild" : false,
    // Produce sourcemaps in dev mode for easier debugging
    sourcemap: !!process.env.TAURI_DEBUG,
    outDir: "dist",
  },
}));
