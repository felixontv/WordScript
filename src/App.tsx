import { Routes, Route, Navigate } from "react-router-dom";
import OverlayWindow from "./windows/OverlayWindow";
import SettingsWindow from "./windows/SettingsWindow";

export default function App() {
  return (
    <Routes>
      <Route path="/overlay"  element={<OverlayWindow />} />
      <Route path="/settings" element={<SettingsWindow />} />
      <Route path="*"         element={<Navigate to="/overlay" replace />} />
    </Routes>
  );
}
