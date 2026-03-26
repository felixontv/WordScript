const VERSION = "0.1.5-alpha";

export function AboutTab() {
  const open = (url: string) => window.open(url);

  return (
    <>
      <div className="tab__title">About</div>

      <p style={{ fontSize: 14, marginBottom: 4 }}>WordScript&nbsp;&nbsp;{VERSION}</p>
      <p className="form-dim" style={{ marginBottom: 0 }}>
        Lightweight speech-to-text for your desktop.
      </p>

      <div className="form-sep" />

      <a className="about-link"
        onClick={() => open("https://github.com/felixontv/WordScript")}>
        GitHub — felixontv/WordScript  ↗
      </a>
      <a className="about-link"
        onClick={() => open("https://console.groq.com")}>
        Groq Console (API keys &amp; usage)  ↗
      </a>

      <div className="form-sep" />
      <p style={{ fontSize: 11, fontWeight: 700, color: "var(--fg-dim)", marginBottom: 4 }}>
        Account / Sync
      </p>
      <p className="form-dim">Account management and cloud sync coming in a future release.</p>
    </>
  );
}
