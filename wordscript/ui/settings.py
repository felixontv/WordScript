"""Settings modal — sidebar-nav layout."""

import webbrowser


def open_settings_modal(root_tk, config_ref: list, on_saved=None) -> None:
    """Open a sidebar-nav dark settings window."""
    try:
        import tkinter as tk
        from tkinter import ttk
        import sounddevice as sd
    except ImportError:
        return

    from ..constants import APP_VERSION
    from .theme import THEME, UI_FONT

    T = THEME
    BG      = T["BG"]
    SURFACE = T["SURFACE"]
    SIDEBAR = T["SIDEBAR"]
    FG      = T["FG"]
    FG_DIM  = T["FG_DIM"]
    ACCENT  = T["ACCENT"]
    BTN_BG  = T["BTN_BG"]
    BORDER  = T["BORDER"]
    GREEN   = T["GREEN"]
    RED     = T["RED"]
    NAV_ACT = T["NAV_ACT"]

    cfg = config_ref[0]
    _no_key = not cfg.groq_api_key

    modal = tk.Toplevel(root_tk)
    modal.title("WordScript – Settings")
    modal.resizable(True, True)
    modal.attributes("-topmost", True)
    modal.attributes("-alpha", 0.96)

    modal.configure(bg=BG)

    # ── Styles ────────────────────────────────────────────────────────────
    style = ttk.Style(modal)
    style.theme_use("clam")
    style.configure("Dark.TCheckbutton", background=BG, foreground=FG,
                    font=(UI_FONT, 10), padding=(4, 2))
    style.map("Dark.TCheckbutton",
              background=[("active", BG)], foreground=[("active", ACCENT)])
    style.configure("Dark.TCombobox",
                    fieldbackground=SURFACE, background=SURFACE,
                    foreground=FG, selectbackground=SURFACE,
                    selectforeground=FG, arrowcolor=FG,
                    borderwidth=0, padding=(8, 5))
    style.map("Dark.TCombobox",
              fieldbackground=[("readonly", SURFACE), ("focus", SURFACE)],
              selectbackground=[("readonly", SURFACE)],
              bordercolor=[("focus", BORDER)])
    style.configure("Dark.TEntry",
                    fieldbackground=SURFACE, foreground=FG,
                    insertcolor=FG, borderwidth=0, padding=(8, 5))
    style.map("Dark.TEntry", bordercolor=[("focus", BORDER)])

    # ── Header ────────────────────────────────────────────────────────────
    header_frame = tk.Frame(modal, bg=BG)
    header_frame.pack(fill="x", padx=24, pady=(18, 0))
    tk.Label(header_frame, text="Settings", bg=BG, fg=ACCENT,
             font=(UI_FONT, 13, "bold")).pack(side="left")

    if _no_key:
        banner = tk.Frame(modal, bg="#1a1200", padx=16, pady=10)
        banner.pack(fill="x", padx=24, pady=(8, 0))
        tk.Label(banner,
                 text="⚠️  No API key found — enter yours on the API & Models tab.",
                 bg="#1a1200", fg="#ffcc44", font=(UI_FONT, 9)).pack(anchor="w")
        link = tk.Label(banner, text="Get a free key at console.groq.com  ↗",
                        bg="#1a1200", fg="#6eb5ff",
                        font=(UI_FONT, 9, "underline"), cursor="hand2")
        link.pack(anchor="w", pady=(3, 0))
        link.bind("<Button-1>", lambda _e: webbrowser.open("https://console.groq.com/keys"))

    # ── Main area: sidebar | divider | content ────────────────────────────
    main_area = tk.Frame(modal, bg=BG)
    main_area.pack(fill="both", expand=True, pady=(12, 0))

    sidebar = tk.Frame(main_area, bg=SIDEBAR, width=160)
    sidebar.pack(side="left", fill="y")
    sidebar.pack_propagate(False)

    tk.Frame(main_area, bg=BORDER, width=1).pack(side="left", fill="y")

    content = tk.Frame(main_area, bg=BG)
    content.pack(side="left", fill="both", expand=True)

    # ── Shared field helpers ──────────────────────────────────────────────
    def _field(parent, r, label, widget_fn):
        tk.Label(parent, text=label, bg=BG, fg=FG,
                 font=(UI_FONT, 10)).grid(
            row=r, column=0, sticky="w", padx=(0, 20), pady=7)
        w = widget_fn(parent)
        w.grid(row=r, column=1, sticky="ew", pady=7, ipady=1)
        parent.columnconfigure(1, weight=1)
        return w

    def _check(parent, r, label, var, state="normal"):
        ttk.Checkbutton(parent, text=label, variable=var,
                        style="Dark.TCheckbutton", state=state).grid(
            row=r, column=0, columnspan=2, sticky="w", pady=5)

    def _sep(parent, r):
        tk.Frame(parent, bg=BORDER, height=1).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=(10, 8))

    def _section(parent, r, text):
        tk.Label(parent, text=text.upper(), bg=BG, fg="#444444",
                 font=(UI_FONT, 7, "bold")).grid(
            row=r, column=0, columnspan=2, sticky="w", pady=(14, 2))

    def _dim(parent, r, text):
        tk.Label(parent, text=text, bg=BG, fg=FG_DIM,
                 font=(UI_FONT, 8)).grid(
            row=r, column=0, columnspan=2, sticky="w", pady=(0, 4))

    # ── Page registry + nav ───────────────────────────────────────────────
    _pages    = {}
    _nav_btns = {}
    _active   = [None]

    def _show_page(name):
        if _active[0]:
            _pages[_active[0]].pack_forget()
            _nav_btns[_active[0]].configure(bg=SIDEBAR, fg=FG_DIM)
        _pages[name].pack(fill="both", expand=True)
        _nav_btns[name].configure(bg=NAV_ACT, fg=ACCENT)
        _active[0] = name

    def _page(name):
        f = tk.Frame(content, bg=BG, padx=28, pady=20)
        _pages[name] = f
        return f

    NAV_ITEMS = ["General", "API & Models", "Input", "Prompts", "AI Assistant", "About"]
    tk.Frame(sidebar, bg=SIDEBAR, height=12).pack()
    for _nav in NAV_ITEMS:
        b = tk.Button(
            sidebar, text=_nav,
            bg=SIDEBAR, fg=FG_DIM,
            activebackground=NAV_ACT, activeforeground=ACCENT,
            font=(UI_FONT, 10), relief="flat", bd=0,
            cursor="hand2", anchor="w", padx=16, pady=8,
            command=lambda k=_nav: _show_page(k),
        )
        b.pack(fill="x")
        _nav_btns[_nav] = b

    # ════════════════════════════════════════════════════════════════════
    # PAGE: General
    # ════════════════════════════════════════════════════════════════════
    pg = _page("General")
    tk.Label(pg, text="General", bg=BG, fg=ACCENT,
             font=(UI_FONT, 13, "bold")).grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

    _section(pg, 1, "Feedback")
    play_sounds_var = tk.BooleanVar(value=cfg.play_sounds)
    _check(pg, 2, "Play sound feedback", play_sounds_var)
    tray_var = tk.BooleanVar(value=cfg.show_tray_icon)
    _check(pg, 3, "Show tray icon", tray_var)

    _section(pg, 4, "Application")
    _check(pg, 5, "Start on system boot",    tk.BooleanVar(value=False), state="disabled")
    _check(pg, 6, "Minimize to system tray", tk.BooleanVar(value=True),  state="disabled")
    _check(pg, 7, "Show in taskbar",         tk.BooleanVar(value=True),  state="disabled")

    _section(pg, 8, "Updates")
    _check(pg, 9, "Automatically download updates", tk.BooleanVar(value=True), state="disabled")
    _dim(pg, 10, "Application behavior settings coming in a future release.")

    # ════════════════════════════════════════════════════════════════════
    # PAGE: API & Models
    # ════════════════════════════════════════════════════════════════════
    pg = _page("API & Models")
    tk.Label(pg, text="API & Models", bg=BG, fg=ACCENT,
             font=(UI_FONT, 13, "bold")).grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

    _section(pg, 1, "Authentication")
    api_key_var = tk.StringVar(value=cfg.groq_api_key or "")
    api_entry = _field(pg, 2, "Groq API Key",
                       lambda p: ttk.Entry(p, textvariable=api_key_var,
                                           show="•", width=36, style="Dark.TEntry"))
    show_key_var = tk.BooleanVar(value=False)

    def _toggle_key():
        api_entry.config(show="" if show_key_var.get() else "•")

    ttk.Checkbutton(pg, text="Show key", variable=show_key_var,
                    command=_toggle_key, style="Dark.TCheckbutton").grid(
        row=3, column=1, sticky="w", pady=(0, 4))

    _section(pg, 4, "Transcription")
    model_var = tk.StringVar(value=cfg.model)
    _field(pg, 5, "Whisper Model",
           lambda p: ttk.Combobox(p, textvariable=model_var,
                                  values=["whisper-large-v3-turbo",
                                          "whisper-large-v3",
                                          "distil-whisper-large-v3-en"],
                                  state="readonly", width=30, style="Dark.TCombobox"))
    lang_var = tk.StringVar(value=cfg.language if cfg.language else "Auto")
    _field(pg, 6, "Language",
           lambda p: ttk.Combobox(p, textvariable=lang_var,
                                  values=["Auto", "en", "de", "fr", "es",
                                          "it", "pt", "nl", "pl", "ru",
                                          "ja", "ko", "zh"],
                                  state="readonly", width=30, style="Dark.TCombobox"))

    _section(pg, 7, "Correction")
    corr_model_var = tk.StringVar(value=cfg.correction_model)
    _field(pg, 8, "Correction Model",
           lambda p: ttk.Combobox(p, textvariable=corr_model_var,
                                  values=["llama-3.3-70b-versatile",
                                          "llama-3.1-8b-instant",
                                          "mixtral-8x7b-32768",
                                          "gemma2-9b-it"],
                                  state="readonly", width=30, style="Dark.TCombobox"))

    # ════════════════════════════════════════════════════════════════════
    # PAGE: Input
    # ════════════════════════════════════════════════════════════════════
    pg = _page("Input")
    tk.Label(pg, text="Input", bg=BG, fg=ACCENT,
             font=(UI_FONT, 13, "bold")).grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

    _section(pg, 1, "Hotkey")
    hotkey_var = tk.StringVar(value=cfg.hotkey)
    _field(pg, 2, "Hotkey Combo",
           lambda p: ttk.Entry(p, textvariable=hotkey_var, width=30,
                               style="Dark.TEntry"))
    _dim(pg, 3, "e.g.  ctrl_l+win  or  ctrl_l+alt_l+space")

    mode_var = tk.StringVar(value=cfg.activation_mode)
    _field(pg, 4, "Activation Mode",
           lambda p: ttk.Combobox(p, textvariable=mode_var,
                                  values=["tap", "hold"],
                                  state="readonly", width=30, style="Dark.TCombobox"))
    _dim(pg, 5, "tap = press once to start, press again to stop\n"
               "hold = hold key to record, release to stop")

    _section(pg, 6, "Audio")
    _devices = ["Default (system microphone)"]
    try:
        for _dev in sd.query_devices():
            if _dev["max_input_channels"] > 0:
                _devices.append(_dev["name"])
    except Exception:
        pass
    _cur_dev = "Default (system microphone)" if not cfg.audio_device else cfg.audio_device
    device_var = tk.StringVar(value=_cur_dev)
    _field(pg, 7, "Input Device",
           lambda p: ttk.Combobox(p, textvariable=device_var,
                                  values=_devices, state="readonly",
                                  width=30, style="Dark.TCombobox"))
    _dim(pg, 8, "Max recording: 12 minutes (fixed)")

    _section(pg, 9, "Behavior")
    auto_paste_var = tk.BooleanVar(value=cfg.auto_paste)
    _check(pg, 10, "Auto-paste into active window (Ctrl+V)", auto_paste_var)
    _check(pg, 11, "Voice auto stop (silence detection)",
           tk.BooleanVar(value=True), state="disabled")

    # ════════════════════════════════════════════════════════════════════
    # PAGE: Prompts
    # ════════════════════════════════════════════════════════════════════
    pg = _page("Prompts")
    tk.Label(pg, text="Prompts", bg=BG, fg=ACCENT,
             font=(UI_FONT, 13, "bold")).grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

    _section(pg, 1, "Transcription Context")
    _dim(pg, 2, "Optional context fed to Whisper for better accuracy.\n"
               "List jargon, names, abbreviations or domain-specific words.")

    prompt_frame = tk.Frame(pg, bg=SURFACE, bd=0)
    prompt_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(0, 6))
    pg.rowconfigure(3, weight=1)
    pg.columnconfigure(0, weight=1)
    pg.columnconfigure(1, weight=1)
    prompt_text = tk.Text(
        prompt_frame, bg=SURFACE, fg=FG, insertbackground=FG,
        font=(UI_FONT, 10), relief="flat", bd=0,
        wrap="word", width=46, height=6,
        padx=10, pady=8,
        selectbackground="#333333", selectforeground=FG,
    )
    prompt_sb = ttk.Scrollbar(prompt_frame, orient="vertical", command=prompt_text.yview)
    prompt_text.configure(yscrollcommand=prompt_sb.set)
    prompt_text.pack(side="left", fill="both", expand=True)
    prompt_sb.pack(side="right", fill="y")
    if cfg.prompt:
        prompt_text.insert("1.0", cfg.prompt)

    _section(pg, 4, "Prompt Library")
    _dim(pg, 5, "Pre-built presets (Ghostwriter, Code Auditor, SEO Optimizer…) — coming soon.")

    _section(pg, 6, "Personal Dictionary")
    _dim(pg, 7, "Custom word list for improved transcription accuracy — coming soon.")

    # ════════════════════════════════════════════════════════════════════
    # PAGE: AI Assistant
    # ════════════════════════════════════════════════════════════════════
    pg = _page("AI Assistant")
    tk.Label(pg, text="AI Assistant", bg=BG, fg=ACCENT,
             font=(UI_FONT, 13, "bold")).grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

    _section(pg, 1, "Post-Processing")
    post_var = tk.BooleanVar(value=cfg.post_process)
    _check(pg, 2, "Enable AI post-correction", post_var)
    _check(pg, 3, "Smart punctuation", tk.BooleanVar(value=True), state="disabled")

    _section(pg, 4, "AI Tone")
    _tone_var = tk.StringVar(value="Professional & Concise")
    _field(pg, 5, "Response Style",
           lambda p: ttk.Combobox(p, textvariable=_tone_var,
                                  values=["Professional & Concise",
                                          "Creative & Flowing",
                                          "Technical & Precise"],
                                  state="disabled", width=30, style="Dark.TCombobox"))

    _section(pg, 6, "Circle to Search")
    _check(pg, 7, "Enable Circle to Search",
           tk.BooleanVar(value=False), state="disabled")
    _check(pg, 8, "Allow AI to read other open apps",
           tk.BooleanVar(value=False), state="disabled")
    _dim(pg, 9, "AI Assistant features are planned for a future release.\n"
               "Post-correction is active and uses the model set in API & Models.")

    # ════════════════════════════════════════════════════════════════════
    # PAGE: About
    # ════════════════════════════════════════════════════════════════════
    pg = _page("About")
    tk.Label(pg, text="About", bg=BG, fg=ACCENT,
             font=(UI_FONT, 13, "bold")).grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

    try:
        _ver = APP_VERSION
    except Exception:
        _ver = "unknown"

    tk.Label(pg, text=f"WordScript  {_ver}",
             bg=BG, fg=FG, font=(UI_FONT, 12)).grid(
        row=1, column=0, columnspan=2, sticky="w", pady=(0, 4))
    _dim(pg, 2, "Lightweight speech-to-text for your desktop.")
    _sep(pg, 3)

    def _link(parent, r, label, url):
        lbl = tk.Label(parent, text=label, bg=BG, fg="#6eb5ff",
                       font=(UI_FONT, 9, "underline"), cursor="hand2")
        lbl.grid(row=r, column=0, columnspan=2, sticky="w", pady=3)
        lbl.bind("<Button-1>", lambda _e: webbrowser.open(url))

    _link(pg, 4, "GitHub — felixontv/WordScript  ↗",
          "https://github.com/felixontv/WordScript")
    _link(pg, 5, "Groq Console (API keys & usage)  ↗",
          "https://console.groq.com")

    _sep(pg, 6)
    tk.Label(pg, text="Account / Sync", bg=BG, fg=FG_DIM,
             font=(UI_FONT, 8, "bold")).grid(
        row=7, column=0, columnspan=2, sticky="w", pady=(0, 2))
    _dim(pg, 8, "Account management and cloud sync coming in a future release.")

    # ── Show initial page ─────────────────────────────────────────────────
    _show_page("API & Models" if _no_key else "General")

    # ── Bottom bar ────────────────────────────────────────────────────────
    tk.Frame(modal, bg=BORDER, height=1).pack(fill="x")
    bottom = tk.Frame(modal, bg=BG, padx=24, pady=12)
    bottom.pack(fill="x")

    status_var = tk.StringVar(value="")
    status_lbl = tk.Label(bottom, textvariable=status_var, bg=BG, fg=FG_DIM,
                          font=(UI_FONT, 8))
    status_lbl.pack(side="left", padx=(0, 16))

    btn_frame = tk.Frame(bottom, bg=BG)
    btn_frame.pack(side="right")

    def _save():
        cfg.groq_api_key     = api_key_var.get().strip()
        cfg.model            = model_var.get()
        cfg.language         = "" if lang_var.get() == "Auto" else lang_var.get()
        cfg.prompt           = prompt_text.get("1.0", "end").strip()
        cfg.post_process     = post_var.get()
        cfg.correction_model = corr_model_var.get()
        _dev = device_var.get()
        cfg.audio_device     = "" if _dev.startswith("Default") else _dev
        cfg.hotkey           = hotkey_var.get().strip()
        cfg.activation_mode  = mode_var.get()
        cfg.auto_paste       = auto_paste_var.get()
        cfg.play_sounds      = play_sounds_var.get()
        cfg.show_tray_icon   = tray_var.get()
        try:
            cfg.save()
            if on_saved:
                on_saved()
            status_var.set("✓  Saved")
            status_lbl.configure(fg=GREEN)
            modal.after(700, modal.destroy)
        except Exception as exc:
            status_var.set(f"✗  {exc}")
            status_lbl.configure(fg=RED)

    def _btn_hover(btn, normal_bg, hover_bg):
        btn.bind("<Enter>", lambda _e: btn.configure(bg=hover_bg))
        btn.bind("<Leave>", lambda _e: btn.configure(bg=normal_bg))

    cancel_btn = tk.Button(
        btn_frame, text="  Cancel  ",
        bg=BTN_BG, fg=FG, activebackground=BORDER,
        font=(UI_FONT, 10), relief="flat", cursor="hand2", bd=0,
        command=modal.destroy, padx=18, pady=7,
    )
    cancel_btn.pack(side="left", padx=(0, 10))
    _btn_hover(cancel_btn, BTN_BG, BORDER)

    save_btn = tk.Button(
        btn_frame, text="  Save  ",
        bg=ACCENT, fg="#000000", activebackground="#d0d0d0",
        font=(UI_FONT, 10, "bold"), relief="flat", cursor="hand2", bd=0,
        padx=18, pady=7,
    )
    save_btn.pack(side="left")
    _btn_hover(save_btn, ACCENT, "#cccccc")

    def _save_animated():
        _save()
        if status_var.get().startswith("✓"):
            save_btn.configure(bg=GREEN, fg="#000000")
            modal.after(600, lambda: save_btn.configure(bg=ACCENT))

    save_btn.configure(command=_save_animated)

    # ── Size and centre ───────────────────────────────────────────────────
    modal.update_idletasks()
    mw = modal.winfo_reqwidth()
    mh = modal.winfo_reqheight()
    sw = modal.winfo_screenwidth()
    sh = modal.winfo_screenheight()
    modal.geometry(
        f"{max(mw, 700)}x{max(mh, 520)}"
        f"+{(sw - max(mw, 700)) // 2}+{(sh - max(mh, 520)) // 2}"
    )
