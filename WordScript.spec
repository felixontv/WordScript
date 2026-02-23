# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['speech_to_text.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['numpy', 'sounddevice', 'groq', 'pyautogui', 'pyperclip', 'pynput', 'pynput.keyboard', 'pystray', 'PIL', 'PIL.Image', 'PIL.ImageDraw', 'PIL.ImageFont', 'tkinter', 'packaging', 'packaging.version', 'plyer', 'plyer.platforms', 'plyer.platforms.win', 'plyer.platforms.win.notification'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='WordScript',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
