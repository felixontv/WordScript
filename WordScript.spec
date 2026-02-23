# -*- mode: python ; coding: utf-8 -*-
import sys

# ── Platform-aware output name & hidden imports ───────────────────────────
if sys.platform == 'win32':
    _name = 'WordScript-windows'
    _console = False
    _plyer_platform = [
        'plyer.platforms.win',
        'plyer.platforms.win.notification',
    ]
elif sys.platform == 'darwin':
    _name = 'WordScript-macos'
    _console = False
    _plyer_platform = [
        'plyer.platforms.macosx',
        'plyer.platforms.macosx.notification',
    ]
else:
    _name = 'WordScript-linux'
    _console = False
    _plyer_platform = [
        'plyer.platforms.linux',
        'plyer.platforms.linux.notification',
    ]

_hidden = [
    'numpy', 'sounddevice', 'groq',
    'pyautogui', 'pyperclip',
    'pynput', 'pynput.keyboard',
    'pystray',
    'PIL', 'PIL.Image', 'PIL.ImageDraw', 'PIL.ImageFont',
    'tkinter',
    'packaging', 'packaging.version',
    'plyer',
] + _plyer_platform


a = Analysis(
    ['speech_to_text.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=_hidden,
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
    name=_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=_console,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
