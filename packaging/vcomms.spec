# PyInstaller spec for V-COMMS (one-folder, windowed build).
#
# Build from the repo root with:  pyinstaller --noconfirm packaging\vcomms.spec
# or just run packaging\build.ps1, which also builds the installer.
#
# Works with either Qt binding: the app currently uses PySide6, and the Fluent
# redesign moves it to PyQt6 + PyQt6-Fluent-Widgets. The binding is detected
# from val_voice_changer.py so only one Qt is bundled (PyInstaller refuses to
# collect two Qt bindings into one app).

import re
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent
ENTRY = ROOT / "val_voice_changer.py"
source = ENTRY.read_text(encoding="utf-8")

uses_pyqt6 = re.search(r"^\s*(from|import)\s+PyQt6\b", source, re.M) is not None
uses_fluent = re.search(r"^\s*(from|import)\s+qfluentwidgets\b", source, re.M) is not None

QT_BINDINGS = {"PySide6", "PyQt6", "PySide2", "PyQt5"}
qt_binding = "PyQt6" if uses_pyqt6 else "PySide6"

hiddenimports = []
datas = []
if uses_fluent:
    # qfluentwidgets and its frameless-window dependency load parts of
    # themselves dynamically, so collect them whole.
    for pkg in ("qfluentwidgets", "qframelesswindow"):
        hiddenimports += collect_submodules(pkg)
        datas += collect_data_files(pkg)

excludes = sorted(QT_BINDINGS - {qt_binding}) + [
    "tkinter",
    "matplotlib",
    "IPython",
    "pytest",
]

a = Analysis(
    [str(ENTRY)],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="V-COMMS",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="V-COMMS",
)
