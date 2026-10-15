# PyInstaller spec for V-COMMS (one-folder, windowed build).
#
# Build from the repo root with:  pyinstaller --noconfirm packaging\vcomms.spec
# or just run packaging\build.ps1, which also builds the installer.
#
# The entry point is vcomms_main.py (single instance, data folders, then the GUI). The Qt binding is detected from
# val_voice_changer.py so only one Qt is bundled (PyInstaller refuses to collect two Qt bindings into one app).
# The version shown in the exe's Properties and in Apps comes from appinfo.APP_VERSION.

import re
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules
from PyInstaller.utils.win32.versioninfo import (
    FixedFileInfo,
    StringFileInfo,
    StringStruct,
    StringTable,
    VarFileInfo,
    VarStruct,
    VSVersionInfo,
)

ROOT = Path(SPECPATH).parent
ENTRY = ROOT / "vcomms_main.py"
source = (ROOT / "val_voice_changer.py").read_text(encoding="utf-8")
appinfo = (ROOT / "appinfo.py").read_text(encoding="utf-8")
APP_VERSION = re.search(r'^APP_VERSION\s*=\s*"([^"]+)"', appinfo, re.M).group(1)
numbers = tuple(int(x) for x in (re.findall(r"\d+", APP_VERSION) + ["0"] * 4)[:4])

uses_pyqt6 = re.search(r"^\s*(from|import)\s+PyQt6\b", source, re.M) is not None
uses_fluent = re.search(r"^\s*(from|import)\s+qfluentwidgets\b", source, re.M) is not None

QT_BINDINGS = {"PySide6", "PyQt6", "PySide2", "PyQt5"}
qt_binding = "PyQt6" if uses_pyqt6 else "PySide6"

hiddenimports = ["PyQt6.QtNetwork"]
datas = [(str(ROOT / "app_icon.ico"), ".")]  # main_window.make_app_icon looks for it next to the code
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

version_info = VSVersionInfo(
    ffi=FixedFileInfo(filevers=numbers, prodvers=numbers),
    kids=[
        StringFileInfo([
            StringTable("040904B0", [
                StringStruct("CompanyName", "Aayush"),
                StringStruct("FileDescription", "V-COMMS voice changer"),
                StringStruct("FileVersion", APP_VERSION),
                StringStruct("InternalName", "V-COMMS"),
                StringStruct("LegalCopyright", "Copyright (c) Aayush"),
                StringStruct("OriginalFilename", "V-COMMS.exe"),
                StringStruct("ProductName", "V-COMMS"),
                StringStruct("ProductVersion", APP_VERSION),
            ])
        ]),
        VarFileInfo([VarStruct("Translation", [0x0409, 1200])]),
    ],
)

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
    icon=str(ROOT / "app_icon.ico"),
    version=version_info,
    uac_admin=False,  # never asks for admin; only the VB-CABLE step of the installer does
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
