"""
Where V-COMMS keeps its files, like any other Windows app, and the one-time move of what older versions left behind.

  %APPDATA%\\V-COMMS        settings.ini, tactical_voice_profile.json, custom_presets.json, hotkeys.json
                           (the player's own data; roams with the Windows account)
  %LOCALAPPDATA%\\V-COMMS   vcomms.log and anything that can be rebuilt (stays on this PC)

Nothing is written next to the exe or in the install folder. Nothing here is shared between Windows users, so
ProgramData is not used.

v1.0.2 was a single portable exe: it saved the profile and the custom presets in the folder it was started from
(usually Downloads) and its window settings in the registry (HKCU\\Software\\V-COMMS). ``migrate_legacy`` finds those
the first time this version runs and copies them in. The old files are left where they are; nothing is deleted.

No Qt at import time (the settings helpers import it lazily), so the launcher can use this before the GUI loads.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from appinfo import APP_NAME

PROFILE_FILENAME = "tactical_voice_profile.json"
PRESETS_FILENAME = "custom_presets.json"
HOTKEYS_FILENAME = "hotkeys.json"
SETTINGS_FILENAME = "settings.ini"
MIGRATION_FILENAME = "migration.json"
LEGACY_DIRS_FILENAME = "legacy_dirs.txt"   # written by the installer: folders where it saw an old V-COMMS.exe
LEGACY_EXE = "V-COMMS.exe"
MUICACHE_KEY = r"Software\Classes\Local Settings\Software\Microsoft\Windows\Shell\MuiCache"


# ------------------------------------------------------------------------------------------------ folders
def _base(env_name: str) -> Path:
    base = os.environ.get(env_name)
    if not base:
        base = str(Path.home() / ("AppData/Roaming" if env_name == "APPDATA" else "AppData/Local"))
    return Path(base) / APP_NAME


def roaming_dir(create: bool = True) -> Path:
    """%APPDATA%\\V-COMMS: settings, profile, presets, hotkeys."""
    path = _base("APPDATA")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def local_dir(create: bool = True) -> Path:
    """%LOCALAPPDATA%\\V-COMMS: the log and caches."""
    path = _base("LOCALAPPDATA")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def profile_path() -> Path:
    return roaming_dir() / PROFILE_FILENAME


def presets_path() -> Path:
    return roaming_dir() / PRESETS_FILENAME


def hotkeys_path() -> Path:
    return roaming_dir() / HOTKEYS_FILENAME


def settings_path() -> Path:
    return roaming_dir() / SETTINGS_FILENAME


def documents_dir() -> Path:
    """Where save and export dialogs start. The user's Documents folder (wherever Windows has moved it)."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
            if ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf) == 0 and buf.value:  # CSIDL_PERSONAL
                return Path(buf.value)
        except Exception:
            pass
    docs = Path.home() / "Documents"
    return docs if docs.is_dir() else Path.home()


def use_documents_as_working_dir() -> None:
    """A relative path in a file dialog (``My preset.json``) starts in the working directory, which for an installed
    app is the install folder. Start in Documents instead, so nothing lands next to the exe by accident."""
    try:
        os.chdir(documents_dir())
    except OSError:
        pass


# ------------------------------------------------------------------------------------------------ settings
def qsettings():
    """The app's QSettings, kept as %APPDATA%\\V-COMMS\\settings.ini (v1.0.2 used the registry)."""
    from PyQt6.QtCore import QSettings

    return QSettings(str(settings_path()), QSettings.Format.IniFormat)


def _copy_registry_settings() -> int:
    """Copies v1.0.2's QSettings (HKCU\\Software\\V-COMMS\\V-COMMS) into settings.ini once. Returns keys copied."""
    from PyQt6.QtCore import QSettings

    old = QSettings(APP_NAME, APP_NAME)
    keys = old.allKeys()
    if not keys:
        return 0
    new = qsettings()
    for key in keys:
        if not new.contains(key):
            new.setValue(key, old.value(key))
    new.sync()
    return len(keys)


# ------------------------------------------------------------------------------------------------ v1.0.2 data
def _muicache_dirs() -> List[Path]:
    """Folders Windows remembers running a V-COMMS.exe from (per user; Explorer records every exe it starts)."""
    if sys.platform != "win32":
        return []
    found: List[Path] = []
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, MUICACHE_KEY) as key:
            i = 0
            while True:
                try:
                    name, _value, _kind = winreg.EnumValue(key, i)
                except OSError:
                    break
                i += 1
                exe = name.rsplit(".", 1)[0] if name.endswith((".FriendlyAppName", ".ApplicationCompany")) else name
                if exe.lower().endswith("\\" + LEGACY_EXE.lower()):
                    found.append(Path(exe).parent)
    except OSError:
        pass
    return found


def _installer_dirs() -> List[Path]:
    path = local_dir(create=False) / LEGACY_DIRS_FILENAME
    try:
        return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return []


def legacy_dirs(extra: Iterable[Path] = ()) -> List[Path]:
    """Every folder an older V-COMMS may have saved its profile or presets in, without duplicates."""
    home = Path.home()
    candidates: List[Path] = list(extra)
    candidates += _installer_dirs() + _muicache_dirs()
    candidates += [Path.cwd(), Path(sys.executable).parent]
    candidates += [home / "Downloads", home / "Desktop", home / "Documents", documents_dir()]
    candidates += [Path(os.environ.get("LOCALAPPDATA", home / "AppData/Local")) / "Programs" / APP_NAME]
    seen, out = set(), []
    data_dir = os.path.normcase(str(roaming_dir(create=False)))
    for c in candidates:
        try:
            key = os.path.normcase(str(c.resolve()))
        except OSError:
            continue
        if key in seen or key == data_dir:
            continue
        seen.add(key)
        if c.is_dir():
            out.append(c)
    return out


def _newest(paths: List[Path]) -> Optional[Path]:
    paths = [p for p in paths if p.is_file()]
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def _merge_presets(sources: List[Path], target: Path) -> List[str]:
    """Adds every preset from ``sources`` to the library at ``target``. A name already taken by a different preset is
    kept as 'Name (2)'; an identical one is skipped. Returns the names added."""
    import presets as pr

    library = pr.PresetLibrary(target)
    library.load()
    if library.load_error:  # never overwrite a library we cannot read
        raise OSError(f"{target.name}: {library.load_error}")
    added: List[str] = []
    for src in sources:
        other = pr.PresetLibrary(src)
        other.load()
        for name, params in other.items.items():
            mine = library.items.get(name)
            if mine is not None and all(getattr(mine, k) == getattr(params, k) for k in pr.PARAM_FIELDS):
                continue
            if len(library.items) >= pr.MAX_CUSTOM_PRESETS:
                return added
            added.append(library.add(pr.unique_name(name, library.names()), params))
    return added


def migrate_legacy(extra_dirs: Iterable[Path] = (), force: bool = False) -> Dict:
    """
    Runs once (a marker file records it). Copies the newest v1.0.2 profile and merges every v1.0.2 preset library
    into %APPDATA%\\V-COMMS, copies the old registry settings into settings.ini, and moves the hotkeys file from
    %LOCALAPPDATA% (where an early build of the hotkey editor kept it). Never deletes the old files.
    """
    marker = roaming_dir() / MIGRATION_FILENAME
    if marker.exists() and not force:
        return {"skipped": True}
    report: Dict = {"version": 1, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "profile": "", "presets_from": [],
                    "presets_added": [], "settings_keys": 0, "hotkeys": "", "errors": []}
    dirs = legacy_dirs(extra_dirs)

    profiles = [d / PROFILE_FILENAME for d in dirs]
    target = profile_path()
    newest = _newest(profiles)
    if newest is not None and not target.exists():
        try:
            shutil.copy2(newest, target)
            report["profile"] = str(newest)
        except OSError as exc:
            report["errors"].append(f"profile: {exc}")

    libraries = [p for p in (d / PRESETS_FILENAME for d in dirs) if p.is_file()]
    if libraries:
        try:
            if not presets_path().exists() and len(libraries) == 1:
                shutil.copy2(libraries[0], presets_path())
                import presets as pr

                lib = pr.PresetLibrary(presets_path())
                lib.load()
                report["presets_added"] = lib.names()
            else:
                report["presets_added"] = _merge_presets(libraries, presets_path())
            report["presets_from"] = [str(p) for p in libraries]
        except Exception as exc:  # keep going: a damaged old file must not stop the app from starting
            report["errors"].append(f"presets: {exc}")

    old_hotkeys = local_dir(create=False) / HOTKEYS_FILENAME
    if old_hotkeys.is_file() and not hotkeys_path().exists():
        try:
            shutil.copy2(old_hotkeys, hotkeys_path())
            report["hotkeys"] = str(old_hotkeys)
        except OSError as exc:
            report["errors"].append(f"hotkeys: {exc}")

    try:
        report["settings_keys"] = _copy_registry_settings()
    except Exception as exc:
        report["errors"].append(f"settings: {exc}")

    try:
        marker.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except OSError:
        pass
    return report


# ------------------------------------------------------------------------------------------------ shell
def open_folder(path: Path) -> bool:
    """Opens ``path`` in File Explorer."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(str(path))  # type: ignore[attr-defined]  (Windows only)
        return True
    except (OSError, AttributeError):
        return False
