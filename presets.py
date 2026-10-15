"""
Custom voice presets: the library file, the naming rules and the import/export format. No Qt in here.

A preset file is untrusted input (a friend sends you one). It is read as plain JSON and never evaluated, it may not
be larger than 64 KB, its format and version must match exactly, every value must be a finite number, unknown keys
are ignored, and every value is clamped to the range of its slider before it can reach the DSP.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import fields
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import engine as eng
import preset_names as names_rules
from preset_names import MAX_CUSTOM_PRESETS, NAME_MAX, normalize_name  # noqa: F401  (re-exported)

PRESET_FORMAT = "v-comms-preset"          # one exported preset
LIBRARY_FORMAT = "v-comms-presets"        # the file the app keeps next to the profile
FORMAT_VERSION = 1
LIBRARY_FILENAME = "custom_presets.json"  # lives in the same folder as the auto-loaded profile
MAX_IMPORT_BYTES = 64 * 1024
PARAM_FIELDS = tuple(f.name for f in fields(eng.VoiceParameters) if f.name != "name")


class PresetError(Exception):
    """A problem with a preset file or a preset name, worded for the player."""


# ------------------------------------------------------------------------------------------------ names
def reserved_names() -> set:
    """The built-in presets' names, short titles and tags (Omen, Radio, KAY/O, Helium...), lower case."""
    names = {"omen", "radio", "kay/o", "helium", "helium-lite"}
    for full in eng.TACTICAL_PRESETS:
        names.add(full.lower())
        m = re.match(r"^(.*?)\s*\((.*)\)\s*$", full)
        if m:
            names.add(m.group(1).lower())
            names.add(m.group(2).lower())
    return names


def check_name(name: str, existing: Iterable[str], ignore: Optional[str] = None, adding: bool = True) -> Optional[str]:
    """Why ``name`` cannot be used for a custom preset, or None when it is fine."""
    return names_rules.check_name(name, existing, reserved_names(), ignore, adding)


def unique_name(base: str, existing: Iterable[str]) -> str:
    """``base`` if it is free, else 'base (2)', 'base (3)'... kept within the length limit."""
    return names_rules.unique_name(base, existing, reserved_names())


# ------------------------------------------------------------------------------------------------ values
def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def clean_params(raw: object, strict: bool = True) -> eng.VoiceParameters:
    """
    Known parameters only, each clamped to its slider range; unknown keys are ignored and missing ones keep their
    neutral default. ``strict`` refuses non-numeric values (imports); the library file is read leniently.
    """
    if not isinstance(raw, dict):
        raise PresetError("The preset has no voice settings.")
    defaults = eng.VoiceParameters()
    values = {}
    found = 0
    for key in PARAM_FIELDS:
        if key not in raw:
            values[key] = getattr(defaults, key)
            continue
        value = raw[key]
        if not _is_number(value):
            if strict:
                raise PresetError(f'"{key}" must be a number.')
            values[key] = getattr(defaults, key)
            continue
        lo, hi = eng.PARAM_RANGES[key]
        values[key] = float(min(max(float(value), lo), hi))
        found += 1
    if not found:
        raise PresetError("The preset has no voice settings.")
    return eng.VoiceParameters(name="Custom", **values)


# ------------------------------------------------------------------------------------------------ files
def export_text(name: str, params: eng.VoiceParameters) -> str:
    return json.dumps(
        {
            "format": PRESET_FORMAT,
            "version": FORMAT_VERSION,
            "name": normalize_name(name)[:NAME_MAX],
            "params": {key: round(float(getattr(params, key)), 6) for key in PARAM_FIELDS},
        },
        indent=2,
    )


def parse_preset_bytes(raw: bytes) -> Tuple[str, eng.VoiceParameters]:
    """Validates and cleans an exported preset. Raises PresetError with a readable reason."""
    if len(raw) > MAX_IMPORT_BYTES:
        raise PresetError("That file is larger than 64 KB, so it is not a V-COMMS preset.")
    try:
        data = json.loads(raw.decode("utf-8-sig"))  # Notepad saves a BOM
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise PresetError("That file is not valid JSON.") from None
    if not isinstance(data, dict):
        raise PresetError("That file is not a V-COMMS preset.")
    if data.get("format") != PRESET_FORMAT:
        raise PresetError("That file is not a V-COMMS preset (wrong format).")
    if data.get("version") != FORMAT_VERSION or isinstance(data.get("version"), bool):
        raise PresetError("That preset was made by a different version of V-COMMS.")
    name = data.get("name")
    name = normalize_name(name)[:NAME_MAX] if isinstance(name, str) else ""
    return name or "Imported preset", clean_params(data.get("params"), strict=True)


def parse_preset_file(path: Path) -> Tuple[str, eng.VoiceParameters]:
    try:
        if os.path.getsize(path) > MAX_IMPORT_BYTES:
            raise PresetError("That file is larger than 64 KB, so it is not a V-COMMS preset.")
        with open(path, "rb") as f:
            raw = f.read(MAX_IMPORT_BYTES + 1)
    except OSError as exc:
        raise PresetError(f"Could not read the file ({exc.strerror or exc}).") from None
    return parse_preset_bytes(raw)


# ------------------------------------------------------------------------------------------------ the library
class PresetLibrary:
    """The player's own presets, in the order they were made, kept in one JSON file next to the profile."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.items: Dict[str, eng.VoiceParameters] = {}
        self.load_error = ""

    def names(self) -> List[str]:
        return list(self.items)

    def get(self, name: str) -> Optional[eng.VoiceParameters]:
        """An independent copy of a preset's parameters, or None when there is no preset by that name."""
        params = self.items.get(name)
        return eng.VoiceParameters(**{**params.__dict__, "name": name}) if params else None

    def load(self) -> None:
        self.items.clear()
        self.load_error = ""
        if not self.path.is_file():
            return
        try:
            raw = self.path.read_bytes()
            if len(raw) > 512 * 1024:
                raise PresetError("the preset file is too large")
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict) or data.get("format") != LIBRARY_FORMAT or data.get("version") != FORMAT_VERSION:
                raise PresetError("the preset file has an unknown format")
            entries = data.get("presets", [])
            if not isinstance(entries, list):
                raise PresetError("the preset file is damaged")
        except (OSError, UnicodeDecodeError, ValueError, RecursionError, PresetError) as exc:
            self.load_error = str(exc) if isinstance(exc, PresetError) else "the preset file could not be read"
            return
        for entry in entries:
            if len(self.items) >= MAX_CUSTOM_PRESETS or not isinstance(entry, dict):
                continue
            name = entry.get("name")
            name = normalize_name(name)[:NAME_MAX] if isinstance(name, str) else ""
            if not name or check_name(name, self.items, adding=False):
                continue
            try:
                self.items[name] = clean_params(entry.get("params"), strict=False)
            except PresetError:
                continue

    def _write(self, items: Dict[str, eng.VoiceParameters]) -> None:
        """Atomic: write a temp file next to the library, then swap it in. Raises OSError when the disk refuses."""
        data = {
            "format": LIBRARY_FORMAT,
            "version": FORMAT_VERSION,
            "presets": [
                {"name": name, "params": {key: round(float(getattr(p, key)), 6) for key in PARAM_FIELDS}}
                for name, p in items.items()
            ],
        }
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def _commit(self, items: Dict[str, eng.VoiceParameters]) -> None:
        """The in-memory library only changes once the file has been written, so a failed save loses nothing."""
        self._write(items)
        self.items = items

    def add(self, name: str, params: eng.VoiceParameters) -> str:
        name = normalize_name(name)
        error = check_name(name, self.items)
        if error:
            raise PresetError(error)
        items = dict(self.items)
        items[name] = clean_params(params.__dict__, strict=False)
        self._commit(items)
        return name

    def rename(self, old: str, new: str) -> str:
        if old not in self.items:
            raise PresetError("That preset no longer exists.")
        new = normalize_name(new)
        error = check_name(new, self.items, ignore=old, adding=False)
        if error:
            raise PresetError(error)
        self._commit({(new if key == old else key): value for key, value in self.items.items()})
        return new

    def delete(self, name: str) -> None:
        if name not in self.items:
            raise PresetError("That preset no longer exists.")
        self._commit({key: value for key, value in self.items.items() if key != name})

    def clear(self) -> None:
        self._commit({})
