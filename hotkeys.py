"""
Global hotkey bindings: which key combination triggers which action, the rules a combination has to follow, how it is
written in the profile and where the player's choices are saved between runs.

Pure Python (no Qt, no Windows calls), so the rules can be tested without a window and be used by the Settings page to
word its messages and by the engine bridge to enforce them. Hotkeys are registered with RegisterHotKey only (see
``engine.Win32HotkeyListener``): keyboard keys with Ctrl, Alt and Shift, never a low-level keyboard or mouse hook.
"""

from __future__ import annotations

import json
import os
import re
import string
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from appinfo import APP_NAME

# The values RegisterHotKey takes. The Windows key is not offered (Windows keeps most Win+key combinations for itself);
# MOD_WIN exists so a Win+key press can be recognised and explained.
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MODIFIERS = MOD_ALT | MOD_CONTROL | MOD_SHIFT

STORE_FILENAME = "hotkeys.json"
STORE_VERSION = 1

VK_F4 = 0x73
VK_F12 = 0x7B


# ======================================================================================================
# the actions
# ======================================================================================================

@dataclass(frozen=True)
class Action:
    id: str        # the key in the profile and in hotkeys.json
    hk_id: int     # the id RegisterHotKey is given (and WM_HOTKEY returns)
    label: str     # how a message names the action when the caller has no better name (presets use their own names)
    default: str   # the combination a first run starts with


ACTIONS: Tuple[Action, ...] = (
    Action("preset_1", 101, "Preset 1", "Ctrl+Alt+F1"),
    Action("preset_2", 102, "Preset 2", "Ctrl+Alt+F2"),
    Action("preset_3", 103, "Preset 3", "Ctrl+Alt+F3"),
    Action("preset_4", 104, "Preset 4", "Ctrl+Alt+F4"),
    Action("bypass", 105, "Bypass toggle", "Ctrl+Alt+F9"),
    Action("mute", 106, "Hardware mute toggle", "Ctrl+Alt+F10"),
)
ACTION_BY_ID: Dict[str, Action] = {a.id: a for a in ACTIONS}
ACTION_BY_HK_ID: Dict[int, Action] = {a.hk_id: a for a in ACTIONS}


# ======================================================================================================
# keys
# ======================================================================================================

# virtual-key code -> (name used in the profile, name shown to the player)
KEYS: Dict[int, Tuple[str, str]] = {}


def _add_key(vk: int, name: str, shown: Optional[str] = None) -> None:
    KEYS[vk] = (name, shown or name)


for _n in range(1, 25):
    _add_key(0x6F + _n, f"F{_n}")                                  # VK_F1 = 0x70 ... VK_F24 = 0x87
for _c in string.ascii_uppercase:
    _add_key(ord(_c), _c)
for _c in string.digits:
    _add_key(ord(_c), _c)
for _n in range(10):
    _add_key(0x60 + _n, f"Num{_n}", f"Numpad {_n}")                # VK_NUMPAD0..9
for _vk, _name, _shown in (
    (0x6A, "NumMultiply", "Numpad *"),
    (0x6B, "NumAdd", "Numpad +"),
    (0x6D, "NumSubtract", "Numpad -"),
    (0x6E, "NumDecimal", "Numpad ."),
    (0x6F, "NumDivide", "Numpad /"),
    (0x20, "Space", "Space"),
    (0x2D, "Insert", "Insert"),
    (0x2E, "Delete", "Delete"),
    (0x24, "Home", "Home"),
    (0x23, "End", "End"),
    (0x21, "PageUp", "Page Up"),
    (0x22, "PageDown", "Page Down"),
    (0x25, "Left", "Left"),
    (0x26, "Up", "Up"),
    (0x27, "Right", "Right"),
    (0x28, "Down", "Down"),
):
    _add_key(_vk, _name, _shown)


def _norm(token: str) -> str:
    return re.sub(r"\s+", "", token.lower())


_BY_NAME: Dict[str, int] = {}
for _vk, (_name, _shown) in KEYS.items():
    _BY_NAME[_norm(_name)] = _vk
    _BY_NAME[_norm(_shown)] = _vk
_BY_NAME.update({"pgup": 0x21, "pgdn": 0x22, "ins": 0x2D, "del": 0x2E})

# Keys that may be used on their own: F1-F11, F13-F24 and the numpad. Everything else (letters, digits, Space, the arrow
# and editing keys) is something a player types or plays with, so it needs Ctrl, Alt or Shift: a registered combination
# is taken away from every program, the game included.
STANDALONE_OK = frozenset(
    [0x6F + n for n in range(1, 25) if 0x6F + n != VK_F12] + list(range(0x60, 0x6B)) + [0x6B, 0x6D, 0x6E, 0x6F]
)
_LETTERS_DIGITS = frozenset(list(range(0x30, 0x3A)) + list(range(0x41, 0x5B)))

KEY_HELP = (
    "Use a letter, a digit, an F key (F1 to F24 except F12), Space, an arrow key, Insert, Delete, Home, End, "
    "Page Up, Page Down or a numpad key."
)


class HotkeyError(ValueError):
    """A combination that cannot be used. ``kind`` is syntax | rule | duplicate; ``other`` names the action in the way."""

    def __init__(self, message: str, kind: str = "rule", other: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.other = other


# ======================================================================================================
# combinations
# ======================================================================================================

def mod_names(mods: int) -> Tuple[str, ...]:
    names = []
    if mods & MOD_CONTROL:
        names.append("Ctrl")
    if mods & MOD_ALT:
        names.append("Alt")
    if mods & MOD_SHIFT:
        names.append("Shift")
    if mods & MOD_WIN:
        names.append("Win")
    return tuple(names)


@dataclass(frozen=True)
class Combo:
    mods: int   # MOD_ALT | MOD_CONTROL | MOD_SHIFT
    vk: int     # Windows virtual-key code

    @property
    def pair(self) -> Tuple[int, int]:
        return (self.mods, self.vk)

    def serial(self) -> str:
        """How it is written in the profile and in hotkeys.json: Ctrl+Alt+F1."""
        name = KEYS.get(self.vk, (f"VK{self.vk:02X}", ""))[0]
        return "+".join(mod_names(self.mods) + (name,))

    def text(self) -> str:
        """How it is shown to the player: Ctrl + Alt + F1."""
        name = KEYS.get(self.vk, ("", f"Key {self.vk:#04x}"))[1]
        return " + ".join(mod_names(self.mods) + (name,))

    def key_text(self) -> str:
        return KEYS.get(self.vk, ("", f"Key {self.vk:#04x}"))[1]


def combo_text(serial: str) -> str:
    """Display text for a stored combination ('' means Not set)."""
    try:
        combo = parse(serial)
    except HotkeyError:
        return "Not set"
    return combo.text() if combo else "Not set"


def parse(text: Any) -> Optional[Combo]:
    """``'Ctrl+Alt+F1'`` -> Combo. ``''``/None -> None (Not set). Anything else that is not a combination raises
    HotkeyError(kind='syntax'). Case and spaces around the '+' do not matter. Only the syntax is checked: see
    ``rule_error`` for what a combination is allowed to be."""
    if text is None:
        return None
    if not isinstance(text, str):
        raise HotkeyError("A hotkey is written like Ctrl+Alt+F1.", "syntax")
    text = text.strip()
    if not text:
        return None
    parts = [p.strip() for p in re.split(r"\s*\+\s*", text)]
    if len(parts) >= 2 and parts[-1] == "":
        parts = parts[:-2] + [parts[-2] + "+"]          # "Ctrl + Numpad +"
    mods = 0
    keys = []
    for part in parts:
        token = _norm(part)
        if token in ("ctrl", "control"):
            mods |= MOD_CONTROL
        elif token == "alt":
            mods |= MOD_ALT
        elif token == "shift":
            mods |= MOD_SHIFT
        elif token in ("win", "windows", "meta", "super", "cmd"):
            mods |= MOD_WIN
        elif token in _BY_NAME:
            keys.append(_BY_NAME[token])
        else:
            raise HotkeyError(f"“{part}” is not a key V-COMMS can use. {KEY_HELP}", "syntax")
    if len(keys) != 1:
        raise HotkeyError("A hotkey needs exactly one key, with Ctrl, Alt or Shift if you like.", "syntax")
    return Combo(mods, keys[0])


# ======================================================================================================
# rules
# ======================================================================================================

def rule_error(combo: Combo) -> Optional[str]:
    """Why ``combo`` cannot be used, or None. These are hard limits, not taste."""
    if combo.mods & MOD_WIN:
        return "The Windows key can't be used in a hotkey, because Windows keeps most Win combinations for itself. Use Ctrl, Alt or Shift."
    if combo.vk not in KEYS:
        return f"That key can't be used. {KEY_HELP}"
    if combo.vk == VK_F12:
        return "F12 is reserved by Windows for debuggers, so it can't be used."
    if combo.mods == MOD_ALT and combo.vk == VK_F4:
        return "Alt + F4 closes windows in Windows, so it can't be used."
    if not (combo.mods & MODIFIERS) and combo.vk not in STANDALONE_OK:
        return (
            f"{combo.key_text()} needs Ctrl, Alt or Shift held with it. On its own it would stop working in every "
            f"program, the game included."
        )
    return None


def rule_warning(combo: Combo) -> Optional[str]:
    """A combination that works but takes a key away from the game: Shift or Ctrl alone with a letter or digit."""
    if combo.vk in _LETTERS_DIGITS and combo.mods in (MOD_SHIFT, MOD_CONTROL):
        if combo.mods == MOD_SHIFT:
            what = "Shift to walk"
        else:
            what = "Ctrl to crouch"
        return (
            f"VALORANT uses {what}, and V-COMMS takes {combo.text()} away from the game while it runs, so that "
            f"move stops working. It was applied anyway. Alt, or two modifiers, avoids this."
        )
    return None


def validate_combo(
    current: Mapping[str, str],
    action_id: str,
    combo: Combo,
    label_of: Optional[Callable[[str], str]] = None,
) -> Optional[str]:
    """Checks ``combo`` as the new binding of ``action_id`` against the rules and against the other actions in
    ``current`` (serialized bindings). Raises HotkeyError; returns a warning to show (or None) when it is accepted."""
    if action_id not in ACTION_BY_ID:
        raise HotkeyError("That hotkey does not exist.", "syntax")
    problem = rule_error(combo)
    if problem:
        raise HotkeyError(problem, "rule")
    for other_id, text in current.items():
        if other_id == action_id or other_id not in ACTION_BY_ID:
            continue
        try:
            other = parse(text)
        except HotkeyError:
            continue
        if other is not None and other.pair == combo.pair:
            name = label_of(other_id) if label_of else ACTION_BY_ID[other_id].label
            raise HotkeyError(f"{combo.text()} is already used by {name}.", "duplicate", other_id)
    return rule_warning(combo)


# ======================================================================================================
# the set of bindings
# ======================================================================================================

def default_bindings() -> Dict[str, str]:
    return {a.id: a.default for a in ACTIONS}


def normalize_bindings(raw: Any) -> Dict[str, str]:
    """
    Bindings from a profile or hotkeys.json, made safe: always one entry per action. An unknown action is ignored; a
    missing, unreadable or not-allowed value falls back to that action's default; ``''``/null means Not set. If two
    actions end up with the same combination the first one (in ``ACTIONS`` order) keeps it and the other goes back to
    its default, or to Not set when that is taken too. Never raises.
    """
    source = raw if isinstance(raw, dict) else {}
    taken = set()
    out: Dict[str, str] = {}
    for action in ACTIONS:
        default = parse(action.default)
        combo: Optional[Combo]
        if action.id not in source:
            combo = default
        else:
            try:
                combo = parse(source[action.id])
                if combo is not None and rule_error(combo):
                    combo = default
            except HotkeyError:
                combo = default
        if combo is not None and combo.pair in taken:
            combo = default if default is not None and default.pair not in taken else None
        if combo is not None:
            taken.add(combo.pair)
        out[action.id] = combo.serial() if combo else ""
    return out


def to_table(bindings: Mapping[str, str]) -> Dict[int, Optional[Tuple[int, int]]]:
    """``{hk_id: (modifiers, virtual key) or None}`` in the form the listener thread takes."""
    table: Dict[int, Optional[Tuple[int, int]]] = {}
    for action in ACTIONS:
        try:
            combo = parse(bindings.get(action.id, action.default))
        except HotkeyError:
            combo = parse(action.default)
        table[action.hk_id] = combo.pair if combo else None
    return table


def from_table_entry(entry: Optional[Tuple[int, int]]) -> str:
    return Combo(*entry).serial() if entry else ""


# ======================================================================================================
# the file the Settings page saves to as soon as a hotkey changes
# ======================================================================================================

def store_path() -> Path:
    """%APPDATA%\\V-COMMS\\hotkeys.json: a per-user setting, so it lives with the user's settings (not with the logs in
    %LOCALAPPDATA%), is writable, and is the same whichever folder the app is started from. The only place this path is
    built."""
    base = os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / APP_NAME / STORE_FILENAME


def load_store(path: Optional[Path] = None) -> Optional[Dict[str, str]]:
    """The saved bindings, validated, or None when there is no file or it is not a hotkeys file."""
    path = path or store_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("hotkeys"), dict):
        return None
    return normalize_bindings(raw["hotkeys"])


def save_store(bindings: Mapping[str, str], path: Optional[Path] = None) -> None:
    """Atomic: the old file stays whole until the new one is complete. Raises OSError."""
    path = path or store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": STORE_VERSION, "hotkeys": normalize_bindings(dict(bindings))}
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
