"""
The naming rules for custom presets. Pure Python with no Qt and no engine, so the dialogs and the bridge share one
implementation and cannot disagree about what a valid name is.
"""

from __future__ import annotations

import unicodedata
from typing import Iterable, Optional

NAME_MAX = 32
MAX_CUSTOM_PRESETS = 24


def normalize_name(name: object) -> str:
    """Trimmed, single-spaced, no control characters."""
    cleaned = "".join(ch for ch in str(name) if not unicodedata.category(ch).startswith("C"))
    return " ".join(cleaned.split())


def clip_name(name: object) -> str:
    """normalize_name cut to NAME_MAX, trimmed again: a cut right after a space would otherwise keep that space, and
    the stored (trimmed) name would no longer be the one whose uniqueness was checked."""
    return normalize_name(name)[:NAME_MAX].rstrip()


def check_name(
    name: str,
    existing: Iterable[str],
    reserved: Iterable[str],
    ignore: Optional[str] = None,
    adding: bool = True,
) -> Optional[str]:
    """
    Why ``name`` cannot be used for a custom preset, or None when it is fine. ``reserved`` holds the lower-case names
    of the built-in presets; ``ignore`` is the preset being renamed, which may keep its own name.
    """
    name = normalize_name(name)
    existing = list(existing)
    if not name:
        return "Give the preset a name."
    if len(name) > NAME_MAX:
        return f"Keep the name to {NAME_MAX} characters or fewer."
    if name.lower() in {r.lower() for r in reserved}:
        return f'"{name}" is the name of a built-in preset, which cannot be replaced.'
    others = [e.lower() for e in existing if ignore is None or e.lower() != ignore.lower()]
    if name.lower() in others:
        return "You already have a preset with that name."
    if adding and len(existing) >= MAX_CUSTOM_PRESETS:
        return f"You can keep up to {MAX_CUSTOM_PRESETS} custom presets. Delete one first."
    return None


def unique_name(base: str, existing: Iterable[str], reserved: Iterable[str], fallback: str = "Imported preset") -> str:
    """``base`` if it is free, else 'base (2)', 'base (3)'... kept within the length limit."""
    taken = {e.lower() for e in existing} | {r.lower() for r in reserved}
    base = clip_name(base) or fallback
    if base.lower() not in taken:
        return base
    n = 2
    while True:
        suffix = f" ({n})"
        candidate = base[: NAME_MAX - len(suffix)].rstrip() + suffix
        if candidate.lower() not in taken:
            return candidate
        n += 1
