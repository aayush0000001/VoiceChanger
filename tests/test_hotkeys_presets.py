"""hotkeys.py, preset_names.py and presets.py: parsing, rules, normalisation, the stores, and fuzzing of the files a
player can hand-edit or receive from a friend."""

from __future__ import annotations

import itertools
import json
import random
import string

import pytest

import engine as eng
import hotkeys as hk
import preset_names
import presets as pr
from conftest import ITERATIONS, SEEDS

MOD_SETS = [0, hk.MOD_CONTROL, hk.MOD_ALT, hk.MOD_SHIFT, hk.MOD_CONTROL | hk.MOD_ALT, hk.MOD_CONTROL | hk.MOD_SHIFT,
            hk.MOD_ALT | hk.MOD_SHIFT, hk.MODIFIERS]


# ------------------------------------------------------------------------------------------------ hotkeys
def test_every_combination_round_trips():
    for mods, vk in itertools.product(MOD_SETS, hk.KEYS):
        combo = hk.Combo(mods, vk)
        assert hk.parse(combo.serial()) == combo
        assert hk.parse(combo.text()) == combo          # "Ctrl + Numpad +" and friends
        assert hk.parse(combo.serial().lower()) == combo
        assert hk.parse("  " + combo.text().replace(" + ", "+") + " ") == combo


@pytest.mark.parametrize("seed", SEEDS)
def test_parse_never_raises_anything_but_hotkeyerror(seed):
    rnd = random.Random(seed)
    alphabet = string.ascii_letters + string.digits + " +-_.*/\t" + "äé日"
    for _ in range(300):
        text = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 25)))
        try:
            combo = hk.parse(text)
        except hk.HotkeyError as exc:
            assert exc.kind == "syntax" and exc.message
            continue
        assert combo is None or isinstance(combo, hk.Combo)
    for value in (None, "", "   "):
        assert hk.parse(value) is None
    for value in (1, 2.5, [], {}, b"Ctrl+F1"):
        with pytest.raises(hk.HotkeyError):
            hk.parse(value)


def test_rules():
    assert hk.rule_error(hk.Combo(hk.MOD_WIN, 0x70))
    assert hk.rule_error(hk.Combo(hk.MOD_CONTROL, hk.VK_F12))
    assert hk.rule_error(hk.Combo(hk.MOD_ALT, hk.VK_F4))
    assert hk.rule_error(hk.Combo(0, ord("A")))          # a lone letter would be taken from the game
    assert hk.rule_error(hk.Combo(hk.MOD_CONTROL, 0x07))  # not a key we offer
    assert hk.rule_error(hk.Combo(0, 0x70)) is None       # F1 alone is fine
    assert hk.rule_error(hk.Combo(0, 0x61)) is None       # Numpad 1 alone is fine
    assert hk.rule_warning(hk.Combo(hk.MOD_SHIFT, ord("W")))
    assert hk.rule_warning(hk.Combo(hk.MOD_CONTROL, ord("1")))
    assert hk.rule_warning(hk.Combo(hk.MOD_CONTROL | hk.MOD_ALT, ord("W"))) is None
    for a in hk.ACTIONS:
        assert hk.rule_error(hk.parse(a.default)) is None


@pytest.mark.parametrize("seed", SEEDS)
def test_normalize_bindings_always_gives_a_clean_set(seed):
    rnd = random.Random(seed)
    junk = [None, "", 5, [], "Ctrl+Alt+F1", "Ctrl+Alt+F2", "F12", "Win+A", "A", "Shift+A", "Ctrl+Numpad +", "nonsense",
            "Ctrl+Alt+F9", "Alt+F4", "Ctrl+Shift+Page Down"]
    raw = {a.id: rnd.choice(junk) for a in hk.ACTIONS if rnd.random() < 0.8}
    raw["not_an_action"] = "Ctrl+X"
    out = hk.normalize_bindings(raw)
    assert set(out) == {a.id for a in hk.ACTIONS}
    pairs = [hk.parse(v).pair for v in out.values() if v]
    assert len(pairs) == len(set(pairs)), "two actions share a combination"
    for v in out.values():
        if v:
            assert hk.rule_error(hk.parse(v)) is None
    assert hk.normalize_bindings("not a dict") == hk.default_bindings()
    table = hk.to_table(out)
    assert set(table) == {a.hk_id for a in hk.ACTIONS}
    assert {a.id: hk.from_table_entry(table[a.hk_id]) for a in hk.ACTIONS} == out


@pytest.mark.parametrize("seed", SEEDS)
def test_validate_combo(seed):
    rnd = random.Random(seed)
    current = hk.default_bindings()
    action = rnd.choice(hk.ACTIONS).id
    other = rnd.choice([a for a in hk.ACTIONS if a.id != action])
    with pytest.raises(hk.HotkeyError) as err:
        hk.validate_combo(current, action, hk.parse(other.default), lambda i: f"<{i}>")
    assert err.value.kind == "duplicate" and err.value.other == other.id and f"<{other.id}>" in err.value.message
    assert hk.validate_combo(current, action, hk.parse(hk.ACTION_BY_ID[action].default)) is None  # its own is fine
    with pytest.raises(hk.HotkeyError):
        hk.validate_combo(current, "nope", hk.Combo(hk.MOD_CONTROL, 0x70))
    assert hk.validate_combo(current, action, hk.Combo(hk.MOD_SHIFT, ord("Q")))  # allowed, with a warning


@pytest.mark.parametrize("seed", SEEDS)
def test_hotkey_store(isolated_home, seed):
    rnd = random.Random(seed)
    bindings = hk.default_bindings()
    bindings[rnd.choice(hk.ACTIONS).id] = ""
    hk.save_store(bindings)
    assert hk.load_store() == hk.normalize_bindings(bindings)
    path = hk.store_path()
    assert path.parent.name == "V-COMMS" and str(path).startswith(str(isolated_home))
    assert [p.name for p in path.parent.iterdir()] == ["hotkeys.json"]
    path.write_text("{ damaged", encoding="utf-8")
    assert hk.load_store() is None
    path.write_text(json.dumps({"hotkeys": "nope"}), encoding="utf-8")
    assert hk.load_store() is None
    path.write_text("﻿" + json.dumps({"version": 1, "hotkeys": {"mute": "Ctrl+M"}}), encoding="utf-8")
    assert hk.load_store()["mute"] == "Ctrl+M"


def test_combo_text_and_names():
    assert hk.combo_text("") == "Not set" and hk.combo_text("garbage") == "Not set"
    assert hk.combo_text("Ctrl+Alt+PageUp") == "Ctrl + Alt + Page Up"
    assert hk.Combo(hk.MOD_CONTROL, 0x07).serial() == "Ctrl+VK07"


# ------------------------------------------------------------------------------------------------ preset names
@pytest.mark.parametrize("seed", SEEDS)
def test_unique_name_is_always_usable(seed):
    rnd = random.Random(seed)
    reserved = pr.reserved_names()
    existing = []
    for _ in range(preset_names.MAX_CUSTOM_PRESETS - 1):
        base = rnd.choice(["My preset", "Omen", "Radio", "x" * 40, "  spaced   out  ", "\x00ctrl\x07", "", "日本語の声"])
        name = pr.unique_name(base, existing)
        assert len(name) <= preset_names.NAME_MAX and name == preset_names.normalize_name(name)
        assert pr.check_name(name, existing) is None, (name, existing)
        existing.append(name)


@pytest.mark.parametrize("seed", SEEDS)
def test_long_imported_name_never_collides_after_trimming(tmp_path, seed):
    """unique_name used to keep a trailing space after cutting to 32 characters; the library trims it on add, so an
    import whose cut name matched an existing preset failed with "You already have a preset with that name"."""
    rnd = random.Random(seed)
    lib = pr.PresetLibrary(tmp_path / pr.LIBRARY_FILENAME)
    words = ["deep", "voice", "for", "ranked", "games", "tonight", "radio", "x"]
    base = ""
    while len(base) < 31:
        base = (base + " " + rnd.choice(words)).strip()
    existing = base[:31].rstrip()
    lib.add(existing, eng.VoiceParameters())
    incoming = existing + " " + " ".join(rnd.choice(words) for _ in range(4))  # cut lands on the space after it
    saved = lib.add(pr.unique_name(incoming, lib.names()), eng.VoiceParameters())
    assert saved != existing and saved == saved.strip() and len(saved) <= pr.NAME_MAX


def test_name_rules():
    assert pr.check_name("", []) and pr.check_name("x" * 33, [])
    assert pr.check_name("omen", []) and pr.check_name("Deep Shadow (Omen)", []) and pr.check_name("KAY/O", [])
    assert pr.check_name("Mine", ["mine"]) and pr.check_name("Mine", ["Mine"], ignore="mine", adding=False) is None
    full = [f"p{i}" for i in range(preset_names.MAX_CUSTOM_PRESETS)]
    assert pr.check_name("new", full) and pr.check_name("p0", full, ignore="p0", adding=False) is None
    assert preset_names.normalize_name("  a \t b\n\x00 ") == "a b"


# ------------------------------------------------------------------------------------------------ preset values and files
@pytest.mark.parametrize("seed", SEEDS)
def test_clean_params_clamps_everything(seed):
    rnd = random.Random(seed)
    raw = {k: rnd.choice([rnd.uniform(-1e6, 1e6), 0, -5, 3.3]) for k in pr.PARAM_FIELDS}
    raw["unknown"] = 1
    p = pr.clean_params(raw)
    for k, (lo, hi) in eng.PARAM_RANGES.items():
        assert lo <= getattr(p, k) <= hi
    with pytest.raises(pr.PresetError):
        pr.clean_params({"pitch_semitones": "high"})
    with pytest.raises(pr.PresetError):
        pr.clean_params({"unrelated": 1})
    with pytest.raises(pr.PresetError):
        pr.clean_params([1, 2])
    lenient = pr.clean_params({"pitch_semitones": "high", "drive_db": 3}, strict=False)
    assert lenient.pitch_semitones == 0.0 and lenient.drive_db == 3.0


@pytest.mark.parametrize("seed", SEEDS)
def test_export_import_round_trip(seed):
    rnd = random.Random(seed)
    params = eng.VoiceParameters(**{k: rnd.uniform(lo, hi) for k, (lo, hi) in eng.PARAM_RANGES.items()})
    name = "".join(rnd.choice(string.ascii_letters + " ") for _ in range(rnd.randint(1, 40)))
    got_name, got = pr.parse_preset_bytes(pr.export_text(name, params).encode("utf-8"))
    assert got_name == (preset_names.clip_name(name) or "Imported preset")
    assert got_name == got_name.strip() and len(got_name) <= pr.NAME_MAX
    for k in pr.PARAM_FIELDS:
        assert abs(getattr(got, k) - getattr(params, k)) < 1e-5


@pytest.mark.parametrize("seed", SEEDS)
def test_preset_import_fuzz(seed):
    """Random bytes and random JSON structures: the only possible failure is a PresetError with a readable reason."""
    rnd = random.Random(seed)

    def value(depth=0):
        kinds = ["int", "float", "str", "bool", "null"] + (["list", "dict"] if depth < 3 else [])
        k = rnd.choice(kinds)
        if k == "int":
            return rnd.randint(-10**9, 10**9)
        if k == "float":
            return rnd.uniform(-1e9, 1e9)
        if k == "str":
            return "".join(rnd.choice("abc{}[]\"\\日") for _ in range(rnd.randint(0, 8)))
        if k == "bool":
            return rnd.random() < 0.5
        if k == "null":
            return None
        if k == "list":
            return [value(depth + 1) for _ in range(rnd.randint(0, 4))]
        return {rnd.choice(list(pr.PARAM_FIELDS) + ["x"]): value(depth + 1) for _ in range(rnd.randint(0, 5))}

    for _ in range(50):
        doc = {"format": rnd.choice([pr.PRESET_FORMAT, "other", 1]), "version": rnd.choice([1, 2, True, "1"]),
               "name": value(), "params": value()}
        for raw in (json.dumps(doc).encode(), bytes(rnd.getrandbits(8) for _ in range(rnd.randint(0, 64)))):
            try:
                name, params = pr.parse_preset_bytes(raw)
            except pr.PresetError as exc:
                assert str(exc)
                continue
            assert isinstance(name, str) and 0 < len(name) <= pr.NAME_MAX
            for k, (lo, hi) in eng.PARAM_RANGES.items():
                assert lo <= getattr(params, k) <= hi
    with pytest.raises(pr.PresetError):
        pr.parse_preset_bytes(b" " * (pr.MAX_IMPORT_BYTES + 1))
    with pytest.raises(pr.PresetError):
        pr.parse_preset_bytes(b"[" * 100000)  # deep nesting


def test_preset_file_limits(tmp_path):
    big = tmp_path / "big.json"
    big.write_bytes(b" " * (pr.MAX_IMPORT_BYTES + 10))
    with pytest.raises(pr.PresetError, match="64 KB"):
        pr.parse_preset_file(big)
    with pytest.raises(pr.PresetError, match="Could not read"):
        pr.parse_preset_file(tmp_path / "missing.json")
    bom = tmp_path / "bom.json"
    bom.write_bytes(b"\xef\xbb\xbf" + pr.export_text("Bom", eng.VoiceParameters()).encode())
    assert pr.parse_preset_file(bom)[0] == "Bom"


# ------------------------------------------------------------------------------------------------ the library
@pytest.mark.parametrize("seed", SEEDS)
def test_library_operations_persist(tmp_path, seed):
    rnd = random.Random(seed)
    path = tmp_path / pr.LIBRARY_FILENAME
    lib = pr.PresetLibrary(path)
    lib.load()
    model = {}
    for _ in range(60):
        op = rnd.choice(["add", "add", "rename", "delete"])
        params = eng.VoiceParameters(pitch_semitones=rnd.uniform(-12, 12))
        try:
            if op == "add":
                name = lib.add(rnd.choice(["A", "B", "C", "Omen", "  D  ", "x" * 40]) + str(rnd.randint(0, 30)), params)
                model[name] = params.pitch_semitones
            elif op == "rename" and model:
                old = rnd.choice(list(model))
                new = lib.rename(old, rnd.choice(["Renamed", "E", old.lower()]) + str(rnd.randint(0, 30)))
                model = {(new if k == old else k): v for k, v in model.items()}
            elif op == "delete" and model:
                victim = rnd.choice(list(model))
                lib.delete(victim)
                del model[victim]
        except pr.PresetError:
            pass
        assert lib.names() == list(model)
    again = pr.PresetLibrary(path)
    again.load()
    assert again.load_error == "" and again.names() == list(model)
    for name, pitch in model.items():
        assert abs(again.get(name).pitch_semitones - pitch) < 1e-5 and again.get(name).name == name
    assert again.get("nope") is None
    again.clear()
    assert again.names() == [] and json.loads(path.read_text())["presets"] == []


def test_library_never_loses_presets_on_a_failed_write(tmp_path, monkeypatch):
    lib = pr.PresetLibrary(tmp_path / pr.LIBRARY_FILENAME)
    lib.add("Keep", eng.VoiceParameters())

    def fail(*_a, **_k):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(pr.os, "replace", fail)
    with pytest.raises(OSError):
        lib.add("New", eng.VoiceParameters())
    assert lib.names() == ["Keep"]
    with pytest.raises(OSError):
        lib.delete("Keep")
    assert lib.names() == ["Keep"]


@pytest.mark.parametrize("content", [b"{", b"[]", json.dumps({"format": "x", "version": 1}).encode(),
                                     json.dumps({"format": pr.LIBRARY_FORMAT, "version": 1, "presets": 5}).encode(),
                                     b"\xff" * 10, b" " * (600 * 1024)])
def test_damaged_library_is_reported_not_overwritten(tmp_path, content):
    path = tmp_path / pr.LIBRARY_FILENAME
    path.write_bytes(content)
    lib = pr.PresetLibrary(path)
    lib.load()
    assert lib.load_error and lib.names() == []


def test_library_skips_bad_entries(tmp_path):
    path = tmp_path / pr.LIBRARY_FILENAME
    entries = [{"name": "Good", "params": {"pitch_semitones": 2}}, {"name": "Omen", "params": {"pitch_semitones": 1}},
               {"name": "", "params": {}}, "junk", {"name": "Good", "params": {"drive_db": 1}},
               {"name": "NoParams"}] + [{"name": f"n{i}", "params": {"drive_db": 1}} for i in range(40)]
    path.write_text(json.dumps({"format": pr.LIBRARY_FORMAT, "version": 1, "presets": entries}))
    lib = pr.PresetLibrary(path)
    lib.load()
    assert lib.load_error == "" and lib.names()[0] == "Good" and "Omen" not in lib.names()
    assert len(lib.names()) == pr.MAX_CUSTOM_PRESETS
