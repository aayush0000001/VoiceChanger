"""
Shared test setup. Every test block runs at least ITERATIONS times (20 by default; set VCOMMS_TEST_ITERATIONS to
change it), either as one pytest case per seed or as a loop inside the test. VCOMMS_TEST_SEED_OFFSET shifts the seeds,
so repeated runs cover new random cases instead of replaying the same ones.

Nothing here touches the real user folders, audio hardware or the network: APPDATA / LOCALAPPDATA / HOME point into
a temporary folder, sounddevice is replaced by tests/fakes.FakeSoundDevice, and downloads go to a local HTTP server.
On Windows no real global hotkey is registered, and the tests that would touch the registry, the Settings app or the
process affinity are skipped. The window tests (test_gui.py) need real windows there and only run with
VCOMMS_GUI_TESTS=1.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))
WINDOWS = sys.platform == "win32"
if not WINDOWS:  # on Windows the frameless window needs real window handles (see test_gui.py)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ITERATIONS = max(20, int(os.environ.get("VCOMMS_TEST_ITERATIONS", "20")))
SEED_OFFSET = int(os.environ.get("VCOMMS_TEST_SEED_OFFSET", "0"))  # repeat runs with new random cases
SEEDS = list(range(SEED_OFFSET, SEED_OFFSET + ITERATIONS))


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """APPDATA, LOCALAPPDATA and HOME inside tmp_path (and the working directory too)."""
    for var, sub in (("APPDATA", "Roaming"), ("LOCALAPPDATA", "Local"), ("HOME", "home"), ("USERPROFILE", "home")):
        folder = tmp_path / sub
        folder.mkdir(exist_ok=True)
        monkeypatch.setenv(var, str(folder))
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    if WINDOWS:
        # Windows finds Documents, the folders an old V-COMMS.exe ran from and the old registry settings through the
        # shell and the registry, not through these variables: keep the migration away from the real ones
        import appdata

        documents = tmp_path / "home" / "Documents"
        documents.mkdir(exist_ok=True)
        monkeypatch.setattr(appdata, "documents_dir", lambda: documents)
        monkeypatch.setattr(appdata, "_muicache_dirs", lambda: [])
        monkeypatch.setattr(appdata, "_copy_registry_settings", lambda: 0)
    return tmp_path


@pytest.fixture
def fake_sd(monkeypatch):
    """engine.sd replaced by a scripted sounddevice (see fakes.FakeSoundDevice)."""
    import engine
    from fakes import FakeSoundDevice

    sd = FakeSoundDevice()
    monkeypatch.setattr(engine, "sd", sd)
    monkeypatch.setattr(engine, "SOUNDDEVICE_AVAILABLE", True)
    return sd


@pytest.fixture(scope="session")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="session")
def onnx_models(tmp_path_factory):
    """Synthetic models with the real input/output contracts (see onnx_models.py), built once per session."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import onnx_models as om

    return om.build_all(tmp_path_factory.mktemp("models"))


@pytest.fixture(autouse=True)
def no_real_hotkeys(monkeypatch):
    """On Windows the listener would register real global hotkeys (Ctrl+Alt+F1 ...) for the length of a test, taking
    them away from a V-COMMS that is running. Its registration logic is tested through a fake user32 instead."""
    if WINDOWS:
        import engine

        monkeypatch.setattr(engine.Win32HotkeyListener, "run", lambda self: self.ready.set())


@pytest.fixture(autouse=True)
def fresh_app_logger():
    """applog keeps one file handler for the whole process; each test gets its own (in its own temp folder)."""
    import logging

    import applog

    logger = logging.getLogger(applog.LOGGER_NAME)

    def drop():
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    drop()
    yield
    drop()
