"""The Fluent window and its pages, offscreen, wired to a real EngineBridge on the fake audio backend."""

from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np
import pytest
from PyQt6.QtCore import QCoreApplication, QEvent, QObject, Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QGraphicsOpacityEffect

import appdata
import engine as eng
import hotkeys as hk
from conftest import ITERATIONS, SEEDS

# On Windows the frameless window makes Win32 calls that need a real window handle, so there these tests open real
# windows (and tray notifications) and take a few minutes: they run only with VCOMMS_GUI_TESTS=1. Close V-COMMS first.
pytestmark = pytest.mark.skipif(
    sys.platform == "win32" and os.environ.get("VCOMMS_GUI_TESTS") != "1",
    reason="opens real windows on Windows; set VCOMMS_GUI_TESTS=1 to run",
)


def pump(until=None, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        QCoreApplication.processEvents()
        if until is not None and until():
            return True
        time.sleep(0.002)
    return until is None


def build_window(first_run_done=True):
    import winapp_ui
    from main_window import MainWindow

    settings = appdata.qsettings()
    settings.setValue(winapp_ui.SETUP_DONE_KEY, first_run_done)
    settings.sync()
    window = MainWindow()
    winapp_ui.attach(None, window)
    window.show()
    assert pump(lambda: window.bridge.engine is not None and window.liveView._tiles and window.devicesView.inputCombo.count())
    pump(timeout=0.05)
    return window


def flush_deletes():
    """deleteLater() only runs inside a running event loop, which tests never start: without this every window of
    every test stays alive and qfluentwidgets restyles all of them each time a new window sets the theme colour."""
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


def close_window(window):
    window.quit_app()
    assert not window.bridge._thread.isRunning()
    window.deleteLater()
    flush_deletes()
    pump(timeout=0.02)


@pytest.fixture(autouse=True)
def slot_exceptions(monkeypatch):
    """Like the app (applog.install_excepthooks): an exception inside a Qt slot is recorded, not a process abort, and
    it fails the test it happened in."""
    import sys

    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda t, v, tb: errors.append(f"{t.__name__}: {v}"))
    yield errors
    pump(timeout=0.02)
    assert errors == [], errors


@pytest.fixture
def window(isolated_home, fake_sd, qapp):
    w = build_window()
    yield w
    close_window(w)


def key(widget, qt_key, mods=Qt.KeyboardModifier.NoModifier, vk=0, release=False):
    event = QKeyEvent(QEvent.Type.KeyRelease if release else QEvent.Type.KeyPress, qt_key, mods, 0, vk, 0)
    QCoreApplication.sendEvent(widget, event)


# ------------------------------------------------------------------------------------------------ lifecycle
def test_window_builds_and_quits_repeatedly(isolated_home, fake_sd, qapp):
    from qfluentwidgets.common.style_sheet import styleSheetManager

    for _ in range(ITERATIONS):
        w = build_window()
        assert w.windowTitle() == "V-COMMS" and len(w.liveView._tiles) == len(eng.TACTICAL_PRESETS)
        assert w.stackedWidget.currentWidget() is w.liveView
        close_window(w)
        del w
        assert len(styleSheetManager.widgets) == 0, "a closed window was not freed"


def test_close_hides_to_tray_and_restores(window):
    window.close()
    assert not window.isVisible() and window.bridge._thread.isRunning()  # still running in the tray
    window.restore_from_tray()
    pump(lambda: window.isVisible())
    assert window.isVisible()


# ------------------------------------------------------------------------------------------------ live page
@pytest.mark.parametrize("seed", SEEDS)
def test_preset_tiles_drive_the_engine(window, seed):
    names = list(eng.TACTICAL_PRESETS)
    name = names[seed % len(names)]
    window.liveView._tiles[name].chosen.emit(name)
    assert pump(lambda: window.bridge.config.active_preset == name)
    assert pump(lambda: window.liveView._tiles[name]._selected)
    assert sum(t._selected for t in window.liveView._tiles.values()) == 1
    assert pump(lambda: window.voiceView.rows["pitch_semitones"].slider.value()
                == int(round(eng.TACTICAL_PRESETS[name].pitch_semitones * 10)))
    in_female = name.startswith("Female")
    assert (window.liveView._tiles[name].parent() is window.liveView.femaleGrid) == in_female


def test_ai_switch_dims_presets_without_a_permanent_effect(window):
    live = window.liveView
    assert live.tileGrid.graphicsEffect() is None  # nothing dims the tiles while the AI voice is off
    live.apply_flags({**window.bridge._flags_payload(), "enable_onnx": True})
    effect = live.tileGrid.graphicsEffect()
    assert isinstance(effect, QGraphicsOpacityEffect) and abs(effect.opacity() - 0.38) < 1e-6
    assert not any(t._selected for t in live._tiles.values()) and "Bypassed" in live.presetsTitle.text()
    for _ in range(ITERATIONS):  # repeated updates reuse the effect instead of piling up new ones
        live.apply_flags({**window.bridge._flags_payload(), "enable_onnx": True})
        assert live.tileGrid.graphicsEffect() is effect
    live.apply_flags({**window.bridge._flags_payload(), "enable_onnx": False})
    assert live.tileGrid.graphicsEffect() is None and live.femaleGrid.graphicsEffect() is None


def test_model_row_text(window):
    live = window.liveView
    live.apply_onnx({"path": "C:/m/voice.onnx", "enabled": False, "message": "NEED_HUBERT:RVC model detected!"})
    assert live.modelRow.contentLabel.fullText() == "voice.onnx (needs the HuBERT base model)"
    live.apply_onnx({"path": "C:/m/voice.onnx", "enabled": True, "message": "RVC Loaded on CPU (CPU Accelerated)"})
    assert live.modelRow.contentLabel.fullText() == "voice.onnx (RVC Loaded on CPU (CPU Accelerated))"
    live.apply_onnx({"path": "C:/m/voice.onnx", "enabled": False, "message": "The voice model has an input V-COMMS does not know"})
    assert live.modelRow.contentLabel.fullText() == "voice.onnx"
    live.apply_onnx({"path": "", "enabled": False, "message": ""})
    assert live.modelRow.contentLabel.fullText() == "No model selected."


def test_engine_button_and_meters(window, fake_sd):
    live = window.liveView
    live.engineButton.click()
    assert pump(lambda: live._state == "running")
    assert live.engineButton.text() == "Stop audio engine" and window.engineNavItem.text() == "Stop engine"
    stream = [s for s in fake_sd.streams if s.kind == "duplex"][-1]
    for _ in range(ITERATIONS):
        stream.run_block(np.stack([np.full(256, 0.2, np.float32)] * 2, axis=1))
        pump(timeout=0.04)
    assert live.micMeter._level > -40 and live.dspStat._value.endswith("ms")
    live.muteSwitch.setChecked(True)
    assert pump(lambda: window.bridge.config.mic_muted)
    assert pump(lambda: "Muted" in live.statusPill._text)
    live.engineButton.click()
    assert pump(lambda: live._state == "stopped")
    assert live.dspStat._value == "-"


# ------------------------------------------------------------------------------------------------ voice page
@pytest.mark.parametrize("seed", SEEDS)
def test_sliders(window, seed):
    rng = np.random.default_rng(seed)
    voice = window.voiceView
    name = list(voice.rows)[seed % len(voice.rows)]
    row = voice.rows[name]
    # the page applies the start-up values a few rows per 4 ms tick: wait until it shows what the engine has
    assert pump(lambda: not voice._row_queue and not row.differs(getattr(window.bridge.config.current_params, name)))
    target = int(rng.integers(row.slider.minimum(), row.slider.maximum() + 1))
    if target == row.slider.value():  # setValue to the same value emits nothing
        target = row.slider.minimum() if target != row.slider.minimum() else row.slider.maximum()
    row.slider.setValue(target)
    assert pump(lambda: abs(getattr(window.bridge.engine.dsp._snapshot.params, name) - target / row.scale) < 1e-6)
    assert row.valueLabel.text() == row._format(target / row.scale)
    voice.resetButton.click()  # back to the preset: the sliders follow without echoing back to the engine
    preset_value = getattr(eng.TACTICAL_PRESETS[window.bridge.config.active_preset], name)
    assert pump(lambda: row.slider.value() == int(round(preset_value * row.scale)))


# ------------------------------------------------------------------------------------------------ devices page
@pytest.mark.parametrize("seed", SEEDS)
def test_device_boxes(window, seed):
    dv = window.devicesView
    combo = [dv.inputCombo, dv.outputCombo, dv.monitorCombo][seed % 3]
    role = ["input", "output", "monitor"][seed % 3]
    index = seed % combo.count()
    combo.setCurrentIndex(index)
    assert pump(lambda: window.bridge._selected[role] == combo.itemData(index))
    dv.blockCombo.setCurrentIndex(seed % 3)
    assert pump(lambda: window.bridge.config.block_size == (128, 256, 512)[seed % 3])
    dv.gainSlider.setValue((seed % 21) * 5)
    assert pump(lambda: abs(window.bridge.config.monitor_volume - (seed % 21) * 5 / 100) < 1e-9)


def test_path_test_results_render(window, fake_sd):
    fake_sd.rec_result = (0.2 * np.sin(2 * np.pi * 300 * np.arange(96000) / 48000)).astype(np.float32)
    fake_sd.playrec_fn = lambda sent: sent * 0.9
    window.devicesView.testButton.click()
    assert pump(lambda: window.devicesView.resultsLayout.count() == 4, timeout=20)  # summary + 3 stages
    assert window.devicesView.testButton.isEnabled()


# ------------------------------------------------------------------------------------------------ settings page
@pytest.mark.parametrize("seed", SEEDS)
def test_hotkey_editor(window, seed):
    sv = window.settingsView
    action = hk.ACTIONS[seed % len(hk.ACTIONS)].id
    row, editor, _ = sv._hotkey_rows[action]
    captured = []
    sv.hotkeyCaptureRequested.connect(captured.append)
    editor.button.click()
    assert captured == [action] and editor._waiting
    sv.begin_hotkey_listening(action)  # what the bridge sends once the listener has switched everything off
    assert editor._listening and editor.chip._text == "Press the new keys"
    key(editor, Qt.Key.Key_Control, Qt.KeyboardModifier.ControlModifier)
    assert editor.chip._text == "Ctrl + …"
    outcome = seed % 3
    chosen, cleared, cancelled = [], [], []
    editor.combinationChosen.connect(lambda *a: chosen.append(a))
    editor.cleared.connect(cleared.append)
    editor.cancelled.connect(cancelled.append)
    if outcome == 0:
        key(editor, Qt.Key.Key_F5, Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier, vk=0x74)
        assert chosen == [(action, hk.MOD_CONTROL | hk.MOD_SHIFT, 0x74)]
        assert pump(lambda: window.bridge.config.hotkeys[action] == "Ctrl+Shift+F5")
    elif outcome == 1:
        key(editor, Qt.Key.Key_Backspace)
        assert cleared == [action]
        assert pump(lambda: window.bridge.config.hotkeys[action] == "")
    else:
        key(editor, Qt.Key.Key_Escape)
        assert cancelled == [action]
    assert not editor.active and editor.button.text() == "Change"


def test_appearance_controls(window):
    sv = window.settingsView
    for i, key_name in enumerate(("light", "dark", "auto")):
        sv.themeCombo.setCurrentIndex((1, 2, 0)[i])
        assert window._theme_key == key_name and window.settings.value("appearance/theme") == key_name
    sv.micaSwitch.setChecked(False)
    window.reset_appearance()
    assert window._theme_key == "auto" and window._mica_pref is True
    sv.liteSwitch.setChecked(True)
    assert pump(lambda: window._lite)
    assert all(t.backgroundColorAni.duration() == 0 for t in window.liveView._tiles.values())


def test_ai_note_is_current(window):
    text = window.settingsView.aiNote.text()
    assert "RVC v2" in text and "will not work" not in text


# ------------------------------------------------------------------------------------------------ dialogs and widgets
def test_name_dialog_validation(window):
    from views.dialogs import NameDialog

    dialog = NameDialog(window, "Save", "Save", ["Mine"], ["omen"], initial="")
    for text, ok in (("", False), ("Omen", False), ("mine", False), ("x" * 32, True), ("Fresh", True), ("  Fresh  ", True)):
        dialog.edit.setText(text)
        assert dialog.yesButton.isEnabled() is ok, text
        assert dialog.validate() is ok
    assert dialog.name() == "Fresh"
    dialog.deleteLater()
    flush_deletes()


def test_report_dialog(window):
    from views.dialogs import ReportDialog

    dialog = ReportDialog(window, "Diagnostics here")
    assert not dialog.validate() and dialog.errorLabel.text()
    dialog.summaryEdit.setText("Voice cuts out")
    dialog.descriptionEdit.setPlainText("When I switch presets")
    dialog.replyEdit.setText("bad")
    assert not dialog.validate()
    dialog.replyEdit.setText("me@example.com")
    assert dialog.validate()
    payload = dialog.payload()
    assert payload["title"].endswith("Bug: Voice cuts out") and "Diagnostics here" in payload["body"]
    dialog.diagBox.setChecked(False)
    assert "Diagnostics here" not in dialog.payload()["body"]
    assert "Diagnostics" not in dialog.preview.toPlainText()
    dialog.deleteLater()
    flush_deletes()


def test_toast_timers_die_with_their_window(isolated_home, fake_sd, qapp):
    from PyQt6.QtCore import QTimer

    w = build_window()
    w.toasts.show("info", "short-lived", "", duration_ms=300)
    timers = [t for bar in w.toasts._bars for t in bar.findChildren(QTimer) if t.isSingleShot() and t.isActive()]
    assert timers, "the expiry timer should be owned by the toast"
    close_window(w)
    pump(timeout=0.6)  # past the expiry: nothing may fire on the deleted toast (slot_exceptions checks)


def test_toasts_stay_bounded(window):
    for i in range(ITERATIONS):
        window.show_notification({"level": ["success", "info", "warning", "error"][i % 4], "title": f"T{i}", "text": "x"})
        pump(timeout=0.005)
    assert len(window.toasts._bars) <= window.toasts.MAX_VISIBLE
    window.hide()
    window.show_notification({"level": "error", "title": "hidden", "text": "goes to the tray"})


@pytest.mark.parametrize("seed", SEEDS)
def test_meter_and_widgets(qapp, seed):
    from views.widgets import Chip, LevelMeter, ReflowGrid, SliderRow, StatusPill

    rng = np.random.default_rng(seed)
    meter = LevelMeter("Mic")
    for db in rng.uniform(-90, 5, 30):
        meter.set_level(float(db))
        assert 0.0 <= meter.bar._fraction <= 1.0
    meter.reset()
    assert meter._level == -80.0 and meter.valueLabel.text() == "-80.0 dBFS"
    grid = ReflowGrid(min_item_width=100, max_columns=4)
    for _ in range(int(rng.integers(1, 9))):
        grid.add(Chip("x"))
    grid.resize(int(rng.integers(50, 900)), 100)
    grid._place(force=True)
    assert 1 <= grid._cols <= 4
    row = SliderRow("p", "Pitch", -12.0, 12.0, 10.0, "st")
    emitted = []
    row.valueChanged.connect(lambda n, v: emitted.append(v))
    row.set_value(3.5)
    assert emitted == [] and row.valueLabel.text() == "+3.5 st"
    row.slider.setValue(-20)
    assert emitted == [-2.0]
    pill = StatusPill()
    pill.set_status("Live", "ok")
    assert pill._kind == "ok" and pill.width() > 46


def test_first_run_setup(isolated_home, fake_sd, qapp):
    import winapp_ui

    w = build_window(first_run_done=False)
    try:
        assert pump(lambda: w._first_run_dialog is not None, timeout=5)
        dialog = w._first_run_dialog
        labels = [dialog.micCombo.itemText(i) for i in range(dialog.micCombo.count())]
        assert labels and not any("CABLE" in label for label in labels)  # the cable is never offered as the mic
        dialog.yesButton.click()  # Done
        assert pump(lambda: w._first_run_dialog is None)
        assert w.settings.value(winapp_ui.SETUP_DONE_KEY, False, type=bool)
    finally:
        close_window(w)


def test_hubert_download_hands_the_model_to_the_bridge_thread(window, onnx_models, monkeypatch):
    """After the download the model used to be loaded by calling the bridge directly on the GUI thread."""
    import main_window

    class InstantWorker(QObject):
        progress = pyqtSignal(int, float, float)
        finished = pyqtSignal(bool, str)

        def __init__(self, parent=None):
            super().__init__(parent)

        def cancel(self):
            pass

        def start(self):
            self.progress.emit(100, 1.0, 1.0)
            self.finished.emit(True, "ok")

    monkeypatch.setattr(main_window, "HuBERTDownloadWorker", InstantWorker)
    load_threads = []
    real = window.bridge.engine.dsp.onnx_streamer.load_model

    def spy(path):
        load_threads.append(threading.current_thread() is threading.main_thread())
        return real(path)

    monkeypatch.setattr(window.bridge.engine.dsp.onnx_streamer, "load_model", spy)
    window.prompt_hubert_download(str(onnx_models["rvc_okada"]))
    dialogs = [d for d in window.findChildren(main_window.HuBERTDownloadDialog)]
    dialogs[-1].yesButton.click()
    assert pump(lambda: window.bridge.config.enable_onnx, timeout=10)
    assert load_threads == [False]  # loaded on the bridge thread, not the GUI thread


def test_second_launch_shows_the_window(window):
    import vcomms_main

    window.hide()
    answers = []
    t = threading.Thread(target=lambda: answers.append(vcomms_main._send("show", 3000)))
    t.start()
    assert pump(lambda: answers, timeout=10)
    t.join()
    assert answers == [True] and pump(lambda: window.isVisible())
