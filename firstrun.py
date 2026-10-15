"""
First-run setup: the steps an installer cannot do for the player. Pick the microphone, make sure Windows lets desktop
apps use it, confirm VB-CABLE is there, run the Test Audio Path, and show where VALORANT's voice input is set.

Opens by itself until the player presses Done, and again from Settings > Run setup again. Presentation only: the mic
choice goes through the Devices page's own box and the test through its own button, so the engine side is unchanged.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGridLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    ComboBox,
    FluentIcon,
    IconWidget,
    InfoBarIcon,
    PushButton,
    StrongBodyLabel,
    SubtitleLabel,
)

import winintegration as wi
from views.dialogs import FastMessageBox
from views.widgets import muted

PRIVACY_TEXT = {
    "ok": "Windows lets desktop apps use your microphone.",
    "device_off": "Microphone access is turned off for this PC, so V-COMMS hears nothing. Turn it on in Privacy settings.",
    "all_apps_off": "Microphone access is off in Windows, so V-COMMS hears nothing. Turn it on in Privacy settings.",
    "desktop_apps_off": "\"Let desktop apps access your microphone\" is off, so V-COMMS hears nothing. Turn it on in Privacy settings.",
    "unknown": "Couldn't read Windows' microphone privacy setting.",
}


def _is_cable(label: str) -> bool:
    return "CABLE" in label.upper() and "VB-AUDIO" in label.upper()


def _windows_default_mic() -> str:
    """The name of Windows' default recording device (WASAPI), or '' when it cannot be read."""
    try:
        import sounddevice as sd

        for api in sd.query_hostapis():
            if "WASAPI" in api["name"].upper() and api["default_input_device"] >= 0:
                return str(sd.query_devices(api["default_input_device"])["name"])
    except Exception:
        pass
    return ""


class FirstRunDialog(FastMessageBox):
    def __init__(self, window, parent: Optional[QWidget] = None):
        super().__init__(parent or window)
        self._window = window
        self._devices = window.devicesView
        self._test_running = False
        self._engine_running = bool(getattr(window, "_engine_live", False))

        self.viewLayout.addWidget(SubtitleLabel("Set up V-COMMS"))
        intro = muted(CaptionLabel("Five quick checks so VALORANT hears your changed voice. You can redo them from Settings."))
        intro.setWordWrap(True)
        self.viewLayout.addWidget(intro)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(14)
        grid.setColumnStretch(1, 1)
        self.viewLayout.addLayout(grid)
        self._grid = grid
        self._row = 0

        # 1. microphone
        self.micCombo = ComboBox()
        self.micCombo.setMinimumWidth(240)
        self.micCombo.setMaximumWidth(300)
        self.micCombo.currentIndexChanged.connect(self._mic_chosen)
        self.micIcon, self.micText = self._add_step("1. Your microphone", self.micCombo)

        # 2. privacy
        self.privacyButton = PushButton(FluentIcon.SETTING, "Privacy settings")
        self.privacyButton.clicked.connect(lambda: wi.open_uri(wi.PRIVACY_MIC_URI))
        self.privacyIcon, self.privacyText = self._add_step("2. Microphone access", self.privacyButton)

        # 3. cable
        self.cableButton = PushButton(FluentIcon.SYNC, "Check again")
        self.cableButton.clicked.connect(self._cable_action)
        self.cableIcon, self.cableText = self._add_step("3. Virtual cable (VB-CABLE)", self.cableButton)

        # 4. test
        self.testButton = PushButton(FluentIcon.PLAY, "Test audio path")
        self.testButton.clicked.connect(self._run_test)
        self.testIcon, self.testText = self._add_step("4. Test the audio path", self.testButton)

        # 5. VALORANT
        self.valorantIcon, self.valorantText = self._add_step("5. In VALORANT", None)
        self.valorantText.setText(
            "Settings > Audio > Voice Chat > Input Device: choose \"CABLE Output (VB-Audio Virtual Cable)\". "
            "Discord and OBS use the same device as their microphone."
        )
        self._set_icon(self.valorantIcon, "info")

        self.yesButton.setText("Done")
        self.cancelButton.setText("Later")
        self.widget.setMinimumWidth(640)
        self.widget.setMaximumWidth(720)

        bridge = window.bridge
        bridge.path_test_finished.connect(self._test_finished)
        bridge.path_test_state.connect(self._test_state)
        bridge.engine_state_changed.connect(self._engine_state)
        self.finished.connect(self._disconnect)

        self._fill_mics()
        self.refresh()

    # ------------------------------------------------------------------ layout
    def _add_step(self, title: str, action: Optional[QWidget]):
        icon = IconWidget(InfoBarIcon.INFORMATION)
        icon.setFixedSize(18, 18)
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(StrongBodyLabel(title))
        text = muted(CaptionLabel(""))
        text.setWordWrap(True)
        lay.addWidget(text)
        self._grid.addWidget(icon, self._row, 0, Qt.AlignmentFlag.AlignTop)
        self._grid.addWidget(box, self._row, 1)
        if action is not None:
            self._grid.addWidget(action, self._row, 2, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
        self._row += 1
        return icon, text

    @staticmethod
    def _set_icon(icon: IconWidget, state: str) -> None:
        icon.setIcon({"ok": InfoBarIcon.SUCCESS, "warn": InfoBarIcon.WARNING, "error": InfoBarIcon.ERROR}.get(
            state, InfoBarIcon.INFORMATION))

    # ------------------------------------------------------------------ checks
    def refresh(self) -> None:
        privacy = wi.mic_privacy()
        self.privacyText.setText(PRIVACY_TEXT.get(privacy, PRIVACY_TEXT["unknown"]))
        self._set_icon(self.privacyIcon, "ok" if privacy == "ok" else ("info" if privacy == "unknown" else "error"))
        self.privacyButton.setVisible(privacy != "ok")

        cable = wi.cable_status()
        self._cable = cable
        if cable["input"] == "ok" and cable["output"] == "ok":
            self.cableText.setText("CABLE Input found: V-COMMS sends your voice there, and VALORANT hears it on CABLE Output.")
            self._set_icon(self.cableIcon, "ok")
            self.cableButton.setText("Check again")
            self.cableButton.setIcon(FluentIcon.SYNC)
        elif "missing" in (cable["input"], cable["output"]):
            self.cableText.setText("VB-CABLE isn't installed. Run the V-COMMS installer again, or get it free from vb-audio.com, then restart your PC.")
            self._set_icon(self.cableIcon, "error")
            self.cableButton.setText("Get VB-CABLE")
            self.cableButton.setIcon(FluentIcon.DOWNLOAD)
        elif cable["input"] == "unknown":
            self.cableText.setText("Couldn't check for VB-CABLE on this system.")
            self._set_icon(self.cableIcon, "info")
        else:
            self.cableText.setText("VB-CABLE is installed but turned off in Windows. Turn on CABLE Input and CABLE Output in Sound settings.")
            self._set_icon(self.cableIcon, "warn")
            self.cableButton.setText("Sound settings")
            self.cableButton.setIcon(FluentIcon.SETTING)
        self._update_test_enabled()

    def _cable_action(self) -> None:
        if self._cable["input"] == "missing" or self._cable["output"] == "missing":
            wi.open_uri(wi.VB_CABLE_URL)
        elif self._cable["input"] in ("disabled", "unplugged") or self._cable["output"] in ("disabled", "unplugged"):
            wi.open_uri(wi.SOUND_SETTINGS_URI)
        else:
            self.refresh()

    # ------------------------------------------------------------------ microphone
    def _fill_mics(self) -> None:
        source = self._devices.inputCombo
        self.micCombo.blockSignals(True)
        self.micCombo.clear()
        for i in range(source.count()):
            label = source.itemText(i)
            if not _is_cable(label):  # CABLE Output is what VALORANT listens to, never the player's mic
                self.micCombo.addItem(label, userData=i)
        current = source.currentText()
        pick = self.micCombo.findText(current) if current and not _is_cable(current) else -1
        if pick < 0:
            default = _windows_default_mic()
            pick = next((i for i in range(self.micCombo.count()) if default and default in self.micCombo.itemText(i)), 0)
        self.micCombo.setCurrentIndex(pick if self.micCombo.count() else -1)
        self.micCombo.blockSignals(False)
        if self.micCombo.count():
            self._mic_chosen(self.micCombo.currentIndex())
            self.micText.setText("Pick the microphone you talk into. A [Windows WASAPI] entry gives the lowest delay.")
            self._set_icon(self.micIcon, "ok")
        else:
            self.micText.setText("No microphone found. Plug one in, then press Rescan on the Devices page.")
            self._set_icon(self.micIcon, "error")

    def _mic_chosen(self, index: int) -> None:
        if index < 0:
            return
        source_index = self.micCombo.itemData(index)
        if source_index is not None and self._devices.inputCombo.currentIndex() != source_index:
            self._devices.inputCombo.setCurrentIndex(source_index)  # the Devices page tells the engine

    # ------------------------------------------------------------------ test
    def _engine_state(self, data: dict) -> None:
        self._engine_running = data.get("state") in ("running", "starting")
        self._update_test_enabled()

    def _update_test_enabled(self) -> None:
        running = self._engine_running
        self.testButton.setEnabled(not self._test_running and not running)
        if running and not self._test_running:
            self.testText.setText("Stop the audio engine to run the test (it needs the microphone and the cable to itself).")
        elif not self._test_running and not self.testText.text():
            self.testText.setText("Records 2 seconds from your mic, runs it through the voice effects and plays a tone through the cable.")

    def _run_test(self) -> None:
        self.testText.setText("Testing: say something into your microphone...")
        self._set_icon(self.testIcon, "info")
        self._devices.pathTestRequested.emit()

    def _test_state(self, data: dict) -> None:
        self._test_running = bool(data.get("running"))
        self._update_test_enabled()

    def _test_finished(self, data: dict) -> None:
        if data.get("passed"):
            self.testText.setText("Passed: your voice reaches CABLE Output, which is what VALORANT listens to.")
            self._set_icon(self.testIcon, "ok")
            return
        failed = next((s for s in data.get("steps", []) if s.get("status") == "fail"), None)
        why = f"{failed['stage']}: {failed['text']}" if failed and failed.get("stage") else (failed or {}).get("text", "")
        self.testText.setText(f"Failed. {why} More detail is on the Devices page.".strip())
        self._set_icon(self.testIcon, "error")

    def _disconnect(self, *_args) -> None:
        bridge = self._window.bridge
        for signal, slot in ((bridge.path_test_finished, self._test_finished), (bridge.path_test_state, self._test_state),
                             (bridge.engine_state_changed, self._engine_state)):
            try:
                signal.disconnect(slot)
            except TypeError:
                pass
