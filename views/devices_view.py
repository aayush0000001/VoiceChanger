"""Devices page: mic, virtual cable and headphone pickers, Hear Myself, and the audio path test."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    ComboBox,
    FluentIcon,
    IndeterminateProgressBar,
    InfoBar,
    InfoBarIcon,
    InfoBarPosition,
    PrimaryPushButton,
    PushButton,
    Slider,
    SwitchButton,
)

from styles import GAP, GAP_SM
from .widgets import Card, ElidedComboBox, GroupCard, PageBase, muted, set_switch

STAGE_TITLES = {"Mic": "Microphone", "DSP": "Effects timing", "Cable": "Virtual cable"}
STEP_ICONS = {
    "pass": InfoBarIcon.SUCCESS,
    "fail": InfoBarIcon.ERROR,
    "skip": InfoBarIcon.WARNING,
    "info": InfoBarIcon.INFORMATION,
}


class InlineInfoBar(InfoBar):
    """InfoBar that sits in a layout. The stock one shrinks to its text, so ask for the full row width instead."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setGraphicsEffect(None)  # the always-on opacity effect would make every repaint go through an offscreen buffer

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        parent = self.parentWidget()
        return QSize(max(hint.width(), parent.width() if parent else 0), hint.height())


class DevicesView(PageBase):
    deviceSelected = pyqtSignal(str, int)       # role ("input" | "output" | "monitor"), device id
    blockSizeSelected = pyqtSignal(int)         # samples
    rescanRequested = pyqtSignal()
    flagToggled = pyqtSignal(str, bool)         # "hear_myself", new value
    monitorVolumeChanged = pyqtSignal(float)    # 0.0 .. 1.0
    pathTestRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("devicesView", "Devices", "Route your microphone through V-COMMS into the virtual cable.", parent)
        self._applying = False
        self._engine_running = False
        self._test_running = False

        # ---- routing -------------------------------------------------------------------------------
        self.add_section("Audio routing")
        self.routingCard = GroupCard("Devices and buffer")
        self.inputCombo = ElidedComboBox()
        self.outputCombo = ElidedComboBox()
        self.monitorCombo = ElidedComboBox()
        self.blockCombo = ComboBox()
        self.blockCombo.setFixedWidth(220)
        self.inputCombo.currentIndexChanged.connect(lambda i: self._device_chosen("input", self.inputCombo, i))
        self.outputCombo.currentIndexChanged.connect(lambda i: self._device_chosen("output", self.outputCombo, i))
        self.monitorCombo.currentIndexChanged.connect(lambda i: self._device_chosen("monitor", self.monitorCombo, i))
        self.blockCombo.currentIndexChanged.connect(self._block_chosen)

        self.rescanButton = PushButton(FluentIcon.SYNC, "Rescan devices")
        self.rescanButton.clicked.connect(self.rescanRequested)
        self.routingCard.headerLayout.addStretch(1)
        self.routingCard.headerLayout.addWidget(self.rescanButton)

        self.routingCard.add_row(
            FluentIcon.MICROPHONE, "Physical Mic Input", "Your real microphone. Pick a [Windows WASAPI] entry.", self.inputCombo
        )
        self.routingCard.add_row(
            FluentIcon.CONNECT,
            "Virtual Cable Output",
            "Where the processed voice goes. Choose CABLE Input.",
            self.outputCombo,
        )
        self.routingCard.add_row(
            FluentIcon.HEADPHONE,
            "Headphone Monitor",
            "Your headphones, for Hear Myself. Must differ from the cable output.",
            self.monitorCombo,
        )
        self.routingCard.add_row(
            FluentIcon.SPEED_HIGH,
            "Buffer Size (@48kHz)",
            "Smaller is faster, bigger is safer. 256 samples (5.3 ms) is the sweet spot.",
            self.blockCombo,
        )
        self.content.addWidget(self.routingCard)

        # ---- hear myself ------------------------------------------------------------------------------
        self.add_section("Monitoring")
        self.monitorCard = GroupCard("Hear Myself")
        self.hearSwitch = SwitchButton()
        self.hearSwitch.setOnText("On")
        self.hearSwitch.setOffText("Off")
        self.hearSwitch.checkedChanged.connect(lambda checked: self.flagToggled.emit("hear_myself", checked))
        self.gainSlider = Slider(Qt.Orientation.Horizontal)
        self.gainSlider.setRange(0, 100)
        self.gainSlider.setValue(75)
        self.gainSlider.setFixedWidth(200)
        self.gainSlider.setAccessibleName("Monitor gain")
        self.gainLabel = muted(CaptionLabel("75%"))
        self.gainLabel.setFixedWidth(40)
        self.gainLabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.gainSlider.valueChanged.connect(self._gain_changed)
        gain_host = QWidget()
        gain_lay = QHBoxLayout(gain_host)
        gain_lay.setContentsMargins(0, 0, 0, 0)
        gain_lay.setSpacing(GAP_SM)
        gain_lay.addWidget(self.gainSlider)
        gain_lay.addWidget(self.gainLabel)
        self.monitorCard.add_row(
            FluentIcon.VOLUME, "Side-tone loopback", "Hear your processed voice in your headphones, without lag.", self.hearSwitch
        )
        self.monitorCard.add_row(FluentIcon.MIX_VOLUMES, "Monitor gain", "How loud you hear yourself.", gain_host)
        self.content.addWidget(self.monitorCard)

        # ---- test ---------------------------------------------------------------------------------------
        self.add_section("Verification")
        self.testCard = Card("Test Audio Path")
        intro = BodyLabel(
            "Records 2 seconds from your mic, runs it through the current preset, then plays a 1 kHz tone into "
            "CABLE Input and checks that it arrives on CABLE Output, which is what VALORANT hears."
        )
        intro.setWordWrap(True)
        self.testHint = muted(CaptionLabel("Speak while it records. The audio engine must be stopped."))
        self.testButton = PrimaryPushButton(FluentIcon.PLAY, "Test Audio Path")
        self.testButton.setMinimumWidth(188)
        self.testButton.clicked.connect(self.pathTestRequested)
        self.progress = IndeterminateProgressBar()
        self.progress.setVisible(False)
        self.progress.stop()

        action_row = QHBoxLayout()
        action_row.setSpacing(GAP)
        action_row.addWidget(self.testButton)
        action_row.addWidget(self.testHint, 1)

        self.results = QWidget()
        self.resultsLayout = QVBoxLayout(self.results)
        self.resultsLayout.setContentsMargins(0, 0, 0, 0)
        self.resultsLayout.setSpacing(GAP_SM)

        self.testCard.body.addWidget(intro)
        self.testCard.body.addLayout(action_row)
        self.testCard.body.addWidget(self.progress)
        self.testCard.body.addWidget(self.results)
        self.content.addWidget(self.testCard)

        self.tipHost = QWidget()
        tip_layout = QVBoxLayout(self.tipHost)
        tip_layout.setContentsMargins(0, 0, 0, 0)
        self.valorantTip = InlineInfoBar(
            InfoBarIcon.INFORMATION,
            "In VALORANT",
            "Settings > Audio > Voice Chat > Input Device: CABLE Output (VB-Audio Virtual Cable).",
            Qt.Orientation.Horizontal,
            False,
            -1,
            InfoBarPosition.NONE,
            self.tipHost,
        )
        tip_layout.addWidget(self.valorantTip)
        self.content.addWidget(self.tipHost)
        self.finish()
        self._render_buttons()

    # ------------------------------------------------------------------ user actions
    def _device_chosen(self, role: str, combo: ComboBox, index: int) -> None:
        if self._applying or index < 0:
            return
        device_id = combo.itemData(index)
        if device_id is not None:
            self.deviceSelected.emit(role, int(device_id))

    def _block_chosen(self, index: int) -> None:
        if self._applying or index < 0:
            return
        samples = self.blockCombo.itemData(index)
        if samples is not None:
            self.blockSizeSelected.emit(int(samples))

    def _gain_changed(self, value: int) -> None:
        self.gainLabel.setText(f"{value}%")
        if not self._applying:
            self.monitorVolumeChanged.emit(value / 100.0)

    # ------------------------------------------------------------------ slots (bridge -> view)
    def apply_devices(self, data: dict) -> None:
        self._applying = True
        try:
            for role, combo, key in (
                ("input", self.inputCombo, "inputs"),
                ("output", self.outputCombo, "outputs"),
                ("monitor", self.monitorCombo, "outputs"),
            ):
                combo.clear()
                for device_id, label in data[key]:
                    combo.addItem(label, userData=device_id)
                wanted = data["selected"].get(role)
                index = combo.findData(wanted) if wanted is not None else -1
                combo.setCurrentIndex(index if index >= 0 else (0 if combo.count() else -1))

            self.blockCombo.clear()
            for samples in data["block_sizes"]:
                self.blockCombo.addItem(f"{samples} samples ({samples / 48000.0 * 1000.0:.1f} ms)", userData=samples)
            idx = self.blockCombo.findData(data["block_size"])
            self.blockCombo.setCurrentIndex(idx if idx >= 0 else 1)
        finally:
            self._applying = False

    def apply_flags(self, flags: dict) -> None:
        def apply() -> None:
            set_switch(self.hearSwitch, bool(flags["hear_myself"]))
            self._applying = True
            try:
                self.gainSlider.setValue(int(round(flags["monitor_volume"] * 100)))
            finally:
                self._applying = False

        self.update_when_shown("flags", apply)

    def apply_engine_state(self, data: dict) -> None:
        self._engine_running = data["state"] in ("running", "starting")
        self._render_buttons()

    def apply_path_test_state(self, data: dict) -> None:
        self._test_running = bool(data["running"])
        self.progress.setVisible(self._test_running)
        if self._test_running:
            self.progress.start()
            self._clear_results()
            self.testHint.setText("Testing… speak into your mic.")
        else:
            self.progress.stop()
        self._render_buttons()

    def show_path_test_result(self, data: dict) -> None:
        self._clear_results()
        passed = data["passed"]
        self._add_result(
            InfoBarIcon.SUCCESS if passed else InfoBarIcon.ERROR,
            "Audio path test passed" if passed else "Audio path test failed",
            "Your voice reaches the virtual cable that VALORANT listens to."
            if passed
            else "Fix the highlighted step, then run the test again.",
        )
        for step in data["steps"]:
            self._add_result(
                STEP_ICONS.get(step["status"], InfoBarIcon.INFORMATION),
                STAGE_TITLES.get(step["stage"], step["stage"] or "Result"),
                step["text"],
            )
        self.testHint.setText("Speak while it records. The audio engine must be stopped.")

    # ------------------------------------------------------------------ helpers
    def _add_result(self, icon: InfoBarIcon, title: str, content: str) -> None:
        bar = InlineInfoBar(icon, title, content, Qt.Orientation.Horizontal, True, -1, InfoBarPosition.NONE, self.results)
        self.resultsLayout.addWidget(bar)
        bar.show()

    def _clear_results(self) -> None:
        while self.resultsLayout.count():
            item = self.resultsLayout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _render_buttons(self) -> None:
        self.testButton.setEnabled(not self._engine_running and not self._test_running)
        self.rescanButton.setEnabled(not self._engine_running and not self._test_running)
        if self._engine_running and not self._test_running:
            self.testHint.setText("Stop the audio engine to run the test.")
        elif not self._test_running:
            self.testHint.setText("Speak while it records. The audio engine must be stopped.")
