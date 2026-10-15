"""Settings page: appearance, hotkeys, profiles and the experimental ONNX voice model."""

from __future__ import annotations

import platform
import sysconfig
from typing import Dict, Optional

from PyQt6.QtCore import QT_VERSION_STR, PYQT_VERSION_STR, Qt, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QWidget
from qfluentwidgets import BodyLabel, CaptionLabel, ComboBox, FluentIcon, PushButton, SwitchButton
import qfluentwidgets

from styles import GAP_SM, danger_color, success_color
from .widgets import Card, Chip, GroupCard, PageBase, icon_by_name, muted, set_switch

THEMES = (("Use system setting", "auto"), ("Light", "light"), ("Dark", "dark"))


class SettingsView(PageBase):
    themeSelected = pyqtSignal(str)             # "auto" | "light" | "dark"   (handled by the window itself)
    micaToggled = pyqtSignal(bool)              # (handled by the window itself)
    flagToggled = pyqtSignal(str, bool)         # "enable_onnx", new value
    onnxModelSelected = pyqtSignal(str)         # path to a .onnx file
    profileSaveRequested = pyqtSignal(str)      # path
    profileLoadRequested = pyqtSignal(str)      # path
    quitRequested = pyqtSignal()                # (handled by the window itself)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("settingsView", "Settings", "Appearance, hotkeys, profiles and experimental features.", parent)
        self._applying = False
        self._profile_file = "tactical_voice_profile.json"
        self._hotkey_rows: Dict[int, tuple] = {}

        # ---- appearance ----------------------------------------------------------------------------
        self.add_section("Appearance")
        self.appearanceCard = GroupCard("Look and feel")
        self.themeCombo = ComboBox()
        for label, key in THEMES:
            self.themeCombo.addItem(label, userData=key)
        self.themeCombo.setFixedWidth(220)
        self.themeCombo.currentIndexChanged.connect(self._theme_chosen)
        self.micaSwitch = SwitchButton()
        self.micaSwitch.setOnText("On")
        self.micaSwitch.setOffText("Off")
        self.micaSwitch.checkedChanged.connect(self.micaToggled)
        self.appearanceCard.add_row(
            FluentIcon.PALETTE, "Theme", "Follow Windows or force light or dark.", self.themeCombo
        )
        self.micaRow = self.appearanceCard.add_row(
            FluentIcon.TRANSPARENT, "Mica backdrop", "Windows 11 material behind the window.", self.micaSwitch
        )
        self.content.addWidget(self.appearanceCard)

        # ---- hotkeys ---------------------------------------------------------------------------------
        self.add_section("Global hotkeys")
        self.hotkeyCard = GroupCard("Work in fullscreen VALORANT, no keyboard hook")
        self.content.addWidget(self.hotkeyCard)

        # ---- profile -----------------------------------------------------------------------------------
        self.add_section("Profile")
        self.profileCard = GroupCard("Voice profile (JSON)")
        self.saveButton = PushButton(FluentIcon.SAVE, "Save JSON Profile")
        self.loadButton = PushButton(FluentIcon.FOLDER, "Load JSON Profile")
        self.saveButton.clicked.connect(self._save_profile)
        self.loadButton.clicked.connect(self._load_profile)
        self.profileCard.add_row(
            FluentIcon.SAVE, "Save your setup", "Voice settings, devices and buffer size.", self.saveButton
        )
        self.loadRow = self.profileCard.add_row(
            FluentIcon.DOCUMENT, "Load a profile", "tactical_voice_profile.json is loaded on startup.", self.loadButton
        )
        self.content.addWidget(self.profileCard)

        # ---- AI --------------------------------------------------------------------------------------------
        self.add_section("AI voice conversion")
        self.aiCard = GroupCard("ONNX AI voice model")
        self.aiCard.headerLayout.addSpacing(GAP_SM)
        self.aiCard.headerLayout.addWidget(Chip("EXPERIMENTAL", accent=True))
        self.aiCard.headerLayout.addStretch(1)
        self.onnxSwitch = SwitchButton()
        self.onnxSwitch.setOnText("On")
        self.onnxSwitch.setOffText("Off")
        self.onnxSwitch.checkedChanged.connect(lambda checked: self.flagToggled.emit("enable_onnx", checked))
        self.modelButton = PushButton(FluentIcon.FOLDER, "Select .ONNX Model…")
        self.modelButton.clicked.connect(self._pick_model)
        self.aiSwitchRow = self.aiCard.add_row(
            FluentIcon.ROBOT, "Enable ONNX AI voice model", "Runs on DirectML or CUDA when available.", self.onnxSwitch
        )
        self.modelRow = self.aiCard.add_row(FluentIcon.MUSIC, "Model file", "No model selected.", self.modelButton)
        self.modelRow.contentLabel.setElideMode(Qt.TextElideMode.ElideMiddle)  # keep the file name visible
        self.content.addWidget(self.aiCard)
        self.aiNote = muted(
            CaptionLabel(
                "The model is fed the last 4096 samples as one 1 x 4096 float32 input and runs inside the audio "
                "callback, so most real voice-conversion models (RVC and similar) will not work as they are."
            )
        )
        self.aiNote.setWordWrap(True)
        self.content.addWidget(self.aiNote)

        # ---- application -----------------------------------------------------------------------------------------
        self.add_section("Application")
        self.appCard = GroupCard("Window and tray")
        self.quitButton = PushButton(FluentIcon.POWER_BUTTON, "Quit V-COMMS")
        self.quitButton.clicked.connect(self.quitRequested)
        self.appCard.add_row(
            FluentIcon.POWER_BUTTON,
            "Quit V-COMMS",
            "Closing the window only hides it to the tray. Quit stops the audio and the hotkeys.",
            self.quitButton,
        )
        self.content.addWidget(self.appCard)

        # ---- about -------------------------------------------------------------------------------------------
        self.add_section("About")
        self.aboutCard = Card("V-COMMS")
        python_line = (
            f"Python {platform.python_version()} ({sysconfig.get_platform()}) on {platform.machine() or 'unknown'}"
        )
        for text in (
            "Real-time low-latency tactical voice engine for Windows 11.",
            python_line,
            f"Qt {QT_VERSION_STR}  ·  PyQt {PYQT_VERSION_STR}  ·  QFluentWidgets {qfluentwidgets.__version__}",
        ):
            label = BodyLabel(text) if text.startswith("Real-time") else muted(CaptionLabel(text))
            self.aboutCard.body.addWidget(label)
        self.content.addWidget(self.aboutCard)
        self.finish()

    # ------------------------------------------------------------------ user actions
    def _theme_chosen(self, index: int) -> None:
        if not self._applying and index >= 0:
            self.themeSelected.emit(str(self.themeCombo.itemData(index)))

    def _save_profile(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Voice Profile", self._profile_file, "JSON Files (*.json)")
        if path:
            self.profileSaveRequested.emit(path)

    def _load_profile(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Voice Profile", "", "JSON Files (*.json)")
        if path:
            self.profileLoadRequested.emit(path)

    def _pick_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select ONNX Voice Model", "", "ONNX Models (*.onnx)")
        if path:
            self.onnxModelSelected.emit(path)

    # ------------------------------------------------------------------ slots (window / bridge -> view)
    def apply_appearance(self, theme_key: str, mica_supported: bool, mica_enabled: bool) -> None:
        self._applying = True
        try:
            index = next((i for i, (_, key) in enumerate(THEMES) if key == theme_key), 0)
            self.themeCombo.setCurrentIndex(index)
        finally:
            self._applying = False
        set_switch(self.micaSwitch, mica_enabled and mica_supported)
        self.micaSwitch.setEnabled(mica_supported)
        self.micaRow.contentLabel.setText(
            "Windows 11 material behind the window." if mica_supported else "Needs Windows 11. Using a solid background."
        )

    def set_catalog(self, data: dict) -> None:
        self._profile_file = data["profile_file"]
        self.loadRow.contentLabel.setText(f"{data['profile_file']} is loaded on startup.")
        self.onnxSwitch.setEnabled(data["onnx_available"])
        self.modelButton.setEnabled(data["onnx_available"])
        if not data["onnx_available"]:
            self.aiSwitchRow.contentLabel.setText("onnxruntime is not installed, so this is unavailable.")
        for hk in data["hotkeys"]:
            chip = Chip(hk["keys"], min_width=128)
            status = CaptionLabel("")
            status.setFixedWidth(132)
            status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            host = QWidget()
            lay = QHBoxLayout(host)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(GAP_SM + 4)
            lay.addWidget(status)
            lay.addWidget(chip)
            row = self.hotkeyCard.add_row(icon_by_name(hk["icon"], FluentIcon.SETTING), hk["action"], hk["detail"], host)
            if hk["id"] is not None:
                self._hotkey_rows[hk["id"]] = (status, row)

    def apply_hotkeys(self, registered: dict) -> None:
        for hk_id, (status, _row) in self._hotkey_rows.items():
            ok = registered.get(hk_id)
            if ok is None:
                continue
            color = success_color() if ok else danger_color()
            status.setText("Active" if ok else "In use by another app")
            status.setTextColor(color, color)
            status.setToolTip(
                "" if ok else "Windows refused this shortcut because another program already uses it. "
                "That can be a second copy of V-COMMS: check the tray."
            )

    def apply_flags(self, flags: dict) -> None:
        set_switch(self.onnxSwitch, bool(flags["enable_onnx"]))

    def apply_onnx(self, data: dict) -> None:
        path = data["path"]
        self.modelRow.contentLabel.setText(path if path else "No model selected.")
        set_switch(self.onnxSwitch, bool(data["enabled"]))

    def apply_profile(self, data: dict) -> None:
        if data.get("path"):
            kind = "Loaded on startup" if data.get("autoloaded") else "Last loaded"
            self.loadRow.contentLabel.setText(f"{kind}: {data['path']}")
