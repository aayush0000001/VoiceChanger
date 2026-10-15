"""Settings page: appearance, hotkeys, profiles and the experimental ONNX voice model."""

from __future__ import annotations

import platform
import sysconfig
from typing import Dict, Optional

from PyQt6.QtCore import QT_VERSION_STR, PYQT_VERSION_STR, Qt, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QWidget
from qfluentwidgets import BodyLabel, CaptionLabel, ComboBox, FluentIcon, PushButton, SwitchButton
import qfluentwidgets

from styles import GAP_SM, danger_color
from .dialogs import ConfirmDialog
from .widgets import Card, Chip, GroupCard, HotkeyEditor, PageBase, icon_by_name, muted, set_switch

THEMES = (("Use system setting", "auto"), ("Light", "light"), ("Dark", "dark"))


class SettingsView(PageBase):
    themeSelected = pyqtSignal(str)             # "auto" | "light" | "dark"   (handled by the window itself)
    micaToggled = pyqtSignal(bool)              # (handled by the window itself)
    flagToggled = pyqtSignal(str, bool)         # "enable_onnx", new value
    onnxModelSelected = pyqtSignal(str)         # path to a .onnx file
    profileSaveRequested = pyqtSignal(str)      # path
    profileLoadRequested = pyqtSignal(str)      # path
    resetAllRequested = pyqtSignal(bool)        # also delete the custom presets? (the window resets the look itself)
    reportRequested = pyqtSignal()              # open the problem report form (the window builds it)
    quitRequested = pyqtSignal()                # (handled by the window itself)
    hotkeyCaptureRequested = pyqtSignal(str)    # action id: switch every hotkey off so the next key press can be read
    hotkeyChosen = pyqtSignal(str, int, int)    # action id, modifier bits, virtual-key code
    hotkeyCleared = pyqtSignal(str)             # action id: back to "Not set"
    hotkeyCaptureCancelled = pyqtSignal()       # nothing was picked: switch the hotkeys back on
    hotkeysRestoreRequested = pyqtSignal()      # every hotkey back to its default

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("settingsView", "Settings", "Appearance, hotkeys, profiles and experimental features.", parent)
        self._applying = False
        self._profile_file = "tactical_voice_profile.json"
        self._hotkey_rows: Dict[str, tuple] = {}   # action id -> (row, editor, its normal description)

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

        # ---- performance -----------------------------------------------------------------------------
        self.add_section("Performance")
        self.performanceCard = GroupCard("For older or busy PCs")
        self.liteSwitch = SwitchButton()
        self.liteSwitch.setOnText("On")
        self.liteSwitch.setOffText("Off")
        self.liteSwitch.checkedChanged.connect(lambda checked: self.flagToggled.emit("lite_mode", checked))
        self.stabilitySwitch = SwitchButton()
        self.stabilitySwitch.setOnText("On")
        self.stabilitySwitch.setOffText("Off")
        self.stabilitySwitch.checkedChanged.connect(lambda checked: self.flagToggled.emit("stability_buffer", checked))
        self.performanceCard.add_row(
            FluentIcon.SPEED_OFF,
            "Lite mode",
            "Uses less memory and CPU: slower meters, no fades or Mica, memory handed back while the window is hidden.",
            self.liteSwitch,
        )
        self.stabilityRow = self.performanceCard.add_row(
            FluentIcon.SPEED_HIGH,
            "Stability buffer",
            "Rides out longer stalls without a hole in your voice, for roughly 40 ms more delay. Applies at the next start.",
            self.stabilitySwitch,
        )
        self.content.addWidget(self.performanceCard)

        # ---- hotkeys ---------------------------------------------------------------------------------
        self.add_section("Global hotkeys")
        self.hotkeyCard = GroupCard("Work in fullscreen VALORANT, no keyboard hook")
        self.content.addWidget(self.hotkeyCard)
        self.hotkeyNote = muted(
            CaptionLabel(
                "Keyboard only. V-COMMS takes a combination over from every program, the game included, while it runs. "
                "Letters, digits, Space and the arrow, Insert, Delete, Home, End and Page keys need Ctrl, Alt or Shift; "
                "F1 to F11, F13 to F24 and the numpad (with Num Lock on) can stand alone. F12 is reserved by Windows. "
                "Esc cancels a change, Backspace or Delete clears a hotkey."
            )
        )
        self.hotkeyNote.setWordWrap(True)
        self.content.addWidget(self.hotkeyNote)

        # ---- profile -----------------------------------------------------------------------------------
        self.add_section("Profile")
        self.profileCard = GroupCard("Voice profile (JSON)")
        self.saveButton = PushButton(FluentIcon.SAVE, "Save JSON Profile")
        self.loadButton = PushButton(FluentIcon.FOLDER, "Load JSON Profile")
        self.saveButton.clicked.connect(self._save_profile)
        self.loadButton.clicked.connect(self._load_profile)
        self.profileCard.add_row(
            FluentIcon.SAVE, "Save your setup", "Voice settings, devices, buffer size and hotkeys.", self.saveButton
        )
        self.loadRow = self.profileCard.add_row(
            FluentIcon.DOCUMENT, "Load a profile", "tactical_voice_profile.json is loaded on startup.", self.loadButton
        )
        self.content.addWidget(self.profileCard)

        # ---- reset -------------------------------------------------------------------------------------------
        self.add_section("Reset")
        self.resetCard = GroupCard("Start fresh")
        self.resetButton = PushButton(FluentIcon.HISTORY, "Reset all settings…")
        self.resetButton.clicked.connect(self._ask_reset)
        self.resetCard.add_row(
            FluentIcon.HISTORY,
            "Reset all settings",
            "Back to the first-launch defaults. Your custom presets are kept unless you choose to delete them.",
            self.resetButton,
        )
        self.content.addWidget(self.resetCard)

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

        # ---- support -------------------------------------------------------------------------------------------
        self.add_section("Support")
        self.supportCard = GroupCard("Help improve V-COMMS")
        self.reportButton = PushButton(FluentIcon.FEEDBACK, "Report a problem…")
        self.reportButton.clicked.connect(self.reportRequested)
        self.supportCard.add_row(
            FluentIcon.FEEDBACK,
            "Report a problem",
            "A bug, a glitch or an idea. You see exactly what is sent, and nothing leaves your PC until you press Send.",
            self.reportButton,
        )
        self.content.addWidget(self.supportCard)

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

    def _ask_reset(self) -> None:
        dialog = ConfirmDialog(
            self.window(),
            "Reset all settings?",
            "Voice effects, switches, devices, buffer size, hotkeys and the theme go back to how they were on first "
            "launch, and your startup profile is rewritten with those defaults. Your custom presets are kept.",
            "Reset",
            checkbox_text="Also delete my custom presets",
        )
        dialog.accepted.connect(lambda: self.resetAllRequested.emit(dialog.checked()))
        dialog.finished.connect(dialog.deleteLater)
        dialog.open()

    def _pick_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select ONNX Voice Model", "", "ONNX Models (*.onnx)")
        if path:
            self.onnxModelSelected.emit(path)

    # ------------------------------------------------------------------ slots (window / bridge -> view)
    def apply_appearance(self, theme_key: str, mica_supported: bool, mica_enabled: bool, lite: bool = False) -> None:
        self._applying = True
        try:
            index = next((i for i, (_, key) in enumerate(THEMES) if key == theme_key), 0)
            self.themeCombo.setCurrentIndex(index)
        finally:
            self._applying = False
        set_switch(self.micaSwitch, mica_enabled and mica_supported)
        self.micaSwitch.setEnabled(mica_supported and not lite)
        if not mica_supported:
            note = "Needs Windows 11. Using a solid background."
        elif lite:
            note = "Off while Lite mode is on."
        else:
            note = "Windows 11 material behind the window."
        self.micaRow.contentLabel.setText(note)

    def set_catalog(self, data: dict) -> None:
        self._profile_file = data["profile_file"]
        self.loadRow.contentLabel.setText(f"{data['profile_file']} is loaded on startup.")
        self.onnxSwitch.setEnabled(data["onnx_available"])
        self.modelButton.setEnabled(data["onnx_available"])
        if not data["onnx_available"]:
            self.aiSwitchRow.contentLabel.setText("onnxruntime is not installed, so this is unavailable.")
        for hotkey in data["hotkeys"]:
            editor = HotkeyEditor(hotkey["id"], hotkey["action"])
            editor.captureRequested.connect(self._capture_requested)
            editor.combinationChosen.connect(self.hotkeyChosen)
            editor.cleared.connect(self.hotkeyCleared)
            editor.cancelled.connect(lambda _id: self.hotkeyCaptureCancelled.emit())
            row = self.hotkeyCard.add_row(
                icon_by_name(hotkey["icon"], FluentIcon.SETTING), hotkey["action"], hotkey["detail"], editor
            )
            self._hotkey_rows[hotkey["id"]] = (row, editor, hotkey["detail"])
        self.restoreHotkeysButton = PushButton(FluentIcon.HISTORY, "Restore defaults")
        self.restoreHotkeysButton.clicked.connect(self.hotkeysRestoreRequested)
        self.hotkeyCard.add_row(
            FluentIcon.HISTORY,
            "Restore default hotkeys",
            "Ctrl + Alt + F1 to F4 select the first four voices, F9 toggles bypass and F10 toggles mute.",
            self.restoreHotkeysButton,
        )

    def _capture_requested(self, action_id: str) -> None:
        for other_id, (_row, editor, _detail) in self._hotkey_rows.items():
            if other_id != action_id:
                editor.cancel()           # only one hotkey is picked at a time
        self.hotkeyCaptureRequested.emit(action_id)

    def begin_hotkey_listening(self, action_id: str) -> None:
        """The engine has switched the hotkeys off: the row that asked can now read the next key press."""
        entry = self._hotkey_rows.get(action_id)
        if entry is not None:
            entry[1].listen()

    def apply_hotkeys(self, data: dict) -> None:
        def apply() -> None:
            for item in data["items"]:
                entry = self._hotkey_rows.get(item["id"])
                if entry is None:
                    continue
                row, editor, detail = entry
                editor.set_state(item["text"], item["bound"], item["registered"])
                conflict = item["conflict"]
                row.contentLabel.setText(conflict or detail)
                if conflict:   # the combination was refused: the row says so until the next change
                    row.contentLabel.setTextColor(danger_color(), danger_color())
                else:
                    muted(row.contentLabel)
            self.restoreHotkeysButton.setEnabled(not data["all_default"])

        self.update_when_shown("hotkeys", apply)

    def apply_flags(self, flags: dict) -> None:
        def apply() -> None:
            set_switch(self.onnxSwitch, bool(flags["enable_onnx"]))
            set_switch(self.liteSwitch, bool(flags.get("lite_mode")))
            set_switch(self.stabilitySwitch, bool(flags.get("stability_buffer")))

        self.update_when_shown("flags", apply)

    def apply_onnx(self, data: dict) -> None:
        path = data["path"]
        self.modelRow.contentLabel.setText(path if path else "No model selected.")
        set_switch(self.onnxSwitch, bool(data["enabled"]))

    def apply_profile(self, data: dict) -> None:
        if data.get("path"):
            kind = "Loaded on startup" if data.get("autoloaded") else "Last loaded"
            self.loadRow.contentLabel.setText(f"{kind}: {data['path']}")
