"""Voice page: every effect control, grouped by what it shapes."""

from __future__ import annotations

from typing import Dict, List, Optional

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget
from qfluentwidgets import Action, FluentIcon, PushButton, RoundMenu

import preset_names
from styles import GAP
from .dialogs import NameDialog, ask_preset_export_path, ask_preset_import_path
from .widgets import Card, PageBase, QuietDropDownButton, ReflowGrid, SliderRow, icon_by_name

# name, title, min, max, scale (slider steps per unit), unit
GROUPS = [
    (
        "Pitch and formant",
        [
            ("pitch_semitones", "Pitch shift (semitones)", -12.0, 12.0, 10.0, "st"),
            ("formant_semitones", "Formant warp (vocal tract)", -6.0, 6.0, 10.0, "st"),
        ],
    ),
    (
        "Noise gate",
        [
            ("gate_threshold_db", "Noise gate threshold", -70.0, -15.0, 1.0, "dB"),
            ("hpf_cutoff_hz", "Rumble filter (high-pass)", 20.0, 600.0, 1.0, "Hz"),
        ],
    ),
    (
        "Character",
        [
            ("ring_mod_hz", "Ring mod frequency", 0.0, 100.0, 1.0, "Hz"),
            ("ring_mod_mix", "Ring mod wet mix", 0.0, 1.0, 100.0, "%"),
            ("drive_db", "Comms drive / saturation", 0.0, 18.0, 10.0, "dB"),
        ],
    ),
    (
        "Reverb",
        [
            ("reverb_mix", "Dark Shadow reverb", 0.0, 0.6, 100.0, "%"),
        ],
    ),
    (
        "Output",
        [
            ("lpf_cutoff_hz", "Low-pass cutoff", 1500.0, 16000.0, 1.0, "Hz"),
            ("output_gain_db", "Output gain", -12.0, 12.0, 10.0, "dB"),
        ],
    ),
]


ROW_ORDER = [name for _title, specs in GROUPS for name, *_rest in specs]
ROWS_PER_TICK = 2       # a preset change moves the sliders two at a time, 4 ms apart (about 20 ms in all) ...
ROW_TICK_MS = 4         # ... so no single stretch of Python painting is long enough to hold up the audio thread


class VoiceView(PageBase):
    paramChanged = pyqtSignal(str, float)   # voice parameter name, new value
    resetRequested = pyqtSignal()           # re-apply the active preset
    presetRequested = pyqtSignal(str)       # switch to this built-in preset
    presetSaveRequested = pyqtSignal(str)   # save the current settings under this name
    presetImportRequested = pyqtSignal(str)     # path of a preset file
    exportCurrentRequested = pyqtSignal(str)    # path to write the current settings to

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("voiceView", "Voice", "Fine-tune the active preset. Changes apply instantly.", parent)
        self.rows: Dict[str, SliderRow] = {}
        self._custom_names: List[str] = []
        self._reserved: List[str] = []
        self._row_queue: List[tuple] = []
        self._row_timer = QTimer(self)
        self._row_timer.setSingleShot(True)
        self._row_timer.setInterval(ROW_TICK_MS)
        self._row_timer.timeout.connect(self._drain_rows)

        self.presetsButton = QuietDropDownButton(FluentIcon.TAG, "Presets")
        menu = RoundMenu(parent=self)
        # the built-in voices, filled in by set_presets
        self.tacticalMenu = RoundMenu("Tactical voices", self)
        self.tacticalMenu.setIcon(FluentIcon.GAME)
        self.femaleMenu = RoundMenu("Female voices", self)
        self.femaleMenu.setIcon(FluentIcon.PEOPLE)
        menu.addMenu(self.tacticalMenu)
        menu.addMenu(self.femaleMenu)
        menu.addSeparator()
        menu.addAction(Action(FluentIcon.SAVE, "Save as preset…", triggered=self._save_as_preset))
        menu.addAction(Action(FluentIcon.DOWNLOAD, "Import preset…", triggered=self._import_preset))
        menu.addSeparator()
        menu.addAction(Action(FluentIcon.SHARE, "Export current settings…", triggered=self._export_current))
        self.presetsButton.setMenu(menu)
        self.resetButton = PushButton(FluentIcon.SYNC, "Reset to preset")
        self.resetButton.clicked.connect(self.resetRequested)
        self.actions.addWidget(self.presetsButton)
        self.actions.addWidget(self.resetButton)

        # two columns: pitch + character on the left, gate + reverb + output on the right
        columns = ReflowGrid(min_item_width=400, max_columns=2)
        left, right = self._column(), self._column()
        for index, (title, specs) in enumerate(GROUPS):
            card = Card(title)
            for name, label, lo, hi, scale, unit in specs:
                row = SliderRow(name, label, lo, hi, scale, unit)
                row.valueChanged.connect(self._on_row_changed)
                self.rows[name] = row
                card.body.addWidget(row)
            (left if index in (0, 2) else right).layout().addWidget(card)
        for col in (left, right):
            col.layout().addStretch(1)
            columns.add(col)
        self.content.addWidget(columns)
        self.finish()

    @staticmethod
    def _column() -> QWidget:
        col = QWidget()
        lay = QVBoxLayout(col)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(GAP)
        return col

    # ------------------------------------------------------------------ preset actions (dialogs live here)
    def _save_as_preset(self) -> None:
        suggestion = preset_names.unique_name("My preset", self._custom_names, self._reserved)
        dialog = NameDialog(
            self.window(),
            "Save as preset",
            "Save",
            self._custom_names,
            self._reserved,
            initial=suggestion,
            caption="Saves every slider as it is now. It appears next to the built-in voices on the Live page.",
        )
        dialog.accepted.connect(lambda: self.presetSaveRequested.emit(dialog.name()))
        dialog.finished.connect(dialog.deleteLater)
        dialog.open()

    def _import_preset(self) -> None:
        path = ask_preset_import_path(self.window())
        if path:
            self.presetImportRequested.emit(path)

    def _export_current(self) -> None:
        path = ask_preset_export_path(self.window(), self._active_name or "My voice")
        if path:
            self.exportCurrentRequested.emit(path)

    # ------------------------------------------------------------------ slots (bridge -> view)
    _active_name = ""

    def set_catalog(self, data: dict) -> None:
        self._reserved = list(data["preset_rules"]["reserved"])

    def set_presets(self, data: dict) -> None:
        self._custom_names = list(data["custom_names"])
        for sub in (self.tacticalMenu, self.femaleMenu):
            sub.clear()
        for p in data["presets"]:
            sub = {"tactical": self.tacticalMenu, "female": self.femaleMenu}.get(p.get("group"))
            if sub is None:
                continue
            label = f"{p['title']}  ·  {p['tag']}" if p["tag"] else p["title"]
            sub.addAction(
                Action(icon_by_name(p["icon"]), label, triggered=lambda _=False, k=p["key"]: self.presetRequested.emit(k))
            )

    def apply_preset(self, name: str) -> None:
        self._active_name = name
        self.update_when_shown(
            "preset", lambda: self.captionLabel.setText(f"Fine-tune the active preset: {name}. Changes apply instantly.")
        )

    def apply_params(self, params: dict) -> None:
        self.update_when_shown("params", lambda: self._queue_rows(params))

    def _queue_rows(self, params: dict) -> None:
        self._row_queue = [(n, float(params[n])) for n in ROW_ORDER if n in params and self.rows[n].differs(float(params[n]))]
        self._drain_rows()

    def _drain_rows(self) -> None:
        for _ in range(ROWS_PER_TICK):
            if not self._row_queue:
                return
            name, value = self._row_queue.pop(0)
            self.rows[name].set_value(value)
        if self._row_queue:
            self._row_timer.start()

    def _on_row_changed(self, name: str, value: float) -> None:
        self._row_queue = [(n, v) for n, v in self._row_queue if n != name]  # the player's hand wins over a pending update
        self.paramChanged.emit(name, value)
