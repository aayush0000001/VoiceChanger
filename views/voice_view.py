"""Voice page: every effect control, grouped by what it shapes."""

from __future__ import annotations

from typing import Dict, Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget
from qfluentwidgets import FluentIcon, PushButton

from styles import GAP
from .widgets import Card, PageBase, ReflowGrid, SliderRow

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


class VoiceView(PageBase):
    paramChanged = pyqtSignal(str, float)   # voice parameter name, new value
    resetRequested = pyqtSignal()           # re-apply the active preset

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("voiceView", "Voice", "Fine-tune the active preset. Changes apply instantly.", parent)
        self.rows: Dict[str, SliderRow] = {}

        self.resetButton = PushButton(FluentIcon.SYNC, "Reset to preset")
        self.resetButton.clicked.connect(self.resetRequested)
        self.actions.addWidget(self.resetButton)

        # two columns: pitch + character on the left, gate + reverb + output on the right
        columns = ReflowGrid(min_item_width=400, max_columns=2)
        left, right = self._column(), self._column()
        for index, (title, specs) in enumerate(GROUPS):
            card = Card(title)
            for name, label, lo, hi, scale, unit in specs:
                row = SliderRow(name, label, lo, hi, scale, unit)
                row.valueChanged.connect(self.paramChanged)
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

    # ------------------------------------------------------------------ slots (bridge -> view)
    def apply_preset(self, name: str) -> None:
        self.captionLabel.setText(f"Fine-tune the active preset: {name}. Changes apply instantly.")

    def apply_params(self, params: dict) -> None:
        for name, row in self.rows.items():
            if name in params:
                row.set_value(float(params[name]))
