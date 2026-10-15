"""Live page: engine switch, voice presets, mute / bypass, level meters and the latency readout."""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, FluentIcon, PrimaryPushButton, StrongBodyLabel, SwitchButton, TransparentPushButton

from styles import GAP_SM
from .dialogs import ConfirmDialog, NameDialog, ask_preset_export_path, ask_preset_import_path
from .widgets import (
    Card,
    GroupCard,
    LevelMeter,
    MeterBar,
    PageBase,
    PresetTile,
    ReflowGrid,
    StatBlock,
    StatusPill,
    icon_by_name,
    muted,
    set_switch,
)


STATS_INTERVAL_S = 0.25  # how often the latency numbers refresh (the meters run at the full telemetry rate)


class LiveView(PageBase):
    # ---- user intent (the main window connects these to EngineBridge slots) ----
    engineToggleRequested = pyqtSignal()
    presetRequested = pyqtSignal(str)          # preset key
    presetRenameRequested = pyqtSignal(str, str)    # custom preset key, new name
    presetExportRequested = pyqtSignal(str, str)    # preset key, file path
    presetDeleteRequested = pyqtSignal(str)         # custom preset key
    presetImportRequested = pyqtSignal(str)         # file path
    flagToggled = pyqtSignal(str, bool)        # "mic_muted" | "bypass_effects", new value

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("liveView", "Live", "Pick a voice, mute or bypass, and watch your levels.", parent)

        self._state = "stopped"
        self._busy = False
        self._button_key = None
        self._tag = ""
        self._stats_at = 0.0
        self._tiles: Dict[str, PresetTile] = {}
        self._active_preset = ""
        self._custom_names: List[str] = []
        self._reserved: List[str] = []

        # header: status + the one big button
        self.statusPill = StatusPill()
        self.engineButton = PrimaryPushButton(FluentIcon.PLAY, "Start audio engine")
        self.engineButton.setMinimumWidth(188)
        self.engineButton.clicked.connect(self.engineToggleRequested)
        self.actions.addWidget(self.statusPill)
        self.actions.addWidget(self.engineButton)

        # voice presets (tiles are created by set_presets)
        section = QHBoxLayout()
        section.setContentsMargins(0, 0, 0, 0)
        title = StrongBodyLabel("Voice presets")
        title.setContentsMargins(2, GAP_SM, 0, 0)
        self.importButton = TransparentPushButton(FluentIcon.DOWNLOAD, "Import preset…")
        self.importButton.setToolTip("Add a preset someone sent you (a .json file)")
        self.importButton.clicked.connect(self._import_preset)
        section.addWidget(title)
        section.addStretch(1)
        section.addWidget(self.importButton)
        self.content.addLayout(section)
        self.tileGrid = ReflowGrid(min_item_width=176, max_columns=4)
        self.content.addWidget(self.tileGrid)

        # quick controls + levels side by side (stacked on narrow windows)
        row = ReflowGrid(min_item_width=380, max_columns=2)
        self.controlsCard = GroupCard("Quick controls")
        self.muteSwitch = self._make_switch("mic_muted")
        self.bypassSwitch = self._make_switch("bypass_effects")
        self.controlsCard.add_row(
            FluentIcon.MUTE, "Hardware mute", "Ctrl + Alt + F10  ·  Silences the output.", self.muteSwitch
        )
        self.controlsCard.add_row(
            FluentIcon.SPEED_OFF, "Bypass effects", "Ctrl + Alt + F9  ·  Raw voice, no effects.", self.bypassSwitch
        )

        self.levelsCard = Card("Levels")
        self.micMeter = LevelMeter("Microphone input")
        self.cableMeter = LevelMeter("Virtual cable output")
        self.levelsCard.body.addWidget(self.micMeter)
        self.levelsCard.body.addWidget(self.cableMeter)
        row.add(self.controlsCard)
        row.add(self.levelsCard)
        self.content.addWidget(row)

        # latency
        self.latencyCard = Card("Latency")
        stats = ReflowGrid(min_item_width=170, max_columns=4)
        self.dspStat = StatBlock("DSP time per block")
        self.roundtripStat = StatBlock("Estimated round trip")
        self.bufferStat = StatBlock("Buffer")
        self.loadStat = StatBlock("Real-time load")
        for block in (self.dspStat, self.roundtripStat, self.bufferStat, self.loadStat):
            stats.add(block)
        self.latencyCard.body.addWidget(stats)
        self.loadBar = MeterBar()
        self.latencyCard.body.addWidget(self.loadBar)
        self.loadHint = muted(CaptionLabel("Slowest block against the buffer deadline. Keep it well under 100%."))
        self.latencyCard.body.addWidget(self.loadHint)
        self.content.addWidget(self.latencyCard)

        self.finish()
        self._render_engine()

    # ------------------------------------------------------------------ construction helpers
    def _make_switch(self, flag: str) -> SwitchButton:
        switch = SwitchButton()
        switch.setOnText("On")
        switch.setOffText("Off")
        switch.checkedChanged.connect(lambda checked, f=flag: self.flagToggled.emit(f, checked))
        return switch

    # ------------------------------------------------------------------ slots (bridge -> view)
    def set_catalog(self, data: dict) -> None:
        self._reserved = list(data["preset_rules"]["reserved"])

    def set_presets(self, data: dict) -> None:
        """Built-in tiles first (Ctrl+Alt+F1..F4), then the player's own."""
        self._custom_names = list(data["custom_names"])
        self.tileGrid.clear()
        self._tiles.clear()
        for p in data["presets"]:
            tile = PresetTile(
                p["key"], p["title"], p["tag"], p["hotkey"], p["blurb"], icon_by_name(p["icon"]), custom=p["custom"]
            )
            tile.chosen.connect(self.presetRequested)
            if p["custom"]:
                tile.renameRequested.connect(self._rename_preset)
                tile.exportRequested.connect(self._export_preset)
                tile.deleteRequested.connect(self._delete_preset)
            self._tiles[p["key"]] = tile
            self.tileGrid.add(tile)
        self.apply_preset(self._active_preset)

    # ------------------------------------------------------------------ custom preset actions (dialogs live here)
    def _rename_preset(self, key: str) -> None:
        dialog = NameDialog(
            self.window(), "Rename preset", "Rename", self._custom_names, self._reserved,
            initial=key, ignore=key, adding=False,
        )
        dialog.accepted.connect(lambda: self.presetRenameRequested.emit(key, dialog.name()))
        dialog.finished.connect(dialog.deleteLater)
        dialog.open()

    def _export_preset(self, key: str) -> None:
        path = ask_preset_export_path(self.window(), key)
        if path:
            self.presetExportRequested.emit(key, path)

    def _delete_preset(self, key: str) -> None:
        dialog = ConfirmDialog(
            self.window(), "Delete this preset?", f"“{key}” will be removed from this PC. This cannot be undone.", "Delete"
        )
        dialog.accepted.connect(lambda: self.presetDeleteRequested.emit(key))
        dialog.finished.connect(dialog.deleteLater)
        dialog.open()

    def _import_preset(self) -> None:
        path = ask_preset_import_path(self.window())
        if path:
            self.presetImportRequested.emit(path)

    def apply_preset(self, name: str) -> None:
        self._active_preset = name
        for key, tile in self._tiles.items():
            tile.set_selected(key == name)

    def apply_flags(self, flags: dict) -> None:
        set_switch(self.muteSwitch, bool(flags["mic_muted"]))
        set_switch(self.bypassSwitch, bool(flags["bypass_effects"]))

    def apply_engine_state(self, data: dict) -> None:
        self._state = data["state"]
        self.bufferStat.set(
            f"{data['block_size']} samples", f"{data['block_ms']:.1f} ms at {data['sample_rate'] / 1000:g} kHz"
        )
        if self._state != "running":
            self.micMeter.reset()
            self.cableMeter.reset()
            self.dspStat.set("-")
            self.roundtripStat.set("-")
            self.loadStat.set("-")
            self.loadBar.set_state(0.0)
        self._render_engine()

    def set_busy(self, busy: bool) -> None:
        """True while the audio path test owns the devices."""
        self._busy = busy
        self._render_engine()

    def update_status(self, data: dict) -> None:
        """~30 Hz telemetry snapshot from the bridge."""
        tag = {"MUTED": "Muted", "BYPASS": "Bypass"}.get(data["tag"], data["tag"])
        if tag != self._tag:
            self._tag = tag
            self._render_engine()
        if not self.isVisible():
            return
        self.micMeter.set_level(data["in_db"])
        self.cableMeter.set_level(data["out_db"])
        now = time.monotonic()
        if data["running"] and now - self._stats_at >= STATS_INTERVAL_S:
            # numbers that change every block are unreadable at 30 Hz and repainting them is not free
            self._stats_at = now
            self.dspStat.set(f"{data['dsp_ms']:.2f} ms", f"slowest {data['peak_dsp_ms']:.2f} ms")
            self.roundtripStat.set(f"~{data['roundtrip_ms']:.1f} ms", "two buffers + DSP + ~6 ms driver")
            load = data["load"]
            self.loadStat.set(f"{load * 100:.0f}%", f"{data['xruns']} dropouts")
            self.loadBar.set_state(load, "danger" if load >= 0.85 else ("warn" if load >= 0.6 else "ok"))

    # ------------------------------------------------------------------ rendering
    def _render_engine(self) -> None:
        state = self._state
        if state == "running":
            self.statusPill.set_status(f"Live  ·  {self._tag}" if self._tag else "Live", "ok")
        elif state == "starting":
            self.statusPill.set_status("Starting…", "busy")
        elif state == "error":
            self.statusPill.set_status("Couldn't start", "error")
        else:
            self.statusPill.set_status("Stopped", "idle")

        # a preset or mute change only re-words the pill; re-creating the button's icon is not free
        key = (state, self._busy)
        if key == self._button_key:
            return
        self._button_key = key
        if state == "running":
            self.engineButton.setText("Stop audio engine")
            self.engineButton.setIcon(FluentIcon.PAUSE)
        elif state == "starting":
            self.engineButton.setText("Starting…")
        else:
            self.engineButton.setText("Start audio engine")
            self.engineButton.setIcon(FluentIcon.PLAY)
        self.engineButton.setEnabled(state != "starting" and not self._busy)
