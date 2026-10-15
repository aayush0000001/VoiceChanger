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

    MUTE_CAPTION = "Silences the output."
    BYPASS_CAPTION = "Raw voice, no effects."

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("liveView", "Live", "Pick a voice, mute or bypass, and watch your levels.", parent)

        self._state = "stopped"
        self._busy = False
        self._button_key = None
        self._tag = ""
        self._stats_at = 0.0
        self._tiles: Dict[str, PresetTile] = {}
        self._tile_actions: Dict[str, str] = {}   # built-in tile key -> the hotkey action that selects it
        self._active_preset = ""
        self._custom_names: List[str] = []
        self._reserved: List[str] = []
        self._lite = False

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
        self.femaleTitle = StrongBodyLabel("Female voices")
        self.femaleTitle.setContentsMargins(2, GAP_SM, 0, 0)
        self.content.addWidget(self.femaleTitle)
        self.femaleGrid = ReflowGrid(min_item_width=176, max_columns=4)
        self.content.addWidget(self.femaleGrid)

        # quick controls + levels side by side (stacked on narrow windows)
        row = ReflowGrid(min_item_width=380, max_columns=2)
        self.controlsCard = GroupCard("Quick controls")
        self.muteSwitch = self._make_switch("mic_muted")
        self.bypassSwitch = self._make_switch("bypass_effects")
        self.muteRow = self.controlsCard.add_row(
            FluentIcon.MUTE, "Hardware mute", self.MUTE_CAPTION, self.muteSwitch
        )
        self.bypassRow = self.controlsCard.add_row(
            FluentIcon.SPEED_OFF, "Bypass effects", self.BYPASS_CAPTION, self.bypassSwitch
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
        """Built-in tiles first (Ctrl+Alt+F1..F4), then the player's own; the female voices get a section of their own."""
        self._custom_names = list(data["custom_names"])
        self.tileGrid.clear()
        self.femaleGrid.clear()
        self._tiles.clear()
        self._tile_actions.clear()
        for p in data["presets"]:
            tile = PresetTile(
                p["key"], p["title"], p["tag"], p["hotkey"], p["blurb"], icon_by_name(p["icon"]), custom=p["custom"]
            )
            if p.get("hotkey_action"):
                self._tile_actions[p["key"]] = p["hotkey_action"]
            tile.chosen.connect(self.presetRequested)
            if self._lite:
                tile.backgroundColorAni.setDuration(0)
            if p["custom"]:
                tile.renameRequested.connect(self._rename_preset)
                tile.exportRequested.connect(self._export_preset)
                tile.deleteRequested.connect(self._delete_preset)
            self._tiles[p["key"]] = tile
            (self.femaleGrid if p.get("group") == "female" else self.tileGrid).add(tile)
        has_female = any(p.get("group") == "female" for p in data["presets"])
        self.femaleTitle.setVisible(has_female)
        self.femaleGrid.setVisible(has_female)
        self.apply_preset(self._active_preset)

    def apply_hotkeys(self, data: dict) -> None:
        """The hotkeys shown next to the voices, mute and bypass follow the ones set in Settings."""
        def apply() -> None:
            by_id = {item["id"]: item for item in data["items"]}
            for row, action, caption in (
                (self.muteRow, "mute", self.MUTE_CAPTION),
                (self.bypassRow, "bypass", self.BYPASS_CAPTION),
            ):
                item = by_id.get(action)
                keys = item["text"] if item and item["bound"] else ""
                row.contentLabel.setText(f"{keys}  ·  {caption}" if keys else caption)
                switch = self.muteSwitch if action == "mute" else self.bypassSwitch
                switch.setToolTip(f"Hotkey: {keys}" if keys else "No hotkey is set. Choose one in Settings > Global hotkeys.")
            for key, action in self._tile_actions.items():
                item = by_id.get(action)
                tile = self._tiles.get(key)
                if tile is not None:
                    tile.set_hotkey(item["text"] if item and item["bound"] else "No hotkey")

        self.update_when_shown("hotkeys", apply)

    def set_lite(self, lite: bool) -> None:
        """Lite mode: hover and selection colours switch at once instead of fading (every fade frame is Python painting)."""
        self._lite = lite
        for tile in self._tiles.values():
            tile.backgroundColorAni.setDuration(0 if lite else 120)

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
        self.update_when_shown("preset", self._select_tiles)

    def _select_tiles(self) -> None:
        for key, tile in self._tiles.items():
            tile.set_selected(key == self._active_preset)

    def apply_flags(self, flags: dict) -> None:
        def apply() -> None:
            set_switch(self.muteSwitch, bool(flags["mic_muted"]))
            set_switch(self.bypassSwitch, bool(flags["bypass_effects"]))

        self.update_when_shown("flags", apply)

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
            self.roundtripStat.set(f"~{data['roundtrip_ms']:.1f} ms", f"two buffers + DSP + {data['driver_ms']:.0f} ms driver")
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
