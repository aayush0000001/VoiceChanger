"""
Fluent window shell: a FluentWindow with Mica, a collapsible navigation pane, the four pages, a tray icon and
the wiring between the pages and the EngineBridge. All engine traffic goes through signals and slots.
"""

from __future__ import annotations

import ctypes
import gc
import sys
import time
from pathlib import Path

from PyQt6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR, QRectF, QSettings, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices, QGuiApplication, QIcon, QPainter, QPalette, QPixmap
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon
import qfluentwidgets
from qfluentwidgets import (
    Action,
    FluentIcon,
    FluentWindow,
    NavigationItemPosition,
    SystemThemeListener,
    SystemTrayMenu,
    Theme,
    isDarkTheme,
    setTheme,
    setThemeColor,
)

import reporting
from appinfo import APP_NAME
from engine_bridge import EngineBridge
from views import DevicesView, LiveView, SettingsView, VoiceView
from views.dialogs import ReportDialog
from views.widgets import Toasts

THEME_KEYS = {"auto": Theme.AUTO, "light": Theme.LIGHT, "dark": Theme.DARK}
MICA_MIN_BUILD = 22000  # Windows 11


def mica_supported() -> bool:
    return sys.platform == "win32" and sys.getwindowsversion().build >= MICA_MIN_BUILD


def trim_working_set() -> None:
    """Hands the memory this process is not touching back to Windows (its pages stay in the standby list and come back
    on demand). Used by Lite mode while the window is hidden, so the audio engine and hotkeys cost the least RAM."""
    if sys.platform != "win32":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        ctypes.windll.psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
        ctypes.windll.psapi.EmptyWorkingSet(kernel32.GetCurrentProcess())
    except Exception:
        pass


def make_app_icon() -> QIcon:
    """Rounded red square with a five-bar voice waveform, drawn at every size the shell asks for."""
    candidates = [
        Path(getattr(sys, "_MEIPASS", "")) / "app_icon.ico",
        Path(sys.executable).parent / "app_icon.ico",
        Path(__file__).resolve().parent / "app_icon.ico",
        Path.cwd() / "app_icon.ico",
    ]
    for c in candidates:
        if c.is_file():
            ico = QIcon(str(c))
            if not ico.isNull():
                return ico

    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#FF4655"))
        p.drawRoundedRect(QRectF(0, 0, size, size), size * 0.22, size * 0.22)
        p.setBrush(QColor("#FFFFFF"))
        heights = (0.30, 0.52, 0.72, 0.52, 0.30)
        bar_w, gap = size * 0.09, size * 0.065
        total = bar_w * len(heights) + gap * (len(heights) - 1)
        x = (size - total) / 2
        for h in heights:
            bh = size * h
            p.drawRoundedRect(QRectF(x, (size - bh) / 2, bar_w, bh), bar_w / 2, bar_w / 2)
            x += bar_w + gap
        p.end()
        icon.addPixmap(pm)
    return icon


class MainWindow(FluentWindow):
    # window -> bridge (the only way the shell talks to the engine besides the views' own signals)
    engineToggleRequested = pyqtSignal()
    telemetryActive = pyqtSignal(bool)
    reportPrepareRequested = pyqtSignal(str)    # the Qt / library versions line (the engine side does not know the UI)
    reportSubmitRequested = pyqtSignal(object)  # {"title", "body", "reply_to"} from the report form
    errorOccurred = pyqtSignal(str)             # raised by the exception hooks, which can fire on any thread

    # Qt sends change events while the base class is still being constructed, before __init__ gets to set these
    _lite = False
    _trim_timer = None

    def __init__(self) -> None:
        settings = QSettings(APP_NAME, APP_NAME)
        saved_theme = str(settings.value("appearance/theme", "auto"))
        if saved_theme not in THEME_KEYS:
            saved_theme = "auto"
        setTheme(THEME_KEYS[saved_theme])  # before any widget exists, so nothing flashes the wrong theme
        accent = QGuiApplication.palette().color(QPalette.ColorRole.Accent)  # the Windows accent colour
        setThemeColor(accent if accent.isValid() and accent.alpha() > 0 else QColor("#0078D4"))

        super().__init__()
        self.toasts = Toasts(self)
        self.settings = settings
        self._quitting = False
        self._session_ending = False
        self._engine_live = False
        self._first_show = True
        self._theme_key = saved_theme
        self._mica_pref = settings.value("appearance/mica", True, type=bool)
        self._last_error_toast = 0.0
        self._lite = False   # Lite mode: set by apply_flags once the bridge reports the saved setting
        self._trim_timer = QTimer(self)
        self._trim_timer.setSingleShot(True)
        self._trim_timer.setInterval(3000)
        self._trim_timer.timeout.connect(self._trim_memory)

        self.bridge = EngineBridge()
        self.themeListener = SystemThemeListener(self)
        self.liveView = LiveView(self)
        self.voiceView = VoiceView(self)
        self.devicesView = DevicesView(self)
        self.settingsView = SettingsView(self)

        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(make_app_icon())
        self.setMinimumSize(880, 620)
        self._place_window()

        self._init_navigation()
        self._wire()
        self._init_tray()
        self._init_backdrop()

        self.bridge.start()
        self.themeListener.start()

    # ------------------------------------------------------------------ construction
    def _place_window(self) -> None:
        geometry = self.settings.value("window/geometry")
        if geometry is not None and self.restoreGeometry(geometry):
            return
        self.resize(1100, 780)
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.center().x() - self.width() // 2, screen.center().y() - self.height() // 2)

    def _init_navigation(self) -> None:
        self.addSubInterface(self.liveView, FluentIcon.MICROPHONE, "Live")
        self.addSubInterface(self.voiceView, FluentIcon.MIX_VOLUMES, "Voice")
        self.addSubInterface(self.devicesView, FluentIcon.HEADPHONE, "Devices")
        self.engineNavItem = self.navigationInterface.addItem(
            routeKey="engineToggle",
            icon=FluentIcon.POWER_BUTTON,
            text="Start engine",
            onClick=self._on_engine_nav_clicked,
            selectable=False,
            position=NavigationItemPosition.BOTTOM,
            tooltip="Start or stop the audio engine",
        )
        self.addSubInterface(self.settingsView, FluentIcon.SETTING, "Settings", NavigationItemPosition.BOTTOM)
        self.navigationInterface.setExpandWidth(220)
        self.navigationInterface.setReturnButtonVisible(False)
        self.stackedWidget.currentChanged.connect(self._sync_telemetry)

    def _wire(self) -> None:
        b, lv, vv, dv, sv = self.bridge, self.liveView, self.voiceView, self.devicesView, self.settingsView

        # views -> bridge  (user intent; queued onto the bridge thread)
        lv.engineToggleRequested.connect(b.toggle_engine)
        self.engineToggleRequested.connect(b.toggle_engine)
        lv.presetRequested.connect(b.apply_preset)
        lv.presetRenameRequested.connect(b.rename_preset)
        lv.presetExportRequested.connect(b.export_preset)
        lv.presetDeleteRequested.connect(b.delete_preset)
        lv.presetImportRequested.connect(b.import_preset)
        lv.flagToggled.connect(b.set_flag)
        vv.paramChanged.connect(b.set_param)
        vv.resetRequested.connect(b.reset_to_preset)
        vv.presetRequested.connect(b.apply_preset)
        vv.presetSaveRequested.connect(b.save_preset)
        vv.presetImportRequested.connect(b.import_preset)
        vv.exportCurrentRequested.connect(b.export_current)
        dv.deviceSelected.connect(b.select_device)
        dv.blockSizeSelected.connect(b.set_block_size)
        dv.rescanRequested.connect(b.rescan_devices)
        dv.flagToggled.connect(b.set_flag)
        dv.monitorVolumeChanged.connect(b.set_monitor_volume)
        dv.pathTestRequested.connect(b.run_path_test)
        sv.flagToggled.connect(b.set_flag)
        sv.onnxModelSelected.connect(b.load_onnx_model)
        sv.profileSaveRequested.connect(b.save_profile)
        sv.profileLoadRequested.connect(b.load_profile)
        sv.resetAllRequested.connect(b.reset_all)
        sv.hotkeyCaptureRequested.connect(b.begin_hotkey_capture)
        sv.hotkeyChosen.connect(b.change_hotkey)
        sv.hotkeyCleared.connect(b.clear_hotkey)
        sv.hotkeyCaptureCancelled.connect(b.cancel_hotkey_capture)
        sv.hotkeysRestoreRequested.connect(b.restore_default_hotkeys)
        sv.reportRequested.connect(self.request_report)
        self.reportPrepareRequested.connect(b.prepare_report)
        self.reportSubmitRequested.connect(b.submit_report)
        self.telemetryActive.connect(b.set_telemetry_active)

        # bridge -> views  (state; delivered on the GUI thread)
        b.catalog_ready.connect(lv.set_catalog)
        b.catalog_ready.connect(vv.set_catalog)
        b.catalog_ready.connect(sv.set_catalog)
        b.presets_changed.connect(lv.set_presets)
        b.presets_changed.connect(vv.set_presets)
        b.preset_changed.connect(lv.apply_preset)
        b.preset_changed.connect(vv.apply_preset)
        b.params_changed.connect(vv.apply_params)
        b.flags_changed.connect(lv.apply_flags)
        b.flags_changed.connect(dv.apply_flags)
        b.flags_changed.connect(sv.apply_flags)
        b.flags_changed.connect(self.apply_flags)
        b.devices_changed.connect(dv.apply_devices)
        b.engine_state_changed.connect(lv.apply_engine_state)
        b.engine_state_changed.connect(dv.apply_engine_state)
        b.engine_state_changed.connect(self.apply_engine_state)
        b.status_updated.connect(lv.update_status)
        b.path_test_state.connect(dv.apply_path_test_state)
        b.path_test_state.connect(self.apply_path_test_state)
        b.path_test_finished.connect(dv.show_path_test_result)
        b.path_test_finished.connect(self.on_path_test_finished)
        b.onnx_changed.connect(sv.apply_onnx)
        b.profile_changed.connect(sv.apply_profile)
        b.hotkeys_changed.connect(sv.apply_hotkeys)
        b.hotkeys_changed.connect(lv.apply_hotkeys)
        b.hotkey_capture_ready.connect(sv.begin_hotkey_listening)
        b.notify.connect(self.show_notification)
        b.report_ready.connect(self.open_report_dialog)
        b.report_finished.connect(self.on_report_finished)
        self.errorOccurred.connect(self.on_unhandled_error)

        # appearance is the window's own business (no engine involved)
        sv.themeSelected.connect(self.set_theme)
        sv.micaToggled.connect(self.set_mica)
        sv.resetAllRequested.connect(self.reset_appearance)
        sv.quitRequested.connect(self.quit_app)
        self.themeListener.systemThemeChanged.connect(self._on_system_theme_changed)

    def _init_tray(self) -> None:
        self.tray = QSystemTrayIcon(self.windowIcon(), self)
        self.tray.setToolTip(APP_NAME)
        menu = SystemTrayMenu(parent=self)
        menu.addAction(Action(FluentIcon.HOME, "Restore window", triggered=self.restore_from_tray))
        menu.addSeparator()
        menu.addAction(Action(FluentIcon.CLOSE, f"Exit {APP_NAME}", triggered=self.quit_app))
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()

    def _init_backdrop(self) -> None:
        self._apply_backdrop()

    def _apply_backdrop(self) -> None:
        """Mica when the player wants it and the PC has it; a solid theme background otherwise, and in Lite mode."""
        supported = mica_supported()
        self.setMicaEffectEnabled(supported and self._mica_pref and not self._lite)
        self.settingsView.apply_appearance(self._theme_key, supported, self._mica_pref, self._lite)

    # ------------------------------------------------------------------ appearance
    def set_theme(self, key: str) -> None:
        if key not in THEME_KEYS:
            return
        self._theme_key = key
        self.settings.setValue("appearance/theme", key)
        setTheme(THEME_KEYS[key])

    def set_mica(self, enabled: bool) -> None:
        self._mica_pref = enabled
        self.settings.setValue("appearance/mica", enabled)
        if mica_supported():
            self.setMicaEffectEnabled(enabled and not self._lite)

    def reset_appearance(self, *_args) -> None:
        """Part of 'Reset all settings': follow Windows' theme again and bring Mica back."""
        self.set_theme("auto")
        self.set_mica(True)
        self.settingsView.apply_appearance("auto", mica_supported(), True, self._lite)

    def _on_system_theme_changed(self) -> None:
        setTheme(Theme.AUTO)

    def _onThemeChangedFinished(self) -> None:
        super()._onThemeChangedFinished()
        # DWM sometimes ignores the first re-application after a theme flip; the library's own demo retries too
        if self.isMicaEffectEnabled():
            QTimer.singleShot(100, lambda: self.windowEffect.setMicaEffect(self.winId(), isDarkTheme()))

    # ------------------------------------------------------------------ slots
    def _apply_animation_policy(self) -> None:
        """Page transitions repaint a whole page for 300 ms, enough to delay a live audio block (measured), so they
        only play while the audio engine is stopped."""
        self.stackedWidget.setAnimationEnabled(not (self._engine_live or self._lite))

    def apply_flags(self, flags: dict) -> None:
        """Lite mode is a window-level matter too: no Mica, no fades, and memory handed back while hidden."""
        lite = bool(flags.get("lite_mode"))
        if lite == self._lite:
            return
        self._lite = lite
        self._apply_backdrop()
        self._apply_animation_policy()
        self.liveView.set_lite(lite)
        self._sync_telemetry()   # starts the memory trim when the window is already hidden

    def _trim_memory(self) -> None:
        if self._lite and not (self.isVisible() and not self.isMinimized()):
            if not self._engine_live:
                gc.collect()  # a full collection holds the GIL for tens of ms, which a running audio callback would feel
            trim_working_set()

    def apply_engine_state(self, data: dict) -> None:
        running = data["state"] == "running"
        self._engine_live = data["state"] in ("running", "starting")
        self._apply_animation_policy()
        self.engineNavItem.setText("Stop engine" if running else "Start engine")
        self.engineNavItem.setIcon(FluentIcon.PAUSE if running else FluentIcon.POWER_BUTTON)
        self.engineNavItem.setEnabled(data["state"] != "starting")
        self.tray.setToolTip(f"{APP_NAME}  ·  {'live' if running else 'stopped'}")

    def apply_path_test_state(self, data: dict) -> None:
        self.liveView.set_busy(bool(data["running"]))
        self.engineNavItem.setEnabled(not data["running"])

    def on_path_test_finished(self, data: dict) -> None:
        if data["passed"]:
            self.show_notification({"level": "success", "title": "Audio path test passed", "text": ""})
        else:
            self.show_notification(
                {"level": "error", "title": "Audio path test failed", "text": "Details are on the Devices page."}
            )

    def show_notification(self, data: dict) -> None:
        level, title, text = data["level"], data["title"], data.get("text", "")
        if not self.isVisible():
            if level in ("error", "warning"):
                self.tray.showMessage(title, text, QSystemTrayIcon.MessageIcon.Warning, 4000)
            return
        self.toasts.show(level, title, text, 8000 if level in ("error", "warning") else 4000)

    # ------------------------------------------------------------------ problem reports
    def request_report(self) -> None:
        versions = f"Qt {QT_VERSION_STR}  |  PyQt {PYQT_VERSION_STR}  |  QFluentWidgets {qfluentwidgets.__version__}"
        self.reportPrepareRequested.emit(versions)

    def open_report_dialog(self, data: dict) -> None:
        if data["sending"]:
            self.show_notification({"level": "info", "title": "Still sending", "text": "The previous report has not finished yet."})
            return
        if data["wait_s"]:
            text = f"You can send another report in {data['wait_s']} seconds."
            self.show_notification({"level": "warning", "title": "Please wait a moment", "text": text})
            return
        dialog = ReportDialog(self, data["diagnostics"])
        dialog.accepted.connect(lambda: self.reportSubmitRequested.emit(dialog.payload()))
        dialog.finished.connect(dialog.deleteLater)
        dialog.open()

    def on_report_finished(self, result: dict) -> None:
        status = result.get("status")
        if status == "sent":
            self.show_notification({"level": "success", "title": "Report sent", "text": "Thank you. It reached the developer."})
        elif status == "rate_limited":
            text = f"You can send another report in {result.get('wait_s', 60)} seconds."
            self.show_notification({"level": "warning", "title": "Please wait a moment", "text": text})
        else:
            self._report_by_mail(result)

    def _report_by_mail(self, result: dict) -> None:
        """The form service is not set up or did not answer: the same report goes out through the mail program, or
        onto the clipboard when there is no address to mail or no mail program."""
        text, address = result.get("text", ""), result.get("address", "")
        if result.get("reason") == "no_key":
            why = "The report service is not set up yet."
        else:
            why = f"The report service could not be reached ({result.get('detail', 'unknown error')})."
        if reporting.is_placeholder_address(address):
            QApplication.clipboard().setText(text)
            self.show_notification(
                {"level": "info", "title": "Report copied", "text": f"{why} It is on your clipboard: paste it into an email or message to the developer."}
            )
        elif QDesktopServices.openUrl(QUrl.fromEncoded(result["mailto"].encode("ascii"))):
            self.show_notification(
                {"level": "info", "title": "Finish in your mail program", "text": f"{why} Your mail program opened with the report ready to send."}
            )
        else:
            QApplication.clipboard().setText(text)
            self.show_notification(
                {"level": "warning", "title": "Couldn't open a mail program", "text": f"The report is on your clipboard: paste it into an email to {address}."}
            )

    def on_unhandled_error(self, text: str) -> None:
        """A bug in a slot or a thread: it is already in the log; the player gets one calm message, not a crash."""
        now = time.monotonic()
        if now - self._last_error_toast < 5.0:
            return
        self._last_error_toast = now
        short = text if len(text) <= 140 else text[:137] + "..."
        self.show_notification(
            {"level": "error", "title": "Something went wrong", "text": f"{short}. It was logged. Settings > Report a problem sends it to the developer."}
        )

    def _on_engine_nav_clicked(self, *_args) -> None:
        self.engineToggleRequested.emit()

    # ------------------------------------------------------------------ tray & lifecycle
    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.restore_from_tray()

    def restore_from_tray(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _sync_telemetry(self, *_args) -> None:
        """The meters only run while the Live page is on screen; a hidden or minimized window, or another page,
        costs the audio engine nothing (hotkeys and audio keep running regardless)."""
        window_shown = self.isVisible() and not self.isMinimized()
        if window_shown:  # state changes that arrived while nobody could see them (hotkeys in a match)
            if self._trim_timer is not None:
                self._trim_timer.stop()
            for page in (self.liveView, self.voiceView, self.devicesView, self.settingsView):
                page.flush_deferred()
        elif self._lite and self._trim_timer is not None:
            self._trim_timer.start()  # a few seconds after the window went away, give the memory back
        self.telemetryActive.emit(window_shown and self.stackedWidget.currentWidget() is self.liveView)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        toasts = getattr(self, "toasts", None)
        if toasts is not None:
            toasts.relayout()

    def showEvent(self, e) -> None:
        super().showEvent(e)
        if self._first_show:
            self._first_show = False
            QTimer.singleShot(0, lambda: self.navigationInterface.expand(useAni=False))  # labels visible by default
        self._sync_telemetry()

    def hideEvent(self, e) -> None:
        super().hideEvent(e)
        self._sync_telemetry()

    def changeEvent(self, e) -> None:
        super().changeEvent(e)
        self._sync_telemetry()

    def closeEvent(self, e) -> None:
        if self._quitting or self._session_ending:
            super().closeEvent(e)
            return
        # closing only hides the window: audio and hotkeys keep running from the tray
        e.ignore()
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.hide()
        self.tray.showMessage(
            f"{APP_NAME} is still running",
            "Meters are paused. Your hotkeys stay active.",
            QSystemTrayIcon.MessageIcon.Information,
            2000,
        )

    def on_session_end(self, *_args) -> None:
        """Windows is logging off or shutting down: let the window really close instead of vetoing the shutdown."""
        self._session_ending = True

    def shutdown_engine(self) -> None:
        """Stops audio and hotkeys. Safe to call more than once; also wired to QApplication.aboutToQuit."""
        self.themeListener.terminate()
        self.themeListener.wait(500)
        self.bridge.request_shutdown()

    def quit_app(self) -> None:
        if self._quitting:
            return
        self._quitting = True
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.hide()
        self.tray.hide()
        self.shutdown_engine()
        QApplication.quit()
