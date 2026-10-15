"""
Qt side of being a normal Windows app, attached to the main window with one call:
  - the instance server: a second launch asks this copy to show its window (or, with --quit, to exit)
  - Settings > Application: Start with Windows, Open data folder, Run setup again, and the version in About
  - the first-run setup, shown until the player presses Done
"""

from __future__ import annotations

import ctypes
import sys

from PyQt6.QtCore import QEvent, QObject, QTimer
from PyQt6.QtNetwork import QLocalServer, QLocalSocket
from qfluentwidgets import CaptionLabel, FluentIcon, PushButton, SwitchButton

import appdata
import winintegration as wi
from appinfo import APP_VERSION
from views.widgets import muted, set_switch

SETUP_DONE_KEY = "setup/completed"


def send_command(command: str, timeout_ms: int = 1500) -> bool:
    """Sends 'show' or 'quit' to the running copy. True when it answered."""
    sock = QLocalSocket()
    sock.connectToServer(wi.server_name())
    if not sock.waitForConnected(timeout_ms):
        return False
    sock.write(command.encode("ascii") + b"\n")
    sock.waitForBytesWritten(timeout_ms)
    answered = sock.waitForReadyRead(timeout_ms) and bytes(sock.readAll()).startswith(b"ok")
    sock.disconnectFromServer()
    return answered


class InstanceServer(QObject):
    def __init__(self, window) -> None:
        super().__init__(window)
        self._window = window
        self._server = QLocalServer(self)
        self._server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)  # only this Windows user
        self._server.newConnection.connect(self._on_connection)
        if not self._server.listen(wi.server_name()):
            QLocalServer.removeServer(wi.server_name())
            self._server.listen(wi.server_name())

    def _on_connection(self) -> None:
        sock = self._server.nextPendingConnection()
        if sock is None:
            return
        sock.readyRead.connect(lambda s=sock: self._on_ready(s))
        sock.disconnected.connect(sock.deleteLater)

    def _on_ready(self, sock: QLocalSocket) -> None:
        if not sock.canReadLine():
            return
        command = bytes(sock.readLine()).strip().decode("ascii", "replace")
        sock.write(b"ok\n")
        sock.flush()
        if command == "show":
            bring_to_front(self._window)
        elif command == "quit":
            QTimer.singleShot(0, self._window.quit_app)


def bring_to_front(window) -> None:
    window.restore_from_tray()
    if sys.platform == "win32":
        try:
            ctypes.windll.user32.SetForegroundWindow(int(window.winId()))
        except Exception:
            pass


def _add_settings_rows(window) -> None:
    sv = window.settingsView
    sv._profile_file = str(appdata.profile_path())  # Save JSON Profile starts on the profile that loads at startup

    autostart = SwitchButton()
    autostart.setOnText("On")
    autostart.setOffText("Off")
    set_switch(autostart, wi.autostart_enabled())

    def toggled(checked: bool) -> None:
        if not wi.set_autostart(checked):
            set_switch(autostart, wi.autostart_enabled())
            window.show_notification({"level": "error", "title": "Couldn't change Start with Windows", "text": ""})

    autostart.checkedChanged.connect(toggled)
    sv.appCard.add_row(
        FluentIcon.POWER_BUTTON,
        "Start with Windows",
        "Starts V-COMMS in the tray when you sign in, so your hotkeys are ready before you open VALORANT.",
        autostart,
    )
    sv.autostartSwitch = autostart

    folder = PushButton(FluentIcon.FOLDER, "Open data folder")
    folder.clicked.connect(lambda: appdata.open_folder(appdata.roaming_dir()))
    sv.appCard.add_row(
        FluentIcon.FOLDER,
        "Your data",
        f"Settings, profile and custom presets are in {appdata.roaming_dir()}. The log is in {appdata.local_dir(create=False)}.",
        folder,
    )

    setup = PushButton(FluentIcon.HELP, "Run setup again")
    setup.clicked.connect(lambda: show_first_run(window))
    sv.appCard.add_row(
        FluentIcon.HELP,
        "First-run setup",
        "Microphone, privacy, VB-CABLE, the audio path test and VALORANT's input device.",
        setup,
    )

    sv.aboutCard.body.addWidget(muted(CaptionLabel(f"Version {APP_VERSION}")))


def show_first_run(window) -> None:
    from firstrun import FirstRunDialog

    if getattr(window, "_first_run_dialog", None) is not None:
        return
    bring_to_front(window)
    dialog = FirstRunDialog(window)
    window._first_run_dialog = dialog

    def finished(code: int) -> None:
        if code:  # Done
            window.settings.setValue(SETUP_DONE_KEY, True)
        window._first_run_dialog = None
        dialog.deleteLater()

    dialog.finished.connect(finished)
    dialog.open()


class _FirstRunTrigger(QObject):
    """Opens the first-run setup once both the engine side is ready (so the device boxes are filled) and the window is
    on screen. A copy started at sign-in stays in the tray, so the setup waits until the player opens the window."""

    def __init__(self, window) -> None:
        super().__init__(window)
        self._window = window
        self._ready = False
        self._shown = False
        self._done = False
        window.bridge.ready.connect(self._on_ready)
        window.installEventFilter(self)

    def _on_ready(self) -> None:
        self._ready = True
        self._maybe_open()

    def eventFilter(self, obj, event) -> bool:
        if obj is self._window and event.type() == QEvent.Type.Show:
            self._shown = True
            self._maybe_open()
        return False

    def _maybe_open(self) -> None:
        if self._done or not (self._ready and self._shown):
            return
        self._done = True
        self._window.removeEventFilter(self)
        QTimer.singleShot(400, lambda: show_first_run(self._window))


def attach(app, window, background: bool = False) -> None:
    """Called once by build_gui right after the window is made (before the event loop starts)."""
    window._first_run_dialog = None
    window.instanceServer = InstanceServer(window)
    _add_settings_rows(window)
    if not window.settings.value(SETUP_DONE_KEY, False, type=bool):
        window.firstRunTrigger = _FirstRunTrigger(window)
