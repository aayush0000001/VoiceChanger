"""
V-COMMS // Real-Time Low-Latency Tactical Voice Engine for Windows 11

Launcher:  python val_voice_changer.py            (Fluent GUI)
           python val_voice_changer.py --self-test (DSP verification and latency benchmark, no GUI needed)

Layout:
  engine.py          audio engine, DSP, hotkeys, profiles, self-test. No Qt anywhere.
  engine_bridge.py   EngineBridge (QObject): the only door between the UI and the engine (signals and slots).
  main_window.py     FluentWindow shell: navigation, Mica, tray.
  views/             one module per page (Live, Voice, Devices, Settings) plus shared widgets.
  styles/            design tokens and the few QSS rules.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import io
import sys

import applog
from appinfo import APP_NAME, APP_VERSION
from engine import *  # noqa: F401,F403  (keeps `import val_voice_changer` exposing the engine API)
from engine import configure_windows_process_isolation, run_production_self_test


def build_gui(background: bool = False):
    """Creates the QApplication and the main window (not shown yet). Returns ``(app, window)``."""
    # When running as a standalone frozen executable (PyInstaller), Windows already associates
    # the process and taskbar icon directly with V-COMMS.exe. Calling SetCurrentProcessExplicitAppUserModelID
    # with an unregistered custom string on an uninstalled portable executable tells Windows to search
    # for a registered Shell Link, which causes the taskbar to show a blank/missing icon.
    # Only set it when running from python.exe.
    if sys.platform == "win32" and not getattr(sys, "frozen", False):
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("V-COMMS.VoiceEngine")
        except Exception:
            pass

    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication

    # crisp fractional scaling (125%, 150%) instead of rounding up to the next whole step
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    with contextlib.redirect_stdout(io.StringIO()):  # qfluentwidgets prints a promo banner when imported
        from qfluentwidgets import setFontFamilies

        from main_window import MainWindow, make_app_icon

    # A bug in a slot must not silently end the app (PyQt6 aborts on an unhandled exception): log it, and tell the player
    applog.setup_logging()
    applog.install_excepthooks()  # logging only for now; the window adds its toast once it exists

    app = QApplication(sys.argv)
    app.setApplicationName("V-COMMS")
    app.setOrganizationName("V-COMMS")
    app.setQuitOnLastWindowClosed(False)  # closing the window hides it to the tray; Exit lives in the tray menu

    # Set application icon immediately upon QApplication creation
    app_icon = make_app_icon()
    app.setWindowIcon(app_icon)

    # Windows 11's own UI font first, then the classic one
    setFontFamilies(["Segoe UI Variable Text", "Segoe UI", "Microsoft YaHei", "PingFang SC"])
    from PyQt6.QtGui import QFont

    base_font = QFont()
    base_font.setFamilies(["Segoe UI Variable Text", "Segoe UI"])
    base_font.setPixelSize(14)
    app.setFont(base_font)

    window = MainWindow()
    window.setWindowIcon(app_icon)
    applog.install_excepthooks(on_error=window.errorOccurred.emit)  # any thread; the window shows one calm message
    import winapp_ui

    winapp_ui.attach(app, window, background)  # single instance, Start with Windows, data folder, first-run setup
    applog.get_logger().info("%s %s started", APP_NAME, APP_VERSION)
    app.commitDataRequest.connect(window.on_session_end, Qt.ConnectionType.DirectConnection)  # logoff / shutdown
    app.aboutToQuit.connect(window.shutdown_engine)
    return app, window


def run_gui(background: bool = False) -> int:
    app, window = build_gui(background)
    if not background:  # started at sign-in: stay in the tray
        window.show()

    # Explicitly set the native Win32 window icon handles on the HWND to ensure
    # the taskbar button displays the icon under all Windows 11 shell states.
    if sys.platform == "win32":
        try:
            hwnd = int(window.winId())
            WM_SETICON = 0x0080
            ICON_SMALL = 0
            ICON_BIG = 1

            h_instance = ctypes.windll.kernel32.GetModuleHandleW(None)
            # PyInstaller embeds the primary icon as resource ID 1
            h_icon_big = ctypes.windll.user32.LoadImageW(
                h_instance, ctypes.c_wchar_p(1), 1, 0, 0, 0x0040 | 0x8000  # LR_DEFAULTSIZE | LR_SHARED
            )
            h_icon_small = ctypes.windll.user32.LoadImageW(
                h_instance, ctypes.c_wchar_p(1), 1, 16, 16, 0x8000  # LR_SHARED
            )

            if h_icon_big:
                ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, h_icon_big)
            if h_icon_small:
                ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, h_icon_small)
        except Exception:
            pass

    return app.exec()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="V-COMMS Real-Time Tactical Voice Engine")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run automated DSP verification, memory audit, and latency benchmark.",
    )
    parser.add_argument("--background", action="store_true", help="Start in the system tray.")
    args = parser.parse_args(argv)

    configure_windows_process_isolation()

    if args.self_test:
        return run_production_self_test()
    return run_gui(args.background)


if __name__ == "__main__":
    sys.exit(main())
