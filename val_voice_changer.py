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

from engine import *  # noqa: F401,F403  (keeps `import val_voice_changer` exposing the engine API)
from engine import configure_windows_process_isolation, run_production_self_test


def build_gui():
    """Creates the QApplication and the main window (not shown yet). Returns ``(app, window)``."""
    if sys.platform == "win32":
        try:  # own taskbar identity (and icon) instead of python.exe's
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("V-COMMS.VoiceEngine")
        except Exception:
            pass

    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication

    # crisp fractional scaling (125%, 150%) instead of rounding up to the next whole step
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    with contextlib.redirect_stdout(io.StringIO()):  # qfluentwidgets prints a promo banner when imported
        from qfluentwidgets import setFontFamilies

        from main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("V-COMMS")
    app.setOrganizationName("V-COMMS")
    app.setQuitOnLastWindowClosed(False)  # closing the window hides it to the tray; Exit lives in the tray menu

    # Windows 11's own UI font first, then the classic one
    setFontFamilies(["Segoe UI Variable Text", "Segoe UI", "Microsoft YaHei", "PingFang SC"])
    from PyQt6.QtGui import QFont

    base_font = QFont()
    base_font.setFamilies(["Segoe UI Variable Text", "Segoe UI"])
    base_font.setPixelSize(14)
    app.setFont(base_font)

    window = MainWindow()
    app.setWindowIcon(window.windowIcon())
    app.commitDataRequest.connect(window.on_session_end, Qt.ConnectionType.DirectConnection)  # logoff / shutdown
    app.aboutToQuit.connect(window.shutdown_engine)
    return app, window


def run_gui() -> int:
    app, window = build_gui()
    window.show()
    return app.exec()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="V-COMMS Real-Time Tactical Voice Engine")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run automated DSP verification, memory audit, and latency benchmark.",
    )
    args = parser.parse_args(argv)

    configure_windows_process_isolation()

    if args.self_test:
        return run_production_self_test()
    return run_gui()


if __name__ == "__main__":
    sys.exit(main())
