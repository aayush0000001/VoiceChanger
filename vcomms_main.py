"""
Launcher for the installed app (the exe's entry point; also works from source: python vcomms_main.py).

Before anything heavy loads it makes sure only one copy runs per Windows user: a second launch brings the running
window forward and exits. Then it moves v1.0.2's data into %APPDATA%\\V-COMMS (once) and starts the GUI.

  V-COMMS.exe                 open the window
  V-COMMS.exe --background    start in the tray (Start with Windows uses this)
  V-COMMS.exe --quit          ask the running copy to exit, and wait for it (the installer uses this)
  V-COMMS.exe --self-test     DSP verification and benchmark, no GUI
"""

from __future__ import annotations

import sys
import time

import winintegration as wi

QUIT_ARG = "--quit"


def _send(command: str, timeout_ms: int = 1500) -> bool:
    """Sends 'show' or 'quit' to the running copy. True when it answered. Loads only QtCore and QtNetwork."""
    from PyQt6.QtNetwork import QLocalSocket

    sock = QLocalSocket()
    sock.connectToServer(wi.server_name())
    if not sock.waitForConnected(timeout_ms):
        return False
    sock.write(command.encode("ascii") + b"\n")
    sock.waitForBytesWritten(timeout_ms)
    answered = sock.waitForReadyRead(timeout_ms) and bytes(sock.readAll()).startswith(b"ok")
    sock.disconnectFromServer()
    return answered


def _quit_running(timeout_s: float = 15.0) -> int:
    if not wi.instance_running():
        return 0
    _send("quit")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not wi.instance_running():
            return 0
        time.sleep(0.2)
    return 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if QUIT_ARG in argv:
        return _quit_running()
    if "--self-test" in argv:
        import val_voice_changer

        return val_voice_changer.main(argv)

    background = wi.BACKGROUND_ARG in argv
    if not wi.claim_instance():
        if background:  # started at sign-in while a copy already runs: nothing to do
            return 0
        wi.allow_other_to_take_focus()
        deadline = time.monotonic() + 30.0  # the first copy may still be starting up
        while time.monotonic() < deadline:
            if _send("show"):
                return 0
            time.sleep(0.5)
        return 0

    import appdata
    import applog

    try:
        report = appdata.migrate_legacy()
        if not report.get("skipped"):
            applog.setup_logging()
            applog.get_logger().info("data folder check: %s", {k: v for k, v in report.items() if v})
    except Exception:  # migration must never keep the app from starting
        applog.setup_logging()
        applog.get_logger().exception("moving older V-COMMS data failed")
    appdata.use_documents_as_working_dir()

    import val_voice_changer

    return val_voice_changer.main(argv)


if __name__ == "__main__":
    sys.exit(main())
