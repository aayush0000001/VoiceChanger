# Building the V-COMMS installer

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build.ps1
```

This:

1. creates `.venv-build` (separate from your dev `.venv`) with x64 Python and installs `requirements.txt` plus
   PyInstaller,
2. builds the app folder `dist\V-COMMS\` (`V-COMMS.exe` + `_internal\`) from `vcomms.spec`; the entry point is
   `vcomms_main.py` and the version comes from `APP_VERSION` in `appinfo.py`,
3. downloads VB-Audio's VB-CABLE driver pack into `build\vbcable`, checks it against the pinned SHA-256 and checks that
   its setup programs are signed by VB-Audio,
4. builds `dist\installer\V-COMMS-Setup-<version>.exe` with Inno Setup 6 (`winget install JRSoftware.InnoSetup`).

To release a new version, change `APP_VERSION` in `appinfo.py`. When VB-Audio publishes a new driver pack, update
`$VBCableUrl` and `$VBCableSha256` in `build.ps1`.

## What the installer does

- Installs per user into `%LOCALAPPDATA%\Programs\V-COMMS`, so neither the app nor its upgrades need admin.
- Installs VB-CABLE silently only when CABLE Input or CABLE Output is missing. That is the only UAC prompt. It asks
  for a restart only if the devices don't appear without one.
- Closes a running V-COMMS before an upgrade (`V-COMMS.exe --quit`), including an old portable v1.0.2, after asking.
- Start Menu shortcut, optional desktop shortcut, optional Start with Windows (off by default).
- Uninstall keeps the player's data in `%APPDATA%\V-COMMS` and VB-CABLE unless they tick a box. A silent uninstall
  keeps the data unless `/REMOVEUSERDATA` is passed.

VB-CABLE is bundled as VB-Audio's licensing page allows for donationware use: the installer names VB-Audio,
links to the donation page and shows the credit. The Windows build ships the GPLv3 text (`LICENSE-GPL-3.0.txt`)
because it bundles PyQt6 and PyQt6-Fluent-Widgets.

The installer and exe are not code-signed yet, so SmartScreen warns on first download.
