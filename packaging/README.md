# Building the V-COMMS installer

This folder turns `val_voice_changer.py` into a normal Windows app, so people
can install it without Python or `pip install -r requirements.txt`.

## One-step build

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build.ps1 -Version 1.0.0
```

This:

1. creates `.venv-build` (separate from your dev `.venv`) with x64 Python and
   installs `requirements.txt` plus PyInstaller,
2. builds the app folder `dist\V-COMMS\` (`V-COMMS.exe` + `_internal\`),
3. builds `dist\installer\V-COMMS-Setup-<version>.exe` with Inno Setup 6
   (`winget install JRSoftware.InnoSetup` if you don't have it).

Use `-SkipInstaller` to stop after step 2, or `-Python <path>` to pick the
interpreter used for the build venv.

## What the installer does

- Installs per user to `%LOCALAPPDATA%\Programs\V-COMMS`. No admin prompt, and
  the folder stays writable for `tactical_voice_profile.json`.
- Adds Start Menu shortcuts (app, VB-CABLE download page, uninstall) and an
  optional desktop shortcut.
- Shows up in Settings > Apps for uninstalling. Saved profiles are kept.
- Offers to open the official VB-CABLE download page on the last screen.

## VB-CABLE

V-COMMS needs the VB-CABLE driver, which is not bundled. VB-Audio does allow
bundling VB-CABLE (not the A+B / C+D variants) in other installers, as long as
it is credited as donationware from vb-cable.com
(<https://vb-audio.com/Services/licensing.htm>). It is left out because the
driver install needs admin rights and a reboot, which doesn't fit a per-user
installer.

## Qt binding

`vcomms.spec` reads `val_voice_changer.py` to see whether it imports PySide6
or PyQt6 (plus `qfluentwidgets`) and bundles only that binding, so the same
build works before and after the Fluent redesign.

## Notes

- The build is x64. It also runs on Windows on Arm through emulation.
- The exe is windowed, so `--self-test` output isn't visible from it. Run the
  self-test from source instead.
- The exe isn't code-signed, so SmartScreen will show "Windows protected your
  PC" on first run (More info > Run anyway) until it is signed.
