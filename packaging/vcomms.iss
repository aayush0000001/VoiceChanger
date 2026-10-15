; Inno Setup script for V-COMMS.
; Build the app first (packaging\build.ps1 does both steps), then:
;   iscc packaging\vcomms.iss
; Output: dist\installer\V-COMMS-Setup-<version>.exe

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif

#define AppName "V-COMMS"
#define AppExe "V-COMMS.exe"
#define VBCableUrl "https://vb-audio.com/Cable/"

[Setup]
AppId={{6B0C2F6E-3E8B-4C4D-9A57-0E3E5D2A8F11}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=Aayush
; Per-user install: no admin prompt, and the install folder stays writable,
; which the app needs because it saves tactical_voice_profile.json next to
; itself (working directory = install folder).
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; x64 build; also installs on Windows on Arm, which runs it under emulation.
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir=..\dist\installer
OutputBaseFilename=V-COMMS-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#AppExe}
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\V-COMMS\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"; WorkingDir: "{app}"
Name: "{group}\Get VB-CABLE (required audio driver)"; Filename: "{#VBCableUrl}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{#VBCableUrl}"; Description: "Open the VB-CABLE download page (free virtual audio driver V-COMMS needs, install once per PC)"; Flags: postinstall shellexec skipifsilent
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; WorkingDir: "{app}"; Flags: postinstall nowait skipifsilent unchecked

[UninstallDelete]
; Leave the user's saved profile alone; only clean up runtime caches.
Type: filesandordirs; Name: "{app}\__pycache__"

[Messages]
FinishedLabel=Setup has finished installing [name] on your computer.%n%nV-COMMS needs the free VB-CABLE virtual audio driver by VB-Audio. If you have not installed it yet, download it from the official page, run its setup as administrator, and restart your PC before launching V-COMMS.
