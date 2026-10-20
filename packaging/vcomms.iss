; Inno Setup script for V-COMMS.
; Build the app first (packaging\build.ps1 does every step), then:
;   iscc /DAppVersion=1.1.0 /DVBCableDir=<unzipped VBCABLE_Driver_Pack> packaging\vcomms.iss
; Output: dist\installer\V-COMMS-Setup-<version>.exe
;
; What a player gets:
;   - a per-user install in %LOCALAPPDATA%\Programs\V-COMMS: no admin for the app or its upgrades
;   - VB-CABLE installed only when CABLE Input / CABLE Output are missing; that is the one UAC prompt
;   - Start Menu shortcut, optional desktop shortcut, optional Start with Windows (off), launch at the end
;   - an upgrade closes the running copy first and keeps every setting (they live in AppData, not here)
;   - uninstall asks whether to delete the player's data (kept by default) and VB-CABLE (kept by default)

#ifndef AppVersion
  #define AppVersion "1.1.0"
#endif
#ifndef OutputSuffix
  #define OutputSuffix ""
#endif
#ifndef VBCableDir
  #error Pass /DVBCableDir=<folder with the unzipped VBCABLE_Driver_Pack> (build.ps1 does)
#endif

#define AppName "V-COMMS"
#define AppExe "V-COMMS.exe"
#define AppMutexName "V-COMMS-AppMutex"
#define RunKey "Software\Microsoft\Windows\CurrentVersion\Run"
#define VBCableUrl "https://vb-cable.com"
#define VBDonateUrl "https://shop.vb-audio.com/en/win-apps/11-vb-cable.html"

[Setup]
AppId={{6B0C2F6E-3E8B-4C4D-9A57-0E3E5D2A8F11}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=Aayush
AppPublisherURL=https://github.com/aayush0000001/VoiceChanger
AppSupportURL=https://github.com/aayush0000001/VoiceChanger/issues
AppUpdatesURL=https://github.com/aayush0000001/VoiceChanger/releases
AppComments=Real-time voice changer for VALORANT voice chat
VersionInfoVersion={#AppVersion}
VersionInfoProductName={#AppName}
VersionInfoDescription={#AppName} Setup
; Per-user install: the app and its upgrades never ask for admin. Only the VB-CABLE driver step elevates.
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
DisableDirPage=auto
UsePreviousAppDir=yes
UsePreviousTasks=yes
; x64 build; also installs on Windows on Arm, which runs it under emulation.
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
OutputDir=..\dist\installer
OutputBaseFilename=V-COMMS-Setup-{#AppVersion}{#OutputSuffix}
SetupIconFile=..\app_icon.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
WizardSizePercent=110
InfoBeforeFile=installer_notice.txt
; the app is told to quit through --quit before files are replaced; Restart Manager is the fallback
CloseApplications=force
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "startup"; Description: "Start V-COMMS in the tray when I sign in to Windows"; GroupDescription: "Startup:"; Flags: unchecked

[Files]
Source: "..\dist\V-COMMS\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "LICENSE-GPL-3.0.txt"; DestDir: "{app}"; Flags: ignoreversion
; VB-CABLE by VB-Audio (donationware, bundled as their licensing page allows); unpacked only when it is needed
Source: "{#VBCableDir}\*"; DestDir: "{tmp}\vbcable"; Flags: ignoreversion recursesubdirs; Check: CableWanted

[InstallDelete]
; files from an older build that this version no longer ships
Type: filesandordirs; Name: "{app}\_internal"

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"; Comment: "Voice changer for VALORANT"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "{#RunKey}"; ValueType: string; ValueName: "{#AppName}"; ValueData: """{app}\{#AppExe}"" --background"; Tasks: startup

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: postinstall nowait skipifsilent

[Messages]
FinishedLabel=Setup has finished installing [name].%n%nIn VALORANT, open Settings > Audio > Voice Chat and set Input Device to "CABLE Output (VB-Audio Virtual Cable)". V-COMMS walks you through the rest the first time it starts.

[Code]
const
  MMDevices = 'SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio';
  PKeyName = '{a45c254e-df1c-4efd-8020-67d146a850e0},2';
  PKeyIface = '{b3f8fa53-0004-438e-9003-51a46e139bfc},6';
  VBUninstallKey = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\VB:VBCABLE {87459874-1236-4469}';

var
  CablePage: TWizardPage;
  CableCheck: TNewCheckBox;
  CableMissingAtStart: Boolean;
  CableNeedsRestart: Boolean;
  CableResultText: String;
  RemoveDataChoice, RemoveCableChoice: Boolean;

{ ------------------------------------------------------------------ VB-CABLE detection }

{ 'ok' when an active endpoint called Wanted from VB-Audio exists in Flow (Render/Capture), 'off' when one exists
  but is disabled or unplugged, '' when there is none. Read from Windows' own device list; no admin needed. }
function EndpointState(Flow, Wanted: String): String;
var
  Keys: TArrayOfString;
  I: Integer;
  Name, Iface, Base: String;
  State: Cardinal;
begin
  Result := '';
  Base := MMDevices + '\' + Flow;
  if not RegGetSubkeyNames(HKLM64, Base, Keys) then
    Exit;
  for I := 0 to GetArrayLength(Keys) - 1 do
  begin
    if RegQueryStringValue(HKLM64, Base + '\' + Keys[I] + '\Properties', PKeyName, Name) and (Name = Wanted) and
       RegQueryStringValue(HKLM64, Base + '\' + Keys[I] + '\Properties', PKeyIface, Iface) and (Pos('VB-Audio', Iface) > 0) then
    begin
      if RegQueryDWordValue(HKLM64, Base + '\' + Keys[I], 'DeviceState', State) and (State = 1) then
      begin
        Result := 'ok';
        Exit;
      end;
      Result := 'off';
    end;
  end;
end;

function CablePresent: Boolean;
begin
  Result := (EndpointState('Render', 'CABLE Input') <> '') and (EndpointState('Capture', 'CABLE Output') <> '');
end;

function CableActive: Boolean;
begin
  Result := (EndpointState('Render', 'CABLE Input') = 'ok') and (EndpointState('Capture', 'CABLE Output') = 'ok');
end;

function CableWanted: Boolean;
begin
  Result := CableMissingAtStart and (CableCheck <> nil) and CableCheck.Checked;
end;

{ ------------------------------------------------------------------ an older V-COMMS that is still running }

procedure RememberLegacyDir(Dir: String);
var
  F: String;
begin
  F := ExpandConstant('{localappdata}\V-COMMS\legacy_dirs.txt');
  ForceDirectories(ExtractFileDir(F));
  SaveStringToFile(F, Dir + #13#10, True);
end;

{ v1.0.2 was a portable exe with no way to ask it to quit. Find copies running from anywhere but the install folder, note their
  folder (so the new version can copy the presets saved next to them), and close them once the player agrees. }
function CloseOldPortableCopies(): Boolean;
var
  Locator, Service, Items, Item: Variant;
  I: Integer;
  Path, Found: String;
begin
  Result := True;
  try
    Locator := CreateOleObject('WbemScripting.SWbemLocator');
    Service := Locator.ConnectServer('.', 'root\CIMV2');
    Items := Service.ExecQuery('SELECT ProcessId, ExecutablePath FROM Win32_Process WHERE Name = ''V-COMMS.exe''');
  except
    Exit;
  end;
  Found := '';
  for I := 0 to Items.Count - 1 do
  begin
    Item := Items.ItemIndex(I);
    Path := '';
    try
      Path := Item.ExecutablePath;
    except
    end;
    if (Path <> '') and (Pos(Lowercase(ExpandConstant('{app}')), Lowercase(Path)) <> 1) and
       (Pos('\_mei', Lowercase(Path)) = 0) then
    begin
      RememberLegacyDir(ExtractFileDir(Path));
      Found := Found + #13#10 + Path;
    end;
  end;
  if Found = '' then
    Exit;
  if WizardSilent or (MsgBox('An older V-COMMS is running:' + Found + #13#10#13#10 +
      'Setup will close it so the new version can use your microphone and hotkeys. ' +
      'Your custom presets are copied into the new version the first time it starts.',
      mbConfirmation, MB_OKCANCEL) = IDOK) then
  begin
    for I := 0 to Items.Count - 1 do
    begin
      Item := Items.ItemIndex(I);
      try
        Path := Item.ExecutablePath;
        if (Pos(Lowercase(ExpandConstant('{app}')), Lowercase(Path)) <> 1) then
          Item.Terminate();
      except
      end;
    end;
    Sleep(1000);
  end
  else
    Result := False;
end;

procedure QuitInstalledCopy();
var
  Code: Integer;
begin
  if FileExists(ExpandConstant('{app}\{#AppExe}')) then
    Exec(ExpandConstant('{app}\{#AppExe}'), '--quit', '', SW_HIDE, ewWaitUntilTerminated, Code);
end;

{ ------------------------------------------------------------------ wizard }

procedure InitializeWizard;
var
  Info: TNewStaticText;
  Link: TNewStaticText;
begin
  CableMissingAtStart := not CablePresent;
#ifdef ForceCablePage
  CableMissingAtStart := True;  { screenshot builds only: shows the page on a PC that has VB-CABLE }
#endif
  if not CableMissingAtStart then
    Exit;
  CablePage := CreateCustomPage(wpSelectTasks, 'Virtual audio cable',
    'V-COMMS sends your changed voice to VALORANT through VB-CABLE.');
  Info := TNewStaticText.Create(CablePage);
  Info.Parent := CablePage.Surface;
  Info.WordWrap := True;
  Info.Width := CablePage.SurfaceWidth;
  Info.Caption :=
    'VB-CABLE is not installed on this PC. It is a free virtual audio cable made by VB-Audio Software ' +
    '(www.vb-cable.com), separate from V-COMMS.' + #13#10#13#10 +
    'VB-CABLE is donationware: if you find it useful, you can support VB-Audio with a donation.' + #13#10#13#10 +
    'Installing a driver needs administrator rights, so Windows will ask for permission once.';
  CableCheck := TNewCheckBox.Create(CablePage);
  CableCheck.Parent := CablePage.Surface;
  CableCheck.Top := Info.Top + Info.Height + ScaleY(16);
  CableCheck.Width := CablePage.SurfaceWidth;
  CableCheck.Caption := 'Install VB-CABLE by VB-Audio (recommended)';
  CableCheck.Checked := True;
  Link := TNewStaticText.Create(CablePage);
  Link.Parent := CablePage.Surface;
  Link.Top := CableCheck.Top + CableCheck.Height + ScaleY(12);
  Link.Caption := 'VB-CABLE licence and donations: {#VBDonateUrl}';
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
  QuitInstalledCopy();
  if not CloseOldPortableCopies() then
    Result := 'Close the older V-COMMS (right-click its tray icon > Exit), then run Setup again.';
end;

procedure InstallCable;
var
  Code, Waited: Integer;
  SetupExe: String;
begin
  SetupExe := ExpandConstant('{tmp}\vbcable\VBCABLE_Setup_x64.exe');
  WizardForm.StatusLabel.Caption := 'Installing VB-CABLE by VB-Audio (Windows asks for permission)...';
  if not ShellExec('runas', SetupExe, '-i -h', ExpandConstant('{tmp}\vbcable'), SW_HIDE, ewWaitUntilTerminated, Code) then
  begin
    CableResultText := 'VB-CABLE was not installed (' + SysErrorMessage(Code) + '). V-COMMS will show you how to get it.';
    Exit;
  end;
  { the endpoints usually appear within seconds; if they do not, the driver needs a restart }
  Waited := 0;
  while (not CableActive) and (Waited < 20000) do
  begin
    Sleep(500);
    Waited := Waited + 500;
  end;
  if CableActive then
    CableResultText := 'VB-CABLE by VB-Audio is installed.'
  else
  begin
    CableNeedsRestart := True;
    CableResultText := 'VB-CABLE by VB-Audio is installed. Restart your PC to finish setting it up.';
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and CableWanted then
    InstallCable;
end;

function NeedRestart(): Boolean;
begin
  Result := CableNeedsRestart;
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if (CurPageID = wpFinished) and (CableResultText <> '') then
    WizardForm.FinishedLabel.Caption := CableResultText + #13#10#13#10 + WizardForm.FinishedLabel.Caption;
end;

{ ------------------------------------------------------------------ uninstall }

function AskUninstallChoices(): Boolean;
var
  Form: TSetupForm;
  Text: TNewStaticText;
  DataBox, CableBox: TNewCheckBox;
  OkButton, CancelButton: TNewButton;
begin
  Form := CreateCustomForm(ScaleX(420), ScaleY(150), False, False);
  try
    Form.Caption := 'Uninstall V-COMMS';
    Text := TNewStaticText.Create(Form);
    Text.Parent := Form;
    Text.Left := ScaleX(16);
    Text.Top := ScaleY(16);
    Text.Width := Form.ClientWidth - ScaleX(32);
    Text.WordWrap := True;
    Text.Caption := 'V-COMMS will be removed. Leave both boxes empty to keep your settings and presets for a later reinstall.';
    DataBox := TNewCheckBox.Create(Form);
    DataBox.Parent := Form;
    DataBox.Left := ScaleX(16);
    DataBox.Top := Text.Top + Text.Height + ScaleY(14);
    DataBox.Width := Form.ClientWidth - ScaleX(32);
    DataBox.Caption := 'Also delete my settings, voice profile and custom presets';
    CableBox := TNewCheckBox.Create(Form);
    CableBox.Parent := Form;
    CableBox.Left := ScaleX(16);
    CableBox.Top := DataBox.Top + DataBox.Height + ScaleY(8);
    CableBox.Width := Form.ClientWidth - ScaleX(32);
    CableBox.Caption := 'Also remove VB-CABLE (other apps may use it; needs admin)';
    CableBox.Enabled := RegKeyExists(HKLM64, VBUninstallKey) or RegKeyExists(HKLM32, VBUninstallKey);
    OkButton := TNewButton.Create(Form);
    OkButton.Parent := Form;
    OkButton.Caption := 'Uninstall';
    OkButton.Width := ScaleX(90);
    OkButton.Height := ScaleY(26);
    OkButton.Left := Form.ClientWidth - ScaleX(16 + 90 + 8 + 90);
    OkButton.Top := Form.ClientHeight - ScaleY(16 + 26);
    OkButton.ModalResult := mrOk;
    OkButton.Default := True;
    CancelButton := TNewButton.Create(Form);
    CancelButton.Parent := Form;
    CancelButton.Caption := 'Cancel';
    CancelButton.Width := ScaleX(90);
    CancelButton.Height := ScaleY(26);
    CancelButton.Left := Form.ClientWidth - ScaleX(16 + 90);
    CancelButton.Top := OkButton.Top;
    CancelButton.ModalResult := mrCancel;
    CancelButton.Cancel := True;
    Result := Form.ShowModal() = mrOk;
    RemoveDataChoice := DataBox.Checked;
    RemoveCableChoice := CableBox.Checked and CableBox.Enabled;
  finally
    Form.Free();
  end;
end;

function InitializeUninstall(): Boolean;
var
  Code: Integer;
begin
  Result := True;
  if not UninstallSilent then
    Result := AskUninstallChoices()
  else
    RemoveDataChoice := Pos('/REMOVEUSERDATA', Uppercase(GetCmdTail)) > 0;  { silent: keep data unless asked }
  if Result and FileExists(ExpandConstant('{app}\{#AppExe}')) then
    Exec(ExpandConstant('{app}\{#AppExe}'), '--quit', '', SW_HIDE, ewWaitUntilTerminated, Code);
end;

procedure RemoveCable;
var
  Cmd: String;
  Code: Integer;
begin
  if not RegQueryStringValue(HKLM64, VBUninstallKey, 'UninstallString', Cmd) then
    if not RegQueryStringValue(HKLM32, VBUninstallKey, 'UninstallString', Cmd) then
      Exit;
  Cmd := RemoveQuotes(Cmd);
  if FileExists(Cmd) then
    ShellExec('runas', Cmd, '-u -h', ExtractFileDir(Cmd), SW_HIDE, ewWaitUntilTerminated, Code);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then
    RegDeleteValue(HKCU, '{#RunKey}', '{#AppName}');  { also when Settings turned it on }
  if CurUninstallStep = usPostUninstall then
  begin
    if RemoveDataChoice then
    begin
      DelTree(ExpandConstant('{userappdata}\V-COMMS'), True, True, True);
      DelTree(ExpandConstant('{localappdata}\V-COMMS'), True, True, True);
      RegDeleteKeyIncludingSubkeys(HKCU, 'Software\V-COMMS');  { v1.0.2 kept its window settings here }
    end;
    if RemoveCableChoice then
      RemoveCable;
  end;
end;
