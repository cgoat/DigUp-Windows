; Inno Setup script: a per-user install (no admin rights), a Start menu entry, and an optional start at sign-in.
#define Version "0.1.0"

[Setup]
AppId={{6C2E8C4B-5F0E-4C0B-9B53-2D1E6B9D1A70}
AppName=DigUp
AppVersion={#Version}
AppPublisher=DigUp for Windows contributors
DefaultDirName={localappdata}\Programs\DigUp
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=DigUp-Setup
SetupIconFile=..\digup\app\resources\digup.ico
UninstallDisplayIcon={app}\DigUp.exe
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "..\dist\DigUp\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{userprograms}\DigUp"; Filename: "{app}\DigUp.exe"

[Tasks]
Name: "startup"; Description: "Start DigUp when I sign in"; GroupDescription: "Options:"

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "DigUp"; \
  ValueData: """{app}\DigUp.exe"" --background"; Flags: uninsdeletevalue; Tasks: startup

[Run]
Filename: "{app}\DigUp.exe"; Description: "Open DigUp"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; The index, the model and llama.cpp live in %LOCALAPPDATA%\DigUp; they're left for a reinstall unless removed by hand.
Type: filesandordirs; Name: "{app}"
