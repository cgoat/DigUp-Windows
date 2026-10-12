# PyInstaller spec: one folder with DigUp.exe (the tray app, no console) and digup-cli.exe (the CLI, which the app also
# runs as its embedding worker). llama.cpp's DLLs and the model are not bundled: the app downloads them on first run
# into %LOCALAPPDATA%\DigUp, the right llama.cpp build for the PC's GPU (CUDA, Vulkan or CPU).
# Build: scripts/build.ps1
import os

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = os.path.dirname(SPECPATH)
J = os.path.join
ICON = J(ROOT, "digup", "app", "resources", "digup.ico")

hidden = collect_submodules("winrt") + collect_submodules("digup") + ["pillow_heif", "pypdfium2"]
datas = (collect_data_files("imageio_ffmpeg") + collect_data_files("pypdfium2_raw") + collect_data_files("winrt")
         + [(J(ROOT, "digup", "app", "resources"), "digup/app/resources")])
binaries = collect_dynamic_libs("pypdfium2_raw") + collect_dynamic_libs("pillow_heif")
excludes = ["tkinter", "matplotlib", "pytest", "IPython", "PySide6.QtWebEngineCore", "PySide6.Qt3DCore",
            "PySide6.QtQuick", "PySide6.QtQml", "PySide6.QtPdf", "PySide6.QtCharts", "PySide6.QtDataVisualization"]

app = Analysis([J(ROOT, "packaging", "app_entry.py")], pathex=[ROOT], binaries=binaries, datas=datas,
               hiddenimports=hidden, excludes=excludes)
cli = Analysis([J(ROOT, "packaging", "cli_entry.py")], pathex=[ROOT], binaries=binaries, datas=datas,
               hiddenimports=[h for h in hidden if not h.startswith("digup.app")], excludes=excludes + ["PySide6"])

app_exe = EXE(PYZ(app.pure), app.scripts, [], exclude_binaries=True, name="DigUp", console=False, icon=ICON)
cli_exe = EXE(PYZ(cli.pure), cli.scripts, [], exclude_binaries=True, name="digup-cli", console=True, icon=ICON)
COLLECT(app_exe, app.binaries, app.datas, cli_exe, cli.binaries, cli.datas, name="DigUp")
