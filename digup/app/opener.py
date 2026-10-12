"""Opening a result where it matched: a video or recording at its moment, a PDF at its page, code at its line, and
showing a file in Explorer or copying it to paste anywhere.

Windows has no one way to open a file "at" a place, so each kind looks for an app that takes one: VLC, mpv, MPC-HC
or PotPlayer for a moment; Edge or SumatraPDF for a page; VS Code, Cursor, Zed, Sublime Text, Notepad++ or a JetBrains
IDE for a line. Without one, the file opens in its default app.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import winreg
from pathlib import Path

from ..model import FileKind
from ..searcher import SearchHit

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
DETACHED = getattr(subprocess, "DETACHED_PROCESS", 0)


def _launch(args: list[str]) -> bool:
    try:
        subprocess.Popen(args, close_fds=True, creationflags=DETACHED | NO_WINDOW)
        return True
    except OSError:
        return False


def _program_files(*parts: str) -> str | None:
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"), os.environ.get("LOCALAPPDATA")):
        if base:
            path = Path(base, *parts)
            if path.exists():
                return str(path)
    return None


def _app_path(exe: str) -> str | None:
    """An app registered under App Paths (how Windows itself finds msedge.exe, vlc.exe…)."""
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(hive, rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{exe}") as key:
                value, _ = winreg.QueryValueEx(key, "")
                if value and os.path.exists(value.strip('"')):
                    return value.strip('"')
        except OSError:
            continue
    return None


def find(exe: str, *program_files_paths: tuple[str, ...]) -> str | None:
    found = shutil.which(exe) or _app_path(exe)
    if found:
        return found
    for parts in program_files_paths:
        hit = _program_files(*parts)
        if hit:
            return hit
    return None


# MARK: Media at a moment

PLAYERS = {
    "vlc": lambda: find("vlc.exe", ("VideoLAN", "VLC", "vlc.exe")),
    "mpv": lambda: find("mpv.exe", ("mpv", "mpv.exe")),
    "mpc-hc": lambda: find("mpc-hc64.exe", ("MPC-HC", "mpc-hc64.exe"), ("K-Lite Codec Pack", "MPC-HC64", "mpc-hc64.exe")),
    "potplayer": lambda: find("PotPlayerMini64.exe", ("DAUM", "PotPlayer", "PotPlayerMini64.exe")),
}


def player(choice: str = "auto") -> tuple[str, str] | None:
    names = [choice] if choice in PLAYERS else list(PLAYERS)
    for name in names:
        path = PLAYERS[name]()
        if path:
            return name, path
    return None


def open_media_at(path: str, seconds: float, choice: str = "auto") -> str:
    """Opens audio or video at `seconds`. Returns how (for the status line)."""
    found = player(choice) if choice != "default" else None
    start = max(0.0, seconds)
    if found:
        name, exe = found
        args = {
            "vlc": [exe, f"--start-time={start:.1f}", path],
            "mpv": [exe, f"--start={start:.1f}", path],
            "mpc-hc": [exe, path, "/start", str(int(start * 1000))],
            "potplayer": [exe, path, f"/seek={int(start)}"],
        }[name]
        if _launch(args):
            return f"opened in {name} at {format_time(start)}"
    os.startfile(path)
    return "opened (install VLC or mpv to jump to the moment)"


# MARK: PDFs at a page

def open_pdf_at(path: str, page: int, words: list[str] | None = None, choice: str = "auto") -> str:
    if choice in ("auto", "sumatra"):
        sumatra = find("SumatraPDF.exe", ("SumatraPDF", "SumatraPDF.exe"))
        if sumatra:
            args = [sumatra, "-reuse-instance", "-page", str(page)]
            if words:
                args += ["-search", " ".join(words[:3])]
            if _launch(args + [path]):
                return f"opened at page {page}"
    if choice in ("auto", "edge"):
        edge = find("msedge.exe", ("Microsoft", "Edge", "Application", "msedge.exe"))
        if edge:
            url = Path(path).as_uri() + f"#page={page}"
            if _launch([edge, url]):
                return f"opened in Edge at page {page}"
    os.startfile(path)
    return "opened"


# MARK: Code at a line

EDITORS = {
    "vscode": lambda: find("code.cmd", ("Programs", "Microsoft VS Code", "bin", "code.cmd"),
                           ("Microsoft VS Code", "bin", "code.cmd")),
    "cursor": lambda: find("cursor.cmd", ("Programs", "cursor", "resources", "app", "bin", "cursor.cmd")),
    "zed": lambda: find("zed.exe", ("Programs", "Zed", "zed.exe")),
    "sublime": lambda: find("subl.exe", ("Sublime Text", "subl.exe"), ("Sublime Text 3", "subl.exe")),
    "notepad++": lambda: find("notepad++.exe", ("Notepad++", "notepad++.exe")),
    "jetbrains": lambda: find("idea64.exe") or find("pycharm64.exe") or find("rider64.exe"),
}
EDITOR_NAMES = {"vscode": "VS Code", "cursor": "Cursor", "zed": "Zed", "sublime": "Sublime Text",
                "notepad++": "Notepad++", "jetbrains": "JetBrains IDE", "notepad": "Notepad"}


def editor(choice: str = "auto") -> tuple[str, str] | None:
    names = [choice] if choice in EDITORS else list(EDITORS)
    for name in names:
        path = EDITORS[name]()
        if path:
            return name, path
    return None


def open_code_at(path: str, line: int, choice: str = "auto") -> str:
    found = editor(choice) if choice != "notepad" else None
    if found:
        name, exe = found
        args = {
            "vscode": [exe, "--goto", f"{path}:{line}"],
            "cursor": [exe, "--goto", f"{path}:{line}"],
            "zed": [exe, f"{path}:{line}"],
            "sublime": [exe, f"{path}:{line}"],
            "notepad++": [exe, f"-n{line}", path],
            "jetbrains": [exe, "--line", str(line), path],
        }[name]
        if _launch(args):
            return f"opened in {EDITOR_NAMES[name]} at line {line}"
    _launch(["notepad.exe", path])
    return "opened in Notepad"


# MARK: The result


def format_time(seconds: float) -> str:
    seconds = int(seconds)
    h, m, s = seconds // 3600, seconds % 3600 // 60, seconds % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def open_hit(hit: SearchHit, settings, words: list[str] | None = None) -> str:
    if not os.path.exists(hit.path):
        return "that file isn't there any more"
    if hit.kind in (FileKind.AUDIO, FileKind.VIDEO) and hit.loc:
        return open_media_at(hit.path, hit.loc, settings.media_player)
    if hit.kind == FileKind.PDF and hit.loc and hit.loc > 1:
        return open_pdf_at(hit.path, int(hit.loc), words, settings.pdf_viewer)
    if hit.kind == FileKind.CODE:
        return open_code_at(hit.path, int(hit.focus or hit.loc or 1), settings.editor)
    os.startfile(hit.path)
    return "opened"


def reveal(path: str) -> None:
    """Shows the file selected in an Explorer window."""
    subprocess.Popen(["explorer.exe", "/select,", os.path.normpath(path)])


def copy_file(path: str) -> None:
    """Puts the file on the clipboard, to paste in Explorer, mail, chat (CF_HDROP, as Explorer's Copy does)."""
    from PySide6.QtCore import QMimeData, QUrl
    from PySide6.QtGui import QGuiApplication

    data = QMimeData()
    data.setUrls([QUrl.fromLocalFile(path)])
    data.setText(path)
    QGuiApplication.clipboard().setMimeData(data)
