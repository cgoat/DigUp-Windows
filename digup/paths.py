"""Where DigUp keeps things on Windows: everything under %LOCALAPPDATA%\\DigUp (never roams, never syncs)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def home() -> Path:
    """%LOCALAPPDATA%\\DigUp, or $DIGUP_HOME (tests and portable installs)."""
    chosen = os.environ.get("DIGUP_HOME")
    if chosen:
        return Path(chosen)
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "DigUp"


def models() -> Path:
    """The EmbeddingGemma 2 GGUF files ($DIGUP_MODELS overrides)."""
    chosen = os.environ.get("DIGUP_MODELS")
    return Path(chosen) if chosen else home() / "Models"


def runtime() -> Path:
    """llama.cpp's Windows DLLs, one folder per build tag and backend."""
    return home() / "Runtime"


def index() -> Path:
    return home() / "Index"


def logs() -> Path:
    return home() / "Logs"


def settings_file() -> Path:
    return home() / "settings.json"


def scratch() -> Path:
    """A folder per process for decoded frames, audio windows and resized images (cleared when indexing ends)."""
    return Path(tempfile.gettempdir()) / f"DigUp-{os.getpid()}"


def normalize(path: str | os.PathLike) -> str:
    """The one spelling of a path the index uses: absolute, backslashes, the drive letter upper-case.

    Windows paths are case-insensitive but case-preserving; the index keeps them as the file system reports them and
    compares folders case-insensitively (`is_inside`).
    """
    text = os.path.abspath(os.fspath(path))
    if len(text) >= 2 and text[1] == ":":
        text = text[0].upper() + text[1:]
    return text.rstrip("\\") if len(text) > 3 else text


def is_inside(path: str, folder: str) -> bool:
    """`path` is `folder` or somewhere under it (case-insensitive, as NTFS is)."""
    p, f = normalize(path).lower(), normalize(folder).lower()
    return p == f or p.startswith(f.rstrip("\\") + "\\")


def prefix(folder: str) -> str:
    """`folder` with one trailing separator: the start of every path under it."""
    folder = normalize(folder)
    return folder if folder.endswith("\\") else folder + "\\"
