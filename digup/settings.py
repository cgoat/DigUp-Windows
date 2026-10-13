"""What the person chose: folders, code folders, the hotkey, the editor. One JSON file the app and the CLI share."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import paths
from .model import IndexOptions


@dataclass
class Settings:
    folders: list[str] = field(default_factory=list)
    code_folders: list[str] = field(default_factory=list)
    code_search: bool = False
    excluded_folders: list[str] = field(default_factory=list)
    excluded_extensions: str = ""
    hotkey: str = "Ctrl+Shift+Space"
    editor: str = "auto"  # auto | vscode | cursor | zed | sublime | notepad++ | jetbrains | notepad
    media_player: str = "auto"  # auto | vlc | mpv | mpc-hc | potplayer | default
    pdf_viewer: str = "auto"  # auto | edge | sumatra | default
    index_on_battery: bool = True
    pause_on_battery_saver: bool = True
    paused: bool = False
    launch_at_login: bool = False
    backend: str = "auto"  # auto | cuda | vulkan | cpu
    onboarded: bool = False
    faces: bool = False  # find people in photos (InsightFace, downloaded only when asked for: Settings → People)

    @staticmethod
    def default_folders() -> list[str]:
        """The folders most people keep their things in (those that exist)."""
        home = Path.home()
        names = ["Documents", "Pictures", "Desktop", "Downloads", "Videos", "Music"]
        out = [str(home / n) for n in names if (home / n).is_dir()]
        onedrive = os.environ.get("OneDrive")
        if onedrive:
            out += [str(Path(onedrive) / n) for n in ("Documents", "Pictures", "Desktop") if (Path(onedrive) / n).is_dir()]
        return out

    def options(self, code: bool = False) -> IndexOptions:
        o = IndexOptions(code=code)
        o.excluded_folders = list(self.excluded_folders)
        o.excluded_extensions = IndexOptions.extensions(self.excluded_extensions)
        return o

    @property
    def backend_choice(self) -> str | None:
        return None if self.backend == "auto" else self.backend

    # MARK: Storage

    @classmethod
    def load(cls, file: Path | None = None) -> "Settings":
        file = file or paths.settings_file()
        try:
            data = json.loads(file.read_text("utf-8"))
        except (OSError, ValueError):
            return cls()
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self, file: Path | None = None) -> None:
        file = file or paths.settings_file()
        file.parent.mkdir(parents=True, exist_ok=True)
        tmp = file.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), "utf-8")
        os.replace(tmp, file)
