"""DigUp for Windows: lives in the notification area, opens the search panel on a hotkey, and keeps the index up to
date in the background."""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon

from .. import __version__
from ..settings import Settings
from ..store import CODE_FILE, MAIN_FILE
from . import theme
from .engine import Engine, Status
from .hotkey import Hotkey
from .onboarding import Onboarding
from .panel import SearchPanel
from .query import QueryService
from .settings_window import SettingsWindow

SERVER = f"DigUp-{os.environ.get('USERNAME', 'user')}"


def already_running() -> bool:
    """Another DigUp is running: ask it to open the search panel instead."""
    socket = QLocalSocket()
    socket.connectToServer(SERVER)
    if socket.waitForConnected(300):
        socket.write(b"show\n")
        socket.flush()
        socket.waitForBytesWritten(300)
        return True
    return False


class App:
    def __init__(self, qt: QApplication, background: bool):
        self.qt = qt
        self.settings = Settings.load()
        self.colors = theme.apply(qt)
        qt.setWindowIcon(theme.app_icon())
        self.service = QueryService()
        self.panel = SearchPanel(self.service, self.settings)
        self.panel.open_settings.connect(self.show_settings)
        self.engine = Engine(self.settings)
        self.engine.status.connect(self._engine_status)
        self.engine.changed.connect(self.service.index_changed)
        self.engine.message.connect(lambda text: self.tray.showMessage("DigUp", text, theme.app_icon(), 6000))
        self.settings_window: SettingsWindow | None = None
        self.onboarding: Onboarding | None = None
        self._tray()
        self.hotkey = Hotkey()
        self.hotkey.pressed.connect(self.panel.toggle)
        if not self.hotkey.register(self.settings.hotkey):
            self.tray.showMessage("DigUp", f"{self.settings.hotkey} is taken by another app. Pick another hotkey "
                                  "in Settings, or click the DigUp icon to search.", theme.app_icon(), 8000)
        self.server = QLocalServer()
        QLocalServer.removeServer(SERVER)
        self.server.listen(SERVER)
        self.server.newConnection.connect(self._remote)
        if not self.settings.onboarded:
            self.onboarding = Onboarding(self.settings)
            self.onboarding.done.connect(self._onboarded)
            self.onboarding.show()
        else:
            self.engine.start()
            if not background:
                QTimer.singleShot(300, self.panel.show_panel)

    # MARK: The tray

    def _tray(self) -> None:
        self.tray = QSystemTrayIcon(theme.tray_icon())
        self.tray.setToolTip("DigUp")
        menu = QMenu()
        search = QAction(f"Search…\t{self.settings.hotkey}", menu)
        search.triggered.connect(self.panel.show_panel)
        self.status_action = QAction("Starting…", menu)
        self.status_action.setEnabled(False)
        self.pause_action = QAction("Pause indexing", menu)
        self.pause_action.triggered.connect(self._toggle_pause)
        settings = QAction("Settings…", menu)
        settings.triggered.connect(self.show_settings)
        about = QAction(f"About DigUp {__version__}", menu)
        about.triggered.connect(self._about)
        quit_ = QAction("Quit DigUp", menu)
        quit_.triggered.connect(self.quit)
        for item in (search, None, self.status_action, self.pause_action, None, settings, about, None, quit_):
            menu.addSeparator() if item is None else menu.addAction(item)
        self.pause_action.setText("Resume indexing" if self.settings.paused else "Pause indexing")
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._tray_clicked)
        self.tray.show()
        self.menu = menu

    def _tray_clicked(self, reason) -> None:
        if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick):
            self.panel.toggle()

    def _engine_status(self, status: Status) -> None:
        if status.phase in ("indexing", "finishing") and status.total:
            what = "Reading the rest of long files" if status.phase == "finishing" else f"Indexing {status.note}"
            line = f"{what}: {status.done:,} of {status.total:,}"
            progress = status.done / status.total
        elif status.phase == "syncing":
            line, progress = "Looking for new and changed files…", 0.0
        elif status.phase == "paused":
            line, progress = f"Indexing paused ({status.paused_reason})", None
        elif status.phase == "waiting":
            line, progress = "Waiting for the model download…", None
        elif status.phase == "error":
            line, progress = f"Indexing stopped: {status.note}", None
        else:
            total = sum(c.get("vectors", 0) for c in status.counts.values()) if status.counts else 0
            files = 0
            for name, counts in (status.counts or {}).items():
                for states in counts.get("by_kind_and_state", {}).values():
                    files += states.get("done", 0) + states.get("partial", 0)
            line, progress = (f"Up to date: {files:,} files searchable" if files else "Up to date"), None
            _ = total
        self.status_action.setText(line)
        self.tray.setToolTip(f"DigUp — {line}")
        # Redrawing the icon renders an SVG at seven sizes: only when the arc would visibly move.
        icon_key = (None if progress is None else round(progress * 24), status.phase == "paused")
        if icon_key != getattr(self, "_icon_key", object()):
            self._icon_key = icon_key
            self.tray.setIcon(theme.tray_icon(progress, paused=status.phase == "paused"))
        self.panel.set_engine_line(line if status.phase != "idle" else "")

    def _toggle_pause(self) -> None:
        paused = not self.settings.paused
        self.engine.set_paused(paused)
        self.pause_action.setText("Resume indexing" if paused else "Pause indexing")

    # MARK: Windows

    def _onboarded(self) -> None:
        self.engine.settings = self.settings
        self.engine.start()
        self.tray.showMessage("DigUp is reading your folders",
                              f"Press {self.settings.hotkey} in any app to search. Searching works while it reads.",
                              theme.app_icon(), 8000)
        QTimer.singleShot(500, self.panel.show_panel)

    def show_settings(self) -> None:
        if self.settings_window is None:
            self.settings_window = SettingsWindow(self.settings)
            self.settings_window.set_engine(self.engine)
            self.settings_window.saved.connect(self._settings_saved)
            self.settings_window.destroyed.connect(lambda: setattr(self, "settings_window", None))
            self.settings_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.settings_window.tabs.setCurrentIndex(0)
        if self.settings_window.isMinimized():
            self.settings_window.showNormal()
        self.settings_window.show()
        self.settings_window.raise_()
        self.settings_window.activateWindow()

    def _settings_saved(self, folders_changed: bool, code_changed: bool) -> None:
        if self.hotkey.registered != self.settings.hotkey and not self.hotkey.register(self.settings.hotkey):
            QMessageBox.warning(None, "DigUp", f"{self.settings.hotkey} is taken by another app; pick another.")
        if folders_changed or code_changed:
            self.engine.resync(None if folders_changed and code_changed else code_changed)
            self.service.index_changed(MAIN_FILE)
            self.service.index_changed(CODE_FILE)

    def _about(self) -> None:
        QMessageBox.about(None, "DigUp", f"<b>DigUp {__version__} for Windows</b><br><br>Search the files on your PC "
                          "by what's in them, with Google DeepMind's EmbeddingGemma 2 on llama.cpp.<br><br>"
                          "A Windows port of <a href='https://github.com/ARahim3/DigUp'>ARahim3/DigUp</a>. "
                          "Nothing you index leaves this PC.")

    def _remote(self) -> None:
        socket = self.server.nextPendingConnection()
        if socket:
            socket.waitForReadyRead(200)
            socket.readAll()
            socket.disconnectFromServer()
            self.panel.show_panel()

    def quit(self) -> None:
        self.hotkey.unregister()
        self.panel.hide_panel()
        self.engine.stop()
        self.service.close()
        self.tray.hide()
        self.qt.quit()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure") and sys.stdout:
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    os.environ.setdefault("QT_LOGGING_RULES", "qt.multimedia*=false")
    os.environ.setdefault("QT_FFMPEG_DEBUG", "0")
    qt = QApplication(sys.argv)
    qt.setApplicationName("DigUp")
    qt.setQuitOnLastWindowClosed(False)
    if already_running():
        return 0
    try:  # our own taskbar identity, not python.exe's
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("DigUp.DigUp")
    except Exception:
        pass
    app = App(qt, background="--background" in sys.argv)
    qt.aboutToQuit.connect(lambda: app.engine.stop())
    return qt.exec()


if __name__ == "__main__":
    sys.exit(main())
