"""The app's indexing engine: one background thread keeps both indexes (files, and code when it's on) in step with the
chosen folders. It syncs when the app starts and whenever a watched folder changes, embeds what's pending through a
worker process (which quits a minute after the work runs out, giving its memory back), and pauses when asked, on
battery saver, or on battery when the person said so.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time
import traceback
from dataclasses import dataclass, field

from PySide6.QtCore import QObject, Signal

from .. import downloads, paths
from ..indexer import Indexer
from ..model import WorkerError
from ..settings import Settings
from ..store import CODE_FILE, MAIN_FILE, IndexStore

WORKER_IDLE_SECONDS = 60
WATCH_SETTLE_SECONDS = 2.0  # a burst of changes (a copy, an unzip) is synced once
CODE_SETTLE_SECONDS = 60.0  # code changes all the time while someone works: catch up once it goes quiet


class SYSTEM_POWER_STATUS(ctypes.Structure):
    _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]


def power() -> tuple[bool, bool]:
    """(on battery, battery saver on)."""
    status = SYSTEM_POWER_STATUS()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return False, False
    return status.ACLineStatus == 0, status.SystemStatusFlag == 1


@dataclass
class Status:
    phase: str = "idle"  # idle | syncing | indexing | finishing | paused | waiting | error
    done: int = 0
    total: int = 0
    current: str = ""
    note: str = ""
    pending: int = 0
    counts: dict = field(default_factory=dict)
    paused_reason: str = ""


class Engine(QObject):
    status = Signal(object)  # Status
    changed = Signal(str)  # an index file name: its searcher should reload
    message = Signal(str)

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self._wake = threading.Condition()
        self._dirty: dict[str, float] = {}  # index file -> when to sync (time)
        self._stop = False
        self._observer = None
        self._thread = threading.Thread(target=self._loop, name="digup-engine", daemon=True)
        self._status = Status()
        self._last_emit = 0.0
        self._caches: dict = {}
        self._interrupt = False

    # MARK: Control (any thread)

    def start(self) -> None:
        self._dirty = {MAIN_FILE: 0.0}
        if self.settings.code_search and self.settings.code_folders:
            self._dirty[CODE_FILE] = 0.0
        self._thread.start()
        self._watch()

    def stop(self) -> None:
        with self._wake:
            self._stop = True
            self._wake.notify_all()
        if self._observer:
            self._observer.stop()

    def resync(self, code: bool | None = None) -> None:
        """Folders or settings changed: sync again (both indexes unless `code` says which)."""
        with self._wake:
            if code in (None, False):
                self._dirty[MAIN_FILE] = 0.0
            if code in (None, True) and self.settings.code_search:
                self._dirty[CODE_FILE] = 0.0
            self._interrupt = True
            self._wake.notify_all()
        self._watch()

    def set_paused(self, paused: bool) -> None:
        self.settings.paused = paused
        self.settings.save()
        with self._wake:
            self._interrupt = paused
            self._wake.notify_all()

    # MARK: Watching

    def _watch(self) -> None:
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            return
        if self._observer:
            self._observer.stop()
            self._observer = None
        engine = self

        class Handler(FileSystemEventHandler):
            def __init__(self, index: str):
                self.index = index

            def on_any_event(self, event):
                path = getattr(event, "src_path", "") or ""
                if "\\." in path or path.endswith((".part", ".tmp", ".crdownload")):
                    return
                delay = CODE_SETTLE_SECONDS if self.index == CODE_FILE else WATCH_SETTLE_SECONDS
                with engine._wake:
                    engine._dirty[self.index] = time.time() + delay
                    engine._wake.notify_all()

        observer = Observer()
        watched = 0
        for folder in self.settings.folders:
            if os.path.isdir(folder):
                observer.schedule(Handler(MAIN_FILE), folder, recursive=True)
                watched += 1
        if self.settings.code_search:
            for folder in self.settings.code_folders:
                if os.path.isdir(folder):
                    observer.schedule(Handler(CODE_FILE), folder, recursive=True)
                    watched += 1
        if watched:
            observer.daemon = True
            observer.start()
            self._observer = observer

    # MARK: The loop (the engine thread)

    STATUS_INTERVAL = 0.25  # seconds between status updates while files fly by (the UI redraws on each)

    def _emit(self, **changes) -> None:
        phase_before = self._status.phase
        for key, value in changes.items():
            setattr(self._status, key, value)
        if self._stop:
            return
        # Hundreds of files a second (small pictures skipped) would bury the UI thread in updates: send at most
        # four a second, and every change of phase right away.
        now = time.time()
        if self._status.phase == phase_before and now - self._last_emit < self.STATUS_INTERVAL:
            return
        self._last_emit = now
        try:
            self.status.emit(Status(**self._status.__dict__))
        except RuntimeError:  # the app is shutting down and the Qt object is gone
            self._stop = True

    def _paused_reason(self) -> str:
        if self.settings.paused:
            return "paused"
        on_battery, saver = power()
        if saver and self.settings.pause_on_battery_saver:
            return "battery saver is on"
        if on_battery and not self.settings.index_on_battery:
            return "waiting for a charger"
        return ""

    def _should_stop(self) -> bool:
        with self._wake:
            return self._stop or self._interrupt or bool(self._paused_reason())

    def _loop(self) -> None:
        indexers: dict[str, Indexer] = {}
        last_work = time.time()
        while True:
            with self._wake:
                if self._stop:
                    break
                self._interrupt = False
                now = time.time()
                due = [name for name, when in self._dirty.items() if when <= now]
            reason = self._paused_reason()
            if reason:
                self._emit(phase="paused", paused_reason=reason)
                for ix in indexers.values():
                    ix.close_worker()
                with self._wake:
                    self._wake.wait(10)
                continue
            if not (downloads.models_complete() and downloads.runtime_ready(self._backend())):
                self._emit(phase="waiting", note="waiting for the model download")
                with self._wake:
                    self._wake.wait(5)
                continue
            try:
                worked = False
                for name in (MAIN_FILE, CODE_FILE):
                    if name == CODE_FILE and not (self.settings.code_search and self.settings.code_folders):
                        continue
                    if name not in indexers:
                        indexers[name] = self._make_indexer(name)
                        self._update_cache(name, indexers[name])
                    indexer = indexers[name]
                    if name in due:
                        with self._wake:
                            self._dirty.pop(name, None)
                        worked |= self._sync(indexer, name)
                    worked |= self._index(indexer, name)
                if worked:
                    last_work = time.time()
                    for name, ix in indexers.items():
                        ix.store.checkpoint()
                        self._update_cache(name, ix)
            except WorkerError as error:
                self.message.emit(f"The model stopped: {error}")
                self._emit(phase="error", note=str(error)[:200])
                for ix in indexers.values():
                    ix.close_worker()
                with self._wake:
                    self._wake.wait(30)
                continue
            except Exception:  # noqa: BLE001 - logged, and the loop goes on
                self._log(traceback.format_exc())
            if time.time() - last_work > WORKER_IDLE_SECONDS:
                for ix in indexers.values():
                    ix.close_worker()
            self._emit(phase="idle", current="", note="", counts=self._counts(indexers))
            with self._wake:
                if self._stop:
                    break
                timeout = 30.0
                if self._dirty:
                    timeout = max(0.2, min(self._dirty.values()) - time.time())
                self._wake.wait(timeout)
        for ix in indexers.values():
            ix.finish()

    def _backend(self) -> str:
        return self.settings.backend_choice or downloads.best_backend()

    def _update_cache(self, name: str, indexer: Indexer) -> None:
        """Appends what indexing added to the vector cache, so a search finds it mapped and ready."""
        from ..vectorcache import VectorCache

        try:
            cache = self._caches.get(name) or VectorCache(indexer.store)
            self._caches[name] = cache
            if not cache._read_meta():
                self._emit(phase="syncing", note="preparing search (once)", current="")
            if cache.update(wait=1.0):
                self.changed.emit(name)
        except Exception:  # noqa: BLE001 - search builds it itself if this fails
            self._log(traceback.format_exc())

    def _make_indexer(self, name: str) -> Indexer:
        from ..worker import EmbeddingWorker

        code = name == CODE_FILE
        store = IndexStore(paths.index(), name)
        if self.settings.backend_choice:
            os.environ["DIGUP_BACKEND"] = self.settings.backend_choice
        from ..llama import model_info

        indexer = Indexer(store, self.settings.options(code),
                          lambda: EmbeddingWorker(text_only=False, background=True),
                          embedder_source=model_info(self._backend()).vector_source)
        indexer.reset_on_model_change = lambda old, new: self.message.emit(
            "DigUp's model changed: everything will be read again.")
        indexer.check_model()
        return indexer

    def _sync(self, indexer: Indexer, name: str) -> bool:
        code = name == CODE_FILE
        roots = self.settings.code_folders if code else self.settings.folders
        self._emit(phase="syncing", note="looking for new and changed files", current="")
        indexer.options = self.settings.options(code)
        indexer.store.prune(roots)
        connected = [r for r in roots if os.path.isdir(r)]
        indexer.store.set_scope(connected)
        summary = indexer.sync(connected, self._should_stop)
        if summary.removed > 10_000:
            # A big folder went (or a rule now leaves it out): give the space back to the disk.
            self._emit(phase="syncing", note="compacting the index", current="")
            indexer.store.checkpoint()
            indexer.store.compact_if_worthwhile()
        # What this scan found, for Settings: when, what changed, and what was left out and why (a scan stopped
        # partway, by pausing or quitting, saw only part of the folders and isn't recorded).
        import json

        report = summary.crawl
        if report and report.complete:
            indexer.store.set_meta("last_sync", json.dumps({
                "time": time.time(), "added": summary.added, "changed": summary.changed, "moved": summary.moved,
                "removed": summary.removed, "unchanged": summary.unchanged, "skipped": report.skipped,
                "skipped_folders": report.skipped_folders, "datasets": report.datasets,
                "missing": [r for r in roots if r not in connected],
            }))
        if summary.added or summary.changed or summary.moved or summary.removed:
            self.changed.emit(name)
        return bool(summary.added or summary.changed)

    def _index(self, indexer: Indexer, name: str) -> bool:
        pending = indexer.store.pending_count()
        unfinished, left = indexer.store.unfinished_count()
        if not pending and not unfinished:
            return False
        label = "code" if name == CODE_FILE else "files"
        changed_since = [0]

        def on_event(event) -> None:
            changed_since[0] += 1
            self._emit(phase="indexing" if event.state != "partial" or pending else "finishing",
                       done=event.done, total=event.total, current=event.path, note=label,
                       pending=max(0, event.total - event.done))
            if changed_since[0] >= 25:  # let searches see new files as they come in
                changed_since[0] = 0
                self.changed.emit(name)

        if pending:
            self._emit(phase="indexing", done=0, total=pending, note=label)
            indexer.run(should_stop=self._should_stop, on_event=on_event)
        if not self._should_stop() and indexer.store.pending_count() == 0 and unfinished:
            self._emit(phase="finishing", done=0, total=left, note=f"the rest of {unfinished} long files")
            indexer.finish_long_files(should_stop=self._should_stop, on_step=on_event)
        self.changed.emit(name)
        return True

    def _counts(self, indexers: dict[str, Indexer]) -> dict:
        out = {}
        for name, ix in indexers.items():
            try:
                out[name] = ix.store.counts(light=True)
            except Exception:
                pass
        return out

    def _log(self, text: str) -> None:
        try:
            paths.logs().mkdir(parents=True, exist_ok=True)
            with open(paths.logs() / "engine.log", "a", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + text + "\n")
        except OSError:
            pass
