"""Settings → Overview: the folders DigUp has scanned and what it found in each, what it's doing now, what the last
scan changed and left out, and the files it couldn't read."""

from __future__ import annotations

import json
import os
import threading
import time

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (QAbstractItemView, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
                               QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from .. import downloads, paths
from ..settings import Settings
from ..store import CODE_FILE, MAIN_FILE, IndexStore
from . import opener

COLUMNS = ["Folder", "Searchable", "What's in it", "Waiting", "Couldn't read", "Size", "Last read"]
KIND_NAMES = [("image", "picture", "pictures"), ("screenshot", "screenshot", "screenshots"), ("pdf", "PDF", "PDFs"),
              ("doc", "document", "documents"), ("audio", "recording", "recordings"), ("video", "video", "videos"),
              ("code", "code file", "code files")]
PATH_ROLE = Qt.ItemDataRole.UserRole


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:,.0f} {unit}" if unit in ("B", "KB") else f"{n:,.1f} {unit}"
        n /= 1024
    return ""


def ago(t: float | None) -> str:
    if not t:
        return "—"
    seconds = time.time() - t
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h ago"
    return time.strftime("%d %b %Y", time.localtime(t))


class _Bridge(QObject):
    collected = Signal(object)
    worked = Signal()  # an index-wide change finished


class Card(QFrame):
    def __init__(self, label: str):
        super().__init__()
        self.setObjectName("Card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)
        self.value = QLabel("—")
        self.value.setObjectName("CardValue")
        self.label = QLabel(label)
        self.label.setObjectName("CardLabel")
        layout.addWidget(self.value)
        layout.addWidget(self.label)

    def set(self, value: str, detail: str | None = None) -> None:
        self.value.setText(value)
        if detail is not None:
            self.label.setToolTip(detail)


class OverviewPage(QWidget):
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.engine = None
        self._stores: dict[str, IndexStore] = {}
        self._last_refresh = 0.0
        self._busy = False
        self._again = False
        self._bridge = _Bridge()
        self._bridge.collected.connect(self._apply)
        self._bridge.worked.connect(lambda: (self._scan(), self.refresh()))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(10)

        top = QHBoxLayout()
        self.status = QLabel("—")
        self.status.setWordWrap(True)
        self.updating = QLabel("")
        self.updating.setObjectName("Muted")
        top.addWidget(self.status, 1)
        top.addWidget(self.updating)
        layout.addLayout(top)

        cards = QGridLayout()
        cards.setSpacing(8)
        self.c_files = Card("files searchable")
        self.c_waiting = Card("waiting to be read")
        self.c_failed = Card("couldn't be read")
        self.c_vectors = Card("vectors (pictures, pages, passages, moments)")
        self.c_size = Card("index on disk")
        self.c_model = Card("model runs on")
        for i, card in enumerate((self.c_files, self.c_waiting, self.c_failed, self.c_vectors, self.c_size,
                                  self.c_model)):
            cards.addWidget(card, i // 3, i % 3)
        layout.addLayout(cards)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(COLUMNS)
        self.tree.setRootIsDecorated(True)
        self.tree.setAlternatingRowColors(False)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setUniformRowHeights(True)
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.resizeSection(0, 340)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        for column in (1, 3, 4, 5, 6):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.tree.itemDoubleClicked.connect(self._open_item)
        self.tree.setToolTip("Double-click a folder to open it, or a file to show it in File Explorer")
        layout.addWidget(self.tree, 1)

        self.last_scan = QLabel()
        self.last_scan.setObjectName("Muted")
        self.last_scan.setWordWrap(True)
        self.last_scan.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.last_scan)

        row = QHBoxLayout()
        buttons = [("Refresh", self.refresh), ("Scan now", self._scan), ("Try failed files again", self._retry),
                   ("Read everything again", self._rebuild), ("Open data folder", self._open_data)]
        for label, action in buttons:
            b = QPushButton(label)
            b.clicked.connect(action)
            row.addWidget(b)
        row.addStretch(1)
        layout.addLayout(row)

        self._timer = QTimer(self)
        self._timer.setInterval(15000)
        self._timer.timeout.connect(lambda: self.isVisible() and self.refresh())

    # MARK: Live status

    def set_engine(self, engine) -> None:
        self.engine = engine
        engine.status.connect(self._engine_status)
        if getattr(engine, "_status", None) is not None:
            self._engine_status(engine._status)

    def _engine_status(self, status) -> None:
        if status.phase in ("indexing", "finishing") and status.total:
            what = "Reading the rest of long files" if status.phase == "finishing" else f"Reading {status.note or 'files'}"
            text = f"<b>{what}:</b> {status.done:,} of {status.total:,}"
            if status.current:
                text += f" — {os.path.basename(status.current)}"
        elif status.phase == "faces" and status.total:
            text = f"<b>Finding people in photos:</b> {status.done:,} of {status.total:,}"
        elif status.phase == "syncing":
            text = "<b>Scanning</b> your folders for new and changed files…"
        elif status.phase == "paused":
            text = f"<b>Paused</b> ({status.paused_reason}). Resume from the DigUp icon in the notification area."
        elif status.phase == "waiting":
            text = "<b>Waiting</b> for the model download to finish."
        elif status.phase == "error":
            text = f"<b>Stopped:</b> {status.note}"
        else:
            text = "<b>Up to date.</b> New and changed files are read a few seconds after they land."
        self.status.setText(text)
        # Keep the numbers moving while it works, without re-reading the index on every file.
        if self.isVisible() and time.time() - self._last_refresh > 15:
            self.refresh()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.refresh()
        self._timer.start()

    def hideEvent(self, event) -> None:
        self._timer.stop()
        super().hideEvent(event)

    # MARK: The numbers (read on a background thread: on a big index they take seconds)

    def _store(self, name: str) -> IndexStore | None:
        """The background thread's connection to an index (one refresh runs at a time)."""
        if not (paths.index() / name).exists():
            return None
        if name not in self._stores:
            self._stores[name] = IndexStore(paths.index(), name)
        return self._stores[name]

    def refresh(self) -> None:
        """Starts reading the numbers; the page shows them when they're in. A refresh asked for while one runs
        happens once it's done."""
        self._last_refresh = time.time()
        if self._busy:
            self._again = True
            return
        self._busy = True
        self.updating.setText("Updating…")
        groups = [(MAIN_FILE, list(self.settings.folders), None)]
        if self.settings.code_search and self.settings.code_folders:
            groups.append((CODE_FILE, list(self.settings.code_folders), "Code search"))
        threading.Thread(target=self._collect_in_background, args=(groups,), daemon=True).start()

    def _collect_in_background(self, groups) -> None:
        try:
            data = self._collect(groups)
        except Exception as error:  # noqa: BLE001 - shown on the page
            data = {"error": str(error)}
        try:
            self._bridge.collected.emit(data)
        except RuntimeError:
            pass  # the window closed meanwhile

    def _collect(self, groups) -> dict:
        data = {"groups": [], "vectors": 0, "scans": []}
        for name, folders, title in groups:
            store = self._store(name)
            rows = []
            for folder in folders:
                empty = {"kinds": {}, "states": {}, "files": 0, "bytes": 0, "last": None}
                rows.append({
                    "folder": folder,
                    "connected": os.path.isdir(folder),
                    "stats": store.folder_stats(folder) if store else empty,
                    "problems": store.problems(folder) if store else [],
                })
            data["groups"].append((name, title, rows))
            if store:
                data["vectors"] += store.vector_count()
                data["scans"].append((name, store.meta("last_sync")))
        data["size"] = sum(IndexStore.size_of(paths.index() / n) for n in (MAIN_FILE, CODE_FILE)
                           if (paths.index() / n).exists())
        backend = self.settings.backend_choice or downloads.best_backend()
        data["backend"] = backend
        data["ready"] = downloads.runtime_ready(backend) and downloads.models_complete()
        return data

    def _apply(self, data: dict) -> None:
        self._busy = False
        self.updating.setText("")
        if self._again:
            self._again = False
            QTimer.singleShot(0, self.refresh)
        if "error" in data:
            self.updating.setText(f"Couldn't read the index: {data['error']}")
            return
        expanded = set()
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            for item in [top] + [top.child(j) for j in range(top.childCount())]:
                if item.isExpanded() and item.data(0, PATH_ROLE):
                    expanded.add(item.data(0, PATH_ROLE))
        self.tree.setUpdatesEnabled(False)
        self.tree.clear()
        totals = {"searchable": 0, "waiting": 0, "failed": 0}
        for name, title, rows in data["groups"]:
            parent = None
            if title:
                parent = QTreeWidgetItem([title])
                self.tree.addTopLevelItem(parent)
                parent.setFirstColumnSpanned(True)
                parent.setExpanded(True)
            for row in rows:
                folder, stats = row["folder"], row["stats"]
                states, kinds = stats["states"], stats["kinds"]
                searchable = states.get("done", 0) + states.get("partial", 0)
                waiting, failed, partial = states.get("pending", 0), states.get("failed", 0), states.get("partial", 0)
                totals["searchable"] += searchable
                totals["waiting"] += waiting
                totals["failed"] += failed
                label = folder.replace(os.path.expanduser("~"), "~", 1)
                if not row["connected"]:
                    label += "  (not connected)"
                contents = " · ".join(f"{kinds[k]:,} {one if kinds[k] == 1 else many}"
                                      for k, one, many in KIND_NAMES if kinds.get(k))
                values = [
                    label, f"{searchable:,}", contents or ("nothing read yet" if waiting else "nothing to read"),
                    f"{waiting:,}" + (f" (+{partial:,} long files finishing)" if partial else "") if waiting or partial else "",
                    f"{failed:,}" if failed else "", human_bytes(stats["bytes"]) if stats["bytes"] else "",
                    ago(stats["last"]),
                ]
                item = QTreeWidgetItem(values)
                item.setData(0, PATH_ROLE, folder)
                item.setToolTip(0, folder)
                item.setToolTip(2, contents)
                for column in (1, 3, 4, 5):
                    item.setTextAlignment(column, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                for path, state, detail in row["problems"]:
                    why = "couldn't read" if state == "failed" else "left out"
                    child = QTreeWidgetItem([f"{os.path.relpath(path, folder)}  —  {why}: {detail}"])
                    child.setData(0, PATH_ROLE, path)
                    child.setToolTip(0, f"{path}\n{detail}")
                    item.addChild(child)
                (parent.addChild if parent else self.tree.addTopLevelItem)(item)
                for i in range(item.childCount()):  # spanning works once the row is in the tree
                    item.child(i).setFirstColumnSpanned(True)
                if folder in expanded:
                    item.setExpanded(True)
        if not self.settings.folders:
            empty = QTreeWidgetItem(["No folders yet: add some in the Folders tab."])
            self.tree.addTopLevelItem(empty)
            empty.setFirstColumnSpanned(True)
        self.tree.setUpdatesEnabled(True)

        self.c_files.set(f"{totals['searchable']:,}")
        self.c_waiting.set(f"{totals['waiting']:,}")
        self.c_failed.set(f"{totals['failed']:,}", "Expand a folder below to see which files and why")
        self.c_vectors.set(f"{data['vectors']:,}")
        self.c_size.set(human_bytes(data["size"]), str(paths.index()))
        backend = data["backend"]
        self.c_model.set({"cuda": "NVIDIA GPU", "vulkan": "GPU (Vulkan)", "cpu": "CPU"}.get(backend, backend)
                         + ("" if data["ready"] else " (downloading)"),
                         f"EmbeddingGemma 2 (Q8_0) on llama.cpp {downloads.LLAMA_TAG}, {backend}")
        titles = {MAIN_FILE: "Last scan", CODE_FILE: "Last code scan"}
        self.last_scan.setText("\n".join(self._last_scan_text(raw, titles[name]) for name, raw in data["scans"]))

    def _last_scan_text(self, raw: str | None, title: str) -> str:
        if not raw:
            return f"{title}: not yet."
        try:
            scan = json.loads(raw)
        except ValueError:
            return ""
        changes = [f"{scan[k]:,} {label}" for k, label in (("added", "new"), ("changed", "changed"),
                                                            ("moved", "moved or renamed"), ("removed", "removed"))
                   if scan.get(k)]
        text = (f"{title} {ago(scan.get('time'))}: " + (", ".join(changes) if changes else "nothing new")
                + f", {scan.get('unchanged', 0):,} unchanged.")
        skipped = scan.get("skipped") or {}
        shown = {k: v for k, v in skipped.items() if k != "other type"}
        if shown:
            text += "  Not read: " + ", ".join(f"{k} ({v:,})" for k, v in sorted(shown.items(), key=lambda kv: -kv[1]))
            text += "."
        if skipped.get("other type"):
            text += f"  {skipped['other type']:,} files of other types (apps, archives…) were passed over."
        folders = scan.get("skipped_folders") or {}
        if folders:
            text += "  Folders not walked into: " + ", ".join(
                f"{k} ({v:,})" for k, v in sorted(folders.items(), key=lambda kv: -kv[1])) + "."
        for folder, n in (scan.get("datasets") or {}).items():
            text += f"  Left out as a dataset: {folder} ({n:,} files)."
        missing = scan.get("missing") or []
        if missing:
            text += "  Not connected: " + ", ".join(missing) + "."
        return text

    # MARK: Actions

    def _open_item(self, item: QTreeWidgetItem, _column: int) -> None:
        path = item.data(0, PATH_ROLE)
        if not path:
            return
        if os.path.isdir(path):
            os.startfile(path)
        elif os.path.exists(path):
            opener.reveal(path)

    def _scan(self) -> None:
        if self.engine:
            self.engine.resync()
            self.status.setText("<b>Scanning</b> your folders for new and changed files…")

    def _in_background(self, label: str, work) -> None:
        """Index-wide changes take a while on a big index: run them off the UI thread, then scan and refresh."""
        self.updating.setText(label)

        def run():
            for name in (MAIN_FILE, CODE_FILE):
                if (paths.index() / name).exists():
                    store = IndexStore(paths.index(), name)  # its own connection, in this thread
                    try:
                        work(store)
                    finally:
                        store.close()
            try:
                self._bridge.worked.emit()
            except RuntimeError:
                pass  # the window closed meanwhile

        threading.Thread(target=run, daemon=True).start()

    def _retry(self) -> None:
        self._in_background("Queuing failed files again…", lambda store: store.retry_failed())

    def _rebuild(self) -> None:
        if QMessageBox.question(self, "Read everything again?",
                                "DigUp will read and embed every file again. That takes as long as the first scan.")                 != QMessageBox.StandardButton.Yes:
            return
        self._in_background("Emptying the index…", lambda store: store.reset_all())

    def _open_data(self) -> None:
        paths.home().mkdir(parents=True, exist_ok=True)
        os.startfile(str(paths.home()))
