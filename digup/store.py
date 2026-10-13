"""The on-disk index: one SQLite file with files, their segments (fp16 vectors), and an FTS5 keyword table.

Vectors live in the same database as everything else, so a crash never leaves them out of step with the rows that
describe them. At this scale (tens of thousands of vectors) brute-force search over an in-memory copy takes a few
milliseconds; no vector database needed.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import paths, searchtext
from .model import DIM, INDEXING_ORDER_SQL, FileKind, IndexOptions, Modality, SegmentKind

SCHEMA_VERSION = 1
# Marks (M*) count as letters, so Bengali vowel signs and Arabic harakat stay inside their words.
TOKENIZER = "porter unicode61 remove_diacritics 2 categories 'L* N* Co M*'"
FTS_COLUMNS = f'name, body, original UNINDEXED, tokenize = "{TOKENIZER}"'
MAIN_FILE = "index.sqlite"
CODE_FILE = "code.sqlite"
_END = "\U0010ffff"


SCHEMA = f"""
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE files(
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    inode INTEGER NOT NULL,
    device INTEGER NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',   -- pending | done | partial | skipped | failed
    detail TEXT,
    info TEXT,                                -- JSON: pixels, pages, duration, read_to, read_end
    indexed_at REAL
);
CREATE INDEX files_inode ON files(device, inode);
CREATE INDEX files_state ON files(state, kind);
CREATE TABLE segments(
    id INTEGER PRIMARY KEY,
    file INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    modality TEXT NOT NULL,
    loc REAL,          -- PDF page, start second, or first line
    loc_end REAL,
    excerpt TEXT,
    vec BLOB           -- 768 x float16, L2-normalized
);
CREATE INDEX segments_file ON segments(file);
CREATE VIRTUAL TABLE fts USING fts5({FTS_COLUMNS});
"""


@dataclass(slots=True)  # millions of these during a scan of a big drive: slots halve their memory
class FileRecord:
    id: int
    path: str
    kind: FileKind
    size: int
    mtime: float
    inode: int
    device: int
    state: str


@dataclass(slots=True)
class Candidate:
    """A file the crawler decided to index."""

    path: str
    kind: FileKind
    size: int
    mtime: float
    inode: int
    device: int


def half(vector) -> bytes:
    return np.asarray(vector, dtype=np.float32).astype(np.float16).tobytes()


class IndexStore:
    FILE_COLUMNS = "id, path, kind, size, mtime, inode, device, state"

    def __init__(self, directory: str | Path | None = None, name: str = MAIN_FILE):
        self.directory = Path(directory) if directory else paths.index()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.file = self.directory / name
        self.db = sqlite3.connect(self.file, timeout=5, isolation_level=None, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.execute("PRAGMA synchronous = NORMAL")
        # The log is cut back to this after each checkpoint, instead of keeping its largest size for good.
        self.db.execute("PRAGMA journal_size_limit = 67108864")
        self.db.execute("PRAGMA foreign_keys = ON")
        self._scope_sql = ""
        self._scope_values: list = []
        self._migrate()

    @property
    def is_code(self) -> bool:
        return self.file.name == CODE_FILE

    def close(self) -> None:
        self.db.close()

    @staticmethod
    def size_of(file: Path) -> int:
        return sum(os.path.getsize(p) for p in (str(file), str(file) + "-wal") if os.path.exists(p))

    # MARK: Schema

    def _migrate(self) -> None:
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError(f"this index was made by a newer DigUp (schema {version})")
        if version == SCHEMA_VERSION:
            return
        # The app and its worker can open the index at the same moment: one creates it while the other waits.
        self.db.execute("PRAGMA busy_timeout = 60000")
        with self.transaction():
            if self.db.execute("PRAGMA user_version").fetchone()[0] == 0:
                for statement in SCHEMA.split(";"):
                    if statement.strip():
                        self.db.execute(statement)
                self.db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.db.execute("PRAGMA busy_timeout = 5000")
        # Counting vectors without reading them (modality is 'none' exactly when a segment has no vector), and the
        # log of deleted vectors the vector cache reads (vectorcache.py).
        try:
            self.db.execute("CREATE INDEX IF NOT EXISTS segments_modality ON segments(modality)")
            from .vectorcache import install_trigger

            install_trigger(self.db)
            from . import people

            people.install(self.db)  # faces and the people they belong to (only filled when faces are on)
        except sqlite3.OperationalError:
            pass  # another process holds the write lock; it will be made next time

    class _Transaction:
        def __init__(self, db: sqlite3.Connection):
            self.db = db
            self.outer = False

        def __enter__(self):
            self.outer = not self.db.in_transaction
            if self.outer:
                self.db.execute("BEGIN IMMEDIATE")
            return self

        def __exit__(self, kind, value, tb):
            if self.outer:
                self.db.execute("COMMIT" if kind is None else "ROLLBACK")
            return False

    def transaction(self) -> "IndexStore._Transaction":
        """BEGIN IMMEDIATE ... COMMIT (nested uses join the outer one)."""
        return IndexStore._Transaction(self.db)

    # MARK: Scope (files on drives that are connected right now)

    def set_scope(self, roots: list[str] | None) -> None:
        if roots is None:
            self._scope_sql, self._scope_values = "", []
            return
        if not roots:
            self._scope_sql, self._scope_values = " AND 0", []
            return
        self._scope_sql = " AND (" + " OR ".join("(path >= ? AND path < ?)" for _ in roots) + ")"
        self._scope_values = [v for r in roots for v in (paths.prefix(r), paths.prefix(r) + _END)]

    # MARK: Meta

    def meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str | None) -> None:
        if value is None:
            self.db.execute("DELETE FROM meta WHERE key = ?", (key,))
        else:
            self.db.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # MARK: Files

    @staticmethod
    def _record(row) -> FileRecord:
        return FileRecord(row[0], row[1], FileKind(row[2]), row[3], row[4], row[5], row[6], row[7])

    def files_under(self, roots: list[str]) -> list[FileRecord]:
        out: list[FileRecord] = []
        for root in roots:
            if any(r != root and paths.is_inside(root, r) for r in roots):
                continue  # covered by the folder it's in
            p = paths.prefix(root)
            out += [
                self._record(r)
                for r in self.db.execute(
                    f"SELECT {self.FILE_COLUMNS} FROM files WHERE path >= ? AND path < ?", (p, p + _END)
                )
            ]
        return out

    def insert(self, file: Candidate) -> int:
        cur = self.db.execute(
            "INSERT INTO files(path, kind, size, mtime, inode, device) VALUES(?, ?, ?, ?, ?, ?)",
            (file.path, file.kind.value, file.size, file.mtime, _i64(file.inode), _i64(file.device)),
        )
        self._add_name_row(cur.lastrowid, file.path)
        return cur.lastrowid

    def requeue_changed(self, id: int, file: Candidate) -> None:
        """The file changed on disk: drop what was read from it (its faces too) and queue it again."""
        self._delete_segments(id, keeping_name=True)
        try:
            self.db.execute("DELETE FROM faces WHERE file = ?", (id,))
            self.db.execute("DELETE FROM face_scans WHERE file = ?", (id,))
        except sqlite3.OperationalError:
            pass  # an index from before faces: no such tables
        self.db.execute(
            "UPDATE files SET kind = ?, size = ?, mtime = ?, inode = ?, device = ?, state = 'pending', detail = NULL, "
            "info = NULL, indexed_at = NULL WHERE id = ?",
            (file.kind.value, file.size, file.mtime, _i64(file.inode), _i64(file.device), id),
        )

    def move(self, id: int, path: str) -> None:
        """Same file id, size and date at a new path: a move or rename. Nothing is embedded again."""
        self.db.execute("UPDATE files SET path = ? WHERE id = ?", (path, id))
        self.db.execute(
            "UPDATE fts SET name = ?, body = ? WHERE rowid IN (SELECT id FROM segments WHERE file = ? AND kind = 'name')",
            (searchtext.normalized(name_text(path)), searchtext.normalized(folder_text(path)), id),
        )

    def delete_many(self, ids: list[int], chunk: int = 20_000) -> None:
        """Removes many files at once (a folder taken out, a rule that now leaves it out): set-based, so two million
        rows take minutes rather than hours. Runs inside the caller's transaction."""
        self.db.execute("CREATE TEMP TABLE IF NOT EXISTS gone(id INTEGER PRIMARY KEY)")
        for start in range(0, len(ids), chunk):
            self.db.execute("DELETE FROM gone")
            self.db.executemany("INSERT OR IGNORE INTO gone(id) VALUES(?)", ((i,) for i in ids[start:start + chunk]))
            self.db.execute("DELETE FROM fts WHERE rowid IN (SELECT s.id FROM segments s JOIN gone g ON g.id = s.file)")
            self.db.execute("DELETE FROM segments WHERE file IN (SELECT id FROM gone)")
            self.db.execute("DELETE FROM files WHERE id IN (SELECT id FROM gone)")
        self.db.execute("DELETE FROM gone")

    def delete(self, id: int) -> None:
        self.db.execute("DELETE FROM fts WHERE rowid IN (SELECT id FROM segments WHERE file = ?)", (id,))
        self.db.execute("DELETE FROM files WHERE id = ?", (id,))

    def prune(self, keeping_under: list[str]) -> int:
        """Removes every file that isn't under one of `keeping_under` (folders taken out of the app). One query, not
        a Python loop over every path in the index."""
        roots = [paths.normalize(r) for r in keeping_under]
        outside = " AND ".join("NOT (path >= ? AND path < ?)" for _ in roots) or "1"
        args = [v for r in roots for v in (paths.prefix(r), paths.prefix(r) + _END)]
        gone = [r[0] for r in self.db.execute(f"SELECT id FROM files WHERE {outside}", args).fetchall()]
        if gone:
            with self.transaction():
                for i in range(0, len(gone), 50_000):
                    self.delete_many(gone[i:i + 50_000])
        return len(gone)

    def reset_all(self) -> None:
        """Queues every file again and forgets the fingerprint (after the model or settings changed)."""
        with self.transaction():
            self.db.execute("DELETE FROM fts WHERE rowid IN (SELECT id FROM segments WHERE kind != 'name')")
            self.db.execute("DELETE FROM segments WHERE kind != 'name'")
            self.db.execute("UPDATE files SET state = 'pending', detail = NULL, info = NULL, indexed_at = NULL")
            self.set_meta("fingerprint", None)

    def retry_failed(self) -> int:
        with self.transaction():
            ids = [r[0] for r in self.db.execute("SELECT id FROM files WHERE state = 'failed'")]
            for id in ids:
                self._delete_segments(id, keeping_name=True)
                self.db.execute("UPDATE files SET state = 'pending', detail = NULL, info = NULL WHERE id = ?", (id,))
        return len(ids)

    def pending(self, limit: int | None = None, only: set[int] | None = None) -> list[FileRecord]:
        if only is not None and not only:
            return []
        filter = f" AND id IN ({','.join(str(int(i)) for i in only)})" if only else ""
        rows = self.db.execute(
            f"SELECT {self.FILE_COLUMNS} FROM files WHERE state = 'pending'{filter}{self._scope_sql} "
            f"ORDER BY {INDEXING_ORDER_SQL}, mtime DESC LIMIT ?",
            [*self._scope_values, limit if limit is not None else -1],
        )
        return [self._record(r) for r in rows]

    def pending_count(self) -> int:
        return self.db.execute(
            f"SELECT COUNT(*) FROM files WHERE state = 'pending'{self._scope_sql}", self._scope_values
        ).fetchone()[0]

    def unfinished(self) -> list[FileRecord]:
        rows = self.db.execute(
            f"SELECT {self.FILE_COLUMNS} FROM files WHERE state = 'partial'{self._scope_sql} "
            f"ORDER BY {INDEXING_ORDER_SQL}, mtime DESC",
            self._scope_values,
        )
        return [self._record(r) for r in rows]

    def unfinished_count(self) -> tuple[int, int]:
        row = self.db.execute(
            "SELECT COUNT(*), SUM(MAX(0, CAST(json_extract(info, '$.read_end') AS INTEGER) "
            f"- CAST(json_extract(info, '$.read_to') AS INTEGER))) FROM files WHERE state = 'partial'{self._scope_sql}",
            self._scope_values,
        ).fetchone()
        return row[0] or 0, row[1] or 0

    def info(self, id: int) -> dict[str, str]:
        row = self.db.execute("SELECT info FROM files WHERE id = ?", (id,)).fetchone()
        return json.loads(row[0]) if row and row[0] else {}

    def last_place(self, id: int) -> int | None:
        row = self.db.execute(
            "SELECT MAX(loc) FROM segments WHERE file = ? AND kind IN ('page', 'page_image', 'chunk')", (id,)
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else None

    def finish(self, id: int, state: str, detail: str | None = None, info: dict | None = None) -> None:
        text = json.dumps({k: v for k, v in (info or {}).items() if v is not None}, sort_keys=True) if info else None
        self.db.execute(
            "UPDATE files SET state = ?, detail = ?, info = ?, indexed_at = ? WHERE id = ?",
            (state, detail, text, time.time(), id),
        )

    # MARK: Segments

    def add_segment(self, file: int, kind: SegmentKind, modality: Modality, loc: float | None = None,
                    loc_end: float | None = None, excerpt: str | None = None, text: str | None = None,
                    vector=None) -> int:
        cur = self.db.execute(
            "INSERT INTO segments(file, kind, modality, loc, loc_end, excerpt, vec) VALUES(?, ?, ?, ?, ?, ?, ?)",
            (file, kind.value, (Modality.NONE if vector is None else modality).value, loc, loc_end, excerpt,
             None if vector is None else half(vector)),
        )
        if text:
            body, original = keyword_text(text)
            self.db.execute("INSERT INTO fts(rowid, name, body, original) VALUES(?, '', ?, ?)",
                            (cur.lastrowid, body, original))
        return cur.lastrowid

    def _add_name_row(self, file: int, path: str) -> None:
        cur = self.db.execute("INSERT INTO segments(file, kind, modality) VALUES(?, 'name', 'none')", (file,))
        self.db.execute(
            "INSERT INTO fts(rowid, name, body) VALUES(?, ?, ?)",
            (cur.lastrowid, searchtext.normalized(name_text(path)), searchtext.normalized(folder_text(path))),
        )

    def _delete_segments(self, file: int, keeping_name: bool) -> None:
        filter = "AND kind != 'name'" if keeping_name else ""
        self.db.execute(f"DELETE FROM fts WHERE rowid IN (SELECT id FROM segments WHERE file = ? {filter})", (file,))
        self.db.execute(f"DELETE FROM segments WHERE file = ? {filter}", (file,))

    # MARK: Stats

    def vector_count(self) -> int:
        """How many vectors the index holds (from an index on `modality`: no vector is read)."""
        return self.db.execute("SELECT COUNT(*) FROM segments WHERE modality != 'none'").fetchone()[0]

    def checkpoint(self) -> None:
        """Copies the write-ahead log into the database and cuts the log back (after a burst of writes)."""
        try:
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.OperationalError:
            pass

    def compact_if_worthwhile(self, min_free_bytes: int = 1 << 30, min_share: float = 0.25) -> bool:
        """Gives free space back to the disk (VACUUM) when a lot of the file is unused, as after removing a big
        folder. Takes a while on a big index; the caller runs it off the UI thread."""
        pages, free, size = (self.db.execute(f"PRAGMA {p}").fetchone()[0]
                             for p in ("page_count", "freelist_count", "page_size"))
        if free * size < min_free_bytes or free < pages * min_share:
            return False
        try:
            self.db.execute("VACUUM")
            self.checkpoint()
            return True
        except sqlite3.OperationalError:
            return False

    def folder_stats(self, root: str) -> dict:
        """What the index holds under `root`: files by kind and state, their bytes, vectors, and the last file read."""
        p = paths.prefix(root)
        stats = {"kinds": {}, "states": {}, "files": 0, "bytes": 0, "last": None}
        for kind, state, n, size, last in self.db.execute(
            "SELECT kind, state, COUNT(*), SUM(size), MAX(indexed_at) FROM files WHERE path >= ? AND path < ? "
            "GROUP BY kind, state", (p, p + _END)
        ):
            stats["files"] += n
            stats["bytes"] += size or 0
            stats["states"][state] = stats["states"].get(state, 0) + n
            if state in ("done", "partial"):
                stats["kinds"][kind] = stats["kinds"].get(kind, 0) + n
            if last and (stats["last"] is None or last > stats["last"]):
                stats["last"] = last
        return stats

    def problems(self, root: str, limit: int = 200) -> list[tuple[str, str, str]]:
        """Files under `root` that failed or were skipped after reading, with why: (path, state, detail). Pictures
        left out for being small (icons, thumbnails) are expected and not listed."""
        p = paths.prefix(root)
        return self.db.execute(
            "SELECT path, state, coalesce(detail, '') FROM files WHERE path >= ? AND path < ? "
            "AND state IN ('failed', 'skipped') AND coalesce(detail, '') NOT LIKE 'small image%' "
            "ORDER BY state, path LIMIT ?", (p, p + _END, limit)).fetchall()

    def counts(self, light: bool = False) -> dict:
        """Files by kind and state, long files left, segments and vectors. `light` leaves out the segment count,
        which reads the whole segments table (seconds on a million-segment index)."""
        by: dict[str, dict[str, int]] = {}
        for kind, state, n in self.db.execute(
            f"SELECT kind, state, COUNT(*) FROM files WHERE 1{self._scope_sql} GROUP BY kind, state",
            self._scope_values,
        ):
            by.setdefault(kind, {})[state] = n
        unfinished, left = self.unfinished_count()
        in_scope = "" if not self._scope_sql else f" AND file IN (SELECT id FROM files WHERE 1{self._scope_sql})"
        segments = None if light else self.db.execute(
            f"SELECT COUNT(*) FROM segments WHERE kind != 'name'{in_scope}", self._scope_values).fetchone()[0]
        vectors = self.db.execute(
            f"SELECT COUNT(*) FROM segments WHERE modality != 'none'{in_scope}", self._scope_values).fetchone()[0]
        return {"by_kind_and_state": by, "segments": segments, "vectors": vectors, "unfinished": unfinished,
                "left": left}


def keyword_text(text: str) -> tuple[str, str | None]:
    """`body` and `original` for keyword search: the normalized text, and the text as written if that differs."""
    body = searchtext.normalized(text)
    return body, (None if body == text else text)


def name_text(path: str) -> str:
    """"Screenshot 2026-10-11 093107.png" -> searchable words."""
    return os.path.basename(path)


def folder_text(path: str) -> str:
    """The folders a file sits in, relative to the user's profile ("Downloads Invoices"), as low-weight keywords."""
    folder = os.path.dirname(path)
    profile = str(Path.home())
    if folder.lower().startswith(profile.lower()):
        folder = folder[len(profile):]
    return " ".join(p for p in folder.replace(":", "").split("\\") if p)


def _i64(value: int) -> int:
    """SQLite integers are signed 64-bit; NTFS file ids can use the top bit."""
    value &= (1 << 64) - 1
    return value - (1 << 64) if value >= 1 << 63 else value


__all__ = ["IndexStore", "FileRecord", "Candidate", "IndexOptions", "MAIN_FILE", "CODE_FILE", "DIM"]
