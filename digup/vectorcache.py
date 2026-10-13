"""The vectors search scores, on disk beside the index, memory-mapped: 768 int8 values and a scale per vector.

Holding a big index's vectors in memory took 1.5 GB (a million of them, fp16) and 3 GB while loading, and every
change reloaded them all. Mapped from a file, they cost the search process almost nothing of its own (the pages are
the file system cache's, shared and given back under pressure), they're there instantly on the next start, and an
indexing pass only appends what it added.

int8 with a per-vector scale keeps cosines within 0.0015 of fp16 (mean 0.0002) and the top 200 results the same
(198 of 200 on a million random vectors).

Correctness: SQLite can hand a new segment the id of a deleted one, so the cache is told about deletions by a trigger
(`vec_gone`): every deleted segment that had a vector is logged, and its cached row is marked dead before anything new
is appended. Rows are appended in id order for ids past the last one cached, plus any logged id that's live again.
When more than a third of the rows are dead, the cache is written anew (a new generation folder: Windows can't
replace a file another process has mapped).
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .model import DIM, FileKind, IndexOptions

MODALITY_CODE = {"text": 1, "image": 2, "audio": 3}
KIND_CODE = {kind.value: i + 1 for i, kind in enumerate(FileKind)}  # the order of searcher.KIND_CODES
COLUMNS = {"vectors": np.int8, "scales": np.float32, "ids": np.int64, "files": np.int64, "modality": np.uint8,
           "sets_scale": np.uint8, "kind": np.uint8, "dead": np.uint8}
VERSION = 2
BLOCK = 65536  # rows scored per task: a few dozen tasks over a million rows, each worth a thread
SLICE = 4096  # rows converted to float32 at once within a task (12 MB)
# Six threads score a million vectors in about 0.2 s; each one costs BLAS some scratch memory, so not more.
_pool = ThreadPoolExecutor(max_workers=max(2, min(6, (os.cpu_count() or 4) - 1)), thread_name_prefix="digup-score")


def install_trigger(db) -> None:
    """The deletion log the cache reads (created once per index; cheap: only segments with a vector are logged)."""
    db.execute("CREATE TABLE IF NOT EXISTS vec_gone(seq INTEGER PRIMARY KEY AUTOINCREMENT, id INTEGER NOT NULL)")
    db.execute("""
        CREATE TRIGGER IF NOT EXISTS vec_gone_log AFTER DELETE ON segments
        WHEN OLD.modality != 'none' BEGIN INSERT INTO vec_gone(id) VALUES (OLD.id); END
    """)


def _quantize(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    peak = np.abs(vectors).max(axis=1)
    scales = np.where(peak > 0, peak / 127.0, 1.0).astype(np.float32)
    return np.clip(np.rint(vectors / scales[:, None]), -127, 127).astype(np.int8), scales


class _Lock:
    """One writer at a time across processes (the app's engine, its search, the CLI): a lock on a file."""

    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def acquire(self, wait: float = 0.0) -> bool:
        import msvcrt

        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + wait
        while True:
            try:
                self.handle = open(self.path, "a+b")
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
                return True
            except OSError:
                if self.handle:
                    self.handle.close()
                    self.handle = None
                if time.time() >= deadline:
                    return False
                time.sleep(0.1)

    def release(self) -> None:
        import msvcrt

        if self.handle:
            try:
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            finally:
                self.handle.close()
                self.handle = None


class VectorCache:
    """Read side: the mapped columns of the current generation. Write side: `update` (under the lock)."""

    def __init__(self, store):
        self.store = store
        self.root = store.file.with_suffix(".vectors")
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = _Lock(self.root / "lock")
        self._meta_mtime = None
        self.meta: dict = {}
        self.n = 0
        self.cols: dict[str, np.ndarray] = {}
        self._mutex = threading.Lock()
        try:
            install_trigger(store.db)
        except Exception:  # noqa: BLE001 - busy: `update` tries again, and won't build without it
            pass

    # MARK: Reading

    def _meta_path(self) -> Path:
        return self.root / "current.json"

    def _read_meta(self) -> dict:
        try:
            return json.loads(self._meta_path().read_text("utf-8"))
        except (OSError, ValueError):
            return {}

    def open(self) -> bool:
        """Maps the current generation (again, if a writer moved it on). True when there's anything to search."""
        path = self._meta_path()
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            self.close()
            return False
        if mtime == self._meta_mtime and self.cols:
            return self.n > 0
        meta = self._read_meta()
        folder = self.root / str(meta.get("generation", ""))
        n = int(meta.get("count", 0))
        cols = {}
        if n and meta.get("version") == VERSION:
            try:
                for name, dtype in COLUMNS.items():
                    shape = (n, DIM) if name == "vectors" else (n,)
                    cols[name] = np.memmap(folder / f"{name}.bin", dtype=dtype, mode="r", shape=shape)
            except (OSError, ValueError):
                cols = {}
        with self._mutex:
            self.cols, self.n, self.meta, self._meta_mtime = cols, (n if cols else 0), meta, mtime
        return self.n > 0

    def close(self) -> None:
        with self._mutex:
            self.cols, self.n, self._meta_mtime = {}, 0, None

    def alive(self) -> np.ndarray:
        return self.cols["dead"][: self.n] == 0

    # MARK: Scoring

    def score(self, query) -> np.ndarray:
        """Cosine of `query` with every cached vector (dead rows included; the caller masks them), on all cores."""
        q = np.asarray(query, dtype=np.float32)
        vectors, scales, n = self.cols["vectors"], self.cols["scales"], self.n
        out = np.empty(n, dtype=np.float32)

        def block(start: int) -> None:
            # A slice at a time: a whole block as float32 would be 200 MB per thread.
            for a in range(start, min(start + BLOCK, n), SLICE):
                b = min(a + SLICE, start + BLOCK, n)
                out[a:b] = (vectors[a:b].astype(np.float32) @ q) * scales[a:b]

        list(_pool.map(block, range(0, n, BLOCK)))
        return out

    # MARK: Writing

    def update(self, wait: float = 0.0, budget_rows: int | None = None) -> bool:
        """Brings the cache up to the index: marks deleted rows dead, appends new ones, starts afresh when it's mostly
        dead or the index was emptied. False when another process holds the lock (it's doing the same)."""
        if not self._lock.acquire(wait):
            return False
        try:
            db = self.store.db
            # Without the deletion log the cache can't know what went: never build it then.
            if not db.execute("SELECT 1 FROM sqlite_master WHERE type = 'trigger' AND name = 'vec_gone_log'").fetchone():
                try:
                    install_trigger(db)
                except Exception:  # noqa: BLE001 - the index is busy; next time
                    return False
            meta = self._read_meta()
            fingerprint = self.store.meta("fingerprint") or ""
            if (meta.get("version") != VERSION or meta.get("fingerprint") != fingerprint
                    or not (self.root / str(meta.get("generation", ""))).is_dir()):
                meta = self._new_generation(fingerprint, db)
            folder = self.root / str(meta["generation"])
            count, max_id, cursor = int(meta["count"]), int(meta["max_id"]), int(meta["gone_seq"])

            # Deleted since last time: mark their rows dead.
            gone = db.execute("SELECT seq, id FROM vec_gone WHERE seq > ? ORDER BY seq", (cursor,)).fetchall()
            revived: list[int] = []
            if gone and count:
                gone_ids = np.fromiter((g[1] for g in gone), dtype=np.int64, count=len(gone))
                ids = np.memmap(folder / "ids.bin", dtype=np.int64, mode="r", shape=(count,))
                hit = np.nonzero(np.isin(ids, gone_ids))[0]
                del ids
                if hit.size:
                    dead = np.memmap(folder / "dead.bin", dtype=np.uint8, mode="r+", shape=(count,))
                    dead[hit] = 1
                    dead.flush()
                    del dead
                    meta["dead"] = int(meta.get("dead", 0)) + int(hit.size)
                # A logged id that has a vector again was reused for a new segment: fetch it below.
                revived = sorted({int(i) for i in gone_ids if i <= max_id})
            if gone:
                cursor = int(gone[-1][0])

            # New since last time (and reused ids), streamed in blocks.
            added = self._append(db, folder, max_id, revived, budget_rows)
            count += added["rows"]
            max_id = max(max_id, added["max_id"])
            meta.update(count=count, max_id=max_id, gone_seq=cursor)
            self._write_meta(meta)
            try:
                db.execute("DELETE FROM vec_gone WHERE seq <= ?", (cursor,))
            except Exception:  # noqa: BLE001 - busy: the cursor in the meta keeps them from counting twice
                pass
            if count and int(meta.get("dead", 0)) > count / 3:
                self._new_generation(fingerprint, db, rebuild=True)
            self._sweep()
            return True
        finally:
            self._lock.release()

    def _new_generation(self, fingerprint: str, db, rebuild: bool = False) -> dict:
        """An empty generation folder, filled from scratch (every vector in the index, in id order)."""
        generation = f"g{int(time.time() * 1000)}"
        folder = self.root / generation
        folder.mkdir(parents=True)
        for name in COLUMNS:
            (folder / f"{name}.bin").touch()
        last = db.execute("SELECT COALESCE(MAX(seq), 0) FROM vec_gone").fetchone()[0]
        meta = {"version": VERSION, "generation": generation, "fingerprint": fingerprint, "count": 0, "max_id": 0,
                "gone_seq": last, "dead": 0}
        if rebuild:
            added = self._append(db, folder, 0, [], None)
            meta.update(count=added["rows"], max_id=added["max_id"])
            self._write_meta(meta)
        return meta

    def _append(self, db, folder: Path, after: int, extra_ids: list[int], budget_rows: int | None) -> dict:
        steps = IndexOptions()
        base = ("SELECT s.id, s.file, s.modality, s.vec, s.kind, s.loc, f.kind, "
                "CAST(json_extract(f.info, '$.pages') AS INTEGER) FROM segments s JOIN files f ON f.id = s.file "
                "WHERE s.modality != 'none' AND ")
        queries = [(base + "s.id > ? ORDER BY s.id", (after,))]
        for i in range(0, len(extra_ids), 900):
            chunk = extra_ids[i:i + 900]
            queries.insert(0, (base + f"s.id IN ({','.join('?' * len(chunk))})", tuple(chunk)))
        handles = {name: open(folder / f"{name}.bin", "ab") for name in COLUMNS}
        rows = 0
        top = after
        try:
            for sql, args in queries:
                cursor = db.execute(sql, args)
                while True:
                    batch = cursor.fetchmany(4096)
                    batch = [r for r in batch if r[3] is not None and len(r[3]) == DIM * 2]
                    if not batch:
                        break
                    vectors = np.frombuffer(b"".join(r[3] for r in batch), dtype=np.float16).reshape(-1, DIM)
                    q, scales = _quantize(vectors.astype(np.float32))
                    later = [
                        (r[4] != "chunk" and (r[7] or 0) > steps.big_pdf_pages and int(r[5] or 0) > steps.pdf_step)
                        if r[6] == "pdf" else (r[4] == "chunk" and int(r[5] or 0) > steps.doc_step)
                        for r in batch
                    ]
                    handles["vectors"].write(q.tobytes())
                    handles["scales"].write(scales.tobytes())
                    handles["ids"].write(np.array([r[0] for r in batch], dtype=np.int64).tobytes())
                    handles["files"].write(np.array([r[1] for r in batch], dtype=np.int64).tobytes())
                    handles["modality"].write(bytes(MODALITY_CODE.get(r[2], 0) for r in batch))
                    handles["sets_scale"].write(bytes(0 if l else 1 for l in later))
                    handles["kind"].write(bytes(KIND_CODE.get(r[6], 0) for r in batch))
                    handles["dead"].write(bytes(len(batch)))
                    rows += len(batch)
                    top = max(top, max(r[0] for r in batch))
                    time.sleep(0)  # let the UI thread run between blocks (same process)
                    if budget_rows and rows >= budget_rows and sql.endswith("ORDER BY s.id"):
                        break
        finally:
            for h in handles.values():
                h.close()
        return {"rows": rows, "max_id": top}

    def _write_meta(self, meta: dict) -> None:
        tmp = self._meta_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(meta), "utf-8")
        os.replace(tmp, self._meta_path())

    def _sweep(self) -> None:
        """Removes old generations nobody maps any more (Windows refuses while one is mapped: tried again later)."""
        current = str(self._read_meta().get("generation", ""))
        for folder in self.root.iterdir():
            if folder.is_dir() and folder.name.startswith("g") and folder.name != current:
                shutil.rmtree(folder, ignore_errors=True)
