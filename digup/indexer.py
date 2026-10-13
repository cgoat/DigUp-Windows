"""Keeps the index in step with the chosen folders, then reads and embeds whatever is pending.

Model work goes to an embedder (a worker process, made on first use, freed by `close_worker`). Reading files (Pillow,
OCR, PDFium, ffmpeg) runs on a second thread one job ahead, so reading files and embedding them overlap.
"""

from __future__ import annotations

import os
import shutil
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import codeextract, crawler, extractors, media, paths, searchtext
from .model import FileKind, IndexOptions, Modality, Prompts, SegmentKind, WorkerError
from .store import Candidate, FileRecord, IndexStore


class FingerprintMismatch(Exception):
    def __init__(self, indexed: str, now: str):
        super().__init__(f"this index was built with different settings or model:\n  index: {indexed}\n  now:   {now}\n"
                         "re-run with --reindex (vectors from different settings can't be compared)")


@dataclass
class SyncSummary:
    added: int = 0
    changed: int = 0
    moved: int = 0
    removed: int = 0
    unchanged: int = 0
    crawl: crawler.CrawlReport | None = None
    queued: list[int] = field(default_factory=list)


@dataclass
class Event:
    done: int
    total: int
    path: str
    kind: FileKind
    state: str
    seconds: float
    note: str | None


@dataclass
class Job:
    files: list[FileRecord]
    step: range | None = None  # a long file's later step: pages or chunks (from 0)


@dataclass
class Prepared:
    job: Job
    folder: Path
    outcomes: dict[int, tuple[str, str | None]] = field(default_factory=dict)
    payload: tuple = ("nothing",)


def batches(queue: list[FileRecord]) -> list[list[FileRecord]]:
    """Images and screenshots go to the model 8 at a time; other files one by one."""
    out, i = [], 0
    while i < len(queue):
        batch = [queue[i]]
        if batch[0].kind in (FileKind.IMAGE, FileKind.SCREENSHOT):
            while i + len(batch) < len(queue) and len(batch) < 8 and queue[i + len(batch)].kind == batch[0].kind:
                batch.append(queue[i + len(batch)])
        out.append(batch)
        i += len(batch)
    return out


class Indexer:
    def __init__(self, store: IndexStore, options: IndexOptions, make_embedder: Callable[[], object],
                 embedder_source: str | None = None):
        self.store = store
        self.options = options
        self.make_embedder = make_embedder
        self.embedder_source = embedder_source
        self.worker = None
        # When set, an index built with another model or settings is emptied and embedded again (the app);
        # when None, indexing stops with FingerprintMismatch (the CLI without --reindex).
        self.reset_on_model_change: Callable[[str, str], None] | None = None
        self.scratch = paths.scratch() / uuid.uuid4().hex[:12]
        self.scratch.mkdir(parents=True, exist_ok=True)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="digup-read")
        self._jobs = 0

    def close_worker(self) -> None:
        if self.worker is not None:
            self.worker.close()
            self.worker = None

    def finish(self) -> None:
        self.close_worker()
        self._pool.shutdown(wait=False, cancel_futures=True)
        shutil.rmtree(self.scratch, ignore_errors=True)

    # MARK: Sync

    def sync(self, roots: list[str], should_stop=lambda: False) -> SyncSummary:
        """Walks `roots` and reconciles the index: new files queued, changed ones queued again, moved ones re-pathed
        (nothing embedded again), and files that are gone or newly excluded removed.

        The walk streams into a temporary table and the comparison is SQL, so a drive of millions of files costs
        disk, not memory (holding both sides as Python objects took gigabytes). Only the differences come back."""
        roots = [paths.normalize(r) for r in roots]
        report = crawler.CrawlReport()
        summary = SyncSummary(crawl=report)
        db = self.store.db
        db.execute("CREATE TEMP TABLE IF NOT EXISTS scan(key TEXT PRIMARY KEY, path TEXT, kind TEXT, size INTEGER, "
                   "mtime REAL, inode INTEGER, device INTEGER) WITHOUT ROWID")
        db.execute("CREATE TEMP TABLE IF NOT EXISTS known(key TEXT PRIMARY KEY, id INTEGER, path TEXT, kind TEXT, "
                   "size INTEGER, mtime REAL, inode INTEGER, device INTEGER) WITHOUT ROWID")
        db.execute("CREATE TEMP TABLE IF NOT EXISTS gone(id INTEGER PRIMARY KEY, path TEXT, size INTEGER, mtime REAL, "
                   "inode INTEGER, device INTEGER, moved INTEGER DEFAULT 0)")
        db.execute("CREATE INDEX IF NOT EXISTS temp.gone_inode ON gone(device, inode)")
        for table in ("scan", "known", "gone"):
            db.execute(f"DELETE FROM temp.{table}")

        batch = []
        for c in crawler.walk(roots, self.options, report, should_stop):
            batch.append((c.path.lower(), c.path, c.kind.value, c.size, c.mtime, _i64(c.inode), _i64(c.device)))
            if len(batch) >= 5000:
                db.executemany("INSERT OR IGNORE INTO temp.scan VALUES(?, ?, ?, ?, ?, ?, ?)", batch)
                batch.clear()
        if batch:
            db.executemany("INSERT OR IGNORE INTO temp.scan VALUES(?, ?, ?, ?, ?, ?, ?)", batch)
        ranges = " OR ".join("(path >= ? AND path < ?)" for _ in roots) or "0"
        bounds = [v for r in roots for v in (paths.prefix(r), paths.prefix(r) + "\U0010ffff")]
        db.execute("INSERT OR IGNORE INTO temp.known SELECT lower(path), id, path, kind, size, mtime, inode, device "
                   f"FROM files WHERE {ranges}", bounds)

        with self.store.transaction():
            # Changed: the same path with another size, date or kind.
            changed = db.execute(
                "SELECT k.id, s.path, s.kind, s.size, s.mtime, s.inode, s.device FROM temp.scan s "
                "JOIN temp.known k ON k.key = s.key "
                "WHERE k.size != s.size OR abs(k.mtime - s.mtime) > 0.001 OR k.kind != s.kind").fetchall()
            for fid, path, kind, size, mtime, inode, device in changed:
                self.store.requeue_changed(fid, Candidate(path, FileKind(kind), size, mtime, inode, device))
                summary.queued.append(fid)
            summary.changed = len(changed)
            matched = db.execute("SELECT COUNT(*) FROM temp.scan s JOIN temp.known k ON k.key = s.key").fetchone()[0]
            summary.unchanged = matched - summary.changed

            # Gone from where they were: removed, or moved (found again below by file id, size and date).
            db.execute("INSERT INTO temp.gone(id, path, size, mtime, inode, device) "
                       "SELECT id, path, size, mtime, inode, device FROM temp.known "
                       "WHERE key NOT IN (SELECT key FROM temp.scan)")
            new = db.execute("SELECT path, kind, size, mtime, inode, device FROM temp.scan "
                             "WHERE key NOT IN (SELECT key FROM temp.known)")
            while rows := new.fetchmany(2000):
                for path, kind, size, mtime, inode, device in rows:
                    file = Candidate(path, FileKind(kind), size, mtime, inode, device)
                    match = db.execute(
                        "SELECT id FROM temp.gone WHERE device = ? AND inode = ? AND size = ? "
                        "AND abs(mtime - ?) < 0.001 AND moved = 0 LIMIT 1", (device, inode, size, mtime)
                    ).fetchone() if inode else None
                    if match:
                        self.store.move(match[0], path)
                        db.execute("UPDATE temp.gone SET moved = 1 WHERE id = ?", (match[0],))
                        summary.moved += 1
                    else:
                        summary.queued.append(self.store.insert(file))
                        summary.added += 1

            # Only a walk that reached the end says what's gone (a stopped one saw part of the folders), and a root
            # that went away mid-walk (a drive unplugged) looked emptier than it is: its rows stay.
            if report.complete and not should_stop():
                vanished = [r for r in roots if not os.path.isdir(r)]
                keep = " AND ".join("NOT (path >= ? AND path < ?)" for _ in vanished) or "1"
                keep_args = [v for r in vanished for v in (paths.prefix(r), paths.prefix(r) + "\U0010ffff")]
                cursor = db.execute(f"SELECT id FROM temp.gone WHERE moved = 0 AND {keep}", keep_args)
                gone = [r[0] for r in cursor.fetchall()]
                for i in range(0, len(gone), 50_000):
                    self.store.delete_many(gone[i:i + 50_000])
                summary.removed = len(gone)
        for table in ("scan", "known", "gone"):
            db.execute(f"DELETE FROM temp.{table}")
        return summary

    # MARK: Run

    def run(self, limit: int | None = None, only: set[int] | None = None, should_stop=lambda: False,
            on_event: Callable[[Event], None] = lambda e: None) -> dict[str, int]:
        """Processes pending files, screenshots first, newest first. Long PDFs and documents get their first step
        and stay `partial` for `finish_long_files`. A dead worker or a settings mismatch stops the run (pending files
        stay pending, so the next run resumes)."""
        queue = self.store.pending(limit, only)
        states: dict[str, int] = {}
        done = 0

        def committed(_index, job, results, seconds):
            nonlocal done
            for offset, file in enumerate(job.files):
                state, note = results[offset]
                states[state] = states.get(state, 0) + 1
                on_event(Event(done + offset + 1, len(queue), file.path, file.kind, state, seconds, note))
            done += len(job.files)

        self._process([Job(b) for b in batches(queue)], should_stop, committed)
        return states

    def finish_long_files(self, should_stop=lambda: False, on_step: Callable[[Event], None] = lambda e: None):
        """The later pass: reads the rest of long PDFs and documents, a step at a time, each step one transaction."""
        jobs: list[Job] = []
        for file in self.store.unfinished():
            info = self.store.info(file.id)
            size = self.options.pdf_step if file.kind == FileKind.PDF else self.options.doc_step
            start = int(info["read_to"]) if info.get("read_to") else (self.store.last_place(file.id) or 0)
            end = max(int(info.get("read_end") or 0), start + 1)
            while start < end:
                jobs.append(Job([file], range(start, min(start + size, end))))
                start += size
        through, total = [], 0
        for job in jobs:
            total += len(job.step)
            through.append(total)
        finished: set[int] = set()
        states: dict[str, int] = {}

        def committed(index, job, results, seconds):
            state, note = results[0]
            if state != "partial" and job.files[0].id not in finished:
                finished.add(job.files[0].id)
                states[state] = states.get(state, 0) + 1
            on_step(Event(through[index], total, job.files[0].path, job.files[0].kind, state, seconds, note))

        self._process(jobs, should_stop, committed, skip=lambda job: job.files[0].id in finished)
        return states

    def _process(self, jobs: list[Job], should_stop, committed, skip=lambda job: False) -> None:
        import time

        ahead: Future | None = None
        try:
            for index, job in enumerate(jobs):
                if should_stop():
                    break
                started = time.time()
                prepared = (ahead or self._prepare_async(job)).result()
                ahead = self._prepare_async(jobs[index + 1]) if index + 1 < len(jobs) else None
                try:
                    if skip(job):
                        continue
                    results = self._commit(prepared)
                    committed(index, job, results, (time.time() - started) / len(job.files))
                finally:
                    shutil.rmtree(prepared.folder, ignore_errors=True)
        finally:
            if ahead is not None:
                try:
                    shutil.rmtree(ahead.result().folder, ignore_errors=True)
                except Exception:
                    pass

    def _prepare_async(self, job: Job) -> Future:
        self._jobs += 1
        folder = self.scratch / f"batch-{self._jobs}"
        return self._pool.submit(prepare, job, self.options, folder)

    # MARK: The model

    def check_model(self) -> None:
        if self.embedder_source:
            self._adopt(f"{self.embedder_source} | {self.options.fingerprint}")

    def _adopt(self, fingerprint: str) -> None:
        indexed = self.store.meta("fingerprint")
        if not indexed or indexed == fingerprint:
            return
        if self.reset_on_model_change is None:
            raise FingerprintMismatch(indexed, fingerprint)
        self.store.reset_all()
        self.reset_on_model_change(indexed, fingerprint)

    def full_worker(self):
        if self.worker is not None and getattr(self.worker, "alive", True):
            return self.worker
        started = self.make_embedder()
        fingerprint = f"{started.info.vector_source} | {self.options.fingerprint}"
        try:
            self._adopt(fingerprint)
        except Exception:
            started.close()
            raise
        self.store.set_meta("fingerprint", fingerprint)
        self.store.set_meta("vector_source", started.info.vector_source)
        self.worker = started
        return started

    # MARK: Commit (the indexing thread: the model and the database)

    def _commit(self, prepared: Prepared) -> list[tuple[str, str | None]]:
        batch = prepared.job.files
        step = prepared.job.step
        later = step is not None
        if step is not None and 0 in prepared.outcomes:
            return [self._stop_reading(batch[0], step.start, prepared.outcomes[0][1] or prepared.outcomes[0][0])]
        results: list[tuple[str, str | None]] = [("failed", None)] * len(batch)
        for slot, (state, detail) in prepared.outcomes.items():
            self.store.finish(batch[slot].id, state, detail)
            results[slot] = (state, detail)
        kind, *args = prepared.payload
        try:
            if kind == "images":
                self._images(batch, *args, results)
            elif kind == "pdf":
                results[0] = self._pdf(batch[0], *args, later)
            elif kind == "doc":
                results[0] = self._doc(batch[0], *args, later)
            elif kind == "audio":
                results[0] = self._audio(batch[0], *args)
            elif kind == "video":
                results[0] = self._video(batch[0], *args)
            elif kind == "code":
                results[0] = self._code(batch[0], *args)
        except (WorkerError, FingerprintMismatch):
            raise
        except Exception as error:  # noqa: BLE001 - one file's problem is recorded on it
            if step is not None:
                return [self._stop_reading(batch[0], step.start, str(error))]
            for slot, file in enumerate(batch):
                if slot not in prepared.outcomes:
                    self.store.finish(file.id, "failed", str(error)[:500])
                    results[slot] = ("failed", str(error)[:500])
        return results

    def _stop_reading(self, file: FileRecord, place: int, problem: str) -> tuple[str, str]:
        info = self.store.info(file.id)
        info.pop("read_to", None)
        info.pop("read_end", None)
        unit = "page" if file.kind == FileKind.PDF else "chunk"
        self.store.finish(file.id, "done", f"read to {unit} {place}, not after: {problem}", info)
        return "done", f"stopped before {unit} {place + 1}: {problem}"

    def _images(self, batch, screenshot: bool, ready: list, results) -> None:
        if not ready:
            return
        worker = self.full_worker()
        budget = self.options.screenshot_budget if screenshot else self.options.photo_budget
        vectors = worker.embed_images([r[1].path for r in ready], budget)
        with self.store.transaction():
            for (slot, image, ocr_text), embedding in zip(ready, vectors):
                file = batch[slot]
                if embedding.vector is None:
                    self.store.finish(file.id, "failed", embedding.error)
                    results[slot] = ("failed", embedding.error)
                    continue
                self.store.add_segment(file.id, SegmentKind.IMAGE, Modality.IMAGE, vector=embedding.vector)
                # The words on screen are for keyword search; meaning comes from the picture alone.
                if ocr_text:
                    self.store.add_segment(file.id, SegmentKind.OCR, Modality.TEXT, excerpt=searchtext.excerpt(ocr_text),
                                           text=ocr_text)
                info = {"width": str(image.width), "height": str(image.height)}
                if image.taken:
                    info["taken"] = image.taken
                if screenshot:
                    info["ocr_chars"] = str(len(ocr_text))
                self.store.finish(file.id, "done", info=info)
                results[slot] = ("done", f"ocr {len(ocr_text)} chars" if screenshot else None)

    def _pdf(self, file: FileRecord, pages: list, read: range, total: int, title: str | None, later: bool):
        o = self.options
        end = min(total, o.max_pages)
        unread = read.stop < end
        if not pages and not later and not unread:
            self.store.finish(file.id, "skipped", "no pages")
            return "skipped", "no pages"
        title = title or Path(file.path).stem
        text_pages = [p for p in pages if p.image_path is None]
        word_pages = [p for p in text_pages if searchtext.has_words(p.text)]
        image_pages = [p for p in pages if p.image_path is not None]
        garbled = [p for p in pages if p.garbled is not None]
        text_vectors, image_vectors, garbled_vectors = {}, [], []
        if word_pages or image_pages:
            worker = self.full_worker()
            vs = worker.embed_texts([Prompts.document(title, p.text[:4000]) for p in word_pages])
            text_vectors = {p.number: v for p, v in zip(word_pages, vs)}
            image_vectors = worker.embed_images([p.image_path for p in image_pages], o.page_budget)
            garbled_vectors = worker.embed_texts([Prompts.document(title, p.garbled[:4000]) for p in garbled])
        with self.store.transaction():
            for page, vector in zip(garbled, garbled_vectors):
                self.store.add_segment(file.id, SegmentKind.PAGE, Modality.TEXT, loc=page.number, vector=vector)
            for page in text_pages:
                self.store.add_segment(file.id, SegmentKind.PAGE, Modality.TEXT, loc=page.number,
                                       excerpt=searchtext.excerpt(page.text), text=page.text,
                                       vector=text_vectors.get(page.number))
            for page, embedding in zip(image_pages, image_vectors):
                self.store.add_segment(file.id, SegmentKind.PAGE_IMAGE, Modality.IMAGE, loc=page.number,
                                       excerpt=searchtext.excerpt(page.text) if page.text else None, text=page.text,
                                       vector=embedding.vector)
            info = self.store.info(file.id) if later else {}
            info["pages"] = str(total)
            info["scanned_pages"] = str(int(info.get("scanned_pages", 0)) + len(image_pages))
            info["read_to"] = str(read.stop) if unread else None
            info["read_end"] = str(end) if unread else None
            detail = (f"read to page {read.stop} of {total}" if unread
                      else f"the first {end} of {total} pages" if end < total else None)
            self.store.finish(file.id, "partial" if unread else "done", detail, info)
        span = f"pages {read.start + 1}–{read.stop} of {total}" if later else f"{len(pages)}/{total} pages"
        return ("partial" if unread else "done"), f"{span}, {len(image_pages)} as images"

    def _doc(self, file: FileRecord, chunks: list[str], read: range, total: int, characters: int, later: bool):
        if total == 0:
            self.store.finish(file.id, "skipped", "no text")
            return "skipped", "no text"
        title = Path(file.path).stem
        worded = [i for i, c in enumerate(chunks) if searchtext.has_words(c)]
        vectors = self.full_worker().embed_texts([Prompts.document(title, chunks[i]) for i in worded]) if worded else []
        vector = dict(zip(worded, vectors))
        end = min(total, self.options.max_chunks)
        unread = read.stop < end
        with self.store.transaction():
            for offset, chunk in enumerate(chunks):
                self.store.add_segment(file.id, SegmentKind.CHUNK, Modality.TEXT, loc=read.start + offset + 1,
                                       excerpt=searchtext.excerpt(chunk), text=chunk, vector=vector.get(offset))
            info = {"chunks": str(end), "chars": str(characters),
                    "read_to": str(read.stop) if unread else None, "read_end": str(end) if unread else None}
            detail = f"read to chunk {read.stop} of {end}" if unread else (f"the first {end} chunks" if end < total else None)
            self.store.finish(file.id, "partial" if unread else "done", detail, info)
        note = f"{len(chunks)} chunks" + (f", {len(chunks) - len(worded)} for keywords only" if len(worded) < len(chunks) else "")
        return ("partial" if unread else "done"), note

    def _code(self, file: FileRecord, code: codeextract.CodeFile):
        if not code.pieces:
            self.store.finish(file.id, "skipped", "no code")
            return "skipped", "no code"
        worded = [i for i, p in enumerate(code.pieces) if searchtext.has_words(p.text)]
        vectors = self.full_worker().embed_texts(
            [Prompts.document(code.title, code.pieces[i].text) for i in worded]) if worded else []
        vector = dict(zip(worded, vectors))
        with self.store.transaction():
            for offset, piece in enumerate(code.pieces):
                self.store.add_segment(file.id, SegmentKind.LINES if code.cells is None else SegmentKind.CELLS,
                                       Modality.TEXT, loc=piece.first, loc_end=piece.last,
                                       excerpt=codeextract.summary(piece.text), text=piece.text,
                                       vector=vector.get(offset))
            info = {"lines": str(code.lines), "cells": str(code.cells) if code.cells else None, "repo": code.repo}
            self.store.finish(file.id, "done", info=info)
        return "done", f"{len(code.pieces)} stretches"

    def _audio(self, file: FileRecord, pieces: list, duration: float):
        if not pieces:
            self.store.finish(file.id, "skipped", "silent or shorter than 2 s")
            return "skipped", "silent or too short"
        vectors = self.full_worker().embed_audio([p.path for p in pieces])
        with self.store.transaction():
            for piece, embedding in zip(pieces, vectors):
                self.store.add_segment(file.id, SegmentKind.AUDIO, Modality.AUDIO, loc=piece.start, loc_end=piece.end,
                                       vector=embedding.vector)
            self.store.finish(file.id, "done", info={"duration": f"{duration:.1f}"})
        return "done", f"{len(pieces)} windows, {int(duration)} s"

    def _video(self, file: FileRecord, frames: list, sound: list, duration: float):
        if not frames and not sound:
            self.store.finish(file.id, "skipped", "no frames or sound")
            return "skipped", "no frames or sound"
        worker = self.full_worker()
        frame_vectors = worker.embed_images([f.path for f in frames], self.options.frame_budget)
        sound_vectors = worker.embed_audio([p.path for p in sound])
        with self.store.transaction():
            for frame, embedding in zip(frames, frame_vectors):
                self.store.add_segment(file.id, SegmentKind.FRAME, Modality.IMAGE, loc=frame.start, loc_end=frame.end,
                                       vector=embedding.vector)
            for piece, embedding in zip(sound, sound_vectors):
                self.store.add_segment(file.id, SegmentKind.AUDIO, Modality.AUDIO, loc=piece.start, loc_end=piece.end,
                                       vector=embedding.vector)
            self.store.finish(file.id, "done", info={"duration": f"{duration:.1f}", "frames": str(len(frames)),
                                                     "sound_windows": str(len(sound))})
        return "done", f"{len(frames)} frames, {len(sound)} sound windows, {int(duration)} s"


# MARK: Prepare (the reading thread: files only, no database, no model)


def prepare(job: Job, options: IndexOptions, folder: Path) -> Prepared:
    batch = job.files
    prepared = Prepared(job, folder)
    path = batch[0].path
    try:
        folder.mkdir(parents=True, exist_ok=True)
        kind = batch[0].kind
        if kind in (FileKind.IMAGE, FileKind.SCREENSHOT):
            screenshot = kind == FileKind.SCREENSHOT
            ready = []
            for slot, file in enumerate(batch):
                try:
                    image, full = extractors.prepare_image(file.path, 1536 if screenshot else 1024,
                                                           options.min_image_side, folder, png=screenshot)
                    text = extractors.ocr(full) if screenshot else ""
                    ready.append((slot, image, text))
                except extractors.TooSmall as error:
                    prepared.outcomes[slot] = ("skipped", str(error))
                except Exception as error:  # noqa: BLE001
                    prepared.outcomes[slot] = ("failed", f"unreadable: {error}"[:300])
            prepared.payload = ("images", screenshot, ready)
        elif kind == FileKind.PDF:
            wanted = {}

            def pick(total: int) -> range:
                r = job.step or (range(0, options.pdf_step) if total > options.big_pdf_pages
                                 else range(0, options.max_pages))
                r = range(max(r.start, 0), min(r.stop, total, options.max_pages))
                wanted["r"] = r
                return r

            pages, total, title = extractors.pdf_pages(path, pick, options.min_page_text, 1400, folder)
            prepared.payload = ("pdf", pages, wanted.get("r", range(0)), total, title)
        elif kind == FileKind.DOC:
            text = extractors.doc_text(path)
            chunks = extractors.chunks(text, limit=options.max_chunks + 1)
            r = job.step or range(0, options.doc_step)
            r = range(max(r.start, 0), min(r.stop, len(chunks), options.max_chunks))
            prepared.payload = ("doc", chunks[r.start:r.stop], r, len(chunks), len(text))
        elif kind == FileKind.AUDIO:
            pieces, duration = media.audio_windows(path, options.audio_window, options.audio_hop, folder)
            prepared.payload = ("audio", pieces, duration)
        elif kind == FileKind.VIDEO:
            probe = media.probe(path)
            frames = media.keyframes(path, options.frame_interval, 1024, folder, probe.duration) if probe.has_video else []
            sound = media.audio_windows(path, options.audio_window, options.audio_hop, folder)[0] if probe.has_audio else []
            prepared.payload = ("video", frames, sound, probe.duration)
        elif kind == FileKind.CODE:
            prepared.payload = ("code", codeextract.read(path, options))
    except codeextract.LeftOut as error:
        prepared.outcomes[0] = ("skipped", str(error))
        prepared.payload = ("nothing",)
    except Exception as error:  # noqa: BLE001 - recorded on the file
        message = str(error) or type(error).__name__
        for slot in range(len(batch)):
            prepared.outcomes.setdefault(slot, ("failed", message[:500]))
        prepared.payload = ("nothing",)
    return prepared


def _i64(value: int) -> int:
    value &= (1 << 64) - 1
    return value - (1 << 64) if value >= 1 << 63 else value
