"""Searching from the app: the query encoder (the model's text part, about 300 MB, in a worker process that starts when
the panel opens and quits a few minutes after the last search), and searches run off the UI thread.

A search shows keyword matches as you type, then the full results once your words have a vector.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

from PySide6.QtCore import QObject, QTimer, Signal

from .. import paths
from ..model import CodeQuery, FileKind, Prompts
from ..searcher import Searcher, SearchHit, strong_count, terms
from ..store import CODE_FILE, MAIN_FILE, IndexStore

ENCODER_IDLE_SECONDS = 300


@dataclass
class Results:
    query: str
    hits: list[SearchHit]
    strong: int
    final: bool  # meaning included (False: words only, while the vector is on its way)
    code: bool
    words: list[str]
    millis: float
    more: bool = False  # there are more results past these ("Show more")


class QueryService(QObject):
    results = Signal(object)  # Results
    encoder_state = Signal(str)  # "starting" | "ready" | "stopped" | an error

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._encoder = None
        self._encoder_starting = False
        self._last_used = time.time()
        self._searchers: dict[str, Searcher] = {}
        self._stale: set[str] = set()
        self._generation = 0
        self._vectors: dict[str, list[float]] = {}
        self._image_vectors: dict[tuple, list[float]] = {}  # (path, size, mtime) -> vector: "Show more" is instant
        self._media = None  # the full model (image encoder too), only while searching by picture
        self._media_used = 0.0
        self._media_lock = threading.Lock()
        self._idle = QTimer(self)
        self._idle.setInterval(30_000)
        self._idle.timeout.connect(self._close_if_idle)
        self._idle.start()

    # MARK: The encoder

    def warm_up(self) -> None:
        """Starts the query encoder in the background (the panel is opening)."""
        with self._lock:
            if self._encoder is not None or self._encoder_starting:
                return
            self._encoder_starting = True
        threading.Thread(target=self._start_encoder, name="digup-encoder", daemon=True).start()

    def _start_encoder(self) -> None:
        from .. import downloads
        from ..worker import EmbeddingWorker

        self.encoder_state.emit("starting")
        try:
            if not downloads.locate_models():
                raise RuntimeError("the model isn't downloaded yet")
            encoder = EmbeddingWorker(text_only=True)
        except Exception as error:  # noqa: BLE001
            with self._lock:
                self._encoder_starting = False
            self.encoder_state.emit(f"error: {error}")
            return
        with self._lock:
            self._encoder = encoder
            self._encoder_starting = False
            self._last_used = time.time()
        self.encoder_state.emit("ready")

    def _encode(self, prompt: str, wait: float = 20.0) -> list[float] | None:
        if prompt in self._vectors:
            return self._vectors[prompt]
        self.warm_up()
        deadline = time.time() + wait
        while True:
            with self._lock:
                encoder, starting = self._encoder, self._encoder_starting
            if encoder is not None or not starting or time.time() > deadline:
                break
            time.sleep(0.05)
        if encoder is None:
            return None
        try:
            vector = encoder.embed_texts([prompt])[0]
        except Exception:  # noqa: BLE001 - the encoder died: start a new one next time
            with self._lock:
                self._encoder = None
            return None
        self._last_used = time.time()
        if len(self._vectors) > 200:
            self._vectors.clear()
        self._vectors[prompt] = vector
        return vector

    def _encode_image(self, path: str) -> list[float] | None:
        """A picture as a query: the full model (865 MB) in its own worker, started on first use and stopped when the
        query encoder is. At the screenshots' budget (280), so small details count."""
        from ..worker import EmbeddingWorker

        with self._media_lock:
            if self._media is None or not self._media.alive:
                self.encoder_state.emit("starting the picture encoder")
                try:
                    self._media = EmbeddingWorker(text_only=False)
                except Exception as error:  # noqa: BLE001
                    self.encoder_state.emit(f"error: {error}")
                    return None
            self._media_used = time.time()
            try:
                result = self._media.embed_images([path], 280)[0]
            except Exception:  # noqa: BLE001
                self._media = None
                return None
        if result.vector is None:
            self.encoder_state.emit(f"error: {result.error}")
        return result.vector

    def _close_if_idle(self) -> None:
        with self._media_lock:
            media = self._media if self._media and time.time() - self._media_used > ENCODER_IDLE_SECONDS / 2 else None
            if media:
                self._media = None
        if media:
            threading.Thread(target=media.close, daemon=True).start()
        with self._lock:
            encoder = self._encoder
            idle = time.time() - self._last_used > ENCODER_IDLE_SECONDS
            if encoder is None or not idle:
                return
            self._encoder = None
        threading.Thread(target=encoder.close, daemon=True).start()
        for searcher in self._searchers.values():
            searcher.release_vectors()  # keyword search keeps working; vectors load again on the next search
        self.encoder_state.emit("stopped")

    def close(self) -> None:
        with self._lock:
            encoder, self._encoder = self._encoder, None
        with self._media_lock:
            media, self._media = self._media, None
        for worker in (encoder, media):
            if worker:
                worker.close()

    # MARK: Searching

    def index_changed(self, name: str) -> None:
        self._stale.add(name)

    def _searcher(self, name: str) -> Searcher | None:
        if not (paths.index() / name).exists():
            return None
        searcher = self._searchers.get(name)
        if searcher is None:
            searcher = Searcher(IndexStore(paths.index(), name), load_vectors=False)
            self._searchers[name] = searcher
        elif name in self._stale:
            self._stale.discard(name)
            if searcher.vectors_loaded:
                searcher.reload()
            else:
                searcher._reload_files()
        return searcher

    PAGE = 40  # results per page; "Show more" asks for the next one

    def search(self, text: str, kinds: set[FileKind] | None = None, limit: int = PAGE) -> None:
        """Searches in the background; `results` fires once with words, then with meaning. Past the first page
        ("Show more") it fires once, with meaning: the query's vector is cached, so that's quick."""
        self._generation += 1
        generation = self._generation
        threading.Thread(target=self._run, args=(text, kinds, generation, limit), daemon=True).start()

    def _run(self, text: str, kinds, generation: int, limit: int = PAGE) -> None:
        code_text = CodeQuery.text_of(text)
        code = code_text is not None
        query = code_text if code else text
        if not query.strip():
            self.results.emit(Results(text, [], 0, True, code, [], 0))
            return
        name = CODE_FILE if code else MAIN_FILE
        if not code and self._run_people(text, query, kinds, generation, limit):
            return
        words = terms(query)
        with self._lock_search():
            searcher = self._searcher(name)
            if searcher is None:
                self.results.emit(Results(text, [], 0, True, code, words, 0))
                return
            started = time.time()
            quick = [] if limit > self.PAGE else searcher.search(query, None, kinds=kinds, limit=30, prefix_last=True)
            if quick and generation == self._generation:
                self.results.emit(Results(text, quick, strong_count(quick), False, code, words,
                                          (time.time() - started) * 1000))
        vector = self._encode(Prompts.code_query(query) if code else Prompts.query(query))
        if generation != self._generation:
            return
        with self._lock_search():
            started = time.time()
            hits = searcher.search(query, vector, kinds=kinds, limit=limit + 1, prefix_last=vector is None)
        more, hits = len(hits) > limit, hits[:limit]
        if generation == self._generation:
            self.results.emit(Results(text, hits, strong_count(hits), True, code, words, (time.time() - started) * 1000,
                                      more))

    # MARK: People

    def _run_people(self, text: str, query: str, kinds, generation: int, limit: int) -> bool:
        """A query that names someone ("Srinivas", "Srinivas at the beach"): their photos, ranked by the rest of the
        words when there are any. False when it names nobody."""
        from .. import people

        with self._lock_search():
            searcher = self._searcher(MAIN_FILE)
            if searcher is None:
                return False
            try:
                groups, rest = people.split_names(searcher.store, query)
            except Exception:  # noqa: BLE001 - an index from before faces
                return False
            if not groups:
                return False
            files = people.files_of(searcher.store, groups)
        started = time.time()
        if not rest:
            with self._lock_search():
                rows = searcher.store.db.execute(
                    f"SELECT id FROM files WHERE id IN ({','.join('?' * len(files))}) ORDER BY mtime DESC",
                    files).fetchall() if files else []
                ordered = [r[0] for r in rows]
                hits = searcher.hits_for_files(ordered[:limit + 1])
            if kinds is not None:
                hits = [h for h in hits if h.kind in kinds]
        else:
            vector = self._encode(Prompts.query(rest))
            if generation != self._generation:
                return True
            with self._lock_search():
                # Every photo of theirs qualifies; the words (by meaning, and exactly) put the fitting ones first.
                hits = searcher.search(rest, vector, kinds=kinds, limit=limit + 1, z_gate=-1e9,
                                       only_files=set(files))
        more, hits = len(hits) > limit, hits[:limit]
        if generation == self._generation:
            self.results.emit(Results(text, hits, len(hits), True, False, terms(rest), (time.time() - started) * 1000,
                                      more))
        return True

    def search_group(self, group: int, key: str, kinds=None, limit: int = PAGE) -> None:
        """A group's photos, newest first."""
        self._generation += 1
        generation = self._generation

        def run():
            from .. import people

            started = time.time()
            with self._lock_search():
                searcher = self._searcher(MAIN_FILE)
                files = people.files_of(searcher.store, [group]) if searcher else []
                rows = searcher.store.db.execute(
                    f"SELECT id FROM files WHERE id IN ({','.join('?' * len(files))}) ORDER BY mtime DESC",
                    files).fetchall() if files else []
                hits = searcher.hits_for_files([r[0] for r in rows][:limit + 1]) if searcher else []
            if kinds is not None:
                hits = [h for h in hits if h.kind in kinds]
            more, hits = len(hits) > limit, hits[:limit]
            if generation == self._generation:
                self.results.emit(Results(key, hits, len(hits), True, False, [], (time.time() - started) * 1000, more))

        threading.Thread(target=run, daemon=True).start()

    def search_person(self, face: int, key: str, kinds=None, limit: int = PAGE) -> None:
        """Photos with this face in them (compared with every face, not only its group)."""
        self._generation += 1
        generation = self._generation

        def run():
            from .. import people

            started = time.time()
            with self._lock_search():
                searcher = self._searcher(MAIN_FILE)
                found = people.same_face(searcher.store, face) if searcher else []
                hits = searcher.hits_for_files([f for f, _ in found[:limit + 1]], [s for _, s in found[:limit + 1]]) \
                    if searcher else []
            if kinds is not None:
                hits = [h for h in hits if h.kind in kinds]
            more, hits = len(hits) > limit, hits[:limit]
            if generation == self._generation:
                self.results.emit(Results(key, hits, len(hits), True, False, [], (time.time() - started) * 1000, more))

        threading.Thread(target=run, daemon=True).start()

    # MARK: Searching by picture

    def search_image(self, path: str, kinds: set[FileKind] | None = None, key: str | None = None,
                     limit: int = PAGE) -> None:
        """Pictures, video moments and pages that look like the image at `path` (any picture, indexed or not)."""
        self._generation += 1
        threading.Thread(target=self._run_image, args=(path, kinds, self._generation, key or f"image:{path}", limit),
                         daemon=True).start()

    def search_similar(self, hit: SearchHit, kinds: set[FileKind] | None = None, key: str | None = None,
                       limit: int = PAGE) -> None:
        """Results like `hit`, from the vector the index already holds for it (no model needed)."""
        self._generation += 1
        threading.Thread(target=self._run_similar,
                         args=(hit, kinds, self._generation, key or f"similar:{hit.path}", limit), daemon=True).start()

    def _run_image(self, path: str, kinds, generation: int, key: str, limit: int = PAGE) -> None:
        started = time.time()
        try:
            st = os.stat(path)
            cache_key = (path, st.st_size, st.st_mtime)
        except OSError:
            cache_key = None
        vector = self._image_vectors.get(cache_key) if cache_key else None
        if vector is None:
            vector = self._encode_image(path)
            if vector is not None and cache_key:
                if len(self._image_vectors) > 50:
                    self._image_vectors.clear()
                self._image_vectors[cache_key] = vector
        if generation != self._generation:
            return
        if vector is None:
            self.results.emit(Results(key, [], 0, True, False, [], 0))
            return
        self._search_vector(vector, kinds, generation, key, exclude=path, started=started, limit=limit)

    def _run_similar(self, hit: SearchHit, kinds, generation: int, key: str, limit: int = PAGE) -> None:
        started = time.time()
        name = CODE_FILE if hit.kind == FileKind.CODE else MAIN_FILE
        with self._lock_search():
            searcher = self._searcher(name)
            vector = searcher.vector_of(hit.segment_id, hit.path) if searcher else None
        if vector is None:
            self.results.emit(Results(key, [], 0, True, False, [], 0))
            return
        self._search_vector(vector, kinds, generation, key, exclude=hit.path, started=started,
                            name=name, limit=limit)

    def _search_vector(self, vector, kinds, generation: int, key: str, exclude: str, started: float,
                       name: str = MAIN_FILE, limit: int = PAGE) -> None:
        with self._lock_search():
            searcher = self._searcher(name)
            hits = searcher.search("", vector, kinds=kinds, limit=limit + 2) if searcher else []
        same = os.path.normcase(os.path.abspath(exclude))
        hits = [h for h in hits if os.path.normcase(h.path) != same]
        more, hits = len(hits) > limit, hits[:limit]
        if generation == self._generation:
            self.results.emit(Results(key, hits, strong_count(hits), True, name == CODE_FILE, [],
                                      (time.time() - started) * 1000, more))

    _search_lock = threading.Lock()

    def _lock_search(self):
        return self._search_lock
