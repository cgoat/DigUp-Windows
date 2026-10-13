"""Hybrid search: meaning (EmbeddingGemma's vectors) + words (FTS5).

Meaning is the model's cosine similarity, put on one scale per query (`meaning`): every vector is scored anyway, so the
distribution comes for free, and only segments that stand out (`z_gate`) count as meaning matches. That also gives an
honest "nothing found". Words rescue what a vector can't hold exactly: codes, IDs and names.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from . import codeextract, searchtext
from .model import DIM, FileKind, SegmentKind
from .store import IndexStore


@dataclass
class SearchHit:
    path: str
    kind: FileKind
    score: float
    z: float | None
    cosine: float | None
    keyword_rank: int | None
    segment: SegmentKind
    segment_id: int
    loc: float | None  # PDF page, seconds into audio/video, first line of code
    loc_end: float | None
    excerpt: str | None
    modified: float
    size: int
    info: dict = field(default_factory=dict)
    focus: float | None = None  # code: the line to open at


STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "and", "or", "my", "me", "that", "this",
    "these", "those", "from", "about", "where", "what", "which", "who", "when", "is", "was", "are", "were", "be",
    "it", "its", "some", "any", "find", "show", "file", "files", "photo", "photos", "picture", "image", "images",
    "screenshot", "screenshots", "video", "videos", "pdf", "document", "documents", "recording",
    "এবং", "ও", "এই", "সেই", "একটি", "আমার", "থেকে", "জন্য", "সাথে", "মধ্যে", "কোথায়", "কোন", "কী", "কি", "যে",
    "ছবি", "স্ক্রিনশট", "ভিডিও", "ফাইল", "পিডিএফ",
    "في", "من", "على", "الى", "عن", "مع", "هذا", "هذه", "ذلك", "التي", "الذي", "اين", "ما",
    "صورة", "صور", "لقطة", "شاشة", "فيديو", "ملف", "مستند",
}

KIND_WORDS = {
    "screenshot": FileKind.SCREENSHOT, "screenshots": FileKind.SCREENSHOT, "photo": FileKind.IMAGE,
    "photos": FileKind.IMAGE, "picture": FileKind.IMAGE, "pictures": FileKind.IMAGE, "image": FileKind.IMAGE,
    "images": FileKind.IMAGE, "pdf": FileKind.PDF, "pdfs": FileKind.PDF, "paper": FileKind.PDF,
    "papers": FileKind.PDF, "document": FileKind.DOC, "documents": FileKind.DOC, "note": FileKind.DOC,
    "notes": FileKind.DOC, "video": FileKind.VIDEO, "videos": FileKind.VIDEO, "clip": FileKind.VIDEO,
    "movie": FileKind.VIDEO, "audio": FileKind.AUDIO, "song": FileKind.AUDIO, "podcast": FileKind.AUDIO,
    "voice": FileKind.AUDIO, "sound": FileKind.AUDIO,
    "স্ক্রিনশট": FileKind.SCREENSHOT, "ছবি": FileKind.IMAGE, "পিডিএফ": FileKind.PDF, "নোট": FileKind.DOC,
    "ভিডিও": FileKind.VIDEO, "অডিও": FileKind.AUDIO, "গান": FileKind.AUDIO, "রেকর্ডিং": FileKind.AUDIO,
    "لقطة": FileKind.SCREENSHOT, "صورة": FileKind.IMAGE, "صور": FileKind.IMAGE, "مستند": FileKind.DOC,
    "ملاحظات": FileKind.DOC, "فيديو": FileKind.VIDEO, "مقطع": FileKind.VIDEO, "صوت": FileKind.AUDIO,
    "تسجيل": FileKind.AUDIO, "اغنية": FileKind.AUDIO,
}

WORD_WEIGHT = 2.0
CODE_WORD_WEIGHT = 1.0


def terms(text: str) -> list[str]:
    """The content words of a query, normalized as keyword search keeps its text; kind words and stopwords dropped."""
    seen: set[str] = set()
    out = []
    for word in searchtext.split_words(searchtext.normalized(text).lower()):
        if len(word) >= 2 and word not in STOPWORDS and word not in seen:
            seen.add(word)
            out.append(word)
    return out


def patterns(words: list[str], prefix_last: bool) -> list[str]:
    return [
        f'"{w}"' + ("*" if (prefix_last and i == len(words) - 1) or searchtext.inflects(w) else "")
        for i, w in enumerate(words)
    ]


def fts_query(text: str, prefix_last: bool = False) -> str | None:
    words = terms(text)
    return " OR ".join(patterns(words, prefix_last)) if words else None


def is_lookup(words: list[str]) -> bool:
    """Codes, IDs and long numbers mean nothing to an embedding: such a query weighs words double, meaning half."""
    return bool(words) and all(
        any(c.isdigit() for c in w) and (any(c.isalpha() for c in w) or len(w) >= 4) for w in words
    )


def kind_hints(text: str) -> set[FileKind]:
    out = set()
    for word in searchtext.split_words(searchtext.normalized(text).lower()):
        if word in KIND_WORDS:
            out.add(KIND_WORDS[word])
            continue
        for key, kind in KIND_WORDS.items():
            if searchtext.inflects(key) and len(key) >= 3 and word.startswith(key):
                out.add(kind)
                break
    return out


def strong_count(hits: list[SearchHit], ratio: float = 0.5) -> int:
    """How many hits are strong (at least `ratio` x the top score); the rest fold under "weaker matches"."""
    if not hits or hits[0].score <= 0:
        return len(hits)
    top = hits[0].score
    n = 0
    for h in hits:
        if h.score < top * ratio:
            break
        n += 1
    return n


def meaning(scores: np.ndarray, modality: np.ndarray, sets_scale: np.ndarray | None = None) -> np.ndarray:
    """Each vector's cosine minus its modality's average, in standard deviations of all of them together. The scale
    comes from the vectors in `sets_scale` (later pages of long files are scored on it but don't move it)."""
    use = np.ones(len(scores), dtype=bool) if sets_scale is None else sets_scale
    s = scores.astype(np.float64)
    overall = s[use].mean() if use.any() else 0.0
    means = np.full(len(s), overall)
    for m in np.unique(modality):
        group = (modality == m) & use
        if group.sum() >= 20:
            means[modality == m] = s[group].mean()
    centered = s - means
    base = centered[use] if use.any() else centered
    mean = base.mean() if base.size else 0.0
    deviation = max(float(np.sqrt(((base - mean) ** 2).mean())) if base.size else 0.0, 0.02)
    return (centered - mean) / deviation


KIND_CODES = {kind: i + 1 for i, kind in enumerate(FileKind)}
KINDS_BY_CODE = {i + 1: kind for i, kind in enumerate(FileKind)}


class _Best:
    """Each file's best-scoring vector row for one query, as arrays sorted by file id: (z, cosine, row) per file
    without a Python object for each of the 100,000 files that score above average on a big index."""

    def __init__(self, files=None, z=None, cosine=None, rows=None, kinds=None):
        empty = np.zeros(0)
        self.files = files if files is not None else empty.astype(np.int64)
        self.z, self.cosine = (z if z is not None else empty), (cosine if cosine is not None else empty)
        self.rows = rows if rows is not None else empty.astype(np.int64)
        self.kinds = kinds if kinds is not None else empty.astype(np.uint8)

    def _index(self, file: int) -> int:
        i = int(np.searchsorted(self.files, file))
        return i if i < len(self.files) and self.files[i] == file else -1

    def get(self, file: int):
        i = self._index(file)
        return None if i < 0 else (float(self.z[i]), float(self.cosine[i]), int(self.rows[i]))

    def __contains__(self, file: int) -> bool:
        return self._index(file) >= 0

    def __getitem__(self, file: int):
        found = self.get(file)
        if found is None:
            raise KeyError(file)
        return found

    def above(self, gate: float) -> list[int]:
        return self.files[self.z >= gate].tolist()

    def kind_of(self, file: int):
        i = self._index(file)
        return None if i < 0 else KINDS_BY_CODE.get(int(self.kinds[i]))


class Searcher:
    """Meaning from the vector cache on disk (`VectorCache`, memory-mapped), words from FTS5. Nothing big is held:
    the cache maps the vectors, and file kinds come from the cache's rows and the database."""

    def __init__(self, store: IndexStore, load_vectors: bool = True, update_cache: bool = True):
        from .vectorcache import VectorCache

        self.store = store
        self.cache = VectorCache(store)
        self.update_cache = update_cache  # bring the cache up to date before searching (False: the engine does it)
        self.vector_source: str | None = store.meta("vector_source")
        self.vectors_loaded = False
        if load_vectors:
            self.reload()

    @property
    def is_code(self) -> bool:
        return self.store.is_code

    @property
    def vector_count(self) -> int:
        return int(self.cache.alive().sum()) if self.vectors_loaded and self.cache.n else 0

    def reload(self) -> None:
        """Catches the cache up with the index (appending what's new: quick after the first time) and maps it."""
        if self.update_cache:
            self.cache.update()
        self.cache.open()
        self.vector_source = self.store.meta("vector_source")
        self.vectors_loaded = True

    def release_vectors(self) -> None:
        self.cache.close()
        self.vectors_loaded = False

    def _reload_files(self) -> None:
        self.vector_source = self.store.meta("vector_source")

    def score(self, query) -> np.ndarray:
        if not self.cache.n:
            return np.zeros(0, dtype=np.float32)
        return self.cache.score(query)

    def _kinds_of(self, files) -> dict[int, FileKind]:
        out: dict[int, FileKind] = {}
        files = list(files)
        for i in range(0, len(files), 900):
            chunk = files[i:i + 900]
            for fid, kind in self.store.db.execute(
                    f"SELECT id, kind FROM files WHERE id IN ({','.join('?' * len(chunk))})", chunk):
                out[fid] = FileKind(kind)
        return out

    def search(self, text: str, query_vector=None, kinds: set[FileKind] | None = None, limit: int = 20,
               z_gate: float = 2.0, word_weight: float | None = None, prefix_last: bool = False) -> list[SearchHit]:
        word_weight = word_weight if word_weight is not None else (CODE_WORD_WEIGHT if self.is_code else WORD_WEIGHT)
        if query_vector is not None and not self.vectors_loaded:
            self.reload()
        file_kind: dict[int, FileKind] = {}

        # Meaning: score every vector on this query's scale, keep each file's best segment (vectorized: a million
        # rows take a fraction of a second, all cores).
        words_ = terms(text)
        ident = codeextract.identifier(text) if self.is_code else None
        pats = [codeextract.phrase(ident)] if ident else patterns(words_, prefix_last)
        db = self.store.db

        def fetch_word_rows():
            try:
                return db.execute(
                    "SELECT s.file, s.id, snippet(fts, 1, '«', '»', '…', 12), fts.original FROM fts "
                    "JOIN segments s ON s.id = fts.rowid WHERE fts MATCH ? ORDER BY bm25(fts, 3.0, 1.0) LIMIT ?",
                    (" OR ".join(pats), max(1000, limit * 10)),
                ).fetchall()
            except Exception:  # noqa: BLE001 - a query FTS5 can't parse: no word matches
                return []

        from .vectorcache import _pool

        word_rows = _pool.submit(fetch_word_rows) if pats else None
        best = _Best()
        row_segment = None
        if query_vector is not None and len(query_vector) == DIM and self.cache.n:
            cols = self.cache.cols
            n = self.cache.n
            alive = cols["dead"][:n] == 0
            scores = self.score(query_vector)
            modality = np.asarray(cols["modality"][:n])
            z = meaning(scores, modality, (np.asarray(cols["sets_scale"][:n]) == 1) & alive)
            keep = alive & (z > 0)  # below the average: it neither qualifies nor adds to a score
            if kinds is not None:
                keep &= np.isin(np.asarray(cols["kind"][:n]), [KIND_CODES[k] for k in kinds])
            rows = np.nonzero(keep)[0]
            if rows.size:
                order = rows[np.argsort(-z[rows], kind="stable")]
                uniq, first = np.unique(np.asarray(cols["files"])[order], return_index=True)  # each file's best row
                top = order[first]
                best = _Best(uniq, z[top], scores[top], top, np.asarray(cols["kind"])[top])
                row_segment = cols["ids"]

        def kind_of(file: int):
            return file_kind.get(file) or best.kind_of(file)

        def allowed(file: int) -> bool:
            return kinds is None or kind_of(file) in kinds

        # Words: FTS5 with Porter stemming over names, folders, OCR, page and document text (fetched above).
        word_hits: list[tuple[int, int, str | None, str | None]] = []
        coverage: dict[int, float] = {}
        if pats:
            seen: set[int] = set()
            rows = word_rows.result()
            if kinds is not None:
                file_kind.update(self._kinds_of({r[0] for r in rows if r[0] not in best} - set(file_kind)))
            for file, seg, snip, original in rows:
                if len(word_hits) >= max(100, limit) or not allowed(file) or file in seen:
                    continue
                seen.add(file)
                word_hits.append((file, seg, snip, original))
            if word_hits:
                files = ",".join(str(h[0]) for h in word_hits)
                for pattern in pats:
                    try:
                        for (file,) in db.execute(
                            "SELECT DISTINCT s.file FROM fts JOIN segments s ON s.id = fts.rowid "
                            f"WHERE fts MATCH ? AND s.file IN ({files})", (pattern,)
                        ):
                            coverage[file] = coverage.get(file, 0) + 1 / len(pats)
                    except Exception:
                        pass

        # A file qualifies by standing out in meaning, or by containing at least half of the query's words.
        lookup = is_lookup(words_) or ident is not None
        meaning_weight, words_weight = (0.5, word_weight * 2) if lookup else (1.0, word_weight)
        hinted = kind_hints(text)
        candidates = set(best.above(z_gate)) | {f for f, s in coverage.items() if s >= 0.5}
        if hinted:
            file_kind.update(self._kinds_of({f for f in candidates if f not in best} - set(file_kind)))
        scored: dict[int, float] = {}
        for file in candidates:
            s = meaning_weight * max(best[file][0] if file in best else 0, 0) + words_weight * coverage.get(file, 0) ** 1.5
            if kind_of(file) in hinted:
                s += 1
            scored[file] = s

        word_rank: dict[int, tuple[int, tuple[int, int, str | None]]] = {}
        for i, (file, seg, snip, original) in enumerate(word_hits):
            word_rank.setdefault(file, (i + 1, (file, seg, (snip, original))))
        quoted: dict[int, str | None] = {}

        def snippet_of(match) -> str | None:
            snip, original = match[2]
            if match[1] not in quoted:
                quoted[match[1]] = (searchtext.snippet(original, words_) if original else None) or snip
            return quoted[match[1]]
        hits: list[SearchHit] = []
        for file, s in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]:
            rank = len(hits)
            semantic = best.get(file)
            keyword = word_rank.get(file)
            use_meaning = semantic is not None and (semantic[0] >= z_gate or keyword is None)
            if use_meaning:
                segment_id = int(row_segment[semantic[2]])
            elif keyword:
                segment_id = keyword[1][1]
            else:
                continue
            focus = None
            if self.is_code:
                segment_id, focus = self._code_place(file, segment_id, ident if rank < 10 else None, text, words_,
                                                     (keyword is not None and keyword[1][1] == segment_id) or ident is not None)
            row = db.execute("SELECT path, kind, mtime, size, info FROM files WHERE id = ?", (file,)).fetchone()
            if not row:
                continue
            seg = db.execute("SELECT kind, loc, loc_end, excerpt FROM segments WHERE id = ?", (segment_id,)).fetchone()
            seg_kind, loc, loc_end, stored = (seg if seg else ("name", None, None, None))
            snippet = None
            marked = snippet_of(keyword[1]) if keyword else None
            if keyword and keyword[1][1] == segment_id and marked and "«" in marked:
                snippet = searchtext.excerpt(marked, 400)
            if snippet is None and stored is None and loc is None and marked and "«" in marked:
                matched = db.execute("SELECT kind FROM segments WHERE id = ?", (keyword[1][1],)).fetchone()
                if matched and matched[0] != "name":
                    snippet = searchtext.excerpt(marked, 400)
            hits.append(SearchHit(
                path=row[0], kind=FileKind(row[1]), score=s, z=semantic[0] if semantic else None,
                cosine=semantic[1] if semantic else None, keyword_rank=keyword[0] if keyword else None,
                segment=SegmentKind(seg_kind), segment_id=segment_id, loc=loc, loc_end=loc_end,
                excerpt=snippet or stored, modified=row[2], size=row[3], info=json.loads(row[4]) if row[4] else {},
                focus=focus,
            ))
        return hits

    def _code_place(self, file: int, segment: int, ident: list[str] | None, query: str, words_: list[str],
                    words_here: bool) -> tuple[int, float | None]:
        name = query.strip().replace("()", "")
        db = self.store.db
        if ident:
            best = None
            for seg_id, loc, text in db.execute(
                "SELECT s.id, s.loc, coalesce(fts.original, fts.body) FROM fts JOIN segments s ON s.id = fts.rowid "
                "WHERE fts MATCH ? AND s.file = ? ORDER BY s.loc LIMIT 300", (codeextract.phrase(ident), file)
            ):
                if best and best[2] >= 2:
                    break
                found = codeextract.definition(name, text or "", int(loc or 1))
                if found and found[1] > (best[2] if best else 0):
                    best = (seg_id, found[0], found[1])
            if best:
                return best[0], float(best[1])
        if not (words_here or ident):
            return segment, None
        text = self.text_of_segment(segment)
        if not text:
            return segment, None
        row = db.execute("SELECT loc FROM segments WHERE id = ?", (segment,)).fetchone()
        first = int(row[0] or 1) if row else 1
        line = codeextract.first_line(words_ if ident is None else [name.lower()], text, first)
        return segment, (float(line) if line else None)

    def vector_of(self, segment_id: int, path: str | None = None) -> list[float] | None:
        """A stored vector to search with ("find similar"): the segment's own, else the file's picture (its image,
        a video's first frame, a scanned page), else any vector of the file."""
        db = self.store.db
        row = db.execute("SELECT vec FROM segments WHERE id = ?", (segment_id,)).fetchone()
        blob = row[0] if row else None
        if blob is None and path is not None:
            row = db.execute(
                "SELECT s.vec FROM segments s JOIN files f ON f.id = s.file WHERE f.path = ? AND s.vec IS NOT NULL "
                "ORDER BY CASE s.modality WHEN 'image' THEN 0 WHEN 'text' THEN 1 ELSE 2 END, s.loc LIMIT 1",
                (path,)).fetchone()
            blob = row[0] if row else None
        if blob is None or len(blob) != DIM * 2:
            return None
        return np.frombuffer(blob, dtype=np.float16).astype(np.float32).tolist()

    def text_of_segment(self, id: int) -> str | None:
        row = self.store.db.execute("SELECT coalesce(original, body) FROM fts WHERE rowid = ?", (id,)).fetchone()
        return row[0] if row else None
