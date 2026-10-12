"""The index and search with a stand-in embedder: syncing (new, changed, moved, removed files), the meaning scale,
keyword matches in Bengali and Arabic, and the fingerprint that keeps settings from mixing."""

import hashlib
import os

import numpy as np
import pytest

from digup.indexer import FingerprintMismatch, Indexer
from digup.model import DIM, Embedding, FileKind, IndexOptions, Modality, SegmentKind, WorkerInfo
from digup.searcher import Searcher, meaning
from digup.store import Candidate, IndexStore


def vector_for(text: str) -> list[float]:
    """A stable unit vector per word set: texts that share words point the same way."""
    v = np.zeros(DIM)
    for word in text.lower().split():
        seed = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
        v += np.random.default_rng(seed).standard_normal(DIM)
    return (v / (np.linalg.norm(v) or 1)).tolist()


class FakeEmbedder:
    def __init__(self, runtime="fake-1"):
        self.info = WorkerInfo("fake/model", "0" * 40, runtime, "f32")
        self.calls = 0

    def embed_texts(self, texts):
        self.calls += 1
        return [vector_for(t.split("text: ")[-1]) for t in texts]

    def embed_images(self, paths, budget):
        return [Embedding(vector_for(os.path.basename(p))) for p in paths]

    def embed_audio(self, paths):
        return [Embedding(vector_for("sound")) for _ in paths]

    def close(self):
        pass


@pytest.fixture
def folder(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "sleep.txt").write_text("Most adults need seven to nine hours of sleep every night.", "utf-8")
    (docs / "bread.txt").write_text("Mash three ripe bananas and bake the banana bread for an hour.", "utf-8")
    (docs / "trip.txt").write_text("সুন্দরবনে আমরা কুমিরও দেখেছি। قَالَ إِنَّ الْمَكْتَبَةَ تُغْلَقُ", "utf-8")
    return tmp_path


def make(tmp_path, folder, embedder=None):
    store = IndexStore(tmp_path / "index")
    embedder = embedder or FakeEmbedder()
    return store, Indexer(store, IndexOptions(), lambda: embedder), embedder


def test_sync_index_search(tmp_path, folder):
    store, indexer, _ = make(tmp_path, folder)
    summary = indexer.sync([str(folder)])
    assert summary.added == 3
    states = indexer.run()
    assert states == {"done": 3}
    searcher = Searcher(store)
    hits = searcher.search("seven hours of sleep", vector_for("seven hours of sleep"))
    assert hits and hits[0].path.endswith("sleep.txt")
    bengali = searcher.search("কুমির", None)
    assert [os.path.basename(h.path) for h in bengali] == ["trip.txt"]
    assert "«কুমিরও»" in bengali[0].excerpt  # the whole word
    assert [os.path.basename(h.path) for h in searcher.search("المكتبة", None)] == ["trip.txt"]
    assert searcher.search("কলকাতা", None) == []
    indexer.finish()


def test_changes_moves_and_removals(tmp_path, folder):
    store, indexer, embedder = make(tmp_path, folder)
    indexer.sync([str(folder)])
    indexer.run()
    calls = embedder.calls
    os.rename(folder / "docs" / "bread.txt", folder / "docs" / "banana bread.txt")  # a rename keeps its vectors
    (folder / "docs" / "sleep.txt").write_text("Changed: notes about running a marathon.", "utf-8")
    os.remove(folder / "docs" / "trip.txt")
    summary = indexer.sync([str(folder)])
    assert (summary.moved, summary.changed, summary.removed, summary.added) == (1, 1, 1, 0)
    indexer.run()
    assert embedder.calls == calls + 1  # only the changed file was embedded again
    hits = Searcher(store).search("banana", None)
    assert [os.path.basename(h.path) for h in hits] == ["banana bread.txt"]
    indexer.finish()


def test_settings_never_mix(tmp_path, folder):
    store, indexer, _ = make(tmp_path, folder)
    indexer.sync([str(folder)])
    indexer.run()
    indexer.finish()
    other = Indexer(store, IndexOptions(photo_budget=280), lambda: FakeEmbedder())
    (folder / "docs" / "new.txt").write_text("a new note", "utf-8")
    other.sync([str(folder)])
    with pytest.raises(FingerprintMismatch):
        other.run()
    reset = []
    other.reset_on_model_change = lambda old, new: reset.append(new)
    other.run()  # a run under way finishes its batch; the files queued again go in the next run
    other.run()
    assert reset and store.pending_count() == 0
    assert store.counts()["vectors"] == 4  # everything embedded again under the new settings
    other.finish()


def test_meaning_scale_per_modality():
    rng = np.random.default_rng(1)
    text = 0.71 + rng.normal(0, 0.01, 30)
    image = 0.552 + rng.normal(0, 0.01, 30)
    text[0], image[0] = 0.71, 0.552
    scores = np.concatenate([text, image]).astype(np.float32)
    modality = np.array(["text"] * 30 + ["image"] * 30)
    z = meaning(scores, modality)
    # Each modality is centered on its own average, so a picture isn't buried under text for the same topic.
    assert abs(z[:30].mean()) < 1e-6 and abs(z[30:].mean()) < 1e-6


def test_moves_keep_vectors_in_store(tmp_path):
    store = IndexStore(tmp_path)
    id = store.insert(Candidate("C:\\d\\shot.png", FileKind.SCREENSHOT, 1, 1.0, 7, 1))
    store.add_segment(id, SegmentKind.IMAGE, Modality.IMAGE, vector=vector_for("x"))
    store.add_segment(id, SegmentKind.OCR, Modality.TEXT, text="Error code: K7Q2LM")
    store.move(id, "C:\\e\\renamed.png")
    hits = Searcher(store).search("K7Q2LM", None)
    assert [h.path for h in hits] == ["C:\\e\\renamed.png"]
    assert store.counts()["vectors"] == 1


def test_find_similar_uses_the_stored_vector(tmp_path):
    store = IndexStore(tmp_path)
    ids = {}
    for name, words in (("beach.jpg", "dog beach sand sea"), ("beach2.jpg", "dog beach sand waves"),
                        ("city.jpg", "night city lights towers")):
        ids[name] = store.insert(Candidate(rf"C:\p\{name}", FileKind.IMAGE, 1, 1.0, len(ids) + 1, 1))
        store.add_segment(ids[name], SegmentKind.IMAGE, Modality.IMAGE, vector=vector_for(words))
    # Filler so the meaning scale has a spread, as a real index does.
    for i in range(40):
        f = store.insert(Candidate(rf"C:\p\other{i}.jpg", FileKind.IMAGE, 1, 1.0, 100 + i, 1))
        store.add_segment(f, SegmentKind.IMAGE, Modality.IMAGE, vector=vector_for(f"filler{i} thing{i}"))
    searcher = Searcher(store)
    seg = store.db.execute("SELECT id FROM segments WHERE file = ? AND kind = 'image'", (ids["beach.jpg"],)).fetchone()[0]
    vector = searcher.vector_of(seg)
    assert vector is not None and abs(np.linalg.norm(vector) - 1) < 1e-2
    # A keyword-only segment (the name row) has no vector: the file's picture is used instead.
    name_row = store.db.execute("SELECT id FROM segments WHERE file = ? AND kind = 'name'", (ids["beach.jpg"],)).fetchone()[0]
    assert searcher.vector_of(name_row, r"C:\p\beach.jpg") is not None
    hits = [h.path for h in searcher.search("", vector) if h.path != r"C:\p\beach.jpg"]
    assert hits[0] == r"C:\p\beach2.jpg" and r"C:\p\city.jpg" not in hits


def test_backup_folders_are_left_out_and_removed_in_bulk(tmp_path, folder):
    store, indexer, _ = make(tmp_path, folder)
    backup = folder / "FileHistory" / "user" / "Data" / "$OF"
    backup.mkdir(parents=True)
    for i in range(50):
        (backup / f"{i}.txt").write_text(f"backup copy {i}", "utf-8")
    # As an older DigUp indexed them, before the rule: then the rule removes them in one go.
    from digup.crawler import CrawlReport
    from digup.store import Candidate as C

    with store.transaction():
        for i in range(50):
            p = str(backup / f"{i}.txt")
            store.insert(C(p, FileKind.DOC, 1, 1.0, 10_000 + i, 1))
    summary = indexer.sync([str(folder)])
    assert summary.removed == 50
    assert not [r for r in store.files_under([str(folder)]) if "FileHistory" in r.path]
    assert store.db.execute("SELECT COUNT(*) FROM fts").fetchone()[0] == summary.added  # their name rows went too
    indexer.finish()


def test_a_stopped_scan_removes_nothing(tmp_path, folder):
    store, indexer, _ = make(tmp_path, folder)
    indexer.sync([str(folder)])
    before = len(store.files_under([str(folder)]))
    summary = indexer.sync([str(folder)], should_stop=lambda: True)
    assert summary.removed == 0 and not summary.crawl.complete
    assert len(store.files_under([str(folder)])) == before
    indexer.finish()
