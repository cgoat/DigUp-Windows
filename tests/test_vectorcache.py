"""The vector cache on disk: appends what's new, never serves a deleted vector (even when SQLite reuses its id), and
starts afresh when the index is emptied."""

import numpy as np

from digup.model import FileKind, Modality, SegmentKind
from digup.searcher import Searcher
from digup.store import Candidate, IndexStore
from test_index_and_search import vector_for


def add(store, name, words, inode):
    fid = store.insert(Candidate(rf"C:\p\{name}", FileKind.IMAGE, 1, 1.0, inode, 1))
    store.add_segment(fid, SegmentKind.IMAGE, Modality.IMAGE, vector=vector_for(words))
    return fid


def paths_for(searcher, words):
    return [h.path.split("\\")[-1] for h in searcher.search("", vector_for(words), limit=5)]


def filler(store, n=40):
    for i in range(n):
        add(store, f"other{i}.jpg", f"filler{i} thing{i}", 1000 + i)


def test_reused_ids_never_serve_a_deleted_vector(tmp_path):
    store = IndexStore(tmp_path)
    filler(store)
    beach = add(store, "beach.jpg", "dog beach sand sea", 1)
    searcher = Searcher(store)
    assert paths_for(searcher, "dog beach sand sea")[0] == "beach.jpg"
    n_before = searcher.cache.n

    # The file changes: its segment goes, and the new one gets the same id back (it was the highest).
    old_id = store.db.execute("SELECT MAX(id) FROM segments").fetchone()[0]
    store.requeue_changed(beach, Candidate(r"C:\p\beach.jpg", FileKind.IMAGE, 2, 2.0, 1, 1))
    store.add_segment(beach, SegmentKind.IMAGE, Modality.IMAGE, vector=vector_for("night city lights towers"))
    new_id = store.db.execute("SELECT MAX(id) FROM segments WHERE file = ? AND kind = 'image'", (beach,)).fetchone()[0]
    assert new_id == old_id  # SQLite reused it

    searcher.reload()
    assert "beach.jpg" not in paths_for(searcher, "dog beach sand sea")  # the old vector is dead
    assert paths_for(searcher, "night city lights towers")[0] == "beach.jpg"  # the new one is there
    assert searcher.cache.n == n_before + 1  # appended, not rebuilt
    assert searcher.vector_count == 41


def test_cache_survives_a_restart_and_follows_a_reset(tmp_path):
    store = IndexStore(tmp_path)
    filler(store)
    add(store, "beach.jpg", "dog beach sand sea", 1)
    store.set_meta("fingerprint", "model A")
    Searcher(store)  # builds the cache
    again = Searcher(IndexStore(tmp_path), update_cache=False)
    assert again.cache.n == 41 and paths_for(again, "dog beach sand sea")[0] == "beach.jpg"
    store.reset_all()  # a new model: every vector goes
    store.set_meta("fingerprint", "model B")
    fresh = Searcher(store)
    assert fresh.vector_count == 0 and fresh.search("", vector_for("dog beach"), limit=5) == []


def test_scores_match_the_stored_vectors(tmp_path):
    store = IndexStore(tmp_path)
    filler(store, 100)
    searcher = Searcher(store)
    q = np.asarray(vector_for("thing7 filler7"), dtype=np.float32)
    exact = np.array([np.frombuffer(r[0], dtype=np.float16).astype(np.float32) @ q
                      for r in store.db.execute("SELECT vec FROM segments WHERE vec IS NOT NULL ORDER BY id")])
    assert np.abs(searcher.score(q) - exact).max() < 0.003
