"""People: faces grouped into the people they belong to, named by the person using DigUp, and found by name.

Faces live in the index beside the files they're in (table `faces`, a 512-number ArcFace vector each), and groups in
`people`. Grouping is strict on purpose (`GROUP`): a group should be one person, even if one person ends up in two
groups; naming both the same merges them. A face is never put in a group it doesn't fit, so unknown faces (a
stranger in the background) stay out of everyone's group.

Searches: a name in the words ("Srinivas at the beach") narrows the search to the photos with that person; the rest
of the words rank them by what they show. "Find this person" on a result looks for the face itself across every face
(not only the group), which also finds the photos grouping missed.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np

from .faces import FACE_DIM, half

# Cosine at or above which a face joins a group. ArcFace's usual "same person" cut is 0.4; groups use more, so that
# a group doesn't mix two people who look alike.
GROUP = 0.50
# "Find this person": faces at least this close count as the same person.
FIND = 0.42

SCHEMA = """
CREATE TABLE IF NOT EXISTS people(
    id INTEGER PRIMARY KEY,
    name TEXT,                 -- NULL until someone names the group
    hidden INTEGER NOT NULL DEFAULT 0,
    created REAL
);
CREATE TABLE IF NOT EXISTS faces(
    id INTEGER PRIMARY KEY,
    file INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    x REAL, y REAL, w REAL, h REAL,   -- the box, as fractions of the photo
    score REAL,
    size INTEGER,                     -- pixels across
    vec BLOB NOT NULL,                -- 512 x float16, L2-normalized (ArcFace)
    person INTEGER REFERENCES people(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS faces_file ON faces(file);
CREATE INDEX IF NOT EXISTS faces_person ON faces(person);
CREATE TABLE IF NOT EXISTS face_scans(
    file INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
    faces INTEGER NOT NULL,
    at REAL
)
"""


def install(db) -> None:
    for statement in SCHEMA.split(";"):
        if statement.strip():
            db.execute(statement)


def ensure(store, wait_seconds: int = 600) -> None:
    """Makes the tables if the index doesn't have them yet, waiting for a busy index (an indexing pass can hold it
    for minutes) rather than giving up as the quick try when the index opens does."""
    db = store.db
    if db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'face_scans'").fetchone():
        return
    db.execute(f"PRAGMA busy_timeout = {wait_seconds * 1000}")
    try:
        install(db)
    finally:
        db.execute("PRAGMA busy_timeout = 5000")


def ready(store) -> bool:
    """The index has face tables (readers treat an index without them as having no people)."""
    return bool(store.db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'face_scans'").fetchone())


@dataclass
class Person:
    id: int
    name: str | None
    faces: int
    photos: int
    hidden: bool
    cover_file: str | None  # a photo with the person's biggest face, and the face's box in it
    cover_box: tuple | None


def _vectors(blobs) -> np.ndarray:
    if not blobs:
        return np.zeros((0, FACE_DIM), np.float32)
    return np.frombuffer(b"".join(blobs), dtype=np.float16).reshape(-1, FACE_DIM).astype(np.float32)


# MARK: Reading and writing


def photos_to_scan(store, limit: int = 256) -> list[tuple[int, str]]:
    """Photos indexed but not yet looked at for faces (newest first)."""
    return store.db.execute(
        "SELECT f.id, f.path FROM files f LEFT JOIN face_scans s ON s.file = f.id "
        "WHERE f.kind = 'image' AND f.state = 'done' AND s.file IS NULL ORDER BY f.mtime DESC LIMIT ?",
        (limit,)).fetchall()


def to_scan_count(store) -> int:
    return store.db.execute(
        "SELECT COUNT(*) FROM files f LEFT JOIN face_scans s ON s.file = f.id "
        "WHERE f.kind = 'image' AND f.state = 'done' AND s.file IS NULL").fetchone()[0]


def save(store, file: int, faces: list[dict]) -> None:
    """A photo's faces (as the worker sends them: box, score, size, vector)."""
    db = store.db
    db.execute("DELETE FROM faces WHERE file = ?", (file,))
    for f in faces:
        x, y, w, h = f["box"]
        db.execute("INSERT INTO faces(file, x, y, w, h, score, size, vec) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                   (file, x, y, w, h, f["score"], f["size"], half(f["vector"])))
    db.execute("INSERT OR REPLACE INTO face_scans(file, faces, at) VALUES(?, ?, ?)", (file, len(faces), time.time()))


def forget(store, file: int) -> None:
    """The file changed: its faces go, and it's looked at again."""
    store.db.execute("DELETE FROM faces WHERE file = ?", (file,))
    store.db.execute("DELETE FROM face_scans WHERE file = ?", (file,))


# MARK: Grouping


def group(store) -> dict:
    """Puts faces that have no group yet into one: the closest group they fit (its average face), else a new group
    with the other loose faces they fit. A face that fits nobody stays loose (one photo of a stranger is no group)."""
    db = store.db
    loose = db.execute("SELECT id, vec FROM faces WHERE person IS NULL").fetchall()
    if not loose:
        return {"joined": 0, "new_groups": 0}
    ids = np.array([r[0] for r in loose], dtype=np.int64)
    vectors = _vectors([r[1] for r in loose])
    joined = 0
    # 1. Join existing groups.
    centroids, people_ids = _centroids(db)
    assign = np.full(len(ids), -1, dtype=np.int64)
    if len(people_ids):
        sims = vectors @ centroids.T
        best = sims.argmax(axis=1)
        ok = sims[np.arange(len(ids)), best] >= GROUP
        assign[ok] = people_ids[best[ok]]
        joined = int(ok.sum())
    # 2. The rest among themselves: greedy, biggest neighbourhoods first, so a group starts at a typical face.
    rest = np.where(assign < 0)[0]
    new_groups = 0
    if rest.size > 1:
        sub = vectors[rest]
        taken = np.zeros(rest.size, dtype=bool)
        for start in range(0, rest.size, 4096):
            block = sub[start:start + 4096] @ sub.T
            for row in range(block.shape[0]):
                i = start + row
                if taken[i]:
                    continue
                members = np.where((block[row] >= GROUP) & ~taken)[0]
                if members.size < 2:
                    continue
                taken[members] = True
                cur = db.execute("INSERT INTO people(name, created) VALUES(NULL, ?)", (time.time(),))
                assign[rest[members]] = cur.lastrowid
                new_groups += 1
    with store.transaction():
        for face_id, person in zip(ids[assign >= 0].tolist(), assign[assign >= 0].tolist()):
            db.execute("UPDATE faces SET person = ? WHERE id = ?", (person, face_id))
        db.execute("DELETE FROM people WHERE id NOT IN (SELECT DISTINCT person FROM faces WHERE person IS NOT NULL)")
    return {"joined": joined, "new_groups": new_groups}


def _centroids(db) -> tuple[np.ndarray, np.ndarray]:
    rows = db.execute("SELECT person, vec FROM faces WHERE person IS NOT NULL ORDER BY person").fetchall()
    if not rows:
        return np.zeros((0, FACE_DIM), np.float32), np.zeros(0, dtype=np.int64)
    persons = np.array([r[0] for r in rows], dtype=np.int64)
    vectors = _vectors([r[1] for r in rows])
    unique, start = np.unique(persons, return_index=True)
    sums = np.add.reduceat(vectors, start, axis=0)
    return sums / np.linalg.norm(sums, axis=1, keepdims=True).clip(1e-6), unique


# MARK: People


def people(store, include_hidden: bool = False) -> list[Person]:
    if not ready(store):
        return []
    rows = store.db.execute(
        "SELECT p.id, p.name, p.hidden, COUNT(fc.id), COUNT(DISTINCT fc.file) FROM people p "
        "JOIN faces fc ON fc.person = p.id GROUP BY p.id").fetchall()
    out = []
    for pid, name, hidden, n_faces, n_photos in rows:
        if hidden and not include_hidden:
            continue
        cover = store.db.execute(
            "SELECT f.path, fc.x, fc.y, fc.w, fc.h FROM faces fc JOIN files f ON f.id = fc.file "
            "WHERE fc.person = ? ORDER BY fc.size * fc.score DESC LIMIT 1", (pid,)).fetchone()
        out.append(Person(pid, name, n_faces, n_photos, bool(hidden), cover[0] if cover else None,
                          tuple(cover[1:]) if cover else None))
    out.sort(key=lambda p: (p.name is None, -(p.photos), (p.name or "").lower()))
    return out


def rename(store, person: int, name: str | None) -> int:
    """Names a group. A name another group already has merges the two (they're one person). Returns the group the
    faces are in now."""
    name = (name or "").strip() or None
    db = store.db
    with store.transaction():
        if name:
            other = db.execute("SELECT id FROM people WHERE lower(name) = lower(?) AND id != ?", (name, person)).fetchone()
            if other:
                db.execute("UPDATE faces SET person = ? WHERE person = ?", (other[0], person))
                db.execute("DELETE FROM people WHERE id = ?", (person,))
                db.execute("UPDATE people SET name = ?, hidden = 0 WHERE id = ?", (name, other[0]))
                return other[0]
        db.execute("UPDATE people SET name = ? WHERE id = ?", (name, person))
    return person


def hide(store, person: int, hidden: bool = True) -> None:
    store.db.execute("UPDATE people SET hidden = ? WHERE id = ?", (1 if hidden else 0, person))


def named(store) -> dict[str, list[int]]:
    """Lower-case name -> its group ids (one name is one person; groups with it were merged)."""
    out: dict[str, list[int]] = {}
    if not ready(store):
        return out
    for pid, name in store.db.execute("SELECT id, name FROM people WHERE name IS NOT NULL"):
        out.setdefault(name.lower(), []).append(pid)
    return out


def split_names(store, query: str) -> tuple[list[int], str]:
    """The people named in a query, and the query without their names: "Srinivas at the beach" -> ([7], "at the
    beach"). Longer names first, whole words only, any case."""
    names = named(store)
    found: list[int] = []
    rest = query
    for name in sorted(names, key=len, reverse=True):
        pattern = re.compile(rf"(?<!\w){re.escape(name)}(?:'s)?(?!\w)", re.IGNORECASE)
        if pattern.search(rest):
            found += names[name]
            rest = pattern.sub(" ", rest)
    rest = re.sub(r"\s+", " ", rest).strip(" ,.-&")
    # "photos of <name> at the beach" -> "at the beach": the words that were about the name go with it.
    rest = re.sub(r"^(?:(?:photos?|pictures?|pics?|images?|videos?)\s+)?(?:of|with)(?:\s+|$)", "", rest,
                  flags=re.IGNORECASE).strip()
    if re.fullmatch(r"(?:photos?|pictures?|pics?|images?)", rest, flags=re.IGNORECASE):
        rest = ""
    return found, rest


def files_of(store, groups: list[int]) -> list[int]:
    """Files with any face of these groups."""
    if not groups:
        return []
    marks = ",".join("?" * len(groups))
    return [r[0] for r in store.db.execute(f"SELECT DISTINCT file FROM faces WHERE person IN ({marks})", groups)]


def faces_in(store, file: int) -> list[dict]:
    return [{"id": r[0], "box": r[1:5], "person": r[5], "name": r[6], "size": r[7]} for r in store.db.execute(
        "SELECT fc.id, fc.x, fc.y, fc.w, fc.h, fc.person, p.name, fc.size FROM faces fc "
        "LEFT JOIN people p ON p.id = fc.person WHERE fc.file = ? ORDER BY fc.size DESC", (file,))]


def same_face(store, face: int, threshold: float = FIND, limit: int = 400) -> list[tuple[int, float]]:
    """Files with a face like this one, best first: (file, cosine). Every face is compared, not only its group."""
    row = store.db.execute("SELECT vec, file FROM faces WHERE id = ?", (face,)).fetchone()
    if not row:
        return []
    query = _vectors([row[0]])[0]
    best: dict[int, float] = {}
    cursor = store.db.execute("SELECT file, vec FROM faces WHERE id != ?", (face,))
    while rows := cursor.fetchmany(8192):
        sims = _vectors([r[1] for r in rows]) @ query
        for (file, _), sim in zip(rows, sims.tolist()):
            if sim >= threshold and sim > best.get(file, -1):
                best[file] = sim
    best.pop(row[1], None)
    return sorted(best.items(), key=lambda kv: -kv[1])[:limit]


def vectors_of(store, file: int) -> np.ndarray:
    """The face vectors of a photo in the index (its faces, biggest first)."""
    if not ready(store):
        return np.zeros((0, FACE_DIM), np.float32)
    return _vectors([r[0] for r in store.db.execute("SELECT vec FROM faces WHERE file = ? ORDER BY size DESC",
                                                    (file,))])


def match_faces(store, queries: np.ndarray, threshold: float = FIND, exclude_file: int | None = None,
                limit: int = 2000) -> list[tuple[int, int, float]]:
    """Photos with the people of `queries` (face vectors, e.g. the faces of a picture searched with), one pass over
    every face: (file, how many of the query's people are in it, best cosine), most people first, then closest."""
    if not ready(store) or not len(queries):
        return []
    queries = np.asarray(queries, dtype=np.float32)[:12]  # a crowd's first dozen faces (biggest first)
    found: dict[int, np.ndarray] = {}  # file -> best cosine per query face
    cursor = store.db.execute("SELECT file, vec FROM faces")
    while rows := cursor.fetchmany(16384):
        sims = _vectors([r[1] for r in rows]) @ queries.T  # rows x query faces
        hit = np.nonzero((sims >= threshold).any(axis=1))[0]
        for i in hit.tolist():
            file = rows[i][0]
            if file == exclude_file:
                continue
            best = found.get(file)
            found[file] = sims[i] if best is None else np.maximum(best, sims[i])
    ranked = sorted(((file, int((s >= threshold).sum()), float(s.max())) for file, s in found.items()),
                    key=lambda t: (-t[1], -t[2]))
    return ranked[:limit]


def counts(store) -> dict:
    db = store.db
    if not ready(store):
        return {"faces": 0, "people": 0, "named": 0, "scanned": 0, "to_scan": 0}
    return {
        "faces": db.execute("SELECT COUNT(*) FROM faces").fetchone()[0],
        "people": db.execute("SELECT COUNT(*) FROM people").fetchone()[0],
        "named": db.execute("SELECT COUNT(*) FROM people WHERE name IS NOT NULL").fetchone()[0],
        "scanned": db.execute("SELECT COUNT(*) FROM face_scans").fetchone()[0],
        "to_scan": to_scan_count(store),
    }
