"""People: faces grouped strictly, named (one name merges groups), found by name and by face."""

import numpy as np

from digup import people
from digup.model import FileKind
from digup.store import Candidate, IndexStore

rng = np.random.default_rng(7)


def person_like(center: np.ndarray, spread: float = 0.35) -> np.ndarray:
    """A face of the person at `center`: close to it (ArcFace puts one person's faces near each other)."""
    v = center + spread * rng.standard_normal(center.size) / np.sqrt(center.size)
    return v / np.linalg.norm(v)


def unit(n: int = 512) -> np.ndarray:
    v = rng.standard_normal(n)
    return v / np.linalg.norm(v)


def photo(store, name: str, inode: int, vectors: list[np.ndarray]) -> int:
    fid = store.insert(Candidate(rf"C:\p\{name}", FileKind.IMAGE, 1, float(inode), inode, 1))
    store.finish(fid, "done")
    people.save(store, fid, [{"box": (0.1, 0.1, 0.2, 0.2), "score": 0.9, "size": 120, "vector": v} for v in vectors])
    return fid


def test_grouping_naming_and_finding(tmp_path):
    store = IndexStore(tmp_path)
    alice, bob = unit(), unit()
    a_files = [photo(store, f"alice{i}.jpg", i + 1, [person_like(alice)]) for i in range(6)]
    b_files = [photo(store, f"bob{i}.jpg", i + 100, [person_like(bob)]) for i in range(4)]
    together = photo(store, "both.jpg", 500, [person_like(alice), person_like(bob)])
    stranger = photo(store, "stranger.jpg", 600, [unit()])
    result = people.group(store)
    assert result["new_groups"] == 2
    groups = people.people(store)
    assert sorted(p.photos for p in groups) == [5, 7]  # Bob: 4 + together; Alice: 6 + together
    alice_group = next(p.id for p in groups if p.photos == 7)

    # Named, and found by name; the rest of the words are left for meaning.
    people.rename(store, alice_group, "Alice")
    found, rest = people.split_names(store, "photos of alice at the beach")
    assert found == [alice_group] and rest == "at the beach"
    assert people.split_names(store, "Alice")[1] == "" and people.split_names(store, "pictures of Alice")[1] == ""
    assert people.split_names(store, "alicent's report")[0] == []  # whole words only
    assert set(people.files_of(store, found)) == set(a_files) | {together}
    assert stranger not in people.files_of(store, [p.id for p in people.people(store)])

    # A later photo of Alice joins her group.
    later = photo(store, "alice_later.jpg", 700, [person_like(alice)])
    people.group(store)
    assert later in people.files_of(store, [alice_group])

    # Find this person: by the face itself.
    face = people.faces_in(store, a_files[0])[0]["id"]
    near = [f for f, _ in people.same_face(store, face)]
    assert set(near) >= set(a_files[1:]) and not set(near) & set(b_files)


def test_one_name_merges_two_groups(tmp_path):
    store = IndexStore(tmp_path)
    carol = unit()
    for i in range(3):
        photo(store, f"c{i}.jpg", i + 1, [person_like(carol, 0.2)])
    people.group(store)
    # The same person, younger: far enough to be a second group.
    young = person_like(carol, 2.5)  # cosine about 0.37 with her: under the grouping line
    for i in range(3):
        photo(store, f"young{i}.jpg", i + 10, [person_like(young, 0.2)])
    people.group(store)
    groups = people.people(store)
    assert len(groups) == 2
    first = people.rename(store, groups[0].id, "Carol")
    merged = people.rename(store, groups[1].id, "carol")
    assert merged == first and len(people.people(store)) == 1 and people.people(store)[0].photos == 6


def test_a_changed_photo_is_looked_at_again(tmp_path):
    store = IndexStore(tmp_path)
    fid = photo(store, "x.jpg", 1, [unit()])
    assert people.to_scan_count(store) == 0
    store.requeue_changed(fid, Candidate(r"C:\p\x.jpg", FileKind.IMAGE, 2, 2.0, 1, 1))
    store.finish(fid, "done")
    assert people.faces_in(store, fid) == [] and people.to_scan_count(store) == 1
