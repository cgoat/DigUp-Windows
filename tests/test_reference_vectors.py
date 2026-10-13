"""The Windows pipeline must make the vectors the Mac app makes (llama.cpp b11461 on Metal, vectors version 1).

reference/vectors.json and its inputs come from the macOS repo (Tests/LlamaRuntimeTests/Reference). Runtimes that agree
measure a cosine >= 0.999 with it; less is a change people would see in their results. Needs the model files and a
llama.cpp runtime; skipped without them.
"""

from pathlib import Path

import json

import numpy as np
import pytest

from digup import downloads
from digup.model import Prompts

FOLDER = Path(__file__).parent / "reference"
AGREEMENT = 0.999

TEXTS = [
    ("query", Prompts.query("payment declined error")),
    ("query-bn", Prompts.query("ইলিশ মাছ রান্না")),
    ("query-ar", Prompts.query("متى تغلق المكتبة")),
    ("document", Prompts.document("Tokyo itinerary", "Flight to Tokyo Haneda on November 3. Then the train to Kamakura "
                                  "to see the Great Buddha, and back for dinner in Shibuya.")),
    # About 1,800 tokens: past the sliding attention window, so the global layers count as well.
    ("document-long", Prompts.document(None, " ".join(
        f"Note {i}: the review moved to room {i % 7 + 2} because the projector in the main hall failed again, "
        "and the team agreed to send the slides the night before." for i in range(1, 61)))),
]
IMAGES = [("screen-280", "screen.png", 280), ("photo-140", "photo.jpg", 140), ("frame-70", "photo.jpg", 70)]

ready = downloads.models_complete() and downloads.runtime_ready(downloads.best_backend())


@pytest.mark.skipif(not ready, reason="needs the model files and a llama.cpp runtime")
def test_vectors_match_the_mac_reference():
    from digup.llama import VECTOR_VERSION, LlamaEmbedder

    reference = json.loads((FOLDER / "vectors.json").read_text())
    assert reference["vector_version"] == VECTOR_VERSION
    embedder = LlamaEmbedder(text_only=False, max_tokens=2048)  # the reference's cut
    try:
        made = dict(zip([n for n, _ in TEXTS], embedder.embed_texts([t for _, t in TEXTS])))
        for name, file, budget in IMAGES:
            result = embedder.embed_images([str(FOLDER / file)], budget)[0]
            assert result.vector, result.error
            made[name] = result.vector
        speech = embedder.embed_audio([str(FOLDER / "speech.wav")])[0]
        assert speech.vector, speech.error
        made["speech"] = speech.vector
    finally:
        embedder.close()
    assert set(made) == set(reference["vectors"])
    cosines = {n: float(np.dot(made[n], reference["vectors"][n])) for n in made}
    print(cosines)
    worst = min(cosines, key=cosines.get)
    assert cosines[worst] >= AGREEMENT, f"{worst}: cosine {cosines[worst]:.5f} with the Mac reference"
