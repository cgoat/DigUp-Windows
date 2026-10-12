"""Text the way keyword search keeps it and matches it, so a word is found however it was typed or extracted."""

from __future__ import annotations

import re
import unicodedata

INDIC = range(0x0900, 0x0E00)
ARABIC = range(0x0600, 0x0700)

_DROP = {0x00AD, 0x061C, 0xFEFF, 0x0640, 0x0670}
_DROP.update(range(0x200C, 0x2010))
_DROP.update(range(0x202A, 0x202F))
_DROP.update(range(0x2060, 0x2065))
_DROP.update(range(0x2066, 0x206A))
_DROP.update(range(0x064B, 0x0660))
_ALEF = {0x0622, 0x0623, 0x0625, 0x0671}


def normalized(text: str) -> str:
    """NFKC, invisible formatting dropped (joiners, direction marks), Arabic as people type it (no tatweel or
    harakat, one alef). A zero-width space becomes a word break."""
    out = []
    for ch in unicodedata.normalize("NFKC", text):
        cp = ord(ch)
        if cp == 0x200B:
            out.append(" ")
        elif cp in _DROP:
            continue
        elif cp in _ALEF:
            out.append("ا")
        else:
            out.append(ch)
    return "".join(out)


def is_letter(ch: str) -> bool:
    """Letters and marks (Bengali vowel signs count), as Foundation's CharacterSet.letters."""
    return unicodedata.category(ch)[0] in "LM"


def is_word_char(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "LMN"


def inflects(word: str) -> bool:
    """Words of scripts that attach endings (Bengali, Hindi, Arabic) match as prefixes."""
    return any(ord(c) in INDIC or ord(c) in ARABIC for c in word)


def has_words(text: str) -> bool:
    """At least a fifth of what isn't space is letters. A table of numbers gets keyword search only."""
    letters = others = 0
    for ch in text:
        if ch.isspace():
            continue
        if is_letter(ch):
            letters += 1
        else:
            others += 1
    return letters > 0 and letters * 4 >= others


def split_words(text: str) -> list[str]:
    """Runs of letters, marks and digits (Foundation's alphanumerics)."""
    words, current = [], []
    for ch in text:
        if is_word_char(ch):
            current.append(ch)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return words


def _fold(ch: str) -> str:
    """One character as keyword search compares it: normalized, lower-case, without accents."""
    piece = normalized(ch).lower()
    decomposed = unicodedata.normalize("NFD", piece)
    # Strip combining accents from Latin/Greek/Cyrillic only: Indic vowel signs and Arabic marks are letters there.
    kept = "".join(c for c in decomposed if not (unicodedata.combining(c) and ord(c) < 0x0370))
    return unicodedata.normalize("NFC", kept)


def matches(word: str, text: str) -> list[tuple[int, int]]:
    """Where `word` is in `text` as keyword search would find it, as (start, end) indexes into `text` as written:
    whole words, and for a word of a language that attaches endings, the whole word it starts."""
    needle = "".join(_fold(c) for c in word)
    if not needle:
        return []
    folded, origin = [], []
    for index, ch in enumerate(text):
        piece = _fold(ch)
        folded.append(piece)
        origin.extend([index] * len(piece))
    hay = "".join(folded)
    prefix = inflects(word)
    found: list[tuple[int, int]] = []
    start = 0
    while len(found) < 500:
        hit = hay.find(needle, start)
        if hit < 0:
            break
        start = hit + len(needle)
        first, last = origin[hit], origin[hit + len(needle) - 1] + 1
        if first > 0 and is_word_char(text[first - 1]):
            continue
        if last < len(text) and is_word_char(text[last]):
            if not prefix:
                continue
            while last < len(text) and is_word_char(text[last]):
                last += 1
        if found and found[-1][1] > first:
            continue
        found.append((first, last))
    return found


def snippet(text: str, terms: list[str], words: int = 12) -> str | None:
    """One line around the first of `terms` in `text`, the terms marked « », as SQLite's snippet() marks them."""
    marks = sorted(m for t in terms for m in matches(t, text))
    if not marks:
        return None
    first = marks[0]
    start, before = first[0], 0
    while start > 0 and before < words // 3:
        start -= 1
        if text[start].isspace() and start < first[0] and not text[start + 1].isspace():
            before += 1
    if start > 0 or (start < len(text) and text[start].isspace()):
        start += 1
    end, after = first[1], 0
    while end < len(text) and after < words - before:
        if text[end].isspace():
            after += 1
        end += 1
    out = ["…" if start > 0 else ""]
    cursor = start
    for a, b in marks:
        if a >= start and b <= end and a >= cursor:
            out.append(text[cursor:a] + "«" + text[a:b] + "»")
            cursor = b
    out.append(text[cursor:end] + ("…" if end < len(text) else ""))
    return re.sub(r"\s+", " ", "".join(out)).strip()


def excerpt(text: str, limit: int = 240) -> str:
    """One line, at most `limit` characters: for result lists."""
    line = re.sub(r"\s+", " ", text).strip()
    return line[:limit] + "…" if len(line) > limit else line


# MARK: PDF text layers that can't be used as they are


def _indic_vowel_sign(cp: int) -> bool:
    return (
        0x09BE <= cp <= 0x09C4 or cp in (0x09C7, 0x09C8, 0x09CB, 0x09CC, 0x09D7, 0x09E2, 0x09E3)
        or cp in (0x093A, 0x093B) or 0x093E <= cp <= 0x094C or cp in (0x094E, 0x094F)
        or 0x0955 <= cp <= 0x0957 or cp in (0x0962, 0x0963)
    )


def _indic_consonant(cp: int) -> bool:
    return (
        0x0995 <= cp <= 0x09B9 or cp in (0x09BC, 0x09CE, 0x09DC, 0x09DD, 0x09DF, 0x09F0, 0x09F1)
        or 0x0915 <= cp <= 0x0939 or cp == 0x093C or 0x0958 <= cp <= 0x095F or 0x0979 <= cp <= 0x097F
    )


def page_text_problem(text: str) -> str | None:
    """"glyph_order" for Bengali/Hindi read in glyph order (vowel signs before consonants), "lost_letters" for Arabic
    whose lam ligatures vanished ("ا?ردن"), else None."""
    text = unicodedata.normalize("NFC", text)
    vowel_signs = misplaced = lost = 0
    previous = before_previous = " "
    for ch in text:
        cp = ord(ch)
        if _indic_vowel_sign(cp):
            vowel_signs += 1
            if not _indic_consonant(ord(previous)):
                misplaced += 1
        if before_previous == "ا" and previous in "?.�" and cp in ARABIC and ch.isalpha():
            lost += 1
        before_previous, previous = previous, ch
    if vowel_signs >= 10 and misplaced * 20 >= vowel_signs:
        return "glyph_order"
    if lost >= 2:
        return "lost_letters"
    return None
