"""Keyword search text: normalizing, matching whole words in scripts that attach endings, and PDF text layers that
came out garbled. Cases from the macOS repo's SearchTextTests."""

from digup import searchtext
from digup.model import FileKind
from digup.searcher import fts_query, is_lookup, kind_hints, terms


def test_normalized():
    assert searchtext.normalized("الْمَكْتَبَةَ") == "المكتبة"  # harakat
    assert searchtext.normalized("إلى الأردن") == "الى الاردن"  # alef forms
    assert searchtext.normalized("مـــرحبا") == "مرحبا"  # tatweel
    assert searchtext.normalized("ﻻ") == "لا"  # the lam-alef ligature
    assert searchtext.normalized("ﬁle") == "file"  # ﬁ
    assert searchtext.normalized("র‍্যাব") == "র্যাব"  # zero-width joiner
    assert searchtext.normalized("ো") == "ো"  # ে + া → ো
    assert searchtext.normalized("a​b") == "a b"  # zero-width space breaks words


def test_fts_queries():
    assert fts_query("কুমির") == '"কুমির"*'
    assert fts_query("مدرسة الأطفال") == '"مدرسة"* OR "الاطفال"*'
    assert fts_query("মা") == '"মা"*'
    assert fts_query("payment error") == '"payment" OR "error"'
    assert fts_query("the screenshot of my payment error") == '"payment" OR "error"'
    assert fts_query("zebr", prefix_last=True) == '"zebr"*'
    assert fts_query("a screenshot") is None


def test_kind_hints_and_lookups():
    assert kind_hints("জেব্রার ভিডিওতে") == {FileKind.VIDEO}
    assert kind_hints("صورة حمار وحشي") == {FileKind.IMAGE}
    assert kind_hints("zebra in a video") == {FileKind.VIDEO}
    assert is_lookup(terms("K7Q2LM"))
    assert is_lookup(terms("order 4815162342")) is False  # "order" has no digit
    assert is_lookup(terms("4815162342"))
    assert not is_lookup(terms("a dog on the beach"))


def test_matches_whole_words():
    text = "সুন্দরবনে আমরা কুমিরও দেখেছি।"
    (start, end), = searchtext.matches("কুমির", text)
    assert text[start:end] == "কুমিরও"  # the whole word it starts, never a piece of it
    assert searchtext.matches("ice", "Price notice ice") == [(13, 16)]
    assert searchtext.matches("cafe", "Café au lait") == [(0, 4)]


def test_snippet_marks_words():
    text = "We discussed how many hours of sleep adults need each night, and why caffeine matters."
    snippet = searchtext.snippet(text, ["sleep", "caffeine"])
    assert "«sleep»" in snippet and "«caffeine»" in snippet
    assert searchtext.snippet(text, ["zebra"]) is None


def test_has_words():
    assert searchtext.has_words("The quarterly report shows revenue grew.")
    assert not searchtext.has_words("1,234.56 7,890.12 3,456.78 9,012.34 5,678.90")
    assert searchtext.has_words("কলকাতা ২০২৬")


def test_garbled_page_text_is_noticed():
    glyph_order = "বািড় ভাড়ার চিপ ধানমি ৭ নর রােডর িতন বডরেমর াট মািসক ২৫,০০০ টাকা ভাড়ায় দওয়া হেলা। ভাড়ায়া দুই মােসর"
    bengali = "বাড়ি ভাড়ার চুক্তিপত্র। ধানমন্ডি ৭ নম্বর রোডের তিন বেডরুমের ফ্ল্যাটটি মাসিক ২৫,০০০ টাকা ভাড়ায় দেওয়া হলো।"
    lost = "ارتفعت ا.بيعات بنسبة خمسة عشر با.ئة في الربع الثالث. زرنا مدينة البتراء ا?ثرية في جنوب ا?ردن."
    arabic = "ارتفعت المبيعات بنسبة خمسة عشر بالمئة في الربع الثالث. زرنا مدينة البتراء الأثرية في جنوب الأردن."
    assert searchtext.page_text_problem(glyph_order) == "glyph_order"
    assert searchtext.page_text_problem(bengali) is None
    assert searchtext.page_text_problem(lost) == "lost_letters"
    assert searchtext.page_text_problem(arabic) is None
    assert searchtext.page_text_problem("Revenue grew 20%. See p.4?No. Done.") is None
