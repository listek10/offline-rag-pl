"""
tests/test_chunk.py
Lekkie testy chunkingu — bez zależności (działa i pod pytest, i przez `python test_chunk.py`).

DLACZEGO testujemy chunker: to fundament jakości retrievalu. Chcemy MIERZALNIE potwierdzić,
że (1) nie tniemy w połowie słowa, (2) skróty nie robią fałszywych zdań, (3) overlap działa,
(4) rozmiary chunków trzymają się w rozsądnych widełkach.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ingest.chunk import split_sentences, pack_chunks


def test_split_respektuje_skroty():
    # "m.in." i "2022 r." NIE powinny kończyć zdania.
    text = "W raporcie m.in. o Danii. Wojna trwa od 2022 r. i nie ustaje. Koniec."
    sents = split_sentences(text)
    assert len(sents) == 3, sents
    assert sents[0] == "W raporcie m.in. o Danii."
    assert sents[1].startswith("Wojna trwa od 2022 r. i nie ustaje")


def test_split_inicjaly():
    # Inicjał "K." (pojedyncza litera) nie kończy zdania.
    text = "Autorem jest K. Kubiak. To ważne."
    sents = split_sentences(text)
    assert sents == ["Autorem jest K. Kubiak.", "To ważne."], sents


def test_chunki_nie_tna_slow():
    # Każdy chunk musi zaczynać i kończyć się na pełnym słowie/interpunkcji (nie w środku słowa).
    sentences = [f"To jest zdanie numer {i} w tekście testowym." for i in range(60)]
    chunks = pack_chunks(sentences, target=200, overlap=40, min_chars=50)
    assert len(chunks) > 1
    for c in chunks:
        assert not c.startswith(" ") and not c.endswith(" ")
        # brak "sklejonych" słów na granicy (proste przybliżenie: nie ma podwójnych spacji)
        assert "  " not in c


def test_overlap_istnieje():
    # Ostatnie zdanie chunka N powinno pojawić się na początku chunka N+1 (zakładka).
    sentences = [f"Zdanie {i} jest wystarczająco długie by wymusić podział na chunki." for i in range(40)]
    chunks = pack_chunks(sentences, target=200, overlap=80, min_chars=50)
    assert len(chunks) >= 2
    overlaps = 0
    for a, b in zip(chunks, chunks[1:]):
        last_sentence_of_a = a.split(". ")[-1]
        if last_sentence_of_a[:20] in b:
            overlaps += 1
    assert overlaps >= 1, "Brak wykrytej zakładki między chunkami"


def test_rozmiar_w_widelkach():
    # Chunki nie powinny drastycznie przekraczać targetu (target + jedno długie zdanie).
    sentences = [f"Krótkie zdanie {i}." for i in range(200)]
    target = 300
    chunks = pack_chunks(sentences, target=target, overlap=50, min_chars=50)
    longest_sentence = max(len(s) for s in sentences)
    for c in chunks:
        assert len(c) <= target + longest_sentence + 5, len(c)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}")
    print(f"\n{len(tests)} testów przeszło.")
