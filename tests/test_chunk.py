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
from config import config
from src.ingest.chunk import split_sentences, pack_chunks, split_overlong, chunk_text


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


# --- Etap 14: dokumenty WYPUNKTOWANE (reguła awaryjna) ------------------------
# PROBLEM (zmierzony na 100-stronicowym raporcie `IV ROK WOJNY`): raport pisany wypunktowaniami
# nie ma kropek, więc cały akapit-lista wraca z `split_sentences` jako JEDNO "zdanie" (2867 zn.),
# a `pack_chunks` nie tnie w środku zdania -> chunk 5722 zn. Testy pilnują OBU warstw naprawy
# i — co równie ważne — pilnują, żeby naprawa NIE dotknęła normalnej prozy.

_LISTA = ("Obszar Działania 5. Zagrożenia hybrydowe " + " ".join(
    f"• wzmocnienie zdolności numer {i} w wymiarze wojskowym i cywilnym," for i in range(40)))


def test_dlugie_zdanie_dzielone_na_wypunktowaniach():
    # Fragment bez kropek, ale z wypunktowaniami: dzielimy NA GRANICACH punktów.
    assert len(_LISTA) > 1800, len(_LISTA)
    czesci = split_overlong(_LISTA, max_chars=1800, target=1200)
    assert len(czesci) > 1
    for c in czesci:
        assert len(c) <= 1800, len(c)
    # Cięcie poszło po punktach, nie w losowym miejscu: każda część (poza pierwszą) startuje od "•".
    assert all(c.startswith("•") for c in czesci[1:]), czesci[1:]


def test_dlugie_zdanie_bez_wypunktowan_nie_tnie_slow():
    # Ostatnia deska ratunku: brak wypunktowań (np. wyciąg z tabeli) -> cięcie po spacji.
    tabela = " ".join(f"obwód kirowski Igor Wasiljew Rosstat rotacja {i}" for i in range(60))
    czesci = split_overlong(tabela, max_chars=1000, target=800)
    assert len(czesci) > 1
    for c in czesci:
        assert len(c) <= 1000, len(c)
    # Żadne słowo nie zostało rozcięte: zbiór słów po podziale = zbiór słów przed podziałem.
    assert " ".join(czesci).split() == tabela.split()


def test_krotkie_zdanie_wraca_bez_zmian():
    # DLACZEGO to jest najważniejszy test w tej grupie: gwarantuje, że reguła awaryjna jest
    # AWARYJNA. Zdanie pod limitem musi wrócić bit w bit — na tym stoi ważność liczb z etapu 8.
    zdanie = "Dania przeznaczyła na obronę 1,75% PKB w 2023 r."
    assert split_overlong(zdanie, max_chars=1800, target=1200) == [zdanie]


def test_zakladka_nie_przenosi_molocha():
    # Druga warstwa: pętla zakładki brała ZAWSZE co najmniej jedno zdanie, choćby olbrzymie.
    # Efekt (zmierzony): chunk = moloch + następne zdanie, czyli 2867 + 2854 = 5722 zn.
    moloch = "A" * 1500
    nastepne = "B" * 1400
    chunks = pack_chunks([moloch, nastepne], target=1200, overlap=180,
                         min_chars=250, carry_max=600)
    assert len(chunks) == 2, chunks
    # Moloch NIE może wystąpić w drugim chunku (byłby tam jako zakładka).
    assert moloch not in chunks[1]
    for c in chunks:
        assert len(c) <= 1500, len(c)


def test_zakladka_dziala_dla_normalnej_prozy():
    # Kontrola po drugiej stronie: kap nie może zabić zakładki na zdaniach normalnej długości,
    # bo zakładka 15% to zmierzona decyzja z etapu 8, a nie ozdoba.
    zdania = [f"Zdanie {i} jest wystarczająco długie by wymusić podział na chunki." for i in range(40)]
    chunks = pack_chunks(zdania, target=200, overlap=80, min_chars=50, carry_max=600)
    zakladki = sum(1 for a, b in zip(chunks, chunks[1:]) if a.split(". ")[-1][:20] in b)
    assert zakladki >= 1, "Kap zakładki zabił zakładkę na normalnej prozie"


def test_split_sentences_nietkniete_przez_regule_awaryjna():
    # DLACZEGO: `split_sentences` jest współdzielone z guardem #3 (`verify.to_claims`), guardem #4
    # i harnessem z etapu 13. Reguła awaryjna należy WYŁĄCZNIE do ingestu — gdyby weszła tutaj,
    # po cichu zmieniłaby definicję "tezy" u sędziego i unieważniła pomiary z etapu 13.
    assert split_sentences(_LISTA) == [" ".join(_LISTA.split())]


def test_chunk_text_trzyma_ogon_w_ryzach():
    # Spięcie obu warstw na dokumencie listowym: żaden chunk nie przekracza targetu o więcej
    # niż limit awaryjny (a bez naprawy przekraczał go czterokrotnie).
    rekordy = chunk_text("\n\n".join([_LISTA] * 5), source="listowy")
    assert rekordy
    najdluzszy = max(r["n_chars"] for r in rekordy)
    assert najdluzszy <= config.CHUNK_MAX_SENTENCE_CHARS, najdluzszy


def test_domyslny_kap_jest_skonczony():
    # Bezpiecznik na przyszłość: gdyby ktoś "wyłączył" kap wartością ujemną/None, wraca patologia.
    assert 0 < config.CHUNK_CARRY_MAX_CHARS <= config.CHUNK_TARGET_CHARS
    assert config.CHUNK_TARGET_CHARS < config.CHUNK_MAX_SENTENCE_CHARS


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}")
    print(f"\n{len(tests)} testów przeszło.")
