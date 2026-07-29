"""
tests/test_attribution.py
Guard #4 (przypisanie wielkości do podmiotu) — testy na NIESKRÓCONYCH fragmentach korpusu.

DLACZEGO „NIESKRÓCONYCH" JEST W PIERWSZYM ZDANIU: pierwsza wersja tych testów używała fragmentów
prawdziwych, ale SKRÓCONYCH — i skrót usunął zdanie „…stacja radiolokacyjna Globus III w Vardø
przy granicy z ROSJĄ". Na pełnym tekście reguła „najbliższa wcześniejsza wzmianka w bloku"
przypisywała 1,75% PKB *Rosji* i blokowała POPRAWNĄ odpowiedź o Norwegii. Testy były zielone,
bo mierzyły łatwiejszą rzeczywistość niż prawdziwa. Awarię złapał dopiero pomiar end-to-end.
Lekcja zapisana tutaj, żeby nie wróciła: fragment „prawdziwy, ale skrócony" nie jest prawdziwy.

Guard nie woła LLM-a, więc te testy przechodzą bez GPU, Ollamy i Qdranta.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.generate.attribution import (attributed_entities, check_attribution, entity_in,
                                      quantities)

# Fragment DOSŁOWNY z chunka #98 — z całym akapitem o Norwegii, w tym pułapką „przy granicy
# z Rosją" dwa zdania przed liczbą, oraz wzmianką o Danii wcześniej (to dlatego model je pomylił).
CTX_POLARNY = """[1] (źródło: Raport-OSW_Polarny-kwartet-NATO #98)
Z kolei Dania chce utrzymać równowagę między zwiększaniem aktywności sił zbrojnych na omawianym \
obszarze a dążeniem do redukowania napięć. Norwegia jest najsilniej zaangażowanym militarnie \
w Arktyce członkiem NATO. Obrona północnej części kraju stanowi priorytet jej sił zbrojnych. \
Zdolności rozpoznawczo-obserwacyjne zapewniają okręty zwiadowcze, samoloty P-8 i F-35 oraz \
stacja radiolokacyjna Globus III w Vardø przy granicy z Rosją. Plan rozwoju armii na lata \
2020-2028 zakłada podwyższenie wydatków obronnych o 2 mld dolarów. W 2021 r. wyniosły one \
8,4 mld dolarów, tj. 1,75% PKB.

[2] (źródło: Raport-OSW_Polarny-kwartet-NATO #97)
Dania na forum NATO przedstawia swoje zaangażowanie w Arktyce jako wzmacnianie bezpieczeństwa \
całego obszaru euroatlantyckiego. Ma ono kompensować zbyt małe wydatki obronne (5,3 mld \
dolarów, tj. 1,3% PKB w 2021 r.) i opóźnienia w modernizacji sił zbrojnych."""

# Fragment DOSŁOWNY: przypis 106 — TRZY wielkości i TRZY podmioty w JEDNYM zdaniu.
CTX_BALTYCKI = """[1] (źródło: Raport Nowe pomysły na obronę totalną #114)
Zagregowane wydatki obronne Estonii w tym okresie wyniosły 2,316 mld dolarów, podczas gdy \
łotewskie 1,548 mld, a litewskie 2,048 mld dolarów."""


def test_entity_in_rozpoznaje_odmienione_nazwy():
    assert entity_in("Ile wynosiły wydatki obronne Danii w 2021 r.?") == "Dania"
    assert entity_in("Ile wynosiły wydatki obronne Norwegii w 2021 r.?") == "Norwegia"
    assert entity_in("Ile wyniosły zagregowane wydatki obronne Łotwy?") == "Łotwa"
    assert entity_in("Co to jest reranking?") is None       # brak podmiotu -> abstynencja


def test_wielkosc_to_liczba_z_JEDNOSTKA_a_rok_nia_nie_jest():
    """Rok to modyfikator czasu, nie przypisywana ilość. Gdyby rok był wielkością, przypadkowe
    „2021" w odpowiedzi zaliczałoby trafienie w pytany podmiot i przepuszczało podmianę kwoty
    (wykryte pomiarem, nie rozumowaniem)."""
    assert quantities("W 2021 r. wyniosły one 8,4 mld dolarów, tj. 1,75% PKB.") == {"8.4", "1.75"}
    assert quantities("Akcesja nastąpiła w 2024 r.") == set()


def test_trzy_wielkosci_w_jednym_zdaniu_rozdzielone_poprawnie():
    """Najtrudniejszy przypadek korpusu — i ten, w którym guard realnie rozstrzyga:
    kolejność WEWNĄTRZ zdania wystarcza, bo każdy podmiot stoi bezpośrednio przed swoją liczbą."""
    assert attributed_entities(CTX_BALTYCKI, "2.316") == {"Estonia"}
    assert attributed_entities(CTX_BALTYCKI, "1.548") == {"Łotwa"}
    assert attributed_entities(CTX_BALTYCKI, "2.048") == {"Litwa"}


def test_REGRESJA_przypadkowa_nazwa_panstwa_NIE_porywa_przypisania():
    """Rdzeń naprawy: „przy granicy z Rosją" stoi dwa zdania przed liczbą Norwegii. Reguła
    ograniczona do zdania NIE MOŻE przypisać 1,75% PKB Rosji ani nikomu innemu — musi
    abstynować, bo zdanie z liczbą używa anafory („wyniosły ONE")."""
    assert attributed_entities(CTX_POLARNY, "1.75") == set()
    assert attributed_entities(CTX_POLARNY, "8.4") == set()


def test_ABSTYNENCJA_zamiast_zgadywania_gdy_zdanie_uzywa_anafory():
    """Świadomy wybór precyzji nad czułością: guard woli nie wiedzieć, niż zgadnąć.
    Zmierzona konsekwencja: podmiany Dania/Norwegia ten guard NIE łapie — łapie ją guard #3."""
    ok, det = check_attribution("Ile wynosiły wydatki obronne Danii w 2021 r.?",
                                "Wydatki obronne Danii w 2021 r. wyniosły 8,4 mld dolarów, "
                                "co stanowiło 1,75% PKB [1].", CTX_POLARNY)
    assert ok is None
    assert all(d.get("przypisana") is None for d in det)


def test_NIE_BLOKUJE_poprawnej_odpowiedzi_o_Norwegii():
    """Test kosztu — najważniejszy z całego pliku. Wersja sprzed naprawy blokowała tę odpowiedź
    (przypisywała liczbę Rosji), czyli zamieniała poprawną odpowiedź na odmowę."""
    ok, _ = check_attribution("Ile wynosiły wydatki obronne Norwegii w 2021 r.?",
                              "Wydatki obronne Norwegii w 2021 roku wyniosły 8,4 mld dolarów, "
                              "co stanowiło 1,75% PKB [1].", CTX_POLARNY)
    assert ok is not False


def test_LAPIE_podmiane_tam_GDZIE_ROZSTRZYGA_klaster_baltycki():
    """Pytanie o Estonię, odpowiedź z liczbą Łotwy — i podmiot stoi w tym samym zdaniu,
    więc guard MA podstawę do werdyktu i musi zablokować."""
    ok, det = check_attribution("Ile wyniosły zagregowane wydatki obronne Estonii?",
                                "Zagregowane wydatki obronne Estonii wyniosły 1,548 mld dolarów [1].",
                                CTX_BALTYCKI)
    assert ok is False
    assert [d for d in det if d["ocena"] == "PODMIANA PODMIOTU"][0]["przypisana"] == ["Łotwa"]


def test_przepuszcza_odpowiedz_wymieniajaca_kilka_podmiotow():
    """Odpowiedź podająca wielkości wszystkich trzech państw ma każdą przypisaną poprawnie —
    jest dobrą odpowiedzią o Estonii, tylko rozwlekłą. Guard nie może karać rozwlekłości."""
    ok, _ = check_attribution("Ile wyniosły zagregowane wydatki obronne Estonii?",
                              "Estonii 2,316 mld, łotewskie 1,548 mld, litewskie 2,048 mld [1].",
                              CTX_BALTYCKI)
    assert ok is True


def test_nie_przenosimy_podmiotu_MIEDZY_blokami_kontekstu():
    """Dwa bloki `[n]` to dwa różne fragmenty tekstu — przenoszenie podmiotu między nimi
    byłoby zmyślaniem powiązania."""
    assert "Norwegia" not in attributed_entities(CTX_POLARNY, "5.3")


def test_ABSTYNENCJA_gdy_pytanie_bez_podmiotu_albo_odpowiedz_bez_wielkosci():
    """Guard poza swoim zakresem musi mówić „nie wiem" (None), a nie zgadywać — decyzję
    przejmuje wtedy guard #3."""
    assert check_attribution("Co to jest reranking?", "To drugi etap retrievalu, 2 modele [1].",
                             CTX_BALTYCKI)[0] is None
    assert check_attribution("Ile wynosiły wydatki obronne Danii?",
                             "Nie znajduję odpowiedzi w źródłach.", CTX_POLARNY)[0] is None


def test_ABSTYNENCJA_gdy_wielkosci_nie_ma_w_kontekscie():
    """Liczba zmyślona (nieobecna w kontekście) nie ma przypisania — ten guard jej NIE ocenia.
    To zakres guardu #3 (brak poparcia). Rozdzielenie ról jest celowe."""
    ok, det = check_attribution("Ile wynosiły wydatki obronne Danii?",
                                "Wyniosły 99,9 mld dolarów [1].", CTX_POLARNY)
    assert ok is None
    assert det[0]["przypisana"] is None
