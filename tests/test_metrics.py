"""
tests/test_metrics.py
Lekkie testy metryk (bez zależności; działa pod pytest i przez `python test_metrics.py`).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.eval.metrics import (classify_trap_answer, extract_values, hit_at_k, mean,
                              reciprocal_rank,
                              relevant_by_evidence, value_match)


def test_hit_at_k():
    ret = ["a", "b", "c", "d"]
    assert hit_at_k(ret, ["c"], k=3) == 1.0        # c jest w top-3
    assert hit_at_k(ret, ["c"], k=2) == 0.0        # c poza top-2
    assert hit_at_k(ret, ["x", "b"], k=2) == 1.0   # którykolwiek trafny wystarczy
    assert hit_at_k(ret, ["z"], k=4) == 0.0


def test_reciprocal_rank():
    ret = ["a", "b", "c"]
    assert reciprocal_rank(ret, ["a"]) == 1.0          # 1. pozycja
    assert reciprocal_rank(ret, ["b"]) == 0.5          # 2. pozycja
    assert abs(reciprocal_rank(ret, ["c"]) - 1/3) < 1e-9
    assert reciprocal_rank(ret, ["z"]) == 0.0          # brak
    assert reciprocal_rank(ret, ["c", "b"]) == 0.5     # bierze PIERWSZy trafny (b na poz. 2)


def test_mean():
    assert mean([1.0, 0.0, 0.5]) == 0.5
    assert mean([]) == 0.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}")
    print(f"\n{len(tests)} testów przeszło.")


# --- Prawda przez cytat-dowód (etap 8: A/B chunkingu) ------------------------------------
# Ta definicja zastępuje `chunk_id` przy porównywaniu strategii chunkingu: numeracja chunków
# przesuwa się po zmianie rozmiaru, więc identyfikatory przestają cokolwiek wskazywać.

def _ch(cid: str, text: str) -> dict:
    return {"id": cid, "text": text}


def test_evidence_znajduje_wszystkie_chunki_z_cytatem():
    """WSZYSTKIE, nie pierwszy: przy zakładce ten sam fragment jest w dwóch kolejnych chunkach
    i oba realnie zawierają odpowiedź."""
    chunks = [_ch("a__0", "Wstęp. Dania ma 1,3% PKB na obronę."),
              _ch("a__1", "Dania ma 1,3% PKB na obronę. Dalszy tekst."),
              _ch("a__2", "Nic o wydatkach.")]
    assert relevant_by_evidence(chunks, "Dania ma 1,3% PKB") == ["a__0", "a__1"]


def test_evidence_ignoruje_roznice_bialych_znakow():
    """Parser PDF-a zostawia artefakty typu „fińsko- -estońskiego" i złamania wierszy —
    dopasowanie ma być odporne na odstęp, nie na treść."""
    chunks = [_ch("a__0", "gazociągu\n Balticconnector   (październik 2023) uszkodzono")]
    assert relevant_by_evidence(chunks, "gazociągu Balticconnector (październik 2023)") == ["a__0"]


def test_evidence_bez_trafienia_daje_pusta_liste():
    """Pusta lista = pytanie NIEMOŻLIWE do trafienia przy tym podziale (np. cytat rozcięty
    na granicy chunków). Harness musi to zgłosić, a nie policzyć jako zwykłe pudło."""
    chunks = [_ch("a__0", "Dania ma"), _ch("a__1", "1,3% PKB na obronę.")]
    assert relevant_by_evidence(chunks, "Dania ma 1,3% PKB") == []


# --- Poprawność WARTOŚCI w odpowiedzi (atak na podmianę podmiotu) -------------------------
# DLACZEGO ta metryka istnieje: dotychczasowe metryki odpowiedzi są ŚLEPE na najgroźniejszą
# awarię systemu. Dla pytania „wydatki obronne Danii" pipeline podaje 1,75% PKB (liczbę
# Norwegii), a `faithfulness` liczy to jako WIERNE (sędzia głosuje TAK), `false-refusal` jako
# poprawne (system odpowiedział), `relevance` jako na temat. Żadna nie sprawdza, czy LICZBA
# jest ta właściwa — a `gold["answer"]` leżało w danych nieużywane. To domyka tę dziurę.

def test_wartosci_wyciaga_liczby_i_normalizuje_przecinek():
    assert extract_values("Około 1,3% PKB w 2024 r.") == {"1.3", "2024"}
    assert extract_values("1,30%") == {"1.3"}          # 1,30 i 1,3 to ta sama wartość
    assert extract_values("Bez liczb.") == set()


def test_wartosci_ignoruja_markery_cytowan():
    """Markery [1], [2] to numeracja źródeł, nie fakty — wpuszczone do porównania dawałyby
    fałszywe trafienia (odpowiedź z [3] „zawierałaby" wartość 3)."""
    assert extract_values("Dania przeznacza 1,3% PKB [3].") == {"1.3"}


def test_value_ok_gdy_wszystkie_oczekiwane_wartosci_sa_w_odpowiedzi():
    ok, missing = value_match("Około 13% ropy i 30% gazu.", "Arktyka kryje 13% ropy i 30% gazu [1].")
    assert ok is True and missing == []


def test_value_wrong_lapie_PODMIANE_PODMIOTU():
    """Rdzeń sprawy: h18. Oczekiwane 1,3% (Dania), system podaje 1,75% (Norwegia)."""
    ok, missing = value_match("Około 1,3% PKB.", "Dania przeznacza 1,75% PKB na obronę [2].")
    assert ok is False and missing == ["1.3"]


def test_value_match_bez_liczb_w_oczekiwanej_odpowiedzi_jest_NIEROZSTRZYGNIETY():
    """Pytania bez liczby (np. „które państwa tworzą kwartet") są POZA zakresem tej metryki —
    zwracamy None, żeby wykluczyć je z mianownika, a nie zaliczyć jako darmowe trafienie."""
    assert value_match("USA, Kanada, Dania i Norwegia.", "Kwartet to USA, Kanada, Dania [1].") == (None, [])


# --- Klasyfikacja odpowiedzi na pytania-pułapki (gold_set/entity_traps.jsonl) --------------
# DLACZEGO nie binarne „ok / nie ok": pułapka ma TRZY interesujące wyniki, nie dwa. Odpowiedź
# może podać wartość INNEGO podmiotu (to jest awaria h18, najgroźniejsza), może nie podać
# żadnej znanej wartości (zwykły błąd), albo podać oba podmioty naraz — a to ostatnie zwykle
# jest poprawne i rozwlekłe, nie halucynacja. Zlanie tego w jedną liczbę zgubiłoby różnicę
# między „system skłamał z cytatem" a „system był nieprecyzyjny".

def test_klasyfikacja_poprawna_gdy_wartosc_oczekiwana_bez_pulapki():
    assert classify_trap_answer("5,3 mld dolarów, czyli 1,3% PKB.", ["8.4", "1.75"],
                                "Dania przeznaczyła 5,3 mld dolarów [1].") == "poprawna"


def test_klasyfikacja_wystarczy_JEDNA_z_oczekiwanych_wartosci():
    """„1,3% PKB" bez kwoty to poprawna odpowiedź na pytanie o wydatki — wymaganie WSZYSTKICH
    wartości karałoby zwięzłość."""
    assert classify_trap_answer("5,3 mld dolarów, czyli 1,3% PKB.", ["8.4", "1.75"],
                                "To 1,3% PKB [2].") == "poprawna"


def test_klasyfikacja_PODMIANA_gdy_wartosc_pulapki_zamiast_oczekiwanej():
    """Rdzeń sprawy — h18: pytanie o Danię, odpowiedź z liczbą Norwegii."""
    assert classify_trap_answer("5,3 mld dolarów, czyli 1,3% PKB.", ["8.4", "1.75"],
                                "Dania przeznacza 1,75% PKB na obronę [2].") == "podmiana"


def test_klasyfikacja_niejednoznaczna_gdy_obie_wartosci_obecne():
    """Odpowiedź wymieniająca oba państwa jest zwykle poprawna, tylko rozwlekła — nie wolno jej
    liczyć jako halucynacji, ale trzeba ją wyodrębnić do oględzin."""
    assert classify_trap_answer("5,3 mld dolarów, czyli 1,3% PKB.", ["8.4", "1.75"],
                                "Dania 5,3 mld [1], a Norwegia 8,4 mld [2].") == "niejednoznaczna"


def test_klasyfikacja_inna_bledna_gdy_zadna_znana_wartosc():
    assert classify_trap_answer("5,3 mld dolarów, czyli 1,3% PKB.", ["8.4", "1.75"],
                                "Wydatki są zbyt małe wobec celu NATO [1].") == "inna-bledna"
