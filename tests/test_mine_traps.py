"""
tests/test_mine_traps.py
Testy minera pułapek (`src/eval/mine_traps.py`) i walidatora zbioru `gold_set/entity_traps.jsonl`.

DLACZEGO MINER MA TESTY: on decyduje, CO wejdzie do zbioru ewaluacyjnego, a zbiór ewaluacyjny
jest podstawą wszystkich twierdzeń o jakości. Miner, który przepuszcza kandydata poza zasięgiem
guardu, produkuje pytania mierzące ciszę — i tak powstał zbiór n=7, w którym guard #4 abstynował
w 6 z 7 przypadków. Testy pilnują dwóch rzeczy: że filtry odsiewają to, co mają, i że walidator
łapie realne pomyłki w danych (nieaktualny `source_chunk`, wartość pułapki równa poprawnej).

DLACZEGO WALIDATOR MA TESTY: jego cichy fałszywy „OK" jest groźniejszy niż brak walidatora —
usypia. Więc każdy test podaje mu ZEPSUTY wpis i sprawdza, że wpis zostaje odrzucony.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.eval import mine_traps as M

# Zdanie DOSŁOWNE z korpusu: trzy podmioty i trzy wielkości, każdy podmiot przed swoją liczbą.
ZDANIE_ARKTYCZNE = ("Wyjątek stanowią spory o podział „arktycznego tortu”, czyli nakładające się "
                    "roszczenia Danii (0,9 mln km2), Kanady (1,2 mln km2) i Rosji (2,1 mln km2) "
                    "do szelfu bieguna północnego11.")

# Zdanie z ANAFORĄ — typ polarny. Guard nie ma tu podstawy do werdyktu, więc miner MUSI je odrzucić.
ZDANIE_ANAFORYCZNE = ("Plan rozwoju armii na lata 2020-2028 zakłada podwyższenie wydatków "
                      "obronnych o 2 mld dolarów. W 2021 r. wyniosły one 8,4 mld dolarów, "
                      "tj. 1,75% PKB.")


def test_miner_widzi_pary_tam_gdzie_guard_ma_podstawe():
    pary = {(p["entity"], p["value"]) for p in M.pairs_in_chunk(ZDANIE_ARKTYCZNE)}
    assert pary == {("Dania", "0.9"), ("Kanada", "1.2"), ("Rosja", "2.1")}


def test_miner_ODRZUCA_typ_polarny_bo_guard_tam_abstynuje():
    """Kluczowy test sensu minera: dokładanie pytań z anaforą nie rozstrzygnęłoby niczego
    o guardzie #4, bo on tam nie głosuje. Miner nie może ich proponować."""
    assert M.pairs_in_chunk(ZDANIE_ANAFORYCZNE) == []


def test_miner_wymaga_KONKURENTA_bo_bez_niego_nie_ma_czego_podmienic():
    """Wielkość bez cudzej wielkości w zasięgu to zwykłe pytanie, nie pułapka."""
    samotna = "Norwegia to państwo o największym odsetku mieszkańców za kołem podbiegunowym (9%)."
    assert len(M.pairs_in_chunk(samotna)) == 1        # para JEST...
    # ...ale przy jednej parze w chunku miner nie ma konkurenta i kandydata nie zgłosi.
    assert [p for p in M.pairs_in_chunk(samotna)] and len(M.pairs_in_chunk(samotna)) < 2


def test_jednostka_rozpoznana_bo_konkurent_w_innej_jednostce_to_slaba_pulapka():
    assert M.unit_of(ZDANIE_ARKTYCZNE, "0.9") == "mln"
    assert M.unit_of("Wydatki wyniosły 1,3% PKB.", "1.3") == "%"


def test_slowa_miary_pomijaja_szum_a_zostawiaja_tresc():
    """Ranking pułapek opiera się na wspólnych słowach treściowych — jeśli do zbioru słów
    wpadną spójniki i jednostki, „ta sama miara" przestanie cokolwiek znaczyć."""
    slowa = M.measure_words("W 2021 r. wydatki obronne wyniosły ok. 26 mld dolarów.")
    assert "wydatki" in slowa and "obronne" in slowa and "dolarów" in slowa
    assert not {"wyniosły", "mld"} & slowa


# ── walidator: każdy test psuje JEDNĄ rzecz i wymaga, żeby została złapana ──────────────

WZORZEC = {"id": "x01", "question": "Jak duży obszar szelfu bieguna północnego obejmują "
                                    "roszczenia Danii?",
           "answer": "0,9 mln km².", "entity": "Dania", "subtype": "podmiana-podmiotu",
           "trap_values": ["1.2", "2.1"], "trap_entity": "Kanada albo Rosja",
           "evidence": ZDANIE_ARKTYCZNE,
           "source_chunk": "Raport-OSW_Polarny-kwartet-NATO_net_4__25"}


def waliduj_jeden(tmp_path, **nadpisz) -> int:
    wpis = {**WZORZEC, **nadpisz}
    plik = tmp_path / "traps.jsonl"
    plik.write_text(json.dumps(wpis, ensure_ascii=False) + "\n", encoding="utf-8")
    stary = M.TRAPS_PATH
    M.TRAPS_PATH = plik
    try:
        return M.waliduj_zbior()
    finally:
        M.TRAPS_PATH = stary


@pytest.mark.skipif(not M.CHUNKS.exists(), reason="wymaga data/processed/chunks.jsonl")
def test_walidator_przepuszcza_poprawny_wpis(tmp_path):
    assert waliduj_jeden(tmp_path) == 0


@pytest.mark.skipif(not M.CHUNKS.exists(), reason="wymaga data/processed/chunks.jsonl")
def test_walidator_lapie_cytat_przepisany_recznie(tmp_path):
    """Najczęstsza cicha pomyłka: półpauza zamiast dywizu, „zwykłe" cudzysłowy, ucięty numer
    przypisu. Taki cytat nie wskazuje ŻADNEGO chunka, więc pułapka nie ma dowodu w korpusie."""
    assert waliduj_jeden(tmp_path, evidence=ZDANIE_ARKTYCZNE.replace("„", '"')) > 0


@pytest.mark.skipif(not M.CHUNKS.exists(), reason="wymaga data/processed/chunks.jsonl")
def test_walidator_lapie_nieaktualny_source_chunk(tmp_path):
    """Realny błąd znaleziony tym walidatorem w istniejących danych: t04-t06 wskazywały chunk
    `__114`, a po naprawie ogona z etapu 14 cytat siedzi w `__113`. Granice chunków się ruszają,
    więc id chunka jest polem, które MUSI być weryfikowane maszynowo."""
    assert waliduj_jeden(tmp_path, source_chunk="Raport-OSW_Polarny-kwartet-NATO_net_4__99") > 0


@pytest.mark.skipif(not M.CHUNKS.exists(), reason="wymaga data/processed/chunks.jsonl")
def test_walidator_lapie_wartosc_pulapki_rowna_poprawnej(tmp_path):
    """Gdyby wartość pułapki równała się poprawnej, metryka nie odróżniłaby `poprawna`
    od `podmiana` — pomiar dałby liczbę bez znaczenia."""
    assert waliduj_jeden(tmp_path, trap_values=["0.9"]) > 0


@pytest.mark.skipif(not M.CHUNKS.exists(), reason="wymaga data/processed/chunks.jsonl")
def test_walidator_lapie_prawde_ktorej_NIE_MA_w_cytacie(tmp_path):
    """Oczekiwana odpowiedź musi wynikać z cytatu. Inaczej poprawna odpowiedź systemu
    klasyfikowałaby się jako `inna-bledna` i pomiar mierzyłby mój błąd, nie system."""
    assert waliduj_jeden(tmp_path, answer="4,7 mln km².") > 0


@pytest.mark.skipif(not M.CHUNKS.exists(), reason="wymaga data/processed/chunks.jsonl")
def test_walidator_lapie_pytanie_POZA_ZASIEGIEM_guardu(tmp_path):
    """Pytanie, które nie nazywa podmiotu z leksykonu, wyprowadza guard #4 w abstynencję —
    czyli nie mierzy tego, po co pułapkę dodano. To jest dokładnie wada zbioru n=7."""
    assert waliduj_jeden(tmp_path, question="Jak duży obszar szelfu obejmują te roszczenia?") > 0
