"""
tests/test_ab_prompt.py
Testy kontraktowe harnessu A/B promptu (`src/eval/ab_prompt.py`) — bez GPU, bez LLM-a.

CO TU JEST WARTE TESTU: nie same tabelki, a dwa kontrakty, których złamanie unieważniłoby
pomiar po cichu:
1. DRABINA WARIANTÓW — v3 musi być produkcyjnym promptem PLUS jedno zdanie, a v4 = v3 plus jedno
   zdanie. Gdyby ktoś przepisał v2 zamiast dopisać regułę, w raporcie „przed/po" zmieniłyby się
   DWIE rzeczy naraz i wniosek o efekcie reguły przestałby wynikać z liczb.
2. ODMOWA ZJADA KLASYFIKACJĘ — wiersz odmowny (modelu albo guardu) nie może wpaść do klasy
   wartościowej, bo wtedy „poprawna" liczyłaby odpowiedzi, których system nigdy nie wydał.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.eval.ab_prompt import (WARIANTY, REGULA_FORMY, REGULA_NIEPRZENOSZENIA, klasa,
                                tabela_kosztu, tabela_zysku, zysk_per_pytanie)
from src.generate.answer import SYSTEM_PROMPT

TRAP = {"id": "t05", "entity": "Łotwa", "answer": "1,548 mld dolarów.",
        "trap_values": ["2.316", "2.048"]}


def _row(variant="v3", rep=0, ans="Wydatki Łotwy wyniosły 1,548 mld dolarów [1].",
         refused=False, g3=False):
    return {"variant": variant, "rep": rep, "id": "t05", "answer": ans,
            "refused": refused, "guard3_blokuje": g3}


def test_drabina_wariantow_dokłada_po_jednym_zdaniu():
    assert WARIANTY["v2"] == SYSTEM_PROMPT                      # baseline = dokładnie produkcja
    assert WARIANTY["v3"] == SYSTEM_PROMPT + REGULA_FORMY       # v3 = v2 + JEDNA reguła
    assert WARIANTY["v4"] == WARIANTY["v3"] + REGULA_NIEPRZENOSZENIA
    # Reguły są dopisywane, nie wplatane: stary prompt zostaje prefiksem każdego wariantu.
    for w in WARIANTY.values():
        assert w.startswith(SYSTEM_PROMPT)


def test_v2bis_jest_bit_w_bit_tym_samym_promptem_co_v2():
    """Kontrola szumu ma sens TYLKO wtedy, gdy tekst jest identyczny — inaczej mierzy zmianę."""
    assert WARIANTY["v2bis"] == WARIANTY["v2"]


def test_generacja_przeplata_warianty_a_nie_leci_blokami():
    """Warianty muszą się zmieniać SZYBCIEJ niż powtórzenia.

    Układ blokowy (całe v2, potem całe v3) sprzęga „numer wariantu" z „czasem", a dryf w czasie
    jest w tym systemie zmierzonym faktem (etap 10) — wynik takiego pomiaru jest nierozplątywalny.
    Pierwszy przebieg tego harnessu miał ten błąd, więc kolejność pilnuje test, nie pamięć.
    """
    ab = sys.modules[klasa.__module__]
    oryginal = ab.answer
    ab.answer = lambda q, system="", gate=True, verify=False: {"answer": "x", "sources": []}
    try:
        rows = ab.generuj_zysk(["v2", "v3"], repeats=2)
    finally:
        ab.answer = oryginal
    kolejnosc = [(r["rep"], r["variant"]) for r in rows]
    # dedup z zachowaniem kolejności — interesują nas przejścia między blokami, nie 7 pułapek
    bloki = [k for i, k in enumerate(kolejnosc) if i == 0 or k != kolejnosc[i - 1]]
    assert bloki == [(0, "v2"), (0, "v3"), (1, "v2"), (1, "v3")]


def test_klasa_odmowy_nie_wchodzi_do_klas_wartosciowych():
    assert klasa(TRAP, _row(refused=True, ans="Nie znajduję odpowiedzi w źródłach.")) == "odmowa"
    assert klasa(TRAP, _row(), blocked=True) == "odmowa"        # zablokowana przez guard
    assert klasa(TRAP, _row()) == "poprawna"


def test_klasa_rozpoznaje_podmiane_podmiotu():
    # Wartość INNEGO państwa (2,316 = Estonia) bez właściwej -> klasa niebezpieczna.
    assert klasa(TRAP, _row(ans="Wydatki Łotwy wyniosły 2,316 mld dolarów [1].")) == "podmiana"


def test_guard3_liczy_sie_tylko_w_kolumnie_produkcyjnej():
    rows = [_row(g3=True)]
    assert tabela_zysku([TRAP], rows)["v3"]["poprawna"] == 1            # sam prompt
    assert tabela_zysku([TRAP], rows, z_guardem3=True)["v3"]["odmowa"] == 1   # + guard #3


def test_tabela_zysku_zawiera_wszystkie_klasy_takze_zerowe():
    t = tabela_zysku([TRAP], [_row()])
    assert set(t["v3"]) == {"poprawna", "niejednoznaczna", "podmiana", "inna-bledna", "odmowa"}


def test_zysk_per_pytanie_trzyma_kolejnosc_powtorzen():
    rows = [_row(rep=1, refused=True, ans="Nie znajduję odpowiedzi w źródłach."), _row(rep=0)]
    assert zysk_per_pytanie(rows, [TRAP])["t05"]["v3"] == ["poprawna", "odmowa"]


def test_tabela_kosztu_liczy_refusal_i_wskazuje_przecieki():
    rows = [{"variant": "v3", "rep": 0, "id": "g1", "zbior": "gold", "refused": True},
            {"variant": "v3", "rep": 1, "id": "g1", "zbior": "gold", "refused": False},
            {"variant": "v3", "rep": 0, "id": "h3", "zbior": "held-out", "refused": True}]
    t = tabela_kosztu(rows)
    assert t["v3"]["gold"] == {"n": 2, "odmowy": 1, "przecieki": ["g1(rep2)"], "refusal": 0.5}
    assert t["v3"]["held-out"]["refusal"] == 1.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}")
    print(f"\n{len(tests)} testów przeszło.")
