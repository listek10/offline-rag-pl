"""
tests/test_ab_claims.py
Testy kontraktowe harnessu A/B definicji tezy (`src/eval/ab_claims.py`) — bez GPU, bez LLM-a.

CO TU JEST WARTE TESTU (nie tabelki, a rzeczy, których złamanie unieważniłoby pomiar po cichu):
1. STARA REGUŁA musi zostać stara. Harness trzyma jedyną kopię progu 15 znaków w repo — gdyby
   ktoś „posprzątał" ją do nowej definicji, `roznica()` zwróciłaby pustkę i raport pokazałby
   „zmiana nic nie dotyka", czyli dokładnie odwrotność prawdy.
2. DWA MECHANIZMY, DWA ZBIORY WIERSZY. Poprawka zmienia (a) co jest tezą i (b) co się dzieje przy
   zerze tez. Harness widzący tylko (a) przegapiłby `"[1]"` — właściwą dziurę fail-open (19 z 35
   dotkniętych wierszy). Test pilnuje, że oba mechanizmy są rozpoznawane i rozróżnione.
3. ETYKIETA OCZEKIWANIA rozstrzyga, czy blokada jest zyskiem czy false-refusalem. Pomyłka tutaj
   zamienia wynik w jego przeciwieństwo, a tabela wygląda równie wiarygodnie.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.eval.ab_claims import STARY_PROG, oczekiwanie, roznica, tezy_stara_regula

OOC = {"o05", "h27"}                                  # pytania spoza korpusu
WZORCE = {"g13": "Około 65%.", "t05": "1,548 mld dolarów."}   # id -> wzorcowa odpowiedź


def _row(answer: str, ident: str = "x") -> dict:
    return {"plik": "test.json", "id": ident, "answer": answer, "context": "ctx"}


def test_stara_regula_nadal_odsiewa_ponizej_progu():
    """Jedyna kopia usuniętej linijki — bez niej nie ma „przed" w pomiarze przed/po."""
    assert tezy_stara_regula("65% [1]") == []
    assert tezy_stara_regula("120 [1]") == []
    assert STARY_PROG == 15
    dluga = "Wydatki Łotwy wyniosły 1,548 mld dolarów [1]."
    assert len(tezy_stara_regula(dluga)) == 1          # powyżej progu: stara i nowa reguła zgodne


def test_roznica_rozdziela_dwa_mechanizmy_poprawki():
    wynik = roznica([
        _row("65% [1]"),                                          # nagła wielkość staje się tezą
        _row("[1]"),                                              # zero treści po usunięciu cytatu
        _row("Wydatki Łotwy wyniosły 1,548 mld dolarów [1]."),    # bez zmiany
    ])
    assert [w["powód"] for w in wynik] == ["nowe tezy", "zero tez"]
    assert wynik[0]["tezy_stare"] == [] and wynik[0]["tezy_nowe"] == ["65%"]
    assert wynik[1]["tezy_nowe"] == []                 # pusto w OBU regułach -> decyduje fail-closed


def test_oczekiwanie_brak_tresci_to_blokada_takze_dla_pytania_z_korpusu():
    """`"[1]"` na pytanie Z korpusu nie jest „krótką poprawną odpowiedzią" — to uszkodzony przypis.
    Gdyby etykieta mówiła `przejście`, blokada `g25` policzyłaby się jako false-refusal."""
    assert oczekiwanie(_row("[1]", "g13"), OOC, WZORCE) == "blokada"


def test_oczekiwanie_rozdziela_przeciek_od_poprawnej_przy_tym_samym_kształcie():
    # `120 [1]` (spoza korpusu) i `65% [1]` (z korpusu, wartość wzorcowa) różnią się WYŁĄCZNIE
    # pytaniem — to jest cała treść pomiaru, więc etykietowanie musi to widzieć.
    assert oczekiwanie(_row("120 [1]", "o05"), OOC, WZORCE) == "blokada"
    assert oczekiwanie(_row("65% [1]", "g13"), OOC, WZORCE) == "przejście"


def test_oczekiwanie_nieorzeczone_nie_wpada_do_zadnej_strony():
    """Odpowiedź na temat, ale bez wzorca do porównania, musi zostać `?` — dopisanie jej do
    którejkolwiek kolumny przesuwałoby wynik w wygodną stronę bez dowodu."""
    assert oczekiwanie(_row("Coś tam o obronie [1].", "nieznane"), OOC, WZORCE) == "?"
    # Wartość NIEZGODNA z wzorcem też nie jest `przejście` — poprawności nie zakładamy.
    assert oczekiwanie(_row("9,9% PKB [1].", "g13"), OOC, WZORCE) == "?"
