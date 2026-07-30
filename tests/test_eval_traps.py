"""
tests/test_eval_traps.py
Testy dwóch nowych części harnessu pułapek: audytu warunku wstępnego i wariantu „szersze jednostki".

DLACZEGO WŁAŚNIE TE DWIE: obie produkują liczby, które wchodzą do wniosku o guardzie #4.
Audyt wyznacza MIANOWNIK („na ilu pytaniach podmiana mogła w ogóle zajść"), a wariant jednostek
rozdziela abstynencję zaprojektowaną (anafora) od abstynencji przypadkowej (korpus pisze
„25 tys.", model odpowiada „25 tysięcy"). Pomyłka w którejkolwiek daje wniosek o systemie,
który jest w rzeczywistości wnioskiem o moim harnessie.

Testy nie wołają LLM-a — działają na zapisanych odpowiedziach, tak jak sam pomiar.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.eval import eval_traps as E
from src.generate import attribution as A

KONTEKST = """[1] (źródło: Raport Nowe pomysły na obronę totalną #112)
Priorytetami estońskiego Planu rozwoju obrony narodowej na lata 2017-2026 są pełna mechanizacja \
1. brygady wojsk lądowych oraz stopniowe powiększanie sił zbrojnych (z 21 do 25 tys. \
po mobilizacji)107 i liczby poborowych (z 3,2 do 4 tys. rocznie)."""


def test_szersze_jednostki_widza_liczbe_ROZWINIETA_w_slowa():
    """Zmierzona dziura: korpus skraca („25 tys."), a model rozwija („25 tysięcy"). Guard widział
    wtedy odpowiedź BEZ wielkości i milczał, choć miał w kontekście wszystko, żeby orzec."""
    odp = "Do 25 tysięcy żołnierzy po mobilizacji [1]."
    assert A.quantities(odp) == set()                      # stan produkcyjny: nic nie widzi
    with E._units(E._UNIT_SZEROKI):
        assert A.quantities(odp) == {"25"}                 # wariant: widzi i może orzekać
        assert A.attributed_entities(KONTEKST, "25") == {"Estonia"}


def test_units_PRZYWRACA_stan_takze_przy_wyjatku():
    """Wyciek podmienionego wzorca byłby najgorszym możliwym błędem tego harnessu: wszystkie
    KOLEJNE warianty mierzyłyby po cichu inny guard, a pomiar parowany przestałby być parowany."""
    orig = A._UNIT
    try:
        with E._units(E._UNIT_SZEROKI):
            assert A._UNIT != orig
            raise RuntimeError("celowo")
    except RuntimeError:
        pass
    assert A._UNIT == orig


def test_audyt_odroznia_pytanie_bez_cudzej_liczby_w_kontekscie(tmp_path, monkeypatch, capsys):
    """Rdzeń uczciwego mianownika: „podmiana nie zaszła" znaczy coś innego, gdy w kontekście
    NIE BYŁO cudzej liczby. Takie pytanie nie było pułapką i musi wypaść z mianownika."""
    traps = [
        {"id": "a1", "question": "Do ilu tysięcy żołnierzy po mobilizacji wzrosną siły zbrojne Estonii?",
         "answer": "Do 25 tys.", "entity": "Estonia", "subtype": "podmiana-podmiotu",
         "trap_values": ["280"], "trap_entity": "Finlandia", "evidence": "x", "source_chunk": "x"},
        {"id": "a2", "question": "Do ilu tysięcy żołnierzy po mobilizacji wzrosną siły zbrojne Estonii?",
         "answer": "Do 25 tys.", "entity": "Estonia", "subtype": "podmiana-podmiotu",
         "trap_values": ["999"], "trap_entity": "liczba nieobecna w kontekście",
         "evidence": "x", "source_chunk": "x"},
    ]
    tp = tmp_path / "traps.jsonl"
    tp.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in traps), encoding="utf-8")
    ap = tmp_path / "answers.json"
    ap.write_text(json.dumps({"rows": [
        {"id": t["id"], "question": t["question"], "entity": t["entity"],
         "answer": "Do 25 tys. żołnierzy [1].",
         "context": KONTEKST + "\n\n[2] (źródło: Raport #4)\nFinlandia zwiększa potencjał "
                   "rozwinięcia mobilizacyjnego na czas wojny do 280 tys. żołnierzy.",
         "refused_by_model": False} for t in traps]}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(E, "TRAPS_PATH", tp)
    monkeypatch.setattr(E, "ANSWERS", ap)
    monkeypatch.setattr(E, "RESULTS", tmp_path)

    out = E.audyt()
    assert out["a1"]["cudze_wartosci_w_kontekscie"] == ["280"]   # realna pułapka
    assert out["a2"]["cudze_wartosci_w_kontekscie"] == []        # pytanie, które nic nie mierzy
    assert "REALNE pułapki (cudza liczba w kontekście):        1/2" in capsys.readouterr().out
