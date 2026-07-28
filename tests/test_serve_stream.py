"""
tests/test_serve_stream.py
Kontrakt mostka callback -> iterator. Bez FastAPI, bez modeli, bez sieci — dlatego trwa milisekundy.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.serve.stream import stream_events


def _on_error(exc: BaseException) -> dict:
    return {"where": "test", "message": type(exc).__name__}


def test_zdarzenia_w_kolejnosci_a_final_na_koncu():
    def work(emit):
        emit(event="stage", stage="a", status="start")
        emit(event="stage", stage="a", status="done")
        return {"answer": "gotowe"}

    events = list(stream_events(work, _on_error))

    assert [e.get("stage") for e in events[:2]] == ["a", "a"]
    assert events[-1] == {"event": "final", "answer": "gotowe"}
    assert len(events) == 3


def test_wyjatek_daje_error_a_nie_final_i_nie_ubija_wczesniejszych_zdarzen():
    def work(emit):
        emit(event="stage", stage="a", status="start")
        raise ValueError("bum")

    events = list(stream_events(work, _on_error))

    assert events[0]["stage"] == "a"
    assert events[-1] == {"event": "error", "where": "test", "message": "ValueError"}
    assert not any(e["event"] == "final" for e in events)


def test_iterator_sie_konczy_gdy_work_nic_nie_emituje():
    events = list(stream_events(lambda emit: {"answer": ""}, _on_error))
    assert events == [{"event": "final", "answer": ""}]


def test_zdarzenia_docieraja_W_TRAKCIE_pracy_a_nie_po_niej():
    """To jedyny test, który broni SENSU tego modułu.

    Trzy testy powyżej sprawdzają treść i kolejność — i przeszłyby także implementacji
    BEZ wątku, która zbiera zdarzenia do listy i oddaje je hurtem na końcu. Taka wersja
    działałaby „poprawnie" i jednocześnie pokazywałaby użytkownikowi pusty panel przez 20 s.

    Tu odwracamy zależność czasową: `work` czeka na `pozwolenie`, którego test udziela
    DOPIERO PO odczytaniu pierwszego zdarzenia. Implementacja bez wątku zakleszczyłaby się
    w tym miejscu (pierwsze zdarzenie nigdy by nie przyszło), więc test padłby na timeout.
    """
    pozwolenie = threading.Event()
    skonczone = threading.Event()

    def work(emit):
        emit(event="stage", stage="a", status="start")
        if not pozwolenie.wait(timeout=5):
            raise AssertionError("test nie odblokowal work() — mostek nie strumieniuje")
        skonczone.set()
        return {"answer": "gotowe"}

    it = stream_events(work, _on_error)

    pierwsze = next(it)                     # musi przyjść, choć work() jeszcze NIE skończyła
    assert pierwsze == {"event": "stage", "stage": "a", "status": "start"}
    assert not skonczone.is_set(), "work() zdazyla sie skonczyc => to nie jest strumieniowanie"

    pozwolenie.set()
    assert next(it) == {"event": "final", "answer": "gotowe"}
    with pytest.raises(StopIteration):
        next(it)
