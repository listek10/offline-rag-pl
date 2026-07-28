"""
tests/test_serve_stream.py
Kontrakt mostka callback -> iterator. Bez FastAPI, bez modeli, bez sieci — dlatego trwa milisekundy.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
