"""
tests/test_answer_events.py
Kontrakt zdarzeń emitowanych przez pętlę RAG. Retrieval i LLM są PODMIENIONE — test nie dotyka
GPU, Qdranta ani Ollamy, więc realnie się go uruchamia (test wymagający usług nie jest testem).

Sprawdzamy to, na co UI liczy: kolejność etapów, statusy `skipped`/`not_run` oraz to, że
zwracany słownik został NIEZMIENIONY (inaczej harness ewaluacyjny i runtime by się rozjechały).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import create_autospec

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config
from src.generate import answer as answer_mod
from src.retrieve.hit import make_hit, set_rerank_score
from src.retrieve.hybrid import hybrid_search


def _hit(score: float) -> dict:
    """
    Atrapa trafienia zbudowana PRAWDZIWĄ fabryką (`make_hit`), nie ręcznie napisanym słownikiem.
    DLACZEGO TAK: gdyby kontrakt trafienia się zmienił, ręczna kopia dalej dokładałaby stary
    klucz i test zostałby ZIELONY — a produkcja by padła (`answer.py` czyta `rerank_score`
    z domyślną zerówką, więc utrata tego pola nie krzyczy, tylko zamyka bramkę na wszystkim).
    Argumenty `make_hit` są tylko po nazwie, więc zmiana kontraktu daje tu `TypeError`.
    """
    hit = make_hit(score=0.01, source="raport", chunk_index=7, chunk_id="raport__7",
                   text="Norwegia przeznacza 1,75% PKB na obronę.")
    set_rerank_score(hit, score)
    return hit


def _fake_retrieval(hits: list[dict]):
    """
    Atrapa `hybrid_search` zbudowana Z PRAWDZIWEJ FUNKCJI (`create_autospec`), nie z lambdy.
    DLACZEGO: lambda ma sygnaturę PRZEPISANĄ Z PAMIĘCI — to druga (po kształcie zwrotki)
    powierzchnia dryfu. Autospec sprawdza wywołanie względem prawdziwej sygnatury, więc
    przemianowanie `top_k` czy `rerank` w `hybrid.py` wywala test, a nie produkcję.
    """
    return create_autospec(hybrid_search, return_value=hits)


@pytest.fixture
def collected(monkeypatch):
    """Zbiera zdarzenia i pozwala testowi ustawić zachowanie retrievalu/LLM-a/sędziego."""
    events: list[dict] = []
    monkeypatch.setattr(answer_mod, "hybrid_search", _fake_retrieval([_hit(0.9)]))
    # Podmieniamy DYSPOZYTOR (`call_llm`), nie konkretny backend: od etapu 7 pętla RAG woła
    # generator przez niego, więc to jest jedyny szew, który izoluje test od Ollamy i vLLM naraz.
    monkeypatch.setattr(answer_mod, "call_llm", lambda *a, **k: "Norwegia daje 1,75% PKB [1].")
    return events


def _stages(events: list[dict]) -> list[tuple[str, str]]:
    return [(e["stage"], e["status"]) for e in events if e["event"] == "stage"]


def test_sciezka_pelna_emituje_wszystkie_etapy(collected, monkeypatch):
    monkeypatch.setattr("src.generate.verify.grounded_fraction",
                        lambda ctx, txt: (1.0, [{"claim": "c", "grounded": True, "reason": "r"}]))

    result = answer_mod.answer("pytanie", verify=True, on_event=lambda **e: collected.append(e))

    assert _stages(collected) == [
        ("retrieval", "start"), ("retrieval", "done"),
        ("gate", "done"),
        ("generation", "start"), ("generation", "done"),
        ("verify", "start"), ("verify", "done"),
    ]
    assert result["answer"] == "Norwegia daje 1,75% PKB [1]."
    assert "blocked_by" not in result


def test_retrieval_done_niesie_ms_topscore_i_surowe_hity(collected):
    answer_mod.answer("pytanie", verify=False, on_event=lambda **e: collected.append(e))
    done = next(e for e in collected if e["stage"] == "retrieval" and e["status"] == "done")
    assert isinstance(done["ms"], int) and done["ms"] >= 0
    assert done["top_score"] == pytest.approx(0.9)
    assert done["hits"][0]["chunk_index"] == 7          # surowy hit, bez kluczy UI
    assert "n" not in done["hits"][0]


def test_bramka_blokuje_a_dalsze_etapy_sa_not_run(collected, monkeypatch):
    monkeypatch.setattr(answer_mod, "hybrid_search", _fake_retrieval([_hit(0.01)]))

    result = answer_mod.answer("pytanie", gate=True, on_event=lambda **e: collected.append(e))

    assert _stages(collected) == [
        ("retrieval", "start"), ("retrieval", "done"),
        ("gate", "done"),
        ("generation", "not_run"), ("verify", "not_run"),
    ]
    gate = next(e for e in collected if e["stage"] == "gate")
    assert gate["passed"] is False
    assert gate["threshold"] == config.RELEVANCE_GATE_MIN
    assert result["blocked_by"] == "gate"


def test_wylaczone_guardy_daja_skipped_nie_cisze(collected):
    answer_mod.answer("pytanie", gate=False, verify=False,
                      on_event=lambda **e: collected.append(e))
    assert ("gate", "skipped") in _stages(collected)
    assert ("verify", "skipped") in _stages(collected)


def test_verify_odrzuca_i_zachowuje_odrzucona_odpowiedz(collected, monkeypatch):
    monkeypatch.setattr("src.generate.verify.grounded_fraction",
                        lambda ctx, txt: (0.0, [{"claim": "c", "grounded": False, "reason": "inny podmiot"}]))

    result = answer_mod.answer("pytanie", verify=True, on_event=lambda **e: collected.append(e))

    assert result["blocked_by"] == "verify"
    assert result["rejected_answer"] == "Norwegia daje 1,75% PKB [1]."
    done = next(e for e in collected if e["stage"] == "verify" and e["status"] == "done")
    assert done["grounded_fraction"] == 0.0
    assert done["verdicts"][0]["grounded"] is False


def test_bez_on_event_nic_sie_nie_psuje(collected):
    result = answer_mod.answer("pytanie", verify=False)
    assert result["answer"]
    assert collected == []
