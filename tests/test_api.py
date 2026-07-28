"""
tests/test_api.py
Kontrakt HTTP warstwy serwującej. Qdrant, Ollama i cała pętla RAG są PODMIENIONE — test
sprawdza KONTRAKT (kształt odpowiedzi, kolejność zdarzeń, kody błędów), a nie jakość modelu.

UWAGA na `TestClient(app)` BEZ menedżera kontekstu: dzięki temu `lifespan` się nie uruchamia,
więc rozgrzewka nie próbuje ładować BGE-M3. Test ma trwać milisekundy i działać bez GPU.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
import urllib.error
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config
from src.serve import api as api_mod

client = TestClient(api_mod.app)


def test_health_zwraca_progi_i_modele_z_configu(monkeypatch):
    monkeypatch.setattr(api_mod, "_probe_qdrant",
                        lambda: {"ok": True, "collection": "kolekcja", "points": 1509})
    monkeypatch.setattr(api_mod, "_probe_ollama",
                        lambda: {"ok": True, "models": ["qwen2.5:7b"], "missing": []})

    body = client.get("/api/health").json()

    assert body["qdrant"]["points"] == 1509
    assert body["ollama"]["ok"] is True
    # Progi MUSZĄ pochodzić z configu — inaczej UI kłamałoby po zmianie progu.
    assert body["config"]["gate_min"] == config.RELEVANCE_GATE_MIN
    assert body["config"]["top_k"] == config.TOP_K
    assert body["config"]["judge"] == config.JUDGE_MODEL
    assert body["warm"] is False        # lifespan nie odpalony => nierozgrzane


def test_health_nie_wywala_sie_gdy_uslugi_leza(monkeypatch):
    def boom():
        raise ConnectionError("brak polaczenia")

    monkeypatch.setattr(api_mod, "_probe_qdrant", lambda: api_mod._safe_probe(boom, "qdrant"))
    monkeypatch.setattr(api_mod, "_probe_ollama", lambda: {"ok": False, "models": [], "missing": []})

    resp = client.get("/api/health")

    assert resp.status_code == 200          # health NIGDY nie zwraca 5xx — inaczej UI nie wie nic
    assert resp.json()["qdrant"]["ok"] is False


def _ask(payload: dict) -> list[dict]:
    """Wysyła zapytanie i parsuje strumień NDJSON na listę zdarzeń."""
    resp = client.post("/api/ask", json=payload)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


def _fake_answer(events: list[dict], result: dict):
    """Podmiana `answer()`: emituje zaplanowaną sekwencję zdarzeń i zwraca zadany wynik."""
    def fake(question, top_k=None, gate=True, verify=None, on_event=None, **kw):
        for e in events:
            on_event(**e)
        return result
    return fake


def test_strumien_konczy_sie_zdarzeniem_final(monkeypatch):
    monkeypatch.setattr(api_mod, "answer", _fake_answer(
        [{"event": "stage", "stage": "retrieval", "status": "start"},
         {"event": "stage", "stage": "retrieval", "status": "done", "ms": 12,
          "top_score": 0.9, "hits": [{"source": "raport", "chunk_index": 7,
                                      "score": 0.1, "rerank_score": 0.9, "text": "tekst"}]}],
        {"answer": "odpowiedź [1]", "sources": []}))

    events = _ask({"question": "pytanie"})

    assert events[-1]["event"] == "final"
    assert events[-1]["answer"] == "odpowiedź [1]"
    assert isinstance(events[-1]["total_ms"], int)
    assert sum(1 for e in events if e["event"] == "final") == 1


def test_hity_dostaja_numeracje_n_zgodna_z_cytatami(monkeypatch):
    monkeypatch.setattr(api_mod, "answer", _fake_answer(
        [{"event": "stage", "stage": "retrieval", "status": "done", "ms": 5, "top_score": 0.9,
          "hits": [{"source": "a", "chunk_index": 1, "score": 0.1, "rerank_score": 0.9, "text": "x"},
                   {"source": "b", "chunk_index": 2, "score": 0.1, "rerank_score": 0.7, "text": "y"}]}],
        {"answer": "ok", "sources": []}))

    hits = next(e for e in _ask({"question": "q"}) if e.get("stage") == "retrieval")["hits"]

    assert [h["n"] for h in hits] == [1, 2]              # numeracja jak w build_context()
    assert hits[0]["score"] == 0.9                        # na drucie: score RERANKERA
    assert "rerank_score" not in hits[0]


def test_odmowa_guardu_3_niesie_odrzucona_odpowiedz(monkeypatch):
    monkeypatch.setattr(api_mod, "answer", _fake_answer(
        [], {"answer": "Nie znajduję odpowiedzi w źródłach.", "sources": [],
             "blocked_by": "verify", "rejected_answer": "Dania daje 1,75% PKB [1].",
             "grounding": [{"claim": "c", "grounded": False, "reason": "to Norwegia"}]}))

    final = _ask({"question": "q"})[-1]

    assert final["blocked_by"] == "verify"
    assert final["rejected_answer"] == "Dania daje 1,75% PKB [1]."
    assert final["self_refused"] is False


def test_odmowa_modelu_samego_jest_rozpoznana_jako_guard_1(monkeypatch):
    """Guard #1 nie ustawia `blocked_by` — bez `self_refused` UI zlałby go z 'model nic nie wie'."""
    monkeypatch.setattr(api_mod, "answer", _fake_answer(
        [], {"answer": "Nie znajduję odpowiedzi w źródłach.", "sources": []}))

    final = _ask({"question": "q"})[-1]

    assert final["blocked_by"] is None
    assert final["self_refused"] is True


def test_wyjatek_daje_zdarzenie_error_z_komunikatem_po_polsku(monkeypatch):
    def boom(*a, **k):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(api_mod, "answer", boom)

    events = _ask({"question": "q"})

    assert events[-1]["event"] == "error"
    assert events[-1]["where"] == "ollama"
    assert "ollama serve" in events[-1]["message"]
    assert not any(e["event"] == "final" for e in events)


def _wolny_port() -> int:
    """Port wybrany przez system — żeby test nie kolidował z działającym serwerem na 8000."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_zdarzenia_ida_po_HTTP_w_trakcie_pracy_a_nie_hurtem_na_koncu(monkeypatch):
    """Broni strumieniowania na warstwie HTTP (test w test_serve_stream.py broni go w mostku).

    DLACZEGO PRAWDZIWY UVICORN, A NIE TestClient (sprawdzone, nie założone):
    `httpx/_transports/asgi.py` zbiera CAŁE ciało odpowiedzi do listy `body_parts`, czeka aż
    `more_body` będzie False i dopiero wtedy buduje `Response`. `TestClient` z zasady nie
    potrafi więc zobaczyć strumienia przyrostowo — pod nim ten test zakleszcza się na timeoucie
    nawet przy poprawnej implementacji. Dlatego jedyny sposób sprawdzenia tej właściwości to
    prawdziwy serwer po prawdziwym TCP.

    PO CO w ogóle ten test: pozostałe czytają `resp.text`, czyli całość PO zakończeniu, więc
    przeszłyby także wersji buforowanej (np. po dodaniu kompresji albo zamianie
    StreamingResponse na zwykły JSON). Wtedy panel przebiegu zapalałby się dopiero na końcu —
    funkcja istniałaby na papierze i nie działała.

    Mechanizm: `answer()` czeka na pozwolenie, którego test udziela DOPIERO PO odczytaniu
    pierwszej linii ze strumienia.
    """
    pozwolenie = threading.Event()
    skonczone = threading.Event()

    def fake(question, top_k=None, gate=True, verify=None, on_event=None, **kw):
        on_event(event="stage", stage="retrieval", status="start")
        if not pozwolenie.wait(timeout=10):
            raise AssertionError("test nie odblokowal answer() — HTTP nie strumieniuje")
        skonczone.set()
        return {"answer": "gotowe", "sources": []}

    monkeypatch.setattr(api_mod, "answer", fake)
    monkeypatch.setattr(config, "SERVE_WARMUP", False)   # lifespan bez ładowania BGE-M3

    port = _wolny_port()
    server = uvicorn.Server(uvicorn.Config(api_mod.app, host="127.0.0.1", port=port,
                                           log_level="warning"))
    watek = threading.Thread(target=server.run, daemon=True)
    watek.start()
    try:
        for _ in range(200):                             # czekamy na gotowość (max ~10 s)
            if server.started:
                break
            time.sleep(0.05)
        assert server.started, "uvicorn nie wstal"

        with httpx.stream("POST", f"http://127.0.0.1:{port}/api/ask",
                          json={"question": "q"}, timeout=30) as resp:
            assert resp.headers["content-type"].startswith("application/x-ndjson")
            linie = resp.iter_lines()
            pierwsza = json.loads(next(linie))
            assert pierwsza == {"event": "stage", "stage": "retrieval", "status": "start"}
            assert not skonczone.is_set(), "answer() zdazyla sie skonczyc => to nie strumien"

            pozwolenie.set()
            ostatnia = json.loads([l for l in linie if l.strip()][-1])
    finally:
        server.should_exit = True
        watek.join(timeout=10)

    assert ostatnia["event"] == "final"
    assert ostatnia["answer"] == "gotowe"


@pytest.mark.parametrize("payload", [
    {"question": ""},
    {"question": "   "},
    {"question": "q", "top_k": 0},
    {"question": "q", "top_k": 999},
    {},
])
def test_zle_wejscie_daje_422(payload):
    assert client.post("/api/ask", json=payload).status_code == 422


def test_strona_glowna_serwuje_html_z_formularzem():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert 'id="ask-form"' in resp.text        # kotwica dla app.js
    assert 'id="stages"' in resp.text          # panel przebiegu
