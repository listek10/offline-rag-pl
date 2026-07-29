"""
tests/test_api.py
Kontrakt HTTP warstwy serwującej. Qdrant, Ollama i cała pętla RAG są PODMIENIONE — test
sprawdza KONTRAKT (kształt odpowiedzi, kolejność zdarzeń, kody błędów), a nie jakość modelu.

UWAGA na `TestClient(app)` BEZ menedżera kontekstu: dzięki temu `lifespan` się nie uruchamia,
więc rozgrzewka nie próbuje ładować BGE-M3. Test ma trwać milisekundy i działać bez GPU.
"""

from __future__ import annotations

import io
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
from src.ingest.upload import UploadRejected
from src.retrieve import corpus as corpus_mod
from src.serve import api as api_mod

client = TestClient(api_mod.app)


def test_health_zwraca_progi_i_modele_z_configu(monkeypatch):
    monkeypatch.setattr(api_mod, "_probe_qdrant",
                        lambda: {"ok": True, "collection": "kolekcja", "points": 1509})
    monkeypatch.setattr(api_mod, "_probe_llm",
                        lambda: {"ok": True, "models": ["qwen2.5:7b"], "missing": []})
    monkeypatch.setattr(api_mod, "_probe_judge",
                        lambda: {"ok": True, "models": ["gemma2:9b"], "missing": []})

    body = client.get("/api/health").json()

    assert body["qdrant"]["points"] == 1509
    assert body["llm"]["ok"] is True
    assert body["judge"]["ok"] is True
    # Progi MUSZĄ pochodzić z configu — inaczej UI kłamałoby po zmianie progu.
    assert body["config"]["gate_min"] == config.RELEVANCE_GATE_MIN
    assert body["config"]["top_k"] == config.TOP_K
    assert body["config"]["judge"] == config.JUDGE_MODEL
    assert body["warm"] is False        # lifespan nie odpalony => nierozgrzane


def test_health_nie_wywala_sie_gdy_uslugi_leza(monkeypatch):
    def boom():
        raise ConnectionError("brak polaczenia")

    monkeypatch.setattr(api_mod, "_probe_qdrant", lambda: api_mod._safe_probe(boom, "qdrant"))
    monkeypatch.setattr(api_mod, "_probe_llm", lambda: {"ok": False, "models": [], "missing": []})
    monkeypatch.setattr(api_mod, "_probe_judge", lambda: {"ok": False, "models": [], "missing": []})

    resp = client.get("/api/health")

    assert resp.status_code == 200          # health NIGDY nie zwraca 5xx — inaczej UI nie wie nic
    assert resp.json()["qdrant"]["ok"] is False


# --- Świadomość backendu LLM (etap 7) ----------------------------------------------------
# Diagnostyka MUSI wiedzieć, który silnik jest generatorem: inaczej `/api/health` sprawdza
# nie ten serwer, a komunikat błędu odsyła użytkownika w złe miejsce.

def _fake_urlopen(routes: dict):
    """Atrapa `urlopen`: mapuje fragment URL-a na gotowy JSON. Bez sieci i bez serwerów."""
    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False

    def _open(url, timeout=None):
        target = url if isinstance(url, str) else url.full_url
        for fragment, body in routes.items():
            if fragment in target:
                return _Resp(json.dumps(body).encode("utf-8"))
        raise urllib.error.URLError(f"brak atrapy dla {target}")

    return _open


def test_sonda_generatora_pyta_vllm_gdy_backend_to_vllm(monkeypatch):
    monkeypatch.setattr(config, "LLM_BACKEND", "vllm")
    monkeypatch.setattr(api_mod.urllib.request, "urlopen", _fake_urlopen({
        "/v1/models": {"data": [{"id": config.VLLM_MODEL}]},
    }))

    probe = api_mod._probe_llm()

    assert probe["ok"] is True
    assert probe["missing"] == []
    assert config.VLLM_MODEL in probe["models"]


def test_sonda_generatora_widzi_brak_modelu_na_vllm(monkeypatch):
    monkeypatch.setattr(config, "LLM_BACKEND", "vllm")
    monkeypatch.setattr(api_mod.urllib.request, "urlopen", _fake_urlopen({
        "/v1/models": {"data": [{"id": "inny/model"}]},
    }))

    assert api_mod._probe_llm()["missing"] == [config.VLLM_MODEL]


def test_sonda_generatora_pyta_ollame_gdy_backend_to_ollama(monkeypatch):
    monkeypatch.setattr(config, "LLM_BACKEND", "ollama")
    monkeypatch.setattr(api_mod.urllib.request, "urlopen", _fake_urlopen({
        "/api/tags": {"models": [{"name": config.LLM_MODEL}]},
    }))

    probe = api_mod._probe_llm()

    assert probe["ok"] is True and probe["missing"] == []


def test_sedzia_jest_sprawdzany_w_ollamie_takze_przy_backendzie_vllm(monkeypatch):
    """Guard #3 działa na Gemmie w Ollamie NIEZALEŻNIE od backendu generatora — health musi to
    sprawdzać, inaczej weryfikacja padnie w trakcie pytania, a nie przed nim."""
    monkeypatch.setattr(config, "LLM_BACKEND", "vllm")
    monkeypatch.setattr(api_mod.urllib.request, "urlopen", _fake_urlopen({
        "/api/tags": {"models": [{"name": "cos-innego"}]},
    }))

    probe = api_mod._probe_judge()

    assert probe["ok"] is True                       # Ollama odpowiada...
    assert probe["missing"] == [config.JUDGE_MODEL]  # ...ale sędziego w niej nie ma


def test_health_niesie_backend_i_model_generatora(monkeypatch):
    monkeypatch.setattr(config, "LLM_BACKEND", "vllm")
    monkeypatch.setattr(api_mod, "_probe_qdrant", lambda: {"ok": True, "points": 1})
    monkeypatch.setattr(api_mod, "_probe_llm", lambda: {"ok": True, "models": [], "missing": []})
    monkeypatch.setattr(api_mod, "_probe_judge", lambda: {"ok": True, "models": [], "missing": []})

    cfg = client.get("/api/health").json()["config"]

    assert cfg["backend"] == "vllm"
    # Nazwa modelu w UI musi być nazwą modelu, który REALNIE liczy odpowiedź.
    assert cfg["llm"] == config.VLLM_MODEL
    assert cfg["judge"] == config.JUDGE_MODEL


def test_blad_polaczenia_przy_vllm_wskazuje_vllm_nie_ollame(monkeypatch):
    monkeypatch.setattr(config, "LLM_BACKEND", "vllm")

    out = api_mod._classify_error(urllib.error.URLError("connection refused"))

    assert out["where"] == "vllm"
    assert "vllm_serve.py" in out["message"]
    # Sędzia stoi w Ollamie, więc przy włączonym guardzie #3 to też kandydat na winowajcę.
    assert config.OLLAMA_URL in out["message"]


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


# DLACZEGO te słowniki NIE idą przez `make_hit` (inaczej niż w test_answer_events.py): tutaj
# testujemy TRANSFORMACJĘ, którą robi `_hits_payload` — celowo podajemy hit BEZ `chunk_id`,
# żeby sprawdzić, że warstwa HTTP wysyła tylko pola przewidziane dla UI. Użycie fabryki
# zabetonowałoby tu wejście, którego kształt jest właśnie przedmiotem testu.
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


# --- Etap 2 UI: wgrywanie PDF-u do ulotnej kolekcji --------------------------------------
# Ingest, embeddingi i Qdrant są PODMIENIONE — tu testujemy KONTRAKT HTTP (kody, kształt,
# komunikaty), a nie jakość parsowania (tę sprawdza tests/test_upload.py na prawdziwym PDF-ie).

@pytest.fixture(autouse=True)
def czysty_rejestr_korpusow():
    """Ulotny korpus to stan globalny procesu serwera — nie może przeciekać między testami."""
    corpus_mod.reset()
    yield
    corpus_mod.reset()


def _stat_uploadu(source: str = "Obcy raport") -> dict:
    return {"source": source, "pages": 12, "chars": 41234, "chunks": 34,
            "chars_per_page": 3436, "corpus": corpus_mod.AD_HOC,
            "collection": config.UPLOAD_COLLECTION, "points": 34, "ms": 4200}


def _upload(dane: bytes = b"%PDF-1.7 udawany", nazwa: str = "Obcy raport.pdf"):
    return client.post(f"/api/upload?name={nazwa}", content=dane,
                       headers={"content-type": "application/pdf"})


def test_upload_zwraca_statystyki_wgranego_dokumentu(monkeypatch):
    monkeypatch.setattr(api_mod.ad_hoc, "build", lambda data, filename: _stat_uploadu())

    body = _upload().json()

    assert body["source"] == "Obcy raport"
    assert body["pages"] == 12 and body["chunks"] == 34 and body["points"] == 34
    assert body["corpus"] == corpus_mod.AD_HOC
    assert body["collection"] == config.UPLOAD_COLLECTION


def test_upload_niesie_nazwe_pliku_do_warstwy_ingestu(monkeypatch):
    """Etykieta źródła w cytatach bierze się z nazwy pliku, a ta jedzie w query stringu
    (ciało POST-a to same bajty PDF-a). Bez tego cytaty pokazywałyby „dokument-wgrany"."""
    widziane = {}
    monkeypatch.setattr(api_mod.ad_hoc, "build",
                        lambda data, filename: widziane.update(nazwa=filename, bajty=data)
                        or _stat_uploadu())

    _upload(b"%PDF-tresc", "Raport OSW 2026.pdf")

    assert widziane["nazwa"] == "Raport OSW 2026.pdf"
    assert widziane["bajty"] == b"%PDF-tresc"


def test_odrzucony_plik_daje_400_z_komunikatem_dla_czlowieka(monkeypatch):
    """Skan/uszkodzony plik to wina WEJŚCIA, nie awaria serwera. 400 + zdanie po polsku,
    nigdy 500 z tracebackiem — inaczej użytkownik nie wie, czy to jego plik, czy nasz błąd."""
    def boom(data, filename):
        raise UploadRejected("PDF bez warstwy tekstowej (prawdopodobnie skan) — wymagałby OCR.")

    monkeypatch.setattr(api_mod.ad_hoc, "build", boom)

    resp = _upload()

    assert resp.status_code == 400
    assert "skan" in resp.json()["detail"]


def test_zbyt_duzy_plik_jest_odrzucony_przed_wczytaniem_ciala(monkeypatch):
    """Limit sprawdzamy na nagłówku `Content-Length`, więc 200-megabajtowy plik nie zdąży
    wylądować w pamięci procesu. Test przykręca limit, zamiast wysyłać 25 MB."""
    monkeypatch.setattr(config, "UPLOAD_MAX_BYTES", 8)
    monkeypatch.setattr(api_mod.ad_hoc, "build",
                        lambda data, filename: pytest.fail("ciało nie powinno zostać wczytane"))

    resp = _upload(b"o wiele za duzo bajtow")

    assert resp.status_code == 413
    assert "MB" in resp.json()["detail"]


def test_awaria_serwera_przy_uploadzie_daje_500_bez_tracebacku(monkeypatch):
    def boom(data, filename):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(api_mod.ad_hoc, "build", boom)

    resp = _upload()

    assert resp.status_code == 500
    assert "Traceback" not in resp.text                  # traceback zostaje w logu serwera
    assert "ollama" in resp.json()["detail"].lower() or "model" in resp.json()["detail"].lower()


def test_usuniecie_dokumentu_demo_wola_zamiatacz(monkeypatch):
    wywolania = []
    monkeypatch.setattr(api_mod.ad_hoc, "drop", lambda: wywolania.append(1))

    resp = client.delete("/api/upload")

    assert resp.status_code == 200 and resp.json()["ok"] is True
    assert wywolania == [1]


def test_ask_przekazuje_wybrany_korpus_do_petli_rag(monkeypatch):
    """Bez tego przełącznik korpusu w UI byłby dekoracją: pytanie o wgrany PDF szłoby
    do kolekcji OSW."""
    widziane = {}

    def fake(question, top_k=None, gate=True, verify=None, on_event=None, corpus=None, **kw):
        widziane["corpus"] = corpus
        return {"answer": "ok", "sources": []}

    monkeypatch.setattr(api_mod, "answer", fake)
    corpus_mod.register(corpus_mod.Corpus(name=corpus_mod.AD_HOC,
                                          collection=config.UPLOAD_COLLECTION, records=[]))

    _ask({"question": "q", "corpus": corpus_mod.AD_HOC})

    assert widziane["corpus"].name == corpus_mod.AD_HOC
    assert widziane["corpus"].collection == config.UPLOAD_COLLECTION


def test_ask_bez_korpusu_leci_na_korpus_bazowy(monkeypatch):
    """Zgodność w tył: stare wywołania (i CLI) nie znają pola `corpus`."""
    widziane = {}

    def fake(question, top_k=None, gate=True, verify=None, on_event=None, corpus=None, **kw):
        widziane["corpus"] = corpus
        return {"answer": "ok", "sources": []}

    monkeypatch.setattr(api_mod, "answer", fake)

    _ask({"question": "q"})

    assert widziane["corpus"].name == corpus_mod.DEFAULT
    assert widziane["corpus"].collection == config.QDRANT_COLLECTION


def test_ask_z_nieznanym_korpusem_daje_400_z_podpowiedzia():
    """Ulotny korpus ginie razem z procesem, a przeglądarka może mieć zapamiętany przełącznik.
    Po restarcie serwera użytkownik ma dostać instrukcję („wgraj ponownie"), nie 500."""
    resp = client.post("/api/ask", json={"question": "q", "corpus": corpus_mod.AD_HOC})

    assert resp.status_code == 400
    assert "wgraj" in resp.json()["detail"].lower()


def test_health_pokazuje_wgrany_dokument_i_limity(monkeypatch):
    """UI po odświeżeniu strony musi wiedzieć, że korpus demo istnieje (i co to za dokument),
    a limity uploadu czytać z serwera — wpisane w HTML rozjechałyby się z configiem."""
    monkeypatch.setattr(api_mod, "_probe_qdrant", lambda: {"ok": True, "points": 1})
    monkeypatch.setattr(api_mod, "_probe_llm", lambda: {"ok": True, "models": [], "missing": []})
    monkeypatch.setattr(api_mod, "_probe_judge", lambda: {"ok": True, "models": [], "missing": []})
    corpus_mod.register(corpus_mod.Corpus(
        name=corpus_mod.AD_HOC, collection=config.UPLOAD_COLLECTION, records=[],
        label="Obcy raport", meta={"source": "Obcy raport", "pages": 12, "chunks": 34}))

    body = client.get("/api/health").json()

    assert body["upload"]["active"] is True
    assert body["upload"]["source"] == "Obcy raport"
    assert body["upload"]["chunks"] == 34
    assert body["config"]["upload_max_mb"] == round(config.UPLOAD_MAX_BYTES / (1024 * 1024))
    assert body["config"]["upload_max_pages"] == config.UPLOAD_MAX_PAGES
    # Etykieta korpusu zmierzonego musi nieść PRAWDZIWĄ nazwę kolekcji (A/B chunkingu zmienia ją).
    assert config.QDRANT_COLLECTION in body["config"]["base_corpus_label"]
    # NAZWY korpusów też z serwera: gdyby siedziały w HTML-u, zmiana `corpus.DEFAULT` w Pythonie
    # dawałaby 400 „nie ma takiego korpusu" — dwa opisy jednej prawdy, ta sama klasa dryfu,
    # którą zamknął `hit.py` po stronie trafień.
    assert body["config"]["base_corpus"] == corpus_mod.DEFAULT
    assert body["config"]["upload_corpus"] == corpus_mod.AD_HOC


def test_health_bez_uploadu_mowi_wprost_ze_go_nie_ma(monkeypatch):
    monkeypatch.setattr(api_mod, "_probe_qdrant", lambda: {"ok": True, "points": 1})
    monkeypatch.setattr(api_mod, "_probe_llm", lambda: {"ok": True, "models": [], "missing": []})
    monkeypatch.setattr(api_mod, "_probe_judge", lambda: {"ok": True, "models": [], "missing": []})

    assert client.get("/api/health").json()["upload"]["active"] is False


def test_ulotna_kolekcja_jest_zamiatana_przy_starcie_i_przy_zamknieciu(monkeypatch):
    """Kasowanie przy STARCIE to nie nadmiar ostrożności: `kill -9` (albo padnięcie procesu)
    nie wykonuje kodu zamykania, więc bez tego zamiatacza kolekcja demo przeżyłaby restart
    i „ulotność" byłaby obietnicą zamiast właściwości."""
    wywolania = []
    monkeypatch.setattr(api_mod.ad_hoc, "drop", lambda: wywolania.append(1))
    monkeypatch.setattr(config, "SERVE_WARMUP", False)     # bez ładowania BGE-M3 w teście

    with TestClient(api_mod.app):                          # kontekst = lifespan start + stop
        assert wywolania == [1]                            # start posprzątał sierotę

    assert wywolania == [1, 1]                             # zamknięcie posprzątało po sobie
