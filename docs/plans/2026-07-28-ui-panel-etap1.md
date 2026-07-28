# UI (etap 1) — plan implementacji

> **Dla wykonawcy:** kroki mają składnię checkboxów (`- [ ]`) do odhaczania. Wykonywać zadanie
> po zadaniu, w kolejności, z commitem na końcu każdego. Nie przeskakiwać kroków testowych —
> test pisany PRZED implementacją jest tu regułą, nie sugestią.

**Cel:** dodać lokalny interfejs webowy do istniejącej pętli RAG — jeden ekran pokazujący
jednocześnie odpowiedź z klikalnymi cytatami i diagnostykę (etapy z czasami, wynik bramki,
werdykty sędziego, źródło odmowy).

**Architektura:** FastAPI serwuje jeden statyczny plik HTML + dwa endpointy API. Pętla RAG
(`src/generate/answer.py`) zostaje nietknięta logicznie — dostaje tylko opcjonalny callback
`on_event`, którym raportuje granice etapów. Warstwa serwująca zamienia ten push-owy callback
na pull-owy iterator (wątek + kolejka) i wypluwa strumień NDJSON.

**Stack:** Python 3.11, FastAPI, uvicorn, Pydantic (walidacja wejścia), vanilla JS (bez npm,
bez build-stepu), pytest.

**Spec:** `docs/specs/2026-07-28-ui-panel-design.md` — w razie wątpliwości on jest rozstrzygający.

---

## Global Constraints

- **Offline w runtime.** Zero wywołań do chmury. Wszystko lokalnie (Ollama, Qdrant, modele z cache HF).
- **Interpreter:** `/home/maciek/miniconda3/envs/offline-rag-pl/bin/python`. Powłoka domyślnie startuje
  w condzie `base`, która ma **złego torcha (cu121)** — nigdy nie uruchamiać komend przez samo `python`.
  W każdym kroku poniżej: `PY=/home/maciek/miniconda3/envs/offline-rag-pl/bin/python`.
- **Katalog roboczy:** `/home/maciek/Pulpit/offline-rag-pl-ui` (worktree, branch `ui-panel`).
  `data/` jest tam symlinkiem do głównego katalogu — nie ruszać.
- **Wszystkie parametry w `config/config.py`.** Żadnych progów, portów ani nazw modeli wpisanych
  w kod serwera czy w HTML.
- **Bez ciężkich frameworków.** Nowe zależności dokładnie trzy: `fastapi`, `uvicorn`, `pytest`.
  Żadnego npm, bundlera, frameworka frontendowego.
- **Zero zmian w logice pipeline'u.** W `answer.py` wolno dodać wyłącznie: jeden kwarg `on_event`,
  wywołania `emit(...)` i lokalne pomiary czasu. Kolejność guardów, progi i zwracane klucze
  bez zmian — inaczej `eval_answers.py` przestałby mierzyć to, co robi runtime.
- **Żaden test nie dotyka Ollamy, Qdranta ani GPU.** Zależności podmieniane `monkeypatch`.
- **Wyjście LLM-a jest tekstem niezaufanym.** W JS wstawiać przez `textContent` / węzły DOM.
  `innerHTML` z treścią odpowiedzi lub kontekstu jest zakazane.
- **Komentarze po polsku i o tym „dlaczego", nie „co".** Kod ma być do obrony na rozmowie.
- **Serwer wiąże się z `127.0.0.1`.** Nigdy `0.0.0.0`.

---

## Struktura plików

| plik | odpowiedzialność |
|---|---|
| `src/serve/stream.py` | **Tylko** mostek: „funkcja blokująca wołająca callback" → „iterator zdarzeń". Nie wie nic o FastAPI ani o RAG-u. Dzięki temu testowalny fałszywką w milisekundach. |
| `src/serve/api.py` | Endpointy HTTP, walidacja wejścia, kształt danych na drucie, klasyfikacja błędów na komunikaty po polsku, rozgrzewka w `lifespan`. |
| `src/serve/static/index.html` | Struktura ekranu. Rozwijane źródła na natywnym `<details>`. |
| `src/serve/static/app.js` | Parser NDJSON, render etapów/odpowiedzi/źródeł, linkowanie cytatów `[n]`. |
| `src/serve/static/style.css` | CSS grid, `prefers-color-scheme`. |
| `src/generate/answer.py` | **Modyfikacja addytywna:** kwarg `on_event` + `emit(...)` na granicach etapów. |
| `config/config.py` | **Modyfikacja:** blok `SERVE_*` na końcu pliku. |
| `tests/test_serve_stream.py` | Kontrakt mostka: kolejność, sentinel, wyjątek. |
| `tests/test_answer_events.py` | Kontrakt zdarzeń z pipeline'u (zależności podmienione). |
| `tests/test_api.py` | Kontrakt HTTP: health, strumień NDJSON, ścieżki odmowy, walidacja 422. |

**Dlaczego `stream.py` osobno od `api.py`:** to jedyne nieoczywiste miejsce w całym etapie
(inwersja kontroli). Wydzielone — daje się przetestować bez FastAPI, bez modeli i bez sieci,
a `api.py` zostaje czytelny.

---

## Task 0: przygotowanie środowiska i zielony baseline

**Files:**
- Modify: `requirements.txt`

**Interfaces:**
- Produces: działający `pytest` w env `offline-rag-pl`; potwierdzenie, że istniejące testy przechodzą
  PRZED naszymi zmianami (żeby późniejsza czerwień była na pewno nasza).

- [ ] **Step 1: Zainstaluj brakujące zależności**

Wymaga internetu (jednorazowo, jak pobranie modeli — runtime pozostaje offline).

```bash
PY=/home/maciek/miniconda3/envs/offline-rag-pl/bin/python
$PY -m pip install "fastapi>=0.115" "uvicorn>=0.30" "pytest>=8.0"
```

- [ ] **Step 2: Potwierdź, że są w POPRAWNYM środowisku**

```bash
PY=/home/maciek/miniconda3/envs/offline-rag-pl/bin/python
$PY -c "import sys,torch,fastapi,uvicorn,pytest,httpx; print(sys.prefix); print('torch',torch.__version__)"
```
Oczekiwane: prefix `…/envs/offline-rag-pl`, torch `2.11.0+cu128`, brak `ModuleNotFoundError`.

- [ ] **Step 3: Zielony baseline istniejących testów**

```bash
cd /home/maciek/Pulpit/offline-rag-pl-ui
PY=/home/maciek/miniconda3/envs/offline-rag-pl/bin/python
$PY -m pytest tests/ -q
```
Oczekiwane: wszystkie testy PASS. Jeśli coś jest czerwone JUŻ TERAZ — zanotuj co i nie naprawiaj
w tym planie; to nie nasza regresja.

- [ ] **Step 4: Dopisz zależności do `requirements.txt`**

Dokleić na końcu pliku:

```
# --- Etap 6: warstwa serwująca (UI) ---
# Cienki serwer HTTP nad istniejącą pętlą RAG. Świadomie BEZ Streamlita/Gradio:
# tamte wciągają ~30 zależności tranzytywnych i przeładowują cały skrypt przy każdej
# interakcji (ryzyko ładowania BGE-M3 za każdym kliknięciem). Tu: 2 paczki, proces długo żyjący.
fastapi>=0.115       # routing + walidacja wejścia (Pydantic) + OpenAPI /docs
uvicorn>=0.30        # serwer ASGI
# pytest>=8.0        # testy (dev); httpx (wymagany przez TestClient) jest już w środowisku
```

- [ ] **Step 5: Commit**

```bash
cd /home/maciek/Pulpit/offline-rag-pl-ui
git add requirements.txt
git commit -m "etap 6/UI: zaleznosci warstwy serwujacej (fastapi, uvicorn)"
```

---

## Task 1: `stream.py` — mostek callback → iterator

**Files:**
- Create: `src/serve/__init__.py` (pusty)
- Create: `src/serve/stream.py`
- Test: `tests/test_serve_stream.py`

**Interfaces:**
- Produces:
  - `stream_events(work, on_error) -> Iterator[dict]`
  - `work: Callable[[Callable[..., None]], dict]` — funkcja blokująca; dostaje `emit(**kwargs)`,
    zwraca słownik-payload zdarzenia końcowego (bez klucza `event`).
  - `on_error: Callable[[BaseException], dict]` — mapuje wyjątek na payload zdarzenia błędu
    (bez klucza `event`).
  - Iterator daje kolejno: wszystkie zdarzenia wyemitowane przez `work`, a na końcu **dokładnie
    jedno** `{"event": "final", ...}` albo `{"event": "error", ...}`.

- [ ] **Step 1: Napisz test, który MUSI się wywalić**

Utwórz `tests/test_serve_stream.py`:

```python
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
```

- [ ] **Step 2: Uruchom test — musi być czerwony**

```bash
cd /home/maciek/Pulpit/offline-rag-pl-ui
PY=/home/maciek/miniconda3/envs/offline-rag-pl/bin/python
$PY -m pytest tests/test_serve_stream.py -q
```
Oczekiwane: `ModuleNotFoundError: No module named 'src.serve'` (albo `ImportError`).

- [ ] **Step 3: Implementacja**

Utwórz pusty `src/serve/__init__.py`, a następnie `src/serve/stream.py`:

```python
"""
src/serve/stream.py
Mostek: funkcja BLOKUJĄCA, która raportuje postęp CALLBACKIEM  ->  ITERATOR zdarzeń.

DLACZEGO to w ogóle potrzebne (sedno tego pliku):
`StreamingResponse` w FastAPI musi z czegoś CIĄGNĄĆ (iterator), a nasza pętla RAG WYPYCHA
zdarzenia przez callback. Ktoś musi blokować po drugiej stronie. Gdyby `answer()` leciała
w tym samym wątku co generator, nie dałoby się wyemitować NICZEGO, dopóki nie skończy —
czyli dokładnie to, czego chcemy uniknąć (użytkownik ma widzieć etapy w trakcie, nie po).
Stąd: wątek roboczy + `queue.Queue` + sentinel kończący iterację.

Ten moduł NIE WIE nic o FastAPI ani o RAG-u. To celowe: dzięki temu testuje się fałszywką,
bez modeli i bez sieci, a `api.py` zostaje czytelny.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Iterator

log = logging.getLogger("offline_rag.serve.stream")

_SENTINEL = object()  # znacznik „koniec strumienia" — nie da się go pomylić z żadnym zdarzeniem


def stream_events(
    work: Callable[[Callable[..., None]], dict],
    on_error: Callable[[BaseException], dict],
) -> Iterator[dict]:
    """Uruchamia `work` w wątku i oddaje jej zdarzenia jako iterator.

    `work(emit)` wykonuje robotę i zwraca payload zdarzenia końcowego.
    `on_error(exc)` mapuje wyjątek na payload zdarzenia błędu — klasyfikacja domenowa
    (Ollama? Qdrant?) należy do warstwy serwującej, nie tutaj.
    """
    q: queue.Queue = queue.Queue()

    def emit(**event) -> None:
        q.put(event)

    def runner() -> None:
        try:
            q.put({"event": "final", **work(emit)})
        except BaseException as exc:  # noqa: BLE001 - świadomie łapiemy wszystko
            # Pełny traceback TYLKO do logu serwera; do przeglądarki idzie komunikat
            # zmapowany przez on_error. Surowy traceback w UI to wyciek szczegółów wdrożenia.
            log.exception("blad w watku roboczym strumienia")
            q.put({"event": "error", **on_error(exc)})
        finally:
            # `finally` jest tu krytyczne: bez sentinela konsument zawisłby na q.get() na zawsze.
            q.put(_SENTINEL)

    # daemon=True: wątek nie może blokować zamknięcia procesu (Ctrl+C na uvicornie).
    threading.Thread(target=runner, name="rag-work", daemon=True).start()

    while True:
        item = q.get()
        if item is _SENTINEL:
            return
        yield item
```

- [ ] **Step 4: Uruchom testy — muszą przejść**

```bash
cd /home/maciek/Pulpit/offline-rag-pl-ui
PY=/home/maciek/miniconda3/envs/offline-rag-pl/bin/python
$PY -m pytest tests/test_serve_stream.py -q
```
Oczekiwane: `3 passed`.

- [ ] **Step 5: Commit**

```bash
cd /home/maciek/Pulpit/offline-rag-pl-ui
git add src/serve/__init__.py src/serve/stream.py tests/test_serve_stream.py
git commit -m "etap 6/UI: mostek callback->iterator (watek + kolejka) + testy kontraktu"
```

---

## Task 2: `answer.py` — addytywny `on_event`

**Files:**
- Modify: `src/generate/answer.py` (dodać `import time`, kwarg `on_event`, wywołania `emit`)
- Test: `tests/test_answer_events.py`

**Interfaces:**
- Consumes: nic z poprzednich zadań.
- Produces: `answer(question, top_k=None, system=SYSTEM_PROMPT, gate=True, verify=None, on_event=None) -> dict`
  emitujący zdarzenia:
  - `event="stage", stage="retrieval", status="start"`
  - `event="stage", stage="retrieval", status="done", ms=int, top_score=float, hits=list[dict]`
    (`hits` to **surowe** trafienia z pipeline'u — kształtowanie pod UI robi `api.py`)
  - `event="stage", stage="gate", status="done", passed=bool, top_score=float, threshold=float`
    albo `status="skipped"` (gdy `gate=False`)
  - `event="stage", stage="generation", status="start"` / `status="done", ms=int` / `status="not_run"`
  - `event="stage", stage="verify", status="start"` / `status="done", ms=int, grounded_fraction=float, verdicts=list[dict]`
    / `status="skipped"` (gdy `verify=False`) / `status="not_run"` (gdy wcześniejszy guard przerwał
    pipeline albo model odmówił sam)
  - Zwracany słownik **bez zmian**: `{answer, sources, blocked_by?, rejected_answer?, grounding?}`.

- [ ] **Step 1: Napisz test, który MUSI się wywalić**

Utwórz `tests/test_answer_events.py`:

```python
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

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config
from src.generate import answer as answer_mod


def _hit(score: float) -> dict:
    return {"score": 0.01, "rerank_score": score, "source": "raport", "chunk_index": 7,
            "chunk_id": "raport__7", "text": "Norwegia przeznacza 1,75% PKB na obronę."}


@pytest.fixture
def collected(monkeypatch):
    """Zbiera zdarzenia i pozwala testowi ustawić zachowanie retrievalu/LLM-a/sędziego."""
    events: list[dict] = []
    monkeypatch.setattr(answer_mod, "hybrid_search", lambda q, top_k=None, rerank=False: [_hit(0.9)])
    monkeypatch.setattr(answer_mod, "call_ollama", lambda *a, **k: "Norwegia daje 1,75% PKB [1].")
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
    monkeypatch.setattr(answer_mod, "hybrid_search",
                        lambda q, top_k=None, rerank=False: [_hit(0.01)])

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
```

- [ ] **Step 2: Uruchom test — musi być czerwony**

```bash
cd /home/maciek/Pulpit/offline-rag-pl-ui
PY=/home/maciek/miniconda3/envs/offline-rag-pl/bin/python
$PY -m pytest tests/test_answer_events.py -q
```
Oczekiwane: FAIL — `answer()` nie przyjmuje `on_event` (`TypeError: unexpected keyword argument`).

- [ ] **Step 3: Dodaj import `time` w `src/generate/answer.py`**

Znajdź blok importów (linie 24-27) i dopisz `time`:

```python
import json
import sys
import time
import urllib.request
from pathlib import Path
```

- [ ] **Step 4: Dodaj helper `_ms` nad definicją `answer()`**

Wstawić bezpośrednio po `REFUSAL_TEXT = "Nie znajduję odpowiedzi w źródłach."`:

```python
def _ms(t0: float) -> int:
    """Czas etapu w milisekundach. Mierzymy W ŚRODKU pętli, bo tylko ona zna granice etapów —
    warstwa serwująca zobaczyłaby wyłącznie czas całości."""
    return round((time.perf_counter() - t0) * 1000)
```

- [ ] **Step 5: Zastąp ciało `answer()`**

Zamień całą funkcję `answer()` (linie 85-124 przed zmianą) na poniższą. Docstring i komentarze
guardów zachowane — dochodzą tylko `emit(...)` i pomiary czasu.

```python
def answer(question: str, top_k: int | None = None, system: str = SYSTEM_PROMPT,
           gate: bool = True, verify: bool | None = None,
           on_event=None) -> dict:
    """Pełna pętla: retrieval -> (guard #2 bramka) -> prompt -> LLM -> (guard #3 weryfikacja).

    Zwraca {answer, sources, blocked_by?, rejected_answer?, grounding?}.

    Retrieval: HYBRYDA (dense+BM25/RRF) + reranking — konfiguracja ZMIERZONA jako najlepsza
    (na 29 pytaniach: recall@1 0.83, recall@5 0.97, MRR 0.89 vs dense 0.45/0.76/0.62).
    `system`/`gate`/`verify` można nadpisać — harness mierzy efekt każdego guardu OSOBNO
    (przecieki i false-refusale), bo guard bez zmierzonego kosztu to nie zabezpieczenie.

    `on_event` (opcjonalny) dostaje zdarzenia o granicach etapów — używa go warstwa serwująca,
    żeby pokazać postęp w UI. Domyślnie None => no-op, więc CLI i harness działają bez zmian.
    Emitujemy ETAPY, a nie tokeny odpowiedzi: guard #3 może odpowiedź odrzucić, więc pokazanie
    jej w trakcie generowania zniweczyłoby sens tego guardu.
    """
    verify = config.VERIFY_GROUNDING if verify is None else verify
    emit = on_event or (lambda **_: None)

    emit(event="stage", stage="retrieval", status="start")
    t = time.perf_counter()
    hits = hybrid_search(question, top_k=top_k, rerank=True)
    # top_score liczymy ZAWSZE (nie tylko przy włączonej bramce), bo UI pokazuje tę liczbę
    # także w trybie demo z rozbrojoną bramką — inaczej nie dałoby się porównać „przed/po".
    top_score = max((h.get("rerank_score", 0.0) for h in hits), default=0.0)
    emit(event="stage", stage="retrieval", status="done", ms=_ms(t),
         top_score=top_score, hits=hits)

    # GUARD #2 — BRAMKA TRAFNOŚCI: jeśli nic trafnego nie wróciło (najwyższy score rerankera
    # < próg), odmawiamy BEZ pytania modelu. Odcina pytania spoza tematyki korpusu (obliczenia,
    # wiedza ogólna, inne regiony) niezależnie od tego, co „chciałby" powiedzieć LLM.
    # Czego NIE odcina: pytań NA TEMAT, na które korpus nie ma odpowiedzi — patrz guard #3.
    if gate:
        passed = top_score >= config.RELEVANCE_GATE_MIN
        emit(event="stage", stage="gate", status="done", passed=passed,
             top_score=top_score, threshold=config.RELEVANCE_GATE_MIN)
        if not passed:
            # Etapy, które NIGDY się nie wykonają, muszą to powiedzieć wprost — inaczej panel
            # przebiegu wisiałby w „w toku" i użytkownik czekałby na coś, co nie nadejdzie.
            emit(event="stage", stage="generation", status="not_run")
            emit(event="stage", stage="verify", status="not_run")
            return {"answer": REFUSAL_TEXT, "sources": hits, "blocked_by": "gate"}
    else:
        emit(event="stage", stage="gate", status="skipped")

    context = build_context(hits)
    emit(event="stage", stage="generation", status="start")
    t = time.perf_counter()
    text = call_ollama(build_prompt(question, hits), system=system)
    emit(event="stage", stage="generation", status="done", ms=_ms(t))

    # GUARD #3 — WERYFIKACJA UGRUNTOWANIA: odpowiedź już powstała, więc dopiero TERAZ widać
    # błędy generacji (błędna atrybucja podmiotu, oferta podana jako rozstrzygnięcie). Sędzia
    # (inny model niż generator) ocenia każdą tezę; za mało popartych -> odmowa zamiast publikacji.
    if verify and not refused(text):
        from src.generate.verify import grounded_fraction
        emit(event="stage", stage="verify", status="start")
        t = time.perf_counter()
        frac, verdicts = grounded_fraction(context, text)
        emit(event="stage", stage="verify", status="done", ms=_ms(t),
             grounded_fraction=frac, verdicts=verdicts)
        if frac < config.VERIFY_MIN_SUPPORTED:
            # Zachowujemy odrzuconą odpowiedź i werdykty — bez tego guard byłby czarną skrzynką
            # (nie dałoby się sprawdzić, czy ubił halucynację, czy trafną parafrazę).
            return {"answer": REFUSAL_TEXT, "sources": hits, "blocked_by": "verify",
                    "rejected_answer": text, "grounding": verdicts}
        return {"answer": text, "sources": hits, "grounding": verdicts}

    # `skipped` = wyłączone przełącznikiem; `not_run` = model odmówił sam (guard #1), więc nie
    # ma czego weryfikować. Dwa różne powody, dwa różne komunikaty w UI.
    emit(event="stage", stage="verify", status="skipped" if not verify else "not_run")
    return {"answer": text, "sources": hits}
```

- [ ] **Step 6: Uruchom testy — muszą przejść**

```bash
cd /home/maciek/Pulpit/offline-rag-pl-ui
PY=/home/maciek/miniconda3/envs/offline-rag-pl/bin/python
$PY -m pytest tests/test_answer_events.py tests/ -q
```
Oczekiwane: `6 passed` z nowego pliku i **zero regresji** w pozostałych testach.

- [ ] **Step 7: Sprawdź, że CLI nadal działa bez zmian (bez GPU niepotrzebne — tylko składnia)**

```bash
cd /home/maciek/Pulpit/offline-rag-pl-ui
PY=/home/maciek/miniconda3/envs/offline-rag-pl/bin/python
$PY -c "import src.generate.answer as a; import inspect; print(inspect.signature(a.answer))"
```
Oczekiwane: `(question: str, top_k: int | None = None, system: str = ..., gate: bool = True, verify: bool | None = None, on_event=None) -> dict`

- [ ] **Step 8: Commit**

```bash
cd /home/maciek/Pulpit/offline-rag-pl-ui
git add src/generate/answer.py tests/test_answer_events.py
git commit -m "etap 6/UI: addytywny on_event w answer() (etapy + czasy) bez zmian w logice guardow"
```

---

## Task 3: `api.py` — szkielet + `GET /api/health`

**Files:**
- Create: `src/serve/api.py`
- Modify: `config/config.py` (blok `SERVE_*` na końcu)
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: nic (health nie używa `stream.py`).
- Produces:
  - `src/serve/api.py::app` — instancja `FastAPI`
  - `GET /api/health` → `{"qdrant": {...}, "ollama": {...}, "config": {...}, "warm": bool}`
  - `config.SERVE_HOST`, `config.SERVE_PORT`

- [ ] **Step 1: Dodaj blok `SERVE_*` na końcu `config/config.py`**

```python

# --- Warstwa serwująca / UI (etap 6) ---
# 127.0.0.1, nie 0.0.0.0: aplikacja jest lokalna i nie ma się wystawiać w sieci.
SERVE_HOST = "127.0.0.1"
SERVE_PORT = 8000
# Rozgrzewka przy starcie: jedno zapytanie na śmieci wciąga BGE-M3 i cross-encoder do VRAM,
# żeby PIERWSZE pytanie użytkownika było szybkie (w CLI ten koszt płaci się przy każdym
# uruchomieniu). Wyłączamy w testach — nie chcemy tam GPU.
SERVE_WARMUP = True
```

- [ ] **Step 2: Napisz test health, który MUSI się wywalić**

Utwórz `tests/test_api.py`:

```python
"""
tests/test_api.py
Kontrakt HTTP warstwy serwującej. Qdrant, Ollama i cała pętla RAG są PODMIENIONE — test
sprawdza KONTRAKT (kształt odpowiedzi, kolejność zdarzeń, kody błędów), a nie jakość modelu.

UWAGA na `TestClient(app)` BEZ menedżera kontekstu: dzięki temu `lifespan` się nie uruchamia,
więc rozgrzewka nie próbuje ładować BGE-M3. Test ma trwać milisekundy i działać bez GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
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
```

- [ ] **Step 3: Uruchom test — musi być czerwony**

```bash
cd /home/maciek/Pulpit/offline-rag-pl-ui
PY=/home/maciek/miniconda3/envs/offline-rag-pl/bin/python
$PY -m pytest tests/test_api.py -q
```
Oczekiwane: `ModuleNotFoundError: No module named 'src.serve.api'`.

- [ ] **Step 4: Implementacja `src/serve/api.py`**

```python
"""
src/serve/api.py
Etap 6: warstwa serwująca — CIENKI klient HTTP nad istniejącą pętlą RAG.

ZASADA: ten plik nie zawiera ANI GRAMA logiki RAG. Nie decyduje o progach, nie składa promptu,
nie interpretuje score'ów. Woła `answer()` i tłumaczy jej zdarzenia na format „na drucie".
Gdyby orkiestracja pipeline'u zamieszkała tutaj, mielibyśmy DWIE jego definicje — a wtedy UI
pokazywałoby zachowanie inne niż mierzy `src/eval/eval_answers.py`.

Uruchomienie:
    uvicorn src.serve.api:app --host 127.0.0.1 --port 8000
    (albo: python src/serve/api.py)
"""

from __future__ import annotations

import json
import logging
import sys
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config
from src.generate.answer import answer, refused
from src.serve.stream import stream_events

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("offline_rag.serve")

STATIC_DIR = Path(__file__).resolve().parent / "static"
_state = {"warm": False}


# --- Sondy stanu usług -------------------------------------------------------------------
# Health NIGDY nie zwraca 5xx: gdy Qdrant leży, UI ma pokazać czerwoną kropkę, a nie samo
# nie wystartować. Czerwona kropka PRZED zadaniem pytania jest dużo lepsza niż błąd po 20 s.

def _safe_probe(fn, where: str) -> dict:
    try:
        return {"ok": True, **fn()}
    except Exception as exc:  # noqa: BLE001
        log.warning("sonda %s nieudana: %s", where, exc)
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]}


def _probe_qdrant() -> dict:
    def probe() -> dict:
        # Świadomie używamy TEGO SAMEGO klienta, co pipeline (a nie nowego połączenia) —
        # health ma odzwierciedlać stan, w jakim naprawdę pracuje retrieval.
        from src.retrieve.search import _get_client
        info = _get_client().get_collection(config.QDRANT_COLLECTION)
        return {"collection": config.QDRANT_COLLECTION, "points": info.points_count}

    return _safe_probe(probe, "qdrant")


def _probe_ollama() -> dict:
    def probe() -> dict:
        with urllib.request.urlopen(f"{config.OLLAMA_URL}/api/tags", timeout=3) as resp:
            names = [m["name"] for m in json.loads(resp.read()).get("models", [])]
        required = [config.LLM_MODEL, config.JUDGE_MODEL]
        return {"models": names, "missing": [m for m in required if m not in names]}

    return _safe_probe(probe, "ollama")


# --- Cykl życia aplikacji ----------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    if config.SERVE_WARMUP:
        # Rozgrzewka: pierwszy `hybrid_search` ładuje BGE-M3 i cross-encoder (kilkanaście
        # sekund). Płacimy ten koszt PRZY STARCIE, nie przy pierwszym pytaniu użytkownika —
        # bo to pierwsze pytanie będzie oglądane na demo.
        try:
            from src.retrieve.hybrid import hybrid_search
            t = time.perf_counter()
            hybrid_search("rozgrzewka", rerank=True)
            log.info("modele rozgrzane w %.1f s", time.perf_counter() - t)
            _state["warm"] = True
        except Exception:  # noqa: BLE001
            # Brak rozgrzewki nie może zabić serwera — UI pokaże `warm:false` i czerwoną kropkę.
            log.exception("rozgrzewka nieudana — serwer wstaje mimo to")
    yield


app = FastAPI(title="offline-rag-pl", lifespan=lifespan)


@app.get("/api/health")
def health() -> dict:
    return {
        "qdrant": _probe_qdrant(),
        "ollama": _probe_ollama(),
        # Progi i modele czytane z configu, NIGDY wpisane w HTML — inaczej UI kłamałoby
        # po zmianie progu, a to jest projekt, w którym progi się zmienia i mierzy.
        "config": {
            "top_k": config.TOP_K,
            "gate_min": config.RELEVANCE_GATE_MIN,
            "verify_min_supported": config.VERIFY_MIN_SUPPORTED,
            "verify_default": config.VERIFY_GROUNDING,
            "llm": config.LLM_MODEL,
            "judge": config.JUDGE_MODEL,
            "embed": config.EMBED_MODEL,
            "reranker": config.RERANK_MODEL,
            "max_top_k": config.RERANK_CANDIDATES,
        },
        "warm": _state["warm"],
    }


def main() -> None:
    import uvicorn
    uvicorn.run(app, host=config.SERVE_HOST, port=config.SERVE_PORT)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Uruchom testy — muszą przejść**

```bash
cd /home/maciek/Pulpit/offline-rag-pl-ui
PY=/home/maciek/miniconda3/envs/offline-rag-pl/bin/python
$PY -m pytest tests/test_api.py -q
```
Oczekiwane: `2 passed`.

- [ ] **Step 6: Commit**

```bash
cd /home/maciek/Pulpit/offline-rag-pl-ui
git add config/config.py src/serve/api.py tests/test_api.py
git commit -m "etap 6/UI: endpoint /api/health (progi i modele z configu) + SERVE_* w configu"
```

---

## Task 4: `POST /api/ask` — strumień NDJSON

**Files:**
- Modify: `src/serve/api.py` (dodać model wejścia, klasyfikator błędów, endpoint)
- Modify: `tests/test_api.py` (dodać testy strumienia)

**Interfaces:**
- Consumes: `stream_events(work, on_error)` z Task 1; zdarzenia z `answer(on_event=...)` z Task 2.
- Produces: `POST /api/ask` → `application/x-ndjson`; zdarzenie `final` niesie
  `{answer, blocked_by, self_refused, rejected_answer, total_ms}`.

- [ ] **Step 1: Dopisz testy strumienia do `tests/test_api.py`**

Dokleić na końcu pliku:

```python
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


@pytest.mark.parametrize("payload", [
    {"question": ""},
    {"question": "   "},
    {"question": "q", "top_k": 0},
    {"question": "q", "top_k": 999},
    {},
])
def test_zle_wejscie_daje_422(payload):
    assert client.post("/api/ask", json=payload).status_code == 422
```

Dopisz też brakujące importy na górze `tests/test_api.py`:

```python
import json
import urllib.error
```

- [ ] **Step 2: Uruchom testy — muszą być czerwone**

```bash
cd /home/maciek/Pulpit/offline-rag-pl-ui
PY=/home/maciek/miniconda3/envs/offline-rag-pl/bin/python
$PY -m pytest tests/test_api.py -q
```
Oczekiwane: nowe testy FAIL z 404 (endpoint `/api/ask` nie istnieje); dwa testy health PASS.

- [ ] **Step 3: Dodaj model wejścia w `src/serve/api.py`**

Wstawić po definicji `_state`:

```python
# --- Wejście ------------------------------------------------------------------------------
# Walidację robi Pydantic (złe wejście => 422 bez naszego kodu). Górny limit `top_k` to
# RERANK_CANDIDATES, bo tyle kandydatów wchodzi do rerankera — więcej nie ma skąd wziąć.

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=config.TOP_K, ge=1, le=config.RERANK_CANDIDATES)
    gate: bool = True
    verify: bool = config.VERIFY_GROUNDING

    @field_validator("question")
    @classmethod
    def _niepuste_po_obcieciu(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("pytanie nie może być puste")
        return v
```

- [ ] **Step 4: Dodaj klasyfikator błędów i kształtowanie hitów**

Wstawić po sondach (`_probe_ollama`):

```python
# --- Tłumaczenie awarii na komunikaty dla człowieka --------------------------------------
# Użytkownik ma dostać zdanie, które MÓWI CO ZROBIĆ, a nie traceback. Trzy awarie są realne
# i przewidywalne, więc obsługujemy je z nazwy; reszta leci jako ogólny błąd serwera.

def _classify_error(exc: BaseException) -> dict:
    if isinstance(exc, (urllib.error.URLError, TimeoutError)):
        return {"where": "ollama",
                "message": f"Nie mogę połączyć się z modelem w Ollamie ({config.OLLAMA_URL}) "
                           f"lub przekroczono limit czasu. Sprawdź: ollama serve"}
    if type(exc).__module__.split(".")[0] in {"qdrant_client", "httpx", "grpc", "h11"}:
        return {"where": "qdrant",
                "message": "Baza wektorowa Qdrant nie odpowiada. Uruchom: docker start qdrant-osw"}
    if isinstance(exc, FileNotFoundError):
        return {"where": "dane",
                "message": f"Brak pliku z chunkami ({config.CHUNKS_PATH}). "
                           f"Uruchom src/ingest/chunk.py albo sprawdź symlink data/."}
    return {"where": "server", "message": f"{type(exc).__name__}: {exc}"[:300]}


def _hits_payload(hits: list[dict]) -> list[dict]:
    """Kształt źródeł NA DRUCIE. Numeracja `n` musi być zgodna z `build_context()` w answer.py,
    bo to na nią wskazują cytaty [n] w odpowiedzi modelu. `score` = score RERANKERA, czyli ta
    sama liczba, którą porównuje bramka — żeby UI pokazywało dokładnie to, na czym decyzja stoi.
    """
    return [{"n": i,
             "source": h["source"],
             "chunk_index": h["chunk_index"],
             "score": round(h.get("rerank_score", h.get("score", 0.0)), 4),
             "text": h["text"]}
            for i, h in enumerate(hits, 1)]
```

- [ ] **Step 5: Dodaj endpoint `POST /api/ask`**

Wstawić po `health()`:

```python
@app.post("/api/ask")
def ask(req: AskRequest) -> StreamingResponse:
    """Strumień NDJSON: etapy pipeline'u, a na końcu `final` albo `error`.

    DLACZEGO POST, a nie SSE/GET: `EventSource` nie obsługuje POST, więc wymusiłby pytanie
    w query stringu — czyli treść zapytania w logach dostępowych i w historii przeglądarki.
    Dla systemu, który ma być architekturą pod korpus niejawny, to dyskwalifikuje.

    DLACZEGO nie strumieniujemy tokenów odpowiedzi: guard #3 ocenia odpowiedź PO generacji
    i może ją odrzucić. Streaming pokazałby użytkownikowi halucynację przed werdyktem sędziego.
    """
    def work(emit) -> dict:
        def emit_na_drut(**event) -> None:
            # Pipeline emituje SWOJE obiekty domenowe; format „na drucie" należy do tej warstwy.
            if event.get("stage") == "retrieval" and "hits" in event:
                event["hits"] = _hits_payload(event["hits"])
            emit(**event)

        t0 = time.perf_counter()
        result = answer(req.question, top_k=req.top_k, gate=req.gate,
                        verify=req.verify, on_event=emit_na_drut)
        blocked_by = result.get("blocked_by")
        return {
            "answer": result["answer"],
            "blocked_by": blocked_by,
            # Guard #1 NIE ustawia `blocked_by` (model odmawia sam, z instrukcji w promptcie).
            # Bez tego pola UI zlałby trzy różne odmowy w jedną i zgubił informację, KTÓRA
            # warstwa zadecydowała — a to jest cały sens panelu diagnostycznego.
            "self_refused": blocked_by is None and refused(result["answer"]),
            "rejected_answer": result.get("rejected_answer"),
            "grounding": result.get("grounding"),
            "total_ms": round((time.perf_counter() - t0) * 1000),
        }

    def linie():
        for event in stream_events(work, _classify_error):
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(linie(), media_type="application/x-ndjson")
```

- [ ] **Step 6: Uruchom cały zestaw testów**

```bash
cd /home/maciek/Pulpit/offline-rag-pl-ui
PY=/home/maciek/miniconda3/envs/offline-rag-pl/bin/python
$PY -m pytest tests/ -q
```
Oczekiwane: wszystko PASS (2 health + 9 nowych z `/api/ask` + 3 stream + 7 answer-events + istniejące).

- [ ] **Step 7: Commit**

```bash
cd /home/maciek/Pulpit/offline-rag-pl-ui
git add src/serve/api.py tests/test_api.py
git commit -m "etap 6/UI: POST /api/ask jako strumien NDJSON (etapy, trzy zrodla odmowy, bledy po polsku)"
```

---

## Task 5: frontend — jeden ekran

**Files:**
- Create: `src/serve/static/index.html`
- Create: `src/serve/static/style.css`
- Create: `src/serve/static/app.js`
- Modify: `src/serve/api.py` (montaż statyków + `GET /`)
- Modify: `tests/test_api.py` (test, że `/` serwuje stronę)

**Interfaces:**
- Consumes: `GET /api/health`, `POST /api/ask` z Task 3 i 4.
- Produces: `GET /` → `index.html`; `GET /static/*` → pliki statyczne.

- [ ] **Step 1: Napisz test serwowania strony (dopisz do `tests/test_api.py`)**

```python
def test_strona_glowna_serwuje_html_z_formularzem():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert 'id="ask-form"' in resp.text        # kotwica dla app.js
    assert 'id="stages"' in resp.text          # panel przebiegu
```

- [ ] **Step 2: Uruchom — musi być czerwony**

```bash
cd /home/maciek/Pulpit/offline-rag-pl-ui
PY=/home/maciek/miniconda3/envs/offline-rag-pl/bin/python
$PY -m pytest tests/test_api.py::test_strona_glowna_serwuje_html_z_formularzem -q
```
Oczekiwane: FAIL z 404.

- [ ] **Step 3: Utwórz `src/serve/static/index.html`**

```html
<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>offline-rag-pl</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <header>
    <h1>offline-rag-pl</h1>
    <div id="health" class="health">sprawdzam stan usług…</div>
  </header>

  <form id="ask-form" autocomplete="off">
    <div class="row">
      <input id="question" type="text" maxlength="500" required
             placeholder="Które państwa tworzą arktyczny kwartet NATO?">
      <button id="submit" type="submit">Zapytaj</button>
    </div>
    <div class="row toggles">
      <label><input id="gate" type="checkbox" checked> bramka trafności</label>
      <label><input id="verify" type="checkbox" checked> weryfikacja ugruntowania</label>
      <label>top_k <input id="top_k" type="number" min="1" value="5"></label>
      <span id="config-note" class="muted"></span>
    </div>
  </form>

  <main>
    <section id="answer-panel">
      <h2>Odpowiedź</h2>
      <div id="blocked" class="blocked" hidden></div>
      <div id="answer" class="answer"></div>
      <details id="rejected-box" hidden>
        <summary>Odrzucona odpowiedź (nie została opublikowana)</summary>
        <div id="rejected" class="rejected"></div>
      </details>
      <ul id="verdicts" class="verdicts"></ul>
    </section>

    <aside id="diag">
      <h2>Przebieg</h2>
      <ol id="stages" class="stages"></ol>
      <h2>Źródła</h2>
      <div id="sources" class="sources"></div>
    </aside>
  </main>

  <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 4: Utwórz `src/serve/static/style.css`**

```css
/* Motyw idzie za ustawieniem systemu — bez przełącznika, bo to stan trzymany bez powodu. */
:root {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #6b7280; --line: #e5e7eb;
  --ok: #15803d; --bad: #b91c1c; --accent: #1d4ed8; --panel: #f9fafb;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #111317; --fg: #e5e7eb; --muted: #9ca3af; --line: #2a2f37;
    --ok: #4ade80; --bad: #f87171; --accent: #93b4ff; --panel: #171a1f;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 1.25rem; background: var(--bg); color: var(--fg);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
}
header { display: flex; justify-content: space-between; align-items: baseline;
         border-bottom: 1px solid var(--line); padding-bottom: .6rem; }
h1 { font-size: 1.05rem; margin: 0; letter-spacing: .02em; }
h2 { font-size: .8rem; text-transform: uppercase; letter-spacing: .06em;
     color: var(--muted); margin: 0 0 .5rem; }
.health { font-size: .82rem; color: var(--muted); }
.dot { display: inline-block; width: .55rem; height: .55rem; border-radius: 50%;
       margin: 0 .3rem 0 .7rem; background: var(--muted); }
.dot.ok { background: var(--ok); } .dot.bad { background: var(--bad); }

form { margin: 1rem 0; }
.row { display: flex; gap: .5rem; align-items: center; }
.toggles { margin-top: .5rem; gap: 1.1rem; font-size: .84rem; color: var(--muted); flex-wrap: wrap; }
.toggles input[type=number] { width: 4rem; }
input, button {
  font: inherit; color: var(--fg); background: var(--bg);
  border: 1px solid var(--line); border-radius: 6px; padding: .5rem .6rem;
}
#question { flex: 1; }
button { cursor: pointer; background: var(--accent); color: #fff; border-color: transparent;
         padding: .5rem 1.1rem; }
button:disabled { opacity: .55; cursor: progress; }

main { display: grid; grid-template-columns: minmax(0, 1fr) 22rem; gap: 1.25rem; }
@media (max-width: 900px) { main { grid-template-columns: 1fr; } }

.answer { white-space: pre-wrap; min-height: 3rem; }
.cite { color: var(--accent); text-decoration: none; font-weight: 600; }
.cite:hover { text-decoration: underline; }
.blocked { border-left: 3px solid var(--bad); background: var(--panel);
           padding: .6rem .8rem; margin-bottom: .8rem; font-size: .9rem; }
.rejected { white-space: pre-wrap; color: var(--muted); font-size: .9rem;
            border-left: 3px solid var(--line); padding-left: .7rem; }
.verdicts { list-style: none; padding: 0; margin: .8rem 0 0; font-size: .86rem; }
.verdicts li { border-left: 3px solid var(--bad); padding: .3rem .7rem; margin-bottom: .5rem; }
.verdicts li.ok { border-left-color: var(--ok); }
.verdicts .reason { color: var(--muted); }

.stages { list-style: none; padding: 0; margin: 0 0 1.2rem; font-size: .86rem; }
.stages li { display: flex; justify-content: space-between; gap: .6rem;
             padding: .28rem 0; border-bottom: 1px dashed var(--line); }
.stages .detail { color: var(--muted); font-variant-numeric: tabular-nums; }
.stages li.pending { opacity: .5; }
.stages li.skipped .detail, .stages li.not_run .detail { font-style: italic; }

.sources details { border: 1px solid var(--line); border-radius: 6px;
                   padding: .45rem .6rem; margin-bottom: .5rem; background: var(--panel); }
.sources summary { cursor: pointer; font-size: .86rem; }
.sources .chunk { white-space: pre-wrap; font-size: .84rem; color: var(--muted);
                  margin-top: .5rem; max-height: 16rem; overflow: auto; }
.sources details.hl { outline: 2px solid var(--accent); }
.muted { color: var(--muted); font-size: .8rem; }
```

- [ ] **Step 5: Utwórz `src/serve/static/app.js`**

```javascript
/* app.js — bez frameworka, bez build-stepu.
 *
 * Dwie rzeczy warte uwagi:
 * 1. Parser NDJSON (readNdjson): to cena za POST zamiast SSE. EventSource dałby to za darmo,
 *    ale nie obsługuje POST — pytanie musiałoby jechać w URL, a więc do logów dostępowych.
 * 2. Odpowiedź modelu wstawiamy WYŁĄCZNIE przez textContent / węzły DOM. Wyjście LLM-a to
 *    tekst NIEZAUFANY (w kontekście były znaczniki z PDF-a), więc innerHTML jest tu zakazany.
 */
const $ = (id) => document.getElementById(id);
const STAGES = [
  ["retrieval", "retrieval (hybryda + rerank)"],
  ["gate", "bramka trafności"],
  ["generation", "generacja (LLM)"],
  ["verify", "weryfikacja ugruntowania"],
];
let CFG = null;

/* --- health --------------------------------------------------------------------------- */
async function loadHealth() {
  try {
    const h = await (await fetch("/api/health")).json();
    CFG = h.config;
    const q = h.qdrant, o = h.ollama;
    $("health").textContent = "";
    addDot("Qdrant", q.ok, q.ok ? `${q.points} chunków` : q.error || "brak");
    addDot("Ollama", o.ok, o.ok ? (o.missing.length ? `brak modeli: ${o.missing.join(", ")}` : "modele gotowe") : o.error || "brak");
    if (!h.warm) addDot("modele", false, "nierozgrzane");
    $("top_k").value = CFG.top_k;
    $("top_k").max = CFG.max_top_k;
    $("verify").checked = CFG.verify_default;
    $("config-note").textContent =
      `próg bramki ${CFG.gate_min} · min. poparte tezy ${CFG.verify_min_supported} · ` +
      `${CFG.llm} + sędzia ${CFG.judge}`;
    const gotowe = q.ok && o.ok;
    $("submit").disabled = !gotowe;
    if (!gotowe) $("submit").textContent = "usługi niegotowe";
  } catch (e) {
    $("health").textContent = "serwer nie odpowiada";
  }
}

function addDot(label, ok, detail) {
  const span = document.createElement("span");
  const dot = document.createElement("span");
  dot.className = "dot " + (ok ? "ok" : "bad");
  span.appendChild(dot);
  span.appendChild(document.createTextNode(`${label}: ${detail}`));
  span.title = detail;
  $("health").appendChild(span);
}

/* --- parser strumienia NDJSON --------------------------------------------------------- */
async function* readNdjson(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop();                     // ostatni fragment może być niepełny — zostaw w buforze
    for (const line of lines) if (line.trim()) yield JSON.parse(line);
  }
  if (buf.trim()) yield JSON.parse(buf);
}

/* --- panel przebiegu ------------------------------------------------------------------ */
function resetStages() {
  const ol = $("stages");
  ol.textContent = "";
  for (const [key, label] of STAGES) {
    const li = document.createElement("li");
    li.id = "stage-" + key;
    li.className = "pending";
    const name = document.createElement("span");
    name.textContent = label;
    const detail = document.createElement("span");
    detail.className = "detail";
    detail.textContent = "—";
    li.append(name, detail);
    ol.appendChild(li);
  }
}

function renderStage(ev) {
  const li = $("stage-" + ev.stage);
  if (!li) return;
  const detail = li.querySelector(".detail");
  li.className = ev.status;
  if (ev.status === "start") { detail.textContent = "w toku…"; li.className = "pending"; return; }
  if (ev.status === "skipped") { detail.textContent = "wyłączone"; return; }
  if (ev.status === "not_run") { detail.textContent = "nie uruchomiono"; return; }
  // status === "done"
  const parts = [];
  if (ev.stage === "gate") {
    const znak = ev.passed ? "≥" : "<";
    parts.push(`${ev.top_score.toFixed(3)} ${znak} ${ev.threshold}`);
    parts.push(ev.passed ? "przeszło" : "ODRZUCONE");
  } else {
    if (typeof ev.ms === "number") parts.push(ev.ms >= 1000 ? `${(ev.ms / 1000).toFixed(1)} s` : `${ev.ms} ms`);
    if (ev.stage === "verify" && ev.verdicts) {
      const ok = ev.verdicts.filter((v) => v.grounded).length;
      parts.push(`${ok}/${ev.verdicts.length} tez poparte`);
    }
  }
  detail.textContent = parts.join(" · ");
}

/* --- źródła i cytaty ------------------------------------------------------------------ */
function renderSources(hits) {
  const box = $("sources");
  box.textContent = "";
  for (const h of hits) {
    const d = document.createElement("details");
    d.id = "src-" + h.n;
    const s = document.createElement("summary");
    s.textContent = `[${h.n}] ${h.source} #${h.chunk_index} · score ${h.score}`;
    const body = document.createElement("div");
    body.className = "chunk";
    body.textContent = h.text;            // NIGDY innerHTML — to treść z PDF-a
    d.append(s, body);
    box.appendChild(d);
  }
}

function highlightSource(n) {
  document.querySelectorAll(".sources details").forEach((d) => d.classList.remove("hl"));
  const target = $("src-" + n);
  if (!target) return;
  target.open = true;
  target.classList.add("hl");
  target.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function renderAnswer(text) {
  const box = $("answer");
  box.textContent = "";
  const re = /\[(\d+)\]/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    // `n` MUSI być kopiowane do const wewnątrz pętli: `m` jest nadpisywane przy każdym obrocie,
    // więc domknięcie w addEventListener widziałoby ostatnie dopasowanie, nie swoje.
    const n = m[1];
    box.appendChild(document.createTextNode(text.slice(last, m.index)));
    const a = document.createElement("a");
    a.href = "#src-" + n;
    a.className = "cite";
    a.textContent = m[0];
    a.addEventListener("click", (e) => { e.preventDefault(); highlightSource(n); });
    box.appendChild(a);
    last = re.lastIndex;
  }
  box.appendChild(document.createTextNode(text.slice(last)));
}

/* --- odmowy i werdykty ---------------------------------------------------------------- */
/* Trzy odmowy znaczą TRZY RÓŻNE rzeczy i muszą być rozróżnione, bo inaczej użytkownik nie wie,
 * czy szukać dalej: bramka mówi „w korpusie nie ma nic na ten temat", guard #1 „temat jest, ale
 * model nie widzi odpowiedzi", guard #3 „odpowiedź powstała, ale nie trzyma się źródeł". */
const POWODY = {
  gate: () => "Zablokowane przez BRAMKĘ TRAFNOŚCI (guard #2): najlepszy fragment nie przekroczył "
            + `progu ${CFG.gate_min}. Model nie był w ogóle pytany — w korpusie nie ma nic na ten temat.`,
  verify: () => "Zablokowane przez WERYFIKACJĘ UGRUNTOWANIA (guard #3): odpowiedź powstała, ale "
              + "sędzia nie potwierdził wszystkich jej tez wobec źródeł. Poniżej odrzucona treść.",
  self: () => "Model SAM odmówił (guard #1: instrukcja w promptcie) — retrieval znalazł fragmenty "
            + "na temat, ale model nie widzi w nich odpowiedzi.",
};

function renderFinal(ev) {
  const powod = ev.blocked_by || (ev.self_refused ? "self" : null);
  if (powod) {
    $("blocked").textContent = POWODY[powod]();
    $("blocked").hidden = false;
  }
  renderAnswer(ev.answer);
  if (ev.rejected_answer) {
    $("rejected").textContent = ev.rejected_answer;
    $("rejected-box").hidden = false;
  }
  const ul = $("verdicts");
  for (const v of ev.grounding || []) {
    if (v.grounded) continue;                       // pokazujemy to, co PODWAŻONE
    const li = document.createElement("li");
    const claim = document.createElement("div");
    claim.textContent = "niepoparta teza: " + v.claim;
    const reason = document.createElement("div");
    reason.className = "reason";
    reason.textContent = "sędzia: " + v.reason;
    li.append(claim, reason);
    ul.appendChild(li);
  }
}

/* --- pętla zapytania ------------------------------------------------------------------ */
$("ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = $("submit");
  btn.disabled = true;                             // ~20 s roboty: blokujemy podwójne wysłanie
  const label = btn.textContent;
  btn.textContent = "pracuję…";
  $("blocked").hidden = true;
  $("rejected-box").hidden = true;
  $("answer").textContent = "";
  $("verdicts").textContent = "";
  $("sources").textContent = "";
  resetStages();

  try {
    const resp = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: $("question").value,
        top_k: Number($("top_k").value),
        gate: $("gate").checked,
        verify: $("verify").checked,
      }),
    });
    if (resp.status === 422) { $("answer").textContent = "Nieprawidłowe wejście (puste pytanie lub top_k poza zakresem)."; return; }
    for await (const ev of readNdjson(resp)) {
      if (ev.event === "stage") {
        renderStage(ev);
        if (ev.stage === "retrieval" && ev.hits) renderSources(ev.hits);
      } else if (ev.event === "final") {
        renderFinal(ev);
      } else if (ev.event === "error") {
        $("blocked").textContent = `Błąd (${ev.where}): ${ev.message}`;
        $("blocked").hidden = false;
      }
    }
  } catch (err) {
    $("blocked").textContent = "Utracono połączenie z serwerem: " + err;
    $("blocked").hidden = false;
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
});

loadHealth();
```

- [ ] **Step 6: Zmontuj statyki w `src/serve/api.py`**

Dopisać **po** definicji endpointu `ask()` (kolejność ma znaczenie: `app.mount` na `/static`
nie koliduje z `/api/*`, ale trzymamy to razem na końcu dla czytelności):

```python
# --- Frontend ----------------------------------------------------------------------------
# Statyki serwuje sam FastAPI — nie ma osobnego serwera WWW, bo to aplikacja lokalna.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
```

- [ ] **Step 7: Uruchom cały zestaw testów**

```bash
cd /home/maciek/Pulpit/offline-rag-pl-ui
PY=/home/maciek/miniconda3/envs/offline-rag-pl/bin/python
$PY -m pytest tests/ -q
```
Oczekiwane: wszystko PASS, w tym nowy `test_strona_glowna_serwuje_html_z_formularzem`.

- [ ] **Step 8: Commit**

```bash
cd /home/maciek/Pulpit/offline-rag-pl-ui
git add src/serve/static src/serve/api.py tests/test_api.py
git commit -m "etap 6/UI: frontend (jeden ekran, klikalne cytaty, panel przebiegu i zrodel)"
```

---

## Task 6: uruchomienie end-to-end, README i notatnik

**Files:**
- Modify: `README.md` (sekcja o UI + diagram/instrukcja)
- Modify: `/home/maciek/Pulpit/offline-rag-pl/NOTATKI_REKRUTER.md` (sekcja DOWÓD w „Etap 6")

**Interfaces:**
- Consumes: wszystko z Task 1-5.
- Produces: potwierdzone działanie na prawdziwych modelach + udokumentowany dowód.

- [ ] **Step 1: Podnieś usługi**

```bash
docker start qdrant-osw
curl -s localhost:6333/healthz
curl -s localhost:11434/api/tags | head -c 200
```
Oczekiwane: Qdrant odpowiada, Ollama wypisuje listę modeli z `qwen2.5:7b` i `gemma2:9b`.

- [ ] **Step 2: Uruchom serwer**

```bash
cd /home/maciek/Pulpit/offline-rag-pl-ui
PY=/home/maciek/miniconda3/envs/offline-rag-pl/bin/python
$PY src/serve/api.py
```
Oczekiwane w logu: `modele rozgrzane w NN.N s`, potem `Uvicorn running on http://127.0.0.1:8000`.
Rozgrzewka może zająć 15-30 s — to zamierzone.

- [ ] **Step 3: Sprawdź health z drugiego terminala**

```bash
curl -s localhost:8000/api/health | python -m json.tool | head -30
```
Oczekiwane: `"ok": true` dla Qdranta (z `points: 1509`) i Ollamy, `"warm": true`,
`gate_min: 0.35`.

- [ ] **Step 4: Ręczne demo w przeglądarce — trzy scenariusze do ZOBACZENIA**

Otwórz `http://127.0.0.1:8000` i sprawdź kolejno:

1. **Pytanie z korpusu:** „Które państwa tworzą arktyczny kwartet NATO?"
   Oczekiwane: źródła pojawiają się PRZED odpowiedzią (po ~0,5 s), cytaty `[n]` są klikalne
   i rozwijają właściwy fragment, wszystkie 4 etapy mają czasy, bramka pokazuje liczbę ≥ 0.35.
2. **Pytanie spoza korpusu:** „Ile wynosi pierwiastek z 144?"
   Oczekiwane: bramka pokazuje `score < 0.35 · ODRZUCONE`, generacja i weryfikacja mają
   „nie uruchomiono", komunikat o guardzie #2.
3. **Hard negative z rozbrojonymi guardami** (to jest demo „przed/po"): „Jakie są wydatki
   obronne Danii jako % PKB?" — najpierw z zaznaczonymi oboma przełącznikami, potem z
   odznaczoną **weryfikacją**.
   Oczekiwane: z weryfikacją → odmowa `guard #3` + odrzucona odpowiedź + niepoparta teza
   z uzasadnieniem sędziego; bez weryfikacji → przecieka halucynacja z cytowaniem.
   To jest najmocniejszy moment demo — zanotuj dokładnie, co zobaczyłeś.

- [ ] **Step 5: Zapisz zmierzone czasy**

Z panelu przebiegu przepisz czasy etapów dla scenariusza 1 i 3 (retrieval / generacja /
weryfikacja / suma). Będą potrzebne w README i notatniku jako DOWÓD — do tej pory koszt
guardu #3 był znany tylko jako „~2× latencja", teraz będzie liczbą.

- [ ] **Step 6: Dopisz sekcję do `README.md`**

Wstawić po sekcji „Uruchomienie", przed „Struktura repo":

```markdown
---

## UI (lokalny panel analityka)

```bash
python src/serve/api.py        # potem: http://127.0.0.1:8000
```

Jeden ekran: po lewej odpowiedź z **klikalnymi cytatami** `[n]`, po prawej **panel diagnostyczny**
— etapy pipeline'u z czasami, wynik bramki (`score` vs próg), werdykty sędziego per teza i źródła
z pełnym tekstem chunka. Przełączniki `bramka` / `weryfikacja` / `top_k` pozwalają pokazać
**„przed/po" na żywo**: to samo pytanie z rozbrojonym guardem przecieka halucynacją.

Decyzje warte odnotowania:
- **FastAPI + statyczny HTML/vanilla JS** — dwie zależności, bez npm i build-stepu (Streamlit
  wciąga ~30 paczek tranzytywnych i przeładowuje skrypt przy każdej interakcji).
- **`POST` + NDJSON, nie SSE** — `EventSource` nie obsługuje POST, więc pytanie musiałoby jechać
  w query stringu, a stamtąd do logów dostępowych i historii przeglądarki.
- **Nie strumieniujemy tokenów odpowiedzi, tylko etapy** — guard #3 ocenia odpowiedź po
  generacji i może ją odrzucić; streaming pokazałby halucynację przed werdyktem sędziego.
  Źródła wysyłamy już po retrievalu, więc użytkownik czyta fragmenty w trakcie generowania.
- **Trzy źródła odmowy rozróżnione w UI** — guard #1 (model sam), #2 (bramka), #3 (weryfikacja)
  znaczą różne rzeczy; zlanie ich w jedno „nie znajduję odpowiedzi" gubi informację, czy szukać dalej.
```

- [ ] **Step 7: Uzupełnij sekcję DOWÓD w notatniku**

W `/home/maciek/Pulpit/offline-rag-pl/NOTATKI_REKRUTER.md`, w sekcji „Etap 6 — UI", zamień
treść pod nagłówkiem `### DOWÓD` na zmierzone liczby ze Step 5 i opis tego, co pokazały trzy
scenariusze ze Step 4 (szczególnie scenariusz 3 — przeciek po rozbrojeniu guardu #3).

- [ ] **Step 8: Cały zestaw testów raz jeszcze + commit**

```bash
cd /home/maciek/Pulpit/offline-rag-pl-ui
PY=/home/maciek/miniconda3/envs/offline-rag-pl/bin/python
$PY -m pytest tests/ -q
git add README.md
git commit -m "etap 6/UI: README z sekcja o panelu + zmierzone czasy etapow"
```

---

## Definition of Done (etap 1)

- [ ] `pytest tests/ -q` zielony w env `offline-rag-pl` (nowe testy: stream 3, answer-events 6, api 12)
- [ ] `python src/serve/api.py` wstaje, health pokazuje `warm:true` i 1509 chunków
- [ ] Trzy scenariusze ze Task 5 Step 4 zobaczone w przeglądarce i opisane
- [ ] `git log --oneline` na `ui-panel` pokazuje 6 commitów etapu 1, drzewo czyste
- [ ] README i `NOTATKI_REKRUTER.md` zaktualizowane zmierzonymi liczbami
- [ ] CLI nadal działa bez zmian: `python src/generate/answer.py "pytanie"` daje ten sam wynik
