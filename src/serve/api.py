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

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config
from src.generate.answer import answer, refused
from src.index import ad_hoc
from src.ingest.upload import UploadRejected
from src.retrieve import corpus as corpus_mod
from src.retrieve.hit import get_rerank_score
from src.serve.stream import stream_events

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("offline_rag.serve")

STATIC_DIR = Path(__file__).resolve().parent / "static"
_state = {"warm": False}


# --- Wejście ------------------------------------------------------------------------------
# Walidację robi Pydantic (złe wejście => 422 bez naszego kodu). Górny limit `top_k` to
# RERANK_CANDIDATES, bo tyle kandydatów wchodzi do rerankera — więcej nie ma skąd wziąć.

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=config.TOP_K, ge=1, le=config.RERANK_CANDIDATES)
    gate: bool = True
    verify: bool = config.VERIFY_GROUNDING
    # Korpus podajemy NAZWĄ, nie listą dopuszczalnych wartości w Pydanticu: zbiór korpusów jest
    # DYNAMICZNY (ulotny pojawia się i ginie razem z dokumentem demo), więc walidacja należy do
    # rejestru — i musi umieć powiedzieć „ten korpus już nie istnieje", a nie tylko „zła wartość".
    corpus: str = Field(default=corpus_mod.DEFAULT, max_length=40)

    @field_validator("question")
    @classmethod
    def _niepuste_po_obcieciu(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("pytanie nie może być puste")
        return v


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


def _ollama_models() -> list[str]:
    with urllib.request.urlopen(f"{config.OLLAMA_URL}/api/tags", timeout=3) as resp:
        return [m["name"] for m in json.loads(resp.read()).get("models", [])]


def _generator_model() -> str:
    """Nazwa modelu, który REALNIE liczy odpowiedź (zależna od backendu, etap 7)."""
    return config.VLLM_MODEL if config.LLM_BACKEND == "vllm" else config.LLM_MODEL


def _probe_llm() -> dict:
    """Sonda GENERATORA — pyta ten serwer, który naprawdę generuje (patrz `config.LLM_BACKEND`).

    DLACZEGO zależna od backendu: sonda wbita na sztywno w Ollamę przy `LLM_BACKEND=vllm`
    świeciłaby zielono, choć generator (vLLM) mógłby leżeć — a to jest dokładnie ten rodzaj
    diagnostyki, która szkodzi bardziej niż jej brak, bo usypia.
    """
    if config.LLM_BACKEND == "vllm":
        def probe() -> dict:
            # vLLM wystawia listę wag pod /v1/models (API zgodne z OpenAI), nie /api/tags.
            with urllib.request.urlopen(f"{config.VLLM_URL}/models", timeout=3) as resp:
                names = [m["id"] for m in json.loads(resp.read()).get("data", [])]
            return {"backend": "vllm", "url": config.VLLM_URL, "models": names,
                    "missing": [m for m in [config.VLLM_MODEL] if m not in names]}

        return _safe_probe(probe, "vllm")

    def probe() -> dict:
        names = _ollama_models()
        return {"backend": "ollama", "url": config.OLLAMA_URL, "models": names,
                "missing": [m for m in [config.LLM_MODEL] if m not in names]}

    return _safe_probe(probe, "ollama")


def _probe_judge() -> dict:
    """Sonda SĘDZIEGO guarda #3 — ZAWSZE Ollama, także gdy generator liczy na vLLM.

    Dwa modele 7B i 9B nie zmieszczą się naraz w 12 GB VRAM, więc sędzia zostaje na Ollamie
    (patrz `call_ollama` w llm.py). Sprawdzamy go OSOBNO, bo brak sędziego wywala dopiero
    guard #3 — czyli w połowie odpowiedzi, po kilkunastu sekundach czekania.
    """
    def probe() -> dict:
        names = _ollama_models()
        return {"url": config.OLLAMA_URL, "models": names,
                "missing": [m for m in [config.JUDGE_MODEL] if m not in names]}

    return _safe_probe(probe, "ollama-judge")


# --- Tłumaczenie awarii na komunikaty dla człowieka --------------------------------------
# Użytkownik ma dostać zdanie, które MÓWI CO ZROBIĆ, a nie traceback. Trzy awarie są realne
# i przewidywalne, więc obsługujemy je z nazwy; reszta leci jako ogólny błąd serwera.

def _classify_error(exc: BaseException) -> dict:
    if isinstance(exc, (urllib.error.URLError, TimeoutError)):
        # Komunikat MUSI wskazać właściwy serwer. Przy backendzie vLLM podajemy oba adresy,
        # bo w grze są dwa procesy: generator na vLLM i sędzia guarda #3 na Ollamie —
        # a z samego URLError nie wynika, który z nich nie odpowiedział.
        if config.LLM_BACKEND == "vllm":
            return {"where": "vllm",
                    "message": f"Nie mogę połączyć się z modelem: generator na vLLM "
                               f"({config.VLLM_URL}) albo sędzia w Ollamie ({config.OLLAMA_URL}). "
                               f"Sprawdź: python src/serve/vllm_serve.py oraz ollama serve"}
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
             "score": round(get_rerank_score(h, h.get("score", 0.0)), 4),
             "text": h["text"]}
            for i, h in enumerate(hits, 1)]


# --- Cykl życia aplikacji ----------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Zamiatacz ulotnej kolekcji przy STARCIE. Dlaczego nie wystarczy sprzątanie przy zamknięciu:
    # `kill -9`, OOM albo padnięcie procesu nie wykonują kodu zamykania, więc kolekcja demo
    # przeżyłaby restart — a wtedy „ulotność" byłaby obietnicą, nie właściwością.
    ad_hoc.drop()

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
    # Dokument demo nie ma prawa przeżyć procesu (patrz `src/index/ad_hoc.py`).
    ad_hoc.drop()


app = FastAPI(title="offline-rag-pl", lifespan=lifespan)


@app.get("/api/health")
def health() -> dict:
    return {
        "qdrant": _probe_qdrant(),
        # Dwie OSOBNE sondy zamiast jednej „ollama": generator zależy od backendu, a sędzia
        # guarda #3 stoi zawsze w Ollamie. Zlanie ich w jedną kropkę gubiło informację, KTÓRA
        # usługa leży — a to różne komendy naprawcze i różny wpływ (bez sędziego pipeline
        # nadal odpowiada, tylko bez guarda #3; bez generatora nie odpowiada wcale).
        "llm": _probe_llm(),
        "judge": _probe_judge(),
        # Progi i modele czytane z configu, NIGDY wpisane w HTML — inaczej UI kłamałoby
        # po zmianie progu, a to jest projekt, w którym progi się zmienia i mierzy.
        "config": {
            "top_k": config.TOP_K,
            "gate_min": config.RELEVANCE_GATE_MIN,
            "verify_min_supported": config.VERIFY_MIN_SUPPORTED,
            "verify_default": config.VERIFY_GROUNDING,
            "backend": config.LLM_BACKEND,
            "llm": _generator_model(),
            "judge": config.JUDGE_MODEL,
            "embed": config.EMBED_MODEL,
            "reranker": config.RERANK_MODEL,
            "max_top_k": config.RERANK_CANDIDATES,
            # Limity uploadu też z configu: wpisane w HTML rozjechałyby się po zmianie limitu,
            # a użytkownik dowiedziałby się o odrzuceniu dopiero po wysłaniu 40 MB.
            "upload_max_mb": round(config.UPLOAD_MAX_BYTES / (1024 * 1024)),
            "upload_max_pages": config.UPLOAD_MAX_PAGES,
            # Nazwy korpusów i etykieta bazowego — UI nie ma prawa ich znać z HTML-a, bo wtedy
            # zmiana `corpus.DEFAULT` po stronie Pythona dawałaby 400 przy pierwszym pytaniu.
            "base_corpus": corpus_mod.DEFAULT,
            "upload_corpus": corpus_mod.AD_HOC,
            "base_corpus_label": corpus_mod.resolve(None).label,
        },
        # Stan ulotnego korpusu: po odświeżeniu strony UI musi wiedzieć, czy dokument demo
        # jeszcze żyje (przetrwał F5, ale nie przetrwa restartu serwera — i to jest zamierzone).
        "upload": _upload_state(),
        "warm": _state["warm"],
    }


def _upload_state() -> dict:
    c = corpus_mod.ad_hoc()
    if c is None:
        return {"active": False}
    return {"active": True, "corpus": c.name, "collection": c.collection, **c.meta}


# --- Wgrywanie PDF-u ad hoc (etap 2) -----------------------------------------------------

@app.post("/api/upload")
async def upload(request: Request, name: str = "") -> dict:
    """Bajty PDF-a -> ulotny korpus. Zwraca statystyki dokumentu (strony, chunki, czas).

    DLACZEGO SUROWE BAJTY W CIELE, A NIE `multipart/form-data` (czyli `UploadFile`): Starlette
    trzyma `UploadFile` w `SpooledTemporaryFile(max_size=1 MB)`, więc każdy PDF większy niż
    megabajt ZOSTAŁBY ZAPISANY NA DYSKU w katalogu tymczasowym. To łamie wymóg „plik demo nie
    dotyka dysku" — dokument miałby przeżyć proces w /tmp. Efekt uboczny wyboru: nie ma
    zależności `python-multipart`. Nazwa pliku jedzie w query stringu (`?name=`), bo jest
    potrzebna jako etykieta źródła w cytatach.

    `async def` + `run_in_threadpool`: ciało requestu czytamy asynchronicznie, ale samo
    indeksowanie (PyMuPDF + BGE-M3 na GPU, kilka sekund) jest BLOKUJĄCE. Zostawione w pętli
    zdarzeń zamroziłoby cały serwer, w tym `/api/health`, którym UI sprawdza stan.
    """
    # Limit sprawdzamy NA NAGŁÓWKU, przed `await request.body()` — inaczej 200-megabajtowy plik
    # wylądowałby najpierw w pamięci procesu, a dopiero potem dowiedzielibyśmy się, że jest za duży.
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > config.UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=413, detail=(
            f"Plik ma {int(declared) / (1024 * 1024):.1f} MB, a limit dla dokumentu demo to "
            f"{config.UPLOAD_MAX_BYTES / (1024 * 1024):.0f} MB."))

    data = await request.body()
    try:
        return await run_in_threadpool(ad_hoc.build, data, name)
    except UploadRejected as exc:
        # Wina WEJŚCIA (skan, uszkodzony plik, limit) — 400 i zdanie po polsku.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        # Awaria usługi (Qdrant, GPU) — traceback TYLKO do logu, do przeglądarki komunikat.
        log.exception("upload nieudany")
        raise HTTPException(status_code=500, detail=_classify_error(exc)["message"]) from exc


@app.delete("/api/upload")
def upload_delete() -> dict:
    """Kasuje dokument demo (kolekcja + rejestr) i wraca do korpusu zmierzonego.

    Endpoint istnieje, bo ulotność „przy zamknięciu procesu" to za mało dla demo: po pokazaniu
    obcego PDF-u chcę móc wrócić do korpusu, na którym mam zmierzone liczby, bez restartu serwera.
    """
    ad_hoc.drop()
    return {"ok": True}


@app.post("/api/ask")
def ask(req: AskRequest) -> StreamingResponse:
    """Strumień NDJSON: etapy pipeline'u, a na końcu `final` albo `error`.

    DLACZEGO POST, a nie SSE/GET: `EventSource` nie obsługuje POST, więc wymusiłby pytanie
    w query stringu — czyli treść zapytania w logach dostępowych i w historii przeglądarki.
    Dla systemu, który ma być architekturą pod korpus niejawny, to dyskwalifikuje.

    DLACZEGO nie strumieniujemy tokenów odpowiedzi: guard #3 ocenia odpowiedź PO generacji
    i może ją odrzucić. Streaming pokazałby użytkownikowi halucynację przed werdyktem sędziego.
    """
    # Korpus rozstrzygamy PRZED uruchomieniem strumienia: „nie ma takiego korpusu" to błąd
    # wejścia (400 z podpowiedzią), a nie zdarzenie w środku odpowiedzi. Ulotny korpus ginie
    # razem z procesem, więc przeglądarka z zapamiętanym przełącznikiem trafi tu po restarcie.
    try:
        korpus = corpus_mod.resolve(req.corpus)
    except corpus_mod.UnknownCorpus as exc:
        raise HTTPException(status_code=400, detail=(
            f"Korpus „{req.corpus}” nie istnieje w tym procesie. Dokument demo jest ulotny — "
            f"wgraj PDF ponownie albo przełącz się na korpus zmierzony.")) from exc

    def work(emit) -> dict:
        def emit_na_drut(**event) -> None:
            # Pipeline emituje SWOJE obiekty domenowe; format „na drucie" należy do tej warstwy.
            if event.get("stage") == "retrieval" and "hits" in event:
                event["hits"] = _hits_payload(event["hits"])
            emit(**event)

        t0 = time.perf_counter()
        result = answer(req.question, top_k=req.top_k, gate=req.gate,
                        verify=req.verify, on_event=emit_na_drut, corpus=korpus)
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


# --- Frontend ----------------------------------------------------------------------------
# Statyki serwuje sam FastAPI — nie ma osobnego serwera WWW, bo to aplikacja lokalna.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def main() -> None:
    import uvicorn
    uvicorn.run(app, host=config.SERVE_HOST, port=config.SERVE_PORT)


if __name__ == "__main__":
    main()
