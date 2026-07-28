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
