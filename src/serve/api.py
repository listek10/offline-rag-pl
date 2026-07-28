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
