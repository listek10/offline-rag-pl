"""
src/index/ad_hoc.py
Etap 2 UI: dokument wgrany ad hoc — indeks ULOTNY, izolowany od korpusu zmierzonego.

CO TO UDOWADNIA (i czego NIE udowadnia): nie to, że „model generalizuje na niewidzianych
danych" — wagi modelu są zamrożone, a wiedza wchodzi przez prompt w czasie zapytania.
Udowadnia coś mocniejszego: CAŁY pipeline (parsowanie -> chunking -> embedding -> indeks ->
hybrydowy retrieval -> trzy guardy -> cytowanie) działa na dokumencie nigdy niewidzianym,
BEZ dotykania modelu. To ekonomiczny argument za RAG: nowa wiedza kosztuje sekundy
indeksowania, nie godziny treningu.

DWIE DECYZJE, KTÓRE TRZYMAJĄ TEN MODUŁ W RYZACH:

1. **Osobna kolekcja (`config.UPLOAD_COLLECTION`), kasowana przy starcie I przy zamknięciu
   procesu.** Ten drugi zamiatacz sprawia, że `kill -9` nie zostawia sieroty. Ulotność jest
   STRUKTURALNA, nie dyscyplinarna: dokument demo fizycznie nie przeżywa procesu, więc nie da
   się przypadkiem zaśmiecić korpusu ewaluacyjnego.
   *Odrzucona alternatywa:* embeddingi w pamięci + brute-force cosine (300 chunków x 1024 wymiary
   to ~1,2 MB, więc dokładne przeszukanie bije tu HNSW, który jest aproksymacją potrzebną dopiero
   przy dużym korpusie). Odrzucone, bo demo pokazywałoby ścieżkę OBOK indeksu wektorowego, a nie
   ścieżkę produkcyjną — plus druga implementacja retrievalu to drugie miejsce do rozjechania się.

2. **Odmowa pracy, gdy nazwa kolekcji demo równa się nazwie kolekcji zmierzonej.**
   `index_records` kasuje kolekcję przed zapisem, a `QDRANT_COLLECTION` jest nadpisywalne
   środowiskiem (A/B chunkingu, etap 8). Zbieżność nazw nie dałaby błędu — dałaby CICHO
   przebudowany indeks, na którym stoją liczby z README. To nie jest hipotetyczne: dokładnie
   ta pułapka była powodem, dla którego etap 2 nie mógł po prostu wywołać `build_index.py`.

Model i klient BIERZEMY OD WARSTWY RETRIEVALU (`search._get_model/_get_client`), a nie tworzymy
nowych: drugi egzemplarz BGE-M3 w 12 GB VRAM to OOM (zmierzone w etapie 7 przy vLLM).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config
from src.index.build_index import index_records
from src.ingest.upload import prepare_upload
from src.retrieve import corpus as corpus_mod
from src.retrieve.corpus import AD_HOC, Corpus

log = logging.getLogger("offline_rag.index.ad_hoc")


def _model():
    """Ten SAM model, którym liczy retrieval (leniwy singleton) — patrz docstring modułu."""
    from src.retrieve.search import _get_model
    return _get_model()


def _client():
    from src.retrieve.search import _get_client
    return _get_client()


def _sprawdz_izolacje() -> str:
    """Zwraca nazwę kolekcji demo, upewniając się, że NIE jest to kolekcja zmierzona."""
    if config.UPLOAD_COLLECTION == config.QDRANT_COLLECTION:
        raise RuntimeError(
            f"Kolekcja demo i kolekcja zmierzona mają tę samą nazwę ({config.UPLOAD_COLLECTION}). "
            f"Wgrywanie PDF-u skasowałoby indeks, na którym stoją zmierzone liczby — odmawiam.")
    return config.UPLOAD_COLLECTION


def build(data: bytes, filename: str) -> dict:
    """Bajty PDF-a -> ulotny korpus gotowy do pytań. Zwraca statystyki dla UI.

    Kolejność: polityka i chunking (bez GPU) -> embeddingi + kolekcja -> rejestracja korpusu.
    Rejestracja jest OSTATNIA celowo: gdyby indeksowanie padło w połowie, w rejestrze nie może
    zostać korpus, którego wektorów nie ma w bazie (UI oferowałby pytania w pustkę).
    """
    collection = _sprawdz_izolacje()
    t0 = time.perf_counter()
    records, stats = prepare_upload(data, filename)

    points = index_records(records, collection, model=_model(), client=_client())

    corpus_mod.register(Corpus(
        name=AD_HOC,
        collection=collection,
        records=records,          # rekordy żyją w PAMIĘCI — plik nie dotknął dysku
        label=stats["source"],
        measured=False,           # na tym korpusie NIE są kalibrowane progi guardów
        meta=stats,
    ))
    log.info("wgrano dokument demo '%s': %d chunków, %d punktów", stats["source"], stats["chunks"], points)
    return {**stats, "corpus": AD_HOC, "collection": collection, "points": points,
            "ms": round((time.perf_counter() - t0) * 1000)}


def drop() -> None:
    """Kasuje ulotny korpus: rejestr + cache'e retrievalu + kolekcja w Qdrancie.

    Wołane przy STARCIE procesu (sierota po `kill -9`) i przy ZAMKNIĘCIU. Awaria bazy nie może
    przewrócić ani startu, ani zamykania serwera — ale rejestr czyścimy ZAWSZE, bo inaczej UI
    oferowałoby korpus, o którego wektorach nic nie wiemy.
    """
    collection = _sprawdz_izolacje()
    corpus_mod.drop(AD_HOC)
    try:
        client = _client()
        if client.collection_exists(collection):
            client.delete_collection(collection)
            log.info("skasowano ulotną kolekcję '%s'", collection)
    except Exception as exc:  # noqa: BLE001
        log.warning("nie udało się skasować kolekcji '%s': %s", collection, exc)
