"""
src/retrieve/hit.py
JEDNO miejsce, które wie, czym jest „trafienie" (hit) w tym projekcie.

DLACZEGO TO ISTNIEJE (a nie jest abstrakcją na zapas): kształt trafienia był zapisany
NIEZALEŻNIE w trzech miejscach — `search.py` (dense), `hybrid.py` (RRF) i atrapa `_hit()`
w `tests/test_answer_events.py`. Trzy kopie jednej prawdy dryfują, a dryf DOKŁADNIE tutaj
jest cichy i groźny:

    `answer.py` czyta `rerank_score` z domyślną zerówką (`h.get("rerank_score", 0.0)`),
    więc przemianowanie tego klucza NIE daje KeyError — daje `top_score = 0.0`, czyli
    wartość poniżej progu bramki trafności. Skutek: system odmawia odpowiedzi na KAŻDE
    pytanie. Cicho. A testy zostają zielone, bo atrapa dokłada stary klucz.

MECHANIZM OBRONY: argumenty `make_hit()` są TYLKO PO NAZWIE (`*`). Zmiana kontraktu =
zmiana sygnatury = `TypeError` w atrapie testowej, natychmiast i bez type-checkera.
Atrapa przestaje OPISYWAĆ kształt — zaczyna go PYTAĆ.

DRUGA WARSTWA (etap 15): skąd BIORĄ SIĘ pola trafienia — z payloadu punktu Qdranta albo z rekordu
chunka — wie `src/index/payload.py`. Ten moduł opisuje, czym trafienie JEST; tamten, jak się je
czyta z bazy i z pliku. Dlatego `make_hit()` jest tu wołane tylko przez tamtą fabrykę.

DLACZEGO DICT, A NIE dataclass: te słowniki płyną prosto do UI jako JSON (`_hits_payload`
w `api.py`, zdarzenia `on_event`). Dataclass wymuszałby `asdict()` na każdej granicy —
koszt większy niż zysk. Zysk (jedno źródło nazw) dostajemy bez zmiany typu danych.
"""

from __future__ import annotations

# Nazwa klucza w JEDNYM miejscu — czytają go bramka (`answer.py`), UI (`api.py`)
# i reranker. Wcześniej ten literał był rozsiany po czterech plikach.
RERANK_SCORE_KEY = "rerank_score"


def make_hit(*, score: float, source: str, chunk_index: int, chunk_id: str, text: str) -> dict:
    """
    Buduje trafienie w kanonicznym kształcie. `score` znaczy różne rzeczy zależnie od
    retrievera (cosine w dense, RRF w hybrydzie) — i to jest w porządku, bo bramka i UI
    patrzą na `rerank_score`, gdy ten istnieje.

    Argumenty TYLKO po nazwie — to jest cały mechanizm ochrony przed dryfem (patrz docstring
    modułu): pozycyjne wywołanie nie przeżyłoby zmiany kolejności pól, a nazwane nie przeżyje
    zmiany nazwy pola. Chcemy drugiego.
    """
    return {
        "score": score,
        "source": source,
        "chunk_index": chunk_index,
        "chunk_id": chunk_id,
        "text": text,
    }


def set_rerank_score(hit: dict, score: float) -> None:
    """Dokłada score cross-encodera (etap 2 retrievalu). Mutuje w miejscu — tak działa `rerank_hits`."""
    hit[RERANK_SCORE_KEY] = float(score)


def get_rerank_score(hit: dict, default: float = 0.0) -> float:
    """
    Czyta score rerankera. `default` jest ŚWIADOMY: przy `rerank=False` tego pola po prostu nie ma
    (np. `run_eval.py` mierzy warianty bez rerankingu), a UI i tak musi coś pokazać.
    Cena tej wyrozumiałości: brak klucza jest nieodróżnialny od score 0.0 — dlatego nazwa klucza
    żyje w `RERANK_SCORE_KEY`, a nie w literałach rozsianych po kodzie.
    """
    return hit.get(RERANK_SCORE_KEY, default)


def top_rerank_score(hits: list[dict], default: float = 0.0) -> float:
    """Najwyższy score rerankera w puli — to LICZBA, KTÓRĄ PORÓWNUJE BRAMKA (guard #2)."""
    return max((get_rerank_score(h) for h in hits), default=default)
