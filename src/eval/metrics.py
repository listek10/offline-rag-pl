"""
src/eval/metrics.py
Etap 4 / checkpoint #11: metryki retrievalu (obiektywne, liczone z chunk_id).

- recall@k (hit-rate): czy PRZYNAJMNIEJ JEDEN trafny chunk jest w top-k. To najważniejsze dla
  RAG — mówi, czy właściwa informacja w ogóle trafiła do promptu w obrębie k. (Wariant "czy
  dotarł choć jeden trafny"; przy pytaniach z kilkoma poprawnymi chunkami liczymy trafienie,
  gdy złapiemy którykolwiek.)
- MRR (reciprocal rank): 1 / pozycja pierwszego trafnego chunka. Nagradza ustawienie trafnego
  WYSOKO. recall@k patrzy "czy jest w top-k", MRR — "jak blisko szczytu".
"""

from __future__ import annotations


def hit_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """1.0 jeśli którykolwiek trafny chunk jest w top-k, inaczej 0.0 (dla jednego pytania)."""
    top = retrieved_ids[:k]
    return 1.0 if any(r in top for r in relevant_ids) else 0.0


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """1 / pozycja pierwszego trafnego chunka; 0.0 jeśli nie ma go w ogóle."""
    for i, cid in enumerate(retrieved_ids, 1):
        if cid in relevant_ids:
            return 1.0 / i
    return 0.0


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0
