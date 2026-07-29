"""
src/eval/metrics.py
Etap 4 / checkpoint #11: metryki retrievalu (obiektywne, liczone z chunk_id).

- recall@k (hit-rate): czy PRZYNAJMNIEJ JEDEN trafny chunk jest w top-k. To najważniejsze dla
  RAG — mówi, czy właściwa informacja w ogóle trafiła do promptu w obrębie k. (Wariant "czy
  dotarł choć jeden trafny"; przy pytaniach z kilkoma poprawnymi chunkami liczymy trafienie,
  gdy złapiemy którykolwiek.)
- MRR (reciprocal rank): 1 / pozycja pierwszego trafnego chunka. Nagradza ustawienie trafnego
  WYSOKO. recall@k patrzy "czy jest w top-k", MRR — "jak blisko szczytu".

DWIE DEFINICJE PRAWDY (etap 8) — świadomie rozdzielone:
- `chunk_ids`: trafny = chunk wymieniony w gold secie. Tanie, ale WIĄŻE prawdę z konkretnym
  podziałem tekstu: po zmianie rozmiaru chunka numeracja się przesuwa i identyfikatory
  przestają cokolwiek wskazywać.
- `evidence`: trafny = chunk ZAWIERAJĄCY dosłowny cytat, w którym jest odpowiedź. Definicja
  niezależna od podziału, więc jedyna sensowna przy porównywaniu strategii chunkingu (także
  semantycznego, gdzie granice nie są funkcją liczby znaków). Jest też SUROWSZA: chunk „na
  temat", ale bez odpowiedzi, nie liczy się jako trafienie.
UWAGA: te dwie definicje dają RÓŻNE liczby bezwzględne, więc nie wolno porównywać recall
liczonego jedną z recallem liczonym drugą. Wariant bazowy raportujemy pod OBIEMA.
"""

from __future__ import annotations

import re


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


def _norm(s: str) -> str:
    """Normalizuje białe znaki. Parser PDF-a zostawia miejscami dziwne odstępy (np.
    „fińsko- -estońskiego"), a dopasowanie ma być odporne na odstęp, nie na treść."""
    return re.sub(r"\s+", " ", s).strip()


def relevant_by_evidence(chunks: list[dict], evidence: str) -> list[str]:
    """Zwraca id chunków ZAWIERAJĄCYCH cytat-dowód — prawda niezależna od podziału tekstu.

    Liczy WSZYSTKIE takie chunki, nie tylko pierwszy: przy zakładce ten sam fragment bywa w
    dwóch kolejnych chunkach i oba realnie zawierają odpowiedź. Konsekwencja, którą trzeba
    znać: większa zakładka = więcej chunków z odpowiedzią = statystycznie łatwiejsze trafienie.
    Dlatego w A/B rozmiaru chunka trzymamy UDZIAŁ zakładki stały (15%), żeby nie mieszać
    dwóch efektów w jednej liczbie.
    """
    cyt = _norm(evidence)
    return [c["id"] for c in chunks if cyt in _norm(c["text"])]
