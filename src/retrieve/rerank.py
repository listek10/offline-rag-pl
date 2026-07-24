"""
src/retrieve/rerank.py
Etap 5 / checkpoint #6: reranking cross-encoderem (drugi etap retrievalu).

DLACZEGO: etap 1 (bi-encoder BGE-M3) jest szybki, ale zgrubny — trafny chunk zwykle JEST w
kandydatach, lecz nie na górze (nasz baseline: recall@10=0.92, ale recall@1=0.31). Cross-encoder
bierze (pytanie, chunk) RAZEM przez model — atencja widzi oba naraz — i ocenia trafność dużo
celniej niż porównanie dwóch osobnych wektorów. Koszt: model odpala się na KAŻDEJ parze, więc
robimy to tylko na N kandydatach z etapu 1 (nie na całym korpusie).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config

_reranker = None  # leniwy singleton — model ładujemy raz


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(config.RERANK_MODEL, device=config.EMBED_DEVICE)
    return _reranker


def rerank_hits(query: str, hits: list[dict], top_k: int) -> list[dict]:
    """
    Przesortowuje kandydatów z etapu 1 wg score cross-encodera i zwraca top_k.
    Dodaje pole 'rerank_score'. Jeśli brak kandydatów — zwraca pustą listę.
    """
    if not hits:
        return []
    ce = _get_reranker()
    pairs = [(query, h["text"]) for h in hits]
    scores = ce.predict(pairs)  # wyższy = bardziej trafny
    for h, s in zip(hits, scores):
        h["rerank_score"] = float(s)
    ranked = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)
    return ranked[:top_k]
