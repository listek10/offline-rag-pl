"""
src/retrieve/hybrid.py
Etap 5: hybrydowy retrieval = dense (Qdrant/BGE-M3) + BM25, połączone przez RRF.

DLACZEGO RRF (Reciprocal Rank Fusion): dense (cosine ~0–1) i BM25 (~0–15) mają RÓŻNE SKALE,
więc surowego dodawania score'ów robić nie wolno. RRF łączy po POZYCJI:
    rrf(d) = Σ_r  1 / (RRF_K + rank_r(d))
gdzie r to kolejny retriever (dense, bm25), a rank_r(d) to pozycja chunka d w jego rankingu.
Odporne na skalę, bez strojenia wag.

Po co hybryda skoro mamy dense+rerank: rerank tylko PRZESTAWIA kandydatów z etapu 1 (sufit =
recall@N dense). Hybryda zmienia to, CO wchodzi do puli — BM25 dorzuca trafienia leksykalne
(nazwy własne), których dense nie wyciągnął → może podnieść recall@N (sufit dla rerankingu).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config
from src.index.payload import hit_from_record
from src.retrieve import corpus as corpus_mod
from src.retrieve.bm25 import get_bm25
from src.retrieve.search import search as dense_search

# Cache map chunków, kluczowany korpusem — te same powody co przy `get_bm25()` (patrz bm25.py).
_chunks = corpus_mod.new_cache()


def _get_chunks(corpus: "corpus_mod.Corpus | str | None" = None) -> dict:
    """Mapa chunk_id -> rekord (potrzebna, bo trafienia z samego BM25 nie przechodzą przez Qdrant)."""
    c = corpus_mod.resolve(corpus)
    if c.name not in _chunks:
        _chunks[c.name] = {r["id"]: r for r in c.load_records()}
    return _chunks[c.name]


def rrf_fuse(rankings: list[list[str]], k: int) -> dict[str, float]:
    """Łączy kilka rankingów (list chunk_id od najlepszego) w jeden score RRF."""
    scores: dict[str, float] = defaultdict(float)
    for ids in rankings:
        for rank, cid in enumerate(ids, 1):
            scores[cid] += 1.0 / (k + rank)
    return scores


def hybrid_search(
    query: str,
    top_k: int | None = None,
    candidates: int | None = None,
    rerank: bool = False,
    corpus: "corpus_mod.Corpus | str | None" = None,
) -> list[dict]:
    """
    dense top-N + BM25 top-N -> RRF -> (opcjonalnie) cross-encoder rerank -> top_k.
    Zwraca listę {score, source, chunk_index, chunk_id, text} (score = RRF, chyba że rerank).

    `corpus` = na czym szukamy (`None` => korpus bazowy z configu). Rozstrzygamy go RAZ i
    przekazujemy jako obiekt dalej: dense, BM25 i mapa chunków MUSZĄ patrzeć na ten sam korpus,
    bo inaczej trafienia z jednego indeksowałyby się metadanymi drugiego.
    """
    top_k = top_k or config.TOP_K
    n = candidates or config.RERANK_CANDIDATES
    c = corpus_mod.resolve(corpus)

    dense_ids = [h["chunk_id"] for h in dense_search(query, top_k=n, rerank=False, corpus=c)]
    bm25_ids = [cid for cid, _ in get_bm25(c).search(query, n)]

    fused = rrf_fuse([dense_ids, bm25_ids], k=config.RRF_K)
    ranked_ids = sorted(fused, key=lambda c_id: fused[c_id], reverse=True)[:n]

    chunks = _get_chunks(c)
    # Trafienia z samego BM25 nie przechodzą przez Qdranta, więc metadane bierzemy z REKORDU —
    # ale przez tę samą fabrykę, co odczyt payloadu (src/index/payload.py). Gdyby te dwie ścieżki
    # opisywały pola niezależnie, hybryda mieszałaby trafienia o różnych kształtach metadanych,
    # czyli cytowała zły dokument, nie rzucając błędu.
    hits = [hit_from_record(score=fused[cid], record=chunks[cid]) for cid in ranked_ids]

    if rerank:
        from src.retrieve.rerank import rerank_hits
        hits = rerank_hits(query, hits, top_k)

    return hits[:top_k]
