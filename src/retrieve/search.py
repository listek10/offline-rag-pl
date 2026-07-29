"""
src/retrieve/search.py
Etap 3 / checkpoint #5: DENSE retrieval — pytanie -> embedding -> Qdrant top_k.

CO SIĘ TU DZIEJE (i dlaczego):
- Zamieniamy pytanie na wektor TYM SAMYM modelem co przy indeksowaniu (BGE-M3) i szukamy
  najbliższych chunków w Qdrant (metryka cosine, indeks HNSW).
- To jest nasz BASELINE. Świadomie zaczynamy od samego dense (bez BM25/hybrydy/rerankingu):
  najpierw prosty, działający wariant, potem — jeśli gold set (etap 4) pokaże, że trzeba —
  dokładamy hybrydę i reranking (checkpoint #6). Optymalizacja z DOWODEM, nie na zapas.

DWA WAŻNE "DLACZEGO":
1. normalize_embeddings MUSI być takie samo jak przy indeksowaniu (config.EMBED_NORMALIZE).
   Inaczej score cosine jest bez sensu (porównywalibyśmy wektory w innej skali).
2. Model i klient ładujemy RAZ (leniwe singletony). W pętli zapytań (etap 3) będziemy wołać
   search() wielokrotnie — przeładowywanie BGE-M3 przy każdym pytaniu byłoby wolne.

Uruchomienie:
    python src/retrieve/search.py "Kiedy Szwecja wstąpiła do NATO?"
    python src/retrieve/search.py            # tryb interaktywny
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config
from src.retrieve import corpus as corpus_mod
from src.retrieve.hit import make_hit

# Leniwe singletony — ładowane przy pierwszym użyciu, potem reużywane.
_model = None
_client = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
    return _model


def _get_client():
    global _client
    if _client is None:
        from qdrant_client import QdrantClient
        _client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    return _client


def search(
    query: str,
    top_k: int | None = None,
    rerank: bool = False,
    candidates: int | None = None,
    corpus: "corpus_mod.Corpus | str | None" = None,
) -> list[dict]:
    """
    Zwraca listę najtrafniejszych chunków z metadanymi do cytowania.
    Każdy wynik: {score, source, chunk_index, chunk_id, text} (+ 'rerank_score' gdy rerank=True).

    rerank=False -> czysty dense (baseline): Qdrant zwraca od razu top_k.
    rerank=True  -> dwustopniowo: pobierz `candidates` (etap 1, dense) i przesortuj cross-encoderem
                    (etap 2), zwróć top_k. `candidates` = sufit jakości rerankingu (recall@N etapu 1).

    `corpus` wskazuje KOLEKCJĘ Qdranta (`None` => bazowa z configu). Model embeddingowy zostaje
    ten sam — i to jest sedno etapu 2 UI: nowy dokument kosztuje sekundy indeksowania, a nie
    dotknięcie modelu. Wektory obcego dokumentu muszą leżeć w OSOBNEJ kolekcji, bo kolekcja,
    na której mierzę (recall@1, MRR), musi być zamrożona.
    """
    top_k = top_k or config.TOP_K
    n = (candidates or config.RERANK_CANDIDATES) if rerank else top_k

    model = _get_model()
    client = _get_client()

    # Pytanie -> wektor. Normalizacja identyczna jak przy indeksowaniu (patrz docstring).
    qv = model.encode([query], normalize_embeddings=config.EMBED_NORMALIZE)[0]

    points = client.query_points(
        collection_name=corpus_mod.resolve(corpus).collection,
        query=qv.tolist(),
        limit=n,
    ).points

    # Kształt trafienia pochodzi z `make_hit` (src/retrieve/hit.py) — jedno źródło nazw pól
    # dla dense, hybrydy i atrap w testach. Dlaczego to ważne: patrz docstring tamtego modułu.
    hits = [
        make_hit(
            score=p.score,
            source=p.payload["source"],
            chunk_index=p.payload["chunk_index"],
            chunk_id=p.payload["chunk_id"],
            text=p.payload["text"],
        )
        for p in points
    ]

    if rerank:
        from src.retrieve.rerank import rerank_hits
        hits = rerank_hits(query, hits, top_k)

    return hits[:top_k]


def _print_hits(query: str, hits: list[dict]) -> None:
    print(f'\nPytanie: "{query}"  (top {len(hits)})')
    for i, h in enumerate(hits, 1):
        snippet = " ".join(h["text"][:200].split())
        print(f"  {i}. [{h['score']:.3f}] {h['source'][:45]} #{h['chunk_index']}")
        print(f"     {snippet}…")


def main() -> None:
    args = sys.argv[1:]
    if args:
        query = " ".join(args)
        _print_hits(query, search(query))
    else:
        print("Tryb interaktywny. Wpisz pytanie (puste = koniec).")
        _get_model()  # rozgrzej model raz, żeby pierwsze pytanie nie czekało na ładowanie
        while True:
            try:
                q = input("\n> ").strip()
            except EOFError:
                break
            if not q:
                break
            _print_hits(q, search(q))


if __name__ == "__main__":
    main()
