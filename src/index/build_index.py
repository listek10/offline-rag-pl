"""
src/index/build_index.py
Etap 2 (indeks): liczy embeddingi chunków i ładuje je do Qdrant (checkpointy #3 i #4).

CO SIĘ TU DZIEJE (i dlaczego):
1. Wczytujemy chunki (data/processed/chunks.jsonl).
2. Zamieniamy każdy chunk na WEKTOR modelem BGE-M3 (wielojęzyczny -> dobry polski).
   Wektor to punkt w przestrzeni 1024D, gdzie bliskość = podobieństwo znaczeniowe.
   Normalizujemy wektory -> dzięki temu cosine == dot (patrz config.EMBED_NORMALIZE).
3. Zapisujemy wektory + metadane (źródło, tekst) do Qdrant, do kolekcji z metryką COSINE
   i indeksem HNSW (Qdrant buduje go automatycznie). HNSW = wyszukiwanie przybliżone,
   sub-liniowe O(log N) zamiast brute-force O(N) — po to jest indeks.
4. Smoke test: pytamy o coś i sprawdzamy, czy retrieval zwraca sensowne chunki.

BGE-M3 nie wymaga prefiksów "query:"/"passage:" (w przeciwieństwie do e5) — upraszcza kod.

Uruchomienie (wymaga uruchomionego Qdrant w Dockerze):
    python src/index/build_index.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config


def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[BŁĄD] Brak {path}. Najpierw uruchom chunk.py.")
        sys.exit(1)
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def main() -> None:
    # Ciężkie importy trzymamy w środku, żeby sam import pliku nie ładował torcha.
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    chunks = load_chunks(config.CHUNKS_PATH)
    print(f"[1/4] Wczytano {len(chunks)} chunków.")

    # --- Embeddingi ---
    print(f"[2/4] Ładuję model {config.EMBED_MODEL} na {config.EMBED_DEVICE}...")
    model = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
    texts = [c["text"] for c in chunks]
    vectors = model.encode(
        texts,
        batch_size=config.EMBED_BATCH_SIZE,
        normalize_embeddings=config.EMBED_NORMALIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"      Policzono wektory: {vectors.shape} (oczekiwane wymiary: {config.EMBED_DIM}).")
    assert vectors.shape[1] == config.EMBED_DIM, "Wymiar wektora != config.EMBED_DIM!"

    # --- Qdrant: (re)tworzymy kolekcję ---
    print(f"[3/4] Łączę z Qdrant {config.QDRANT_HOST}:{config.QDRANT_PORT}...")
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

    name = config.QDRANT_COLLECTION
    if client.collection_exists(name):
        client.delete_collection(name)  # świeży indeks przy każdym buildzie (idempotentnie)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE),
    )

    # Punkt Qdrant: id (int), wektor, payload z metadanymi do cytowania i podglądu.
    points = [
        PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={
                "chunk_id": c["id"],
                "source": c["source"],
                "chunk_index": c["chunk_index"],
                "text": c["text"],
            },
        )
        for i, c in enumerate(chunks)
    ]

    # Upsert w partiach — bezpieczniej niż jeden wielki request.
    B = 256
    for start in range(0, len(points), B):
        client.upsert(collection_name=name, points=points[start:start + B])
    count = client.count(collection_name=name).count
    print(f"      Załadowano {count} punktów do kolekcji '{name}'.")

    # --- Smoke test: czy retrieval działa? ---
    print("[4/4] Smoke test retrievalu:")
    query = "Dlaczego Szwecja i Finlandia wstąpiły do NATO?"
    qv = model.encode([query], normalize_embeddings=config.EMBED_NORMALIZE)[0]
    hits = client.query_points(collection_name=name, query=qv.tolist(), limit=3).points
    print(f'      Pytanie: "{query}"')
    for h in hits:
        snippet = h.payload["text"][:140].replace("\n", " ")
        print(f"      [{h.score:.3f}] {h.payload['source'][:35]}… | {snippet}…")


if __name__ == "__main__":
    main()
