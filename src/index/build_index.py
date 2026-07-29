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


def index_records(records: list[dict], collection: str, *, model, client,
                  progress: bool = False) -> int:
    """Liczy embeddingi rekordów i (RE)TWORZY kolekcję `collection`. Zwraca liczbę punktów.

    ⚠️ KASUJE istniejącą kolekcję o tej nazwie — świeży indeks przy każdym buildzie (idempotencja).
    Dlatego wołający MUSI wiedzieć, jaką nazwę podaje: podanie tu kolekcji zmierzonej niszczy
    indeks, na którym stoją liczby z README (patrz `src/index/ad_hoc.py`, który się przed tym broni).

    DLACZEGO TO JEST OSOBNA FUNKCJA (wyciągnięta z `main()` w etapie 2 UI): payload punktu
    (`chunk_id`, `source`, `chunk_index`, `text`) czyta `search.py`, więc jego kształt to kontrakt.
    Wgrywanie PDF-u w UI potrzebuje tej samej ścieżki — a dwa opisy jednego payloadu rozjechałyby
    się przy pierwszej zmianie. `model` i `client` WSTRZYKUJEMY, bo serwer ma już wczytany BGE-M3
    (drugi egzemplarz w 12 GB VRAM = OOM) i otwartego klienta Qdranta.
    """
    from qdrant_client.models import Distance, PointStruct, VectorParams

    vectors = model.encode(
        [r["text"] for r in records],
        batch_size=config.EMBED_BATCH_SIZE,
        normalize_embeddings=config.EMBED_NORMALIZE,
        show_progress_bar=progress,
        convert_to_numpy=True,
    )

    # Kolekcja jest tworzona z wymiarem z configu, więc niezgodność wymiaru wektora dałaby błąd
    # dopiero przy upsercie (albo, gorzej, cichy nonsens w score'ach). Sprawdzamy wcześnie.
    if records and len(vectors[0]) != config.EMBED_DIM:
        raise ValueError(f"Model zwrócił wektor {len(vectors[0])}D, a config mówi {config.EMBED_DIM}D.")

    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE),
    )

    # Punkt Qdrant: id (int), wektor, payload z metadanymi do cytowania i podglądu.
    points = [
        PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={
                "chunk_id": r["id"],
                "source": r["source"],
                "chunk_index": r["chunk_index"],
                "text": r["text"],
            },
        )
        for i, r in enumerate(records)
    ]

    # Upsert w partiach — bezpieczniej niż jeden wielki request.
    B = 256
    for start in range(0, len(points), B):
        client.upsert(collection_name=collection, points=points[start:start + B])
    return client.count(collection_name=collection).count


def main() -> None:
    # Ciężkie importy trzymamy w środku, żeby sam import pliku nie ładował torcha.
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient

    chunks = load_chunks(config.CHUNKS_PATH)
    print(f"[1/4] Wczytano {len(chunks)} chunków.")

    print(f"[2/4] Ładuję model {config.EMBED_MODEL} na {config.EMBED_DEVICE}...")
    model = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)

    print(f"[3/4] Łączę z Qdrant {config.QDRANT_HOST}:{config.QDRANT_PORT}...")
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

    name = config.QDRANT_COLLECTION
    count = index_records(chunks, name, model=model, client=client, progress=True)
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
