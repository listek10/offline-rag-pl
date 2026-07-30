"""
src/index/payload.py
JEDNO miejsce, które wie, co leży w payloadzie punktu Qdranta — i jak się to czyta.

DLACZEGO TO ISTNIEJE (a nie jest abstrakcją na zapas): kształt payloadu był zapisany
NIEZALEŻNIE w trzech miejscach:

    `build_index.index_records()`  — rekord chunka  -> payload            (ZAPIS)
    `search.search()`              — payload        -> trafienie          (ODCZYT z Qdranta)
    `hybrid.hybrid_search()`       — rekord chunka  -> trafienie          (ścieżka BM25)

Trzecia ścieżka istnieje, bo trafienia z samego BM25 nigdy nie przechodzą przez Qdranta —
składamy je z rekordów w pliku. To znaczy, że TO SAMO mapowanie pól opisane jest dwa razy po
stronie odczytu, a raz po stronie zapisu.

DLACZEGO TEN DRYF JEST GROŹNIEJSZY NIŻ WYGLĄDA: jest cichy w obie strony.
- Przemianowanie klucza przy ZAPISIE nie wywala testów — w `pytest` retrieval jest atrapą,
  więc nikt nie czyta prawdziwego payloadu. Wywali się w runtime, na prawdziwym Qdrancie,
  po policzeniu 1500 embeddingów.
- Rozjechanie się ODCZYTU z Qdranta i odczytu z rekordu (BM25) daje coś gorszego niż wyjątek:
  hybryda zwraca trafienia, w których część rekordów ma metadane w jednym kształcie, a część
  w drugim — czyli cytat wskazujący na zły dokument. Nic nie wybucha.

To ta sama klasa błędu, którą dla TRAFIENIA zamknął `src/retrieve/hit.py` (commit 177c36e),
tylko po stronie indeksu — i domykamy ją tym samym mechanizmem: argumenty TYLKO PO NAZWIE (`*`),
więc zmiana kontraktu = zmiana sygnatury = `TypeError` natychmiast, bez type-checkera.

DLACZEGO ZAPIS I ODCZYT SIEDZĄ W JEDNYM MODULE (a nie w dwóch „czystszych"): żeby przemianowanie
klucza łamało OBIE strony w tym samym pliku. Rozdzielenie ich przywróciłoby dokładnie ten dryf,
przed którym ten moduł ma bronić. Test round-trip (`tests/test_payload.py`) jest jedynym testem,
który to wiąże — testy zapisu i odczytu OSOBNO przechodzą także na przesuniętych nazwach.

DLACZEGO DICT, A NIE dataclass: payload płynie do Qdranta jako JSON, a trafienia do UI jako JSON
(patrz `hit.py`) — dataclass wymuszałby konwersję na każdej granicy. Zysk (jedno źródło nazw)
dostajemy bez zmiany typu danych.
"""

from __future__ import annotations

from src.retrieve.hit import make_hit

# Kanoniczne pola payloadu. `text` jest w payloadzie ŚWIADOMIE (duplikuje plik z chunkami):
# bez niego trafienie z Qdranta wymagałoby drugiego odczytu, a UI ma pokazać fragment od razu.
# `n_chars` z rekordu chunka NIE wchodzi — to metadana chunkingu, nie dana do cytowania.
PAYLOAD_FIELDS = ("chunk_id", "source", "chunk_index", "text")

# Nazwa pola, pod którym identyfikator chunka żyje w REKORDZIE (`chunks.jsonl`), a nie w payloadzie.
# To jedyne miejsce, gdzie asymetria `id` (rekord) vs `chunk_id` (payload/trafienie) jest zapisana.
RECORD_ID_FIELD = "id"


def make_payload(*, chunk_id: str, source: str, chunk_index: int, text: str) -> dict:
    """Buduje payload w kanonicznym kształcie. Argumenty tylko po nazwie — patrz docstring modułu."""
    return {"chunk_id": chunk_id, "source": source, "chunk_index": chunk_index, "text": text}


def payload_from_record(record: dict) -> dict:
    """Rekord chunka (`chunks.jsonl`) -> payload punktu Qdranta.

    Indeksowanie po kluczu (a nie `.get()`) jest celowe: rekord bez `chunk_index` ma wywalić
    `KeyError` PRZED policzeniem embeddingów, a nie zapisać kaleki indeks, który cicho
    zwróci trafienie z `None` w metadanych.
    """
    return make_payload(
        chunk_id=record[RECORD_ID_FIELD],
        source=record["source"],
        chunk_index=record["chunk_index"],
        text=record["text"],
    )


def hit_from_payload(*, score: float, payload: dict) -> dict:
    """Payload punktu z Qdranta -> trafienie w kształcie `make_hit` (ścieżka dense)."""
    return make_hit(
        score=score,
        source=payload["source"],
        chunk_index=payload["chunk_index"],
        chunk_id=payload["chunk_id"],
        text=payload["text"],
    )


def hit_from_record(*, score: float, record: dict) -> dict:
    """Rekord chunka -> trafienie (ścieżka BM25, która OMIJA Qdranta — patrz docstring modułu).

    Musi dać wynik identyczny z `hit_from_payload(payload_from_record(...))` — i dokładnie to
    sprawdza test round-trip. Gdyby te dwie ścieżki się rozjechały, hybryda RRF mieszałaby
    trafienia o różnych kształtach metadanych, czyli cytowała zły dokument, nie rzucając błędu.
    """
    return hit_from_payload(score=score, payload=payload_from_record(record))
