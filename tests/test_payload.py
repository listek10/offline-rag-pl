"""
tests/test_payload.py
Kontrakt payloadu punktu Qdranta — JEDEN opis kształtu dla zapisu i odczytu.

DLACZEGO TE TESTY ISTNIEJĄ (klasa błędu, nie hipoteza): kształt payloadu był opisany
NIEZALEŻNIE w trzech miejscach — `index_records()` (zapis), `search.py` (odczyt z Qdranta)
i `hybrid.py` (złożenie trafienia z rekordu, bo trafienia z samego BM25 nie przechodzą przez
Qdranta). Dryf między nimi jest CICHY w najgorszy możliwy sposób: przemianowanie klucza przy
zapisie wywala się dopiero w runtime na prawdziwym Qdrancie, a `pytest` zostaje zielony, bo
w testach retrieval jest atrapą. To dokładnie ta sama pułapka, którą dla trafień zamknął
`src/retrieve/hit.py` (commit 177c36e), tylko po stronie indeksu.

Test kluczowy to ROUND-TRIP: ani test zapisu, ani test odczytu osobno nie łapią dryfu, bo każdy
z nich sprawdza swoją stronę wobec własnego (być może już przesuniętego) wyobrażenia o kluczach.
Dopiero przepuszczenie rekordu przez zapis i odczyt wiąże obie strony jednym twierdzeniem.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.index.payload import (PAYLOAD_FIELDS, hit_from_payload, hit_from_record,
                               make_payload, payload_from_record)

# Rekord chunka w kształcie, jaki produkuje `chunk.chunk_text` (uwaga: klucz `id`, nie `chunk_id`
# — to właśnie mapowanie, które musi mieszkać w jednym miejscu).
REKORD = {
    "id": "raport-x__7",
    "source": "raport-x",
    "chunk_index": 7,
    "n_chars": 42,
    "text": "Dania przeznaczyła na obronę 1,3% PKB.",
}


def test_payload_z_rekordu_ma_dokladnie_kanoniczne_klucze():
    p = payload_from_record(REKORD)
    assert set(p) == set(PAYLOAD_FIELDS)
    # `n_chars` NIE wchodzi do payloadu: to metadana chunkingu, nie dana do cytowania.
    assert "n_chars" not in p
    assert p["chunk_id"] == "raport-x__7"


def test_round_trip_zapis_odczyt_daje_to_samo_trafienie():
    # Serce tego pliku: zapis (rekord -> payload) i odczyt (payload -> trafienie) muszą się
    # spotkać na tych samych nazwach. Ścieżka BM25 (rekord -> trafienie, bez Qdranta) też.
    z_qdranta = hit_from_payload(score=0.9, payload=payload_from_record(REKORD))
    z_bm25 = hit_from_record(score=0.9, record=REKORD)
    assert z_qdranta == z_bm25
    assert z_qdranta["chunk_id"] == REKORD["id"]
    assert z_qdranta["text"] == REKORD["text"]


def test_dryf_nazwy_pola_to_TypeError_a_nie_cicha_zerowka():
    # Mechanizm obrony ten sam co w `hit.py`: argumenty TYLKO po nazwie.
    with pytest.raises(TypeError):
        make_payload(chunk_id="a__0", zrodlo="a", chunk_index=0, text="t")  # noqa
    with pytest.raises(TypeError):
        make_payload("a__0", "a", 0, "t")  # pozycyjnie — też nie wolno


def test_brakujacy_klucz_w_payloadzie_wybucha_zamiast_wpisac_None():
        # Payload z bazy zaindeksowanej STARSZĄ wersją kodu ma prawo nie mieć pola. Chcemy wtedy
        # KeyError, nie trafienia z `None` w źródle — to drugie płynie do UI i do cytatu.
    niepelny = {"chunk_id": "a__0", "source": "a", "text": "t"}   # bez `chunk_index`
    with pytest.raises(KeyError):
        hit_from_payload(score=0.5, payload=niepelny)


def test_rekord_bez_wymaganego_pola_wybucha_przy_zapisie():
    # Lepiej wywalić się przed policzeniem 1500 embeddingów niż zapisać kaleki indeks.
    with pytest.raises(KeyError):
        payload_from_record({"id": "a__0", "source": "a", "text": "t"})   # bez `chunk_index`


# --- kontrola zgodności bazy z plikiem (src/index/verify_payload.py) ----------------------
# DLACZEGO NA ATRAPIE KLIENTA: sprawdzamy LOGIKĘ porównania (co jest sierotą, co brakiem),
# a nie samego Qdranta. Wersja z prawdziwą bazą to `python src/index/verify_payload.py`.

class _KlientAtrapa:
    """Minimalny `scroll` z paginacją — kontrakt, na którym stoi `verify()`."""

    def __init__(self, punkty: list[dict], strona: int = 2):
        self._punkty = punkty
        self._strona = strona

    def scroll(self, collection_name, limit, offset, with_payload, with_vectors):
        i = offset or 0
        okno = self._punkty[i:i + self._strona]
        nastepny = i + self._strona
        return ([type("P", (), {"payload": p})() for p in okno],
                nastepny if nastepny < len(self._punkty) else None)


def _rekordy(n: int) -> list[dict]:
    return [{"id": f"r__{i}", "source": "r", "chunk_index": i, "n_chars": 5, "text": f"tekst {i}"}
            for i in range(n)]


def test_verify_zgodny_indeks_daje_ok():
    from src.index.verify_payload import verify
    rek = _rekordy(5)
    klient = _KlientAtrapa([payload_from_record(r) for r in rek])

    w = verify(rek, "kolekcja-x", client=klient)

    assert w["ok"] is True
    assert (w["zgodnych"], w["roznych"], w["sierot_w_kolekcji"]) == (5, 0, 0)


def test_verify_widzi_payload_zapisany_starym_kodem():
    from src.index.verify_payload import verify
    rek = _rekordy(3)
    payloady = [payload_from_record(r) for r in rek]
    payloady[1] = {**payloady[1], "text": "TEKST Z INNEJ WERSJI PLIKU"}   # dryf w danych
    klient = _KlientAtrapa(payloady)

    w = verify(rek, "kolekcja-x", client=klient)

    assert w["ok"] is False
    assert w["roznych"] == 1
    assert w["przyklady_roznic"][0]["chunk_id"] == "r__1"


def test_verify_widzi_sierote_po_starszym_pliku_chunkow():
    from src.index.verify_payload import verify
    rek = _rekordy(2)
    payloady = [payload_from_record(r) for r in rek]
    payloady.append(make_payload(chunk_id="r__99", source="r", chunk_index=99, text="sierota"))
    klient = _KlientAtrapa(payloady)

    w = verify(rek, "kolekcja-x", client=klient)

    assert w["ok"] is False and w["sierot_w_kolekcji"] == 1


def test_verify_widzi_chunk_ktorego_nie_ma_w_kolekcji():
    from src.index.verify_payload import verify
    rek = _rekordy(4)
    klient = _KlientAtrapa([payload_from_record(r) for r in rek[:2]])   # build przerwany w połowie

    w = verify(rek, "kolekcja-x", client=klient)

    assert w["ok"] is False and w["brakuje_w_kolekcji"] == 2
