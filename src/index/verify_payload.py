"""
src/index/verify_payload.py
Kontrola zgodności: payload LEŻĄCY w Qdrancie vs to, co wypisze dzisiejsza fabryka.

PO CO TO ISTNIEJE (a nie wystarczy test jednostkowy): test sprawdza kod, a ten skrypt sprawdza
DANE. Kolekcja w Qdrancie jest artefaktem zewnętrznym — mogła zostać zapisana starszą wersją
kodu, innym plikiem chunków albo przerwanym buildem. Żaden test w `pytest` tego nie zobaczy,
bo tam retrieval jest atrapą. To jest dokładnie ta różnica, którą w etapie 14 nazwałem
„dowodem zasięgu na danych": kod można przeczytać, ale zgodność bazy z plikiem trzeba zmierzyć.

CZEGO NIE ROBI: nie sprawdza WEKTORÓW (te są kosztowne do przeliczenia i niedeterministyczne
w ostatnich bitach na GPU). Sprawdza metadane, czyli to, z czego powstaje CYTAT — a błąd
w cytacie jest dla tego projektu najgroźniejszą klasą awarii, bo wygląda jak poprawna odpowiedź.

Uruchomienie (wymaga Qdranta):
    python src/index/verify_payload.py                       # kolekcja i chunki z configu
    python src/index/verify_payload.py --json results/payload_contract.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config
from src.index.payload import payload_from_record


def verify(records: list[dict], collection: str, *, client, batch: int = 1000) -> dict:
    """Porównuje payload każdego punktu w kolekcji z `payload_from_record` dla tego chunk_id.

    `client` wstrzykiwany — te same powody co w `index_records`: serwer ma już otwartego klienta,
    a test może podstawić atrapę bez Qdranta.
    """
    po_id = {r["id"]: r for r in records}

    punkty, offset = [], None
    while True:
        batch_points, offset = client.scroll(collection_name=collection, limit=batch,
                                             offset=offset, with_payload=True, with_vectors=False)
        punkty += batch_points
        if offset is None:
            break

    zgodne, rozne, sieroty, brak_id = 0, [], [], 0
    for p in punkty:
        cid = (p.payload or {}).get("chunk_id")
        if cid is None:
            brak_id += 1
            continue
        if cid not in po_id:
            # Punkt, którego nie ma w pliku chunków: indeks jest starszy niż plik (albo odwrotnie).
            sieroty.append(cid)
            continue
        oczekiwany = payload_from_record(po_id[cid])
        if p.payload == oczekiwany:
            zgodne += 1
        else:
            rozne.append({"chunk_id": cid, "w_bazie": p.payload, "z_fabryki": oczekiwany})

    return {
        "collection": collection,
        "chunks_path": str(config.CHUNKS_PATH),
        "punktow_w_kolekcji": len(punkty),
        "rekordow_w_pliku": len(records),
        "zgodnych": zgodne,
        "roznych": len(rozne),
        "bez_chunk_id": brak_id,
        "sierot_w_kolekcji": len(sieroty),
        "brakuje_w_kolekcji": len(po_id) - zgodne - len(rozne),
        "przyklady_roznic": rozne[:5],
        "ok": bool(punkty) and not rozne and not sieroty and not brak_id
             and len(punkty) == len(records),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Zgodność payloadu w Qdrancie z fabryką (etap 15)")
    ap.add_argument("--kolekcja", default=config.QDRANT_COLLECTION)
    ap.add_argument("--json", dest="json_out", help="zapisz raport do pliku JSON")
    a = ap.parse_args()

    from qdrant_client import QdrantClient

    records = [json.loads(l) for l in config.CHUNKS_PATH.open(encoding="utf-8") if l.strip()]
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    w = verify(records, a.kolekcja, client=client)

    print(f"\n=== PAYLOAD: baza vs fabryka — kolekcja '{w['collection']}' ===")
    print(f"  punktów w kolekcji: {w['punktow_w_kolekcji']}   rekordów w pliku: {w['rekordow_w_pliku']}")
    print(f"  zgodnych znak w znak: {w['zgodnych']}")
    print(f"  różnych: {w['roznych']}   bez `chunk_id`: {w['bez_chunk_id']}   "
          f"sierot: {w['sierot_w_kolekcji']}   brakuje w kolekcji: {w['brakuje_w_kolekcji']}")
    for r in w["przyklady_roznic"]:
        print(f"    RÓŻNICA {r['chunk_id']}: {r['w_bazie']} != {r['z_fabryki']}")
    print(f"\n  WYNIK: {'OK — indeks zgadza się z plikiem chunków' if w['ok'] else 'NIEZGODNOŚĆ'}")

    if a.json_out:
        out = Path(a.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(w, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Zapisano -> {out}")

    sys.exit(0 if w["ok"] else 1)   # kod wyjścia, żeby dało się wołać z CI/skryptu


if __name__ == "__main__":
    main()
